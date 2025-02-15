from numba import jit
import numpy as np
import os
from lang_sam import LangSAM
import torch
import os
import numpy as np
import open3d as o3d
from PIL import Image
import time
import cv2
from .vlm_prompts import *
from .img_bbox_utils import *
from .vlm_relation import compile_relation
from .vlm_prompts import crop_max_and_rotate
from matplotlib import pyplot as plt
import time
import json
import ast

from openai import OpenAI

use_langsam = True

current_file_folder = os.path.dirname(os.path.abspath(__file__))
min_pixels = 256 * 28 * 28
max_pixels = 2560 * 28 * 28

def agent_process(img_queue, 
                  depth_queue, 
                  pose_queue, 
                  pcd_queue, 
                  vlm_return_queue,
                  elevation_angle_queue,
                  rotation_angle_queue,
                  vlm_cmd_queue,
                  target_object,
                  ):
    agent = VLMAgent(target_object)
    while True:
        if img_queue.empty() or depth_queue.empty() or pose_queue.empty() or pcd_queue.empty():
            continue
        # fetch data from queues
        img = img_queue.get()
        rotation_angle = rotation_angle_queue.get()
        elevation_angle = elevation_angle_queue.get()
        depth = depth_queue.get()
        pose = pose_queue.get()
        pcd = pcd_queue.get()
        vlm_cmd = vlm_cmd_queue.get()
        # start VLM inference
        print("[VLM]: Start VLM inference with task:", vlm_cmd)
        vlm_return = agent(img, depth, pose, pcd, rotation_angle, elevation_angle, vlm_cmd)
        # put output in queues
        vlm_return_queue.put(vlm_return)
        print("[VLM]: VLM inference done with task:", vlm_cmd)


class VLMAgent():
    def __init__(self, target_object = None):
        self.langsam_model = LangSAM(sam_type="sam2.1_hiera_large") if use_langsam else None
        api_key = "DASHSCOPE_API_KEY"
        self.client = OpenAI(
            api_key=os.getenv(api_key),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.message_history = []
        self.image_history = []
        self.key_not_detected = []
        self.vlm_cmd = None
        self.img_last = None

        assert target_object is not None, "Please specify the target object."
        self.target_object = target_object
    
    def vlm_inference(self, img, depth=None, rotation_angle=0., elevation_angle=0.):
        """
        Inference using VLM model
        """
        # # normalize depth
        # if depth is not None:
        #     depth = (depth.max() - depth) / (depth.max() - depth.min() + 1e-6) * 255.
        #     depth = depth.astype(np.uint8)
        #     depth = np.clip(depth, 0, 255)
        #     depth = depth[:, :, None].repeat(3, axis=2)

        curr_time = time.time()

        # rotate image
        # print("[VLM]: Message history: ", self.message_history)
        img = crop_max_and_rotate(img, -rotation_angle)
        cv2.imwrite(f"{current_file_folder}/img.jpg", img[..., ::-1])
        
        if self.vlm_cmd == "return_prompt_scene":
            self.message_history += return_prompt_scene(img, self.target_object, self.key_not_detected)
            message = self.message_history

        elif self.vlm_cmd in ["return_prompt_guess", "return_prompt_ordered_grasp"]:
            assert len(self.scene_objects.keys()) > 0, "scene analysis should be done before"
            bbox_data = {obj_label: obj_scene.return_dict() for obj_label, obj_scene in self.scene_objects.items()}
            json_data = json_scene_data(bbox_data, rotation_angle, elevation_angle)
            
            message = return_prompt_guess(
                imgs = [self.img_last, img], 
                target_object = self.target_object, 
                json_data = json_data
                )
            
        # give the message to the model
        output_text = self.client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct", 
            messages=message
        )
        output_text = output_text.choices[0].message.content
        print("[VLM]: Output text: ", output_text)
        print(f"[VLM] Time taken: {time.time() - curr_time}")     
        
        if self.vlm_cmd == "return_prompt_scene":
            text_processed = json.loads(output_text.split("```json")[-1].split("```")[0])
            vlm_description_dict = {}
            try:
                for item in text_processed:
                    vlm_description_dict[item["descriptors"][0] + ' ' +
                                        # item["descriptors"][1]  + ' ' +
                                        # item["descriptors"][2] + ' ' + 
                                        item["object"]] = item["descriptors"]
            except:
                print("[VLM]: Error processing output text.")
                print("[VLM]: Output text: ", text_processed)
            return vlm_description_dict
        else:
            return ast.literal_eval(output_text)

    def generate_langsam(self, pcd, img, vlm_description_dict):

        def mark_duplicates(labels):
            label_count = {}
            result = []
            
            for label in labels:
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1
                result.append(f"{label} {label_count[label]}")
            return result
        
        # Process the prompt point cloud
        time_curr = time.time()
        vlm_label_list = list(vlm_description_dict.keys())
        print("[VLM]: Results from VLM: ")
        for key in vlm_label_list:
            print(f"[VLM]: - {key}: {vlm_description_dict[key]}")

        # predict masks with lang_sam
        results = self.langsam_model.predict([Image.fromarray(img)], [". ".join(vlm_label_list)])
        print(f"[LangSAM] Time taken: {time.time() - time_curr}")
                
        # check if there are labels detected
        labels = results[0]["labels"]
        if len(labels) == 0:
            print("[VLM]: No labels detected.")
            return None

        # sort labels by score
        scores = results[0]["scores"]
        labels = [label for _, label in sorted(zip(scores, labels), reverse=True, key=lambda pair: pair[0])]
        masks = [mask for _, mask in sorted(zip(scores, results[0]["masks"]), reverse=True, key=lambda pair: pair[0])]
        scores = sorted(scores, reverse=True)

        # check duplicates in labels, if there are duplicates, mark them with a number
        labels = mark_duplicates(labels)

        print("[VLM]: Results from LangSAM: ", labels)
        print("[VLM]: Scores: ", scores)

        self.scene_objects = {}
        vis_list = []

        # # remove previous image
        # if os.path.exists(f"{current_file_folder}/"):
        #     for file in os.listdir(f"{current_file_folder}/"):
        #         if file.endswith(".jpg"):
        #             os.remove(f"{current_file_folder}/{file}")

        for i, langsam_label in enumerate(labels):
            # only the first instance of the object is considered
            if not "1" in langsam_label:
                print(f"[VLM]: Skipping {langsam_label} as it is not the first instance.")
                continue

            # mask image and point cloud
            mask = masks[i].astype(np.uint8)[:, :, None]
            pcd_masked = pcd[mask.reshape(-1) == 1]
            # filter out noises out of range:
            pcd_masked = pcd_masked[(pcd_masked[:, 2] > 0.02) & (pcd_masked[:, 2] < 0.3)]
            if len(pcd_masked) == 0:
                print(f"[VLM]: No valid points in the masked point cloud for {langsam_label}.")
                continue

            # find adjectives for the object from the VLM output
            adj_list = None
            for vlm_label in vlm_label_list:
                if all(word.lower() in vlm_label.lower() for word in langsam_label[:-2].split(" ")):
                    adj_list = vlm_description_dict[vlm_label]
                    vlm_label_list.remove(vlm_label) 
                    break
            if adj_list is None:
                print(f"[VLM]: No adjectives found for {langsam_label}, set as empty list.")
            
            # compile scene object
            scene_object = SceneObject(pcd_masked, langsam_label, adj_list)
            self.scene_objects[langsam_label] = scene_object
            # print(str(scene_object))

            # visualize the scene object
            vis_list += visualize_pcd_with_obb(pcd_masked, scene_object.bbox_3d)

            # save masked image
            # cv2.imwrite(f"{current_file_folder}/{langsam_label}.jpg", img[..., ::-1] * mask)
        
        o3d.visualization.draw_geometries(vis_list)

        if len(vlm_label_list) > 0:
            print("[VLM]: Objects not detected by LangSAM: ", vlm_label_list)
            self.key_not_detected = vlm_label_list
        else:
            print("[VLM]: All objects detected by LangSAM.")
            self.key_not_detected = []
    
    def __call__(self, img, depth, pose, pcd_raw, rotation_angle, elevation_angle, vlm_cmd="scene"):
        
        self.vlm_cmd = "return_prompt_" + vlm_cmd
        
        # view description by qwen
        vlm_output = self.vlm_inference(img, depth, rotation_angle, elevation_angle)

        self.img_last = img

        if self.vlm_cmd == "return_prompt_scene":
            # generate semantic point cloud
            self.generate_langsam(pcd_raw, img, vlm_output)
            # compile relations between objects
            compile_relation(self.scene_objects.values())
            return self.scene_objects
        else:
            return vlm_output

