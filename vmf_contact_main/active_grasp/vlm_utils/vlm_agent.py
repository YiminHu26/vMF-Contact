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
from .bbox_langsam import *
from .vlm_relation import compile_relation
from .img_preprocess import crop_max_and_rotate
from matplotlib import pyplot as plt
import time
import json

remote_api = True
if not remote_api:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
else:
    from openai import OpenAI

use_langsam = True

current_file_folder = os.path.dirname(os.path.abspath(__file__))
min_pixels = 256 * 28 * 28
max_pixels = 2560 * 28 * 28

def agent_process(img_queue, 
                  depth_queue, 
                  pose_queue, 
                  pcd_queue, 
                  relations_queue,
                  scene_objects_queue,
                  rotation_angle_queue,
                  target_object
                  ):
    agent = VLMAgent(target_object)
    while True:
        if img_queue.empty() or depth_queue.empty() or pose_queue.empty() or pcd_queue.empty():
            continue
        print("[VLM]: Processing image in agent process")
        # fetch data from queues
        img = img_queue.get()
        rotation_angle = rotation_angle_queue.get()
        depth = depth_queue.get()
        pose = pose_queue.get()
        pcd = pcd_queue.get()

        # process image
        relations, scene_objects = agent(img, depth, pose, pcd, rotation_angle)
        
        # put output in queues
        relations_queue.put(relations)
        scene_objects_queue.put(scene_objects)
        print("[VLM]: Done processing image in agent process")

class VLMAgent():
    def __init__(self, target_object = "tennis ball"):
        self.langsam_model = LangSAM(sam_type="sam2.1_hiera_large") if use_langsam else None
        if not remote_api:
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct-AWQ", min_pixels=min_pixels, max_pixels=max_pixels
            )
            self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct-AWQ", torch_dtype=torch.float16, device_map="auto"
                )
        else:
            api_key = "DASHSCOPE_API_KEY"
            self.client = OpenAI(
                api_key=os.getenv(api_key),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

        self.message_history = []
        self.image_history = []
    
    def vlm_inference(self, img, depth=None, rotation_angle=0.):
        # normalize depth
        if depth is not None:
            depth = (depth.max() - depth) / (depth.max() - depth.min() + 1e-6) * 255.
            depth = depth.astype(np.uint8)
            depth = np.clip(depth, 0, 255)
            depth = depth[:, :, None].repeat(3, axis=2)
        curr_time = time.time()

        # print("[VLM]: Message history: ", self.message_history)
        img = crop_max_and_rotate(img, rotation_angle)
        cv2.imwrite(f"{current_file_folder}/img.jpg", img[..., ::-1])
        
        if remote_api:
            self.message_history += return_analysis(img, self.key_not_detected if hasattr(self, "key_not_detected") else None)
            output_text = self.client.chat.completions.create(
                model="qwen2.5-vl-72b-instruct", 
                messages=self.message_history
            )
            output_text = output_text.choices[0].message.content
        else:
            text_prompt = return_analysis_local()  # e.g. "Please provide instructions for objects: ..."
            text = self.processor.apply_chat_template(
                text_prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs_cpu = self.processor(
                text=[text],
                images=[img, depth] if depth is not None else [img],
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.qwen.device) for k, v in inputs_cpu.items()}  # Move all tensors to the model's device
            generated_ids = self.qwen.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        self.message_history.append({ "role": "assistant", "content": [{"type": "text", "text": output_text}] })
        print(output_text)
        print(f"[VLM] Time taken: {time.time() - curr_time}")
        
        descri_langsam_dict = {}
        text_processed = output_text.split("```json")[-1].split("```")[0]
        try:
            for item in json.loads(text_processed):
                descri_langsam_dict[item["descriptors"][0] + ' ' +
                                    # item["descriptors"][1]  + ' ' +
                                    # item["descriptors"][2] + ' ' + 
                                    item["object"]] = item["descriptors"]
        except:
            print("[VLM]: Error processing output text.")
            print("[VLM]: Output text: ", text_processed)

        return descri_langsam_dict

    def generate_langsam(self, pcd, img, descri_langsam_dict):

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
        descri_langsam_key_list = list(descri_langsam_dict.keys())
        print("[VLM]: Results from VLM: ", descri_langsam_key_list)

        # predict masks with lang_sam
        results = self.langsam_model.predict([Image.fromarray(img)], [". ".join(descri_langsam_key_list)])
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

        scene_objects = {}
        vis_list = []
        # Remove previous image
        if os.path.exists(f"{current_file_folder}/"):
            for file in os.listdir(f"{current_file_folder}/"):
                if file.endswith(".jpg"):
                    os.remove(f"{current_file_folder}/{file}")

        for i, text in enumerate(labels):
            # Only the first instance of the object is considered
            if not "1" in text:
                print(f"[VLM]: Skipping {text} as it is not the first instance.")
                continue

            # mask image and point cloud
            mask = masks[i].astype(np.uint8)[:, :, None]
            pcd_masked = pcd[mask.reshape(-1) == 1]
            
            # filter out noises out of range:
            pcd_masked = pcd_masked[(pcd_masked[:, 2] > 0.02) & (pcd_masked[:, 2] < 0.3)]
            if len(pcd_masked) == 0:
                print(f"[VLM]: No valid points in the masked point cloud for {text}.")
                continue

            # find adjectives for the object from the VLM output
            adj_list = None
            for key in descri_langsam_key_list:
                if all(k.lower() in text for k in key.split(" ")):
                    adj_list = descri_langsam_dict[key]
                    break
            if adj_list is None:
                print(f"[VLM]: No adjectives found for {text}.")
                continue
            
            # compile scene object
            scene_object = compute_oriented_bounding_box(pcd_masked, label=text, adjectives=adj_list)
            scene_objects[text] = scene_object
            # print(str(scene_object))

            # remove the key from the list, the remaining keys are not detected
            descri_langsam_key_list.remove(key) 

            # visualize the scene object
            vis_list += visualize_pcd_with_obb(pcd_masked, scene_object.bbox_3d)

            # save masked image
            # cv2.imwrite(f"{current_file_folder}/{text}.jpg", img[..., ::-1] * mask)
        
        o3d.visualization.draw_geometries(vis_list)

        if len(descri_langsam_key_list) > 0:
            print("[VLM]: Objects not detected by LangSAM: ", descri_langsam_key_list)
            self.key_not_detected = descri_langsam_key_list

        return scene_objects
    
    def __call__(self, img, depth, pose, pcd_raw, rotation_angle):
        
        ## direct nbv prediction by qwen
        # direction = self.vlm_inference_nbv(img, depth)
        # return direction
        
        # view description by qwen
        descri_langsam_dict = self.vlm_inference(img, depth, rotation_angle)

        # generate semantic point cloud
        scene_objects = self.generate_langsam(pcd_raw, img, descri_langsam_dict)

        relations = compile_relation(scene_objects)

        return relations, scene_objects


