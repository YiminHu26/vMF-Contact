import itertools
from numba import jit
import numpy as np

from .policy import MultiViewPolicy
from .timer import Timer
from .nbv import get_voxel_at, raycast
from .spatial import SpatialTransform, look_at, ViewHalfSphere
from .perception import UniformTSDFVolume 
from .bbox import AABBox
from vmf_contact_main.train import main_module, parse_args_from_yaml
from scipy.spatial.transform import Rotation
import os
from lang_sam import LangSAM
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import numpy as np
import glob
import open3d as o3d
from PIL import Image
from matplotlib import pyplot as plt
import time
import cv2
from .vlm_utils import return_prompt

use_langsam = True
prompt_input = "baseball"

current_file_folder = os.path.dirname(os.path.abspath(__file__))
O_SIZE = .3
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

def mark_duplicates(labels):
    label_count = {}
    result = []
    
    for label in labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
        result.append(f"{label}_{label_count[label]}")
    
    return result

class VLMPolicy(MultiViewPolicy):
    def __init__(self):
        super().__init__()
        self.min_z_dist = .3
        self.max_views = 80
        self.min_gain = 10
        self.downsample = 10
        self.langsam = LangSAM() if use_langsam else None
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct-AWQ", min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct-AWQ", torch_dtype=torch.float16, device_map="auto"
            )
        self.grasp_agent = main_module(parse_args_from_yaml(current_file_folder + "/../config.yaml"), learning=False)
        self.grasp_buffer = self.grasp_agent.grasp_buffer

    def activate(self, bbox, intrinsic, pcd_shift):
        self.intrinsic = intrinsic
        self.bbox = bbox
        self.view_sphere = ViewHalfSphere(bbox, self.min_z_dist)
        self.views = []
        self.best_grasp = None
        self.x_d = None
        self.done = False
        self.info = {}
        self.pcd_shift = pcd_shift
        self.block = False
        
        if use_langsam:
            self.target_object = prompt_input
            while self.target_object == "":
                self.target_object = input("Please enter what you would like to grasp: ")
            self.target_object = self.target_object.split(".")
            print("Prompt: ", self.target_object)
    
    
    def vlm_inference(self, img, depth, pose):
        # TODO: add vlm inference
        text = self.processor.apply_chat_template(
        return_prompt(self.target_object), tokenize=False, add_generation_prompt=True
        )
        # Tensor image to Image
        # normalize depth image
        depth = depth - depth.min()
        depth = depth / depth.max() * 255
        #triplet to single channel
        depth = np.repeat(depth[..., None], 3, axis=-1)
        
        # limit image range
        depth = Image.fromarray(np.clip(depth, 0, 255).astype(np.uint8))
        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        
        image_inputs = [img, depth]

        # image_inputs, video_inputs = process_vision_info(MESSAGES)
        inputs_cpu = self.processor(
            text=[text],
            images=image_inputs,
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
        )
        return self.process_output_text(output_text)
    
    def process_output_text(self, output_text):
        print("Output text: ", output_text)
        return None, None

    def update_viewsphere(self, box_center):
        box_min = [box_center[0] - O_SIZE, box_center[1] - O_SIZE, box_center[2] - O_SIZE]
        box_max = [box_center[0] + O_SIZE, box_center[1] + O_SIZE, box_center[2] + O_SIZE]
        self.bbox = AABBox(box_min, box_max)
        self.view_sphere = ViewHalfSphere(self.bbox, self.min_z_dist)
            
    def generate_view(self, img, depth, pose):
        # Extract the current pose

        # TODO: add vlm inference
        self.block = True
        pos_new, gaze_point_new = self.vlm_inference(img, depth, pose)
        self.block = False
        
        pos_new = pose.translation # TODO: remove this line, this is a test for gazing at middle of the desk
        gaze_point_new = [-0.74, 0.1, 0.031] # TODO: remove this line, this is a test for gazing at middle of the desk
        self.update_viewsphere(gaze_point_new) # TODO: remove this line, this is a test for gazing at middle of the desk
        
        # Compute the transformation from the current position to the gaze point
        up = np.r_[-1.0, 0.0, 0.0]
        transformation: SpatialTransform = look_at(pos_new, gaze_point_new, up)

        return transformation, 1000

    def generate_langsam(self, pcd, img):
        # Process the prompt point cloud
        time_curr = time.time()
    
        self.masked_pcd_dict = {}
        # predict masks with lang_sam
        results = self.langsam.predict([Image.fromarray(img)], [". ".join(self.target_object)])

        print(f"Time taken for inference: {time.time() - time_curr}")
        
        # check if there are labels detected
        labels = results[0]["labels"]
        if len(labels) == 0:
            print("No labels detected.")
            return pcd
        # check duplicates in labels, if there are duplicates, mark them with a number
        labels = mark_duplicates(labels)
        print("Results: ", labels)
        print("Scores: ", results[0]["scores"])

        # mask point cloud and image
        for i, text in enumerate(labels):
            mask = results[0]["masks"][i].astype(np.uint8)[:, :, None]
            # mask image and point cloud
            pcd_masked = pcd[mask.reshape(-1) == 1]
            self.masked_pcd_dict[text] = pcd_masked
            # save masked image
            # cv2.imwrite(f"{current_file_folder}/{text}.jpg", img[..., ::-1] * mask)
        pcd_from_prompt = []
        for label in self.masked_pcd_dict:
            pcd_from_prompt.append(self.masked_pcd_dict[label])
            break
        pcd_from_prompt = np.concatenate(pcd_from_prompt, axis=0)
        pcd_from_prompt = (pcd_from_prompt - self.pcd_shift)
        print("Prompt point cloud: ", pcd_from_prompt.shape)
        return pcd_from_prompt
    
    def update(self, img, depth, pcd_raw, x):
        pos, quat = [x.position.x, x.position.y, x.position.z], [x.orientation.x, x.orientation.y, x.orientation.z, x.orientation.w]
        x = SpatialTransform.from_translation(pos)
        x.rotation = Rotation.from_quat(quat)

        # Process the prompt point cloud
        # pcd_from_prompt = self.generate_langsam(pcd_raw, img) if use_langsam else None

        # TODO: add criteria for grasp execution
        # Process the point cloud
        # pcd = (pcd_raw - self.pcd_shift)
        # pcd = pcd[(pcd[:, 0] > -O_SIZE) & (pcd[:, 0] < O_SIZE)]
        # pcd = pcd[(pcd[:, 1] > -O_SIZE) & (pcd[:, 1] < O_SIZE)]
        # pcd = pcd[(pcd[:, 2] > 0.03) & (pcd[:, 2] < 0.45)]

        # print("Processed point cloud: ", pcd.shape)
            
        if len(self.views) > self.max_views or self.best_grasp_prediction_is_stable():
            # self.done = True
            pass # TODO: remove this line, this is a test for gazing at middle of the desk
        else:
            # with Timer("grasp_prediction"):
            #     self.best_grasp = self.grasp_agent.inference(pcd, 
            #                             pcd_from_prompt=pcd_from_prompt,
            #                             shift=self.pcd_shift,
            #                             graspness_th=0.6)
            with Timer("view_generation"):
                nbv, gain = self.generate_view(img, depth, x)
            # utilities = gains / np.sum(gains) - costs / np.sum(costs)
            # i = np.argmax(utilities)
            # nbv, gain = views[i], gains[i]

            print("NBV: ", nbv.translation, "Gain: ", gain)

            if gain < self.min_gain and len(self.views) > self.T:
                # self.done = True
                pass # TODO: remove this line, this is a test for gazing at middle of the desk

            self.x_d = nbv
    
    def update_grasp(self, pcd_raw):
        if not self.done and not self.block:
            # TODO: add criteria for grasp execution
            # Process the point cloud
            pcd = (pcd_raw - self.pcd_shift)
            pcd = pcd[(pcd[:, 0] > -O_SIZE) & (pcd[:, 0] < O_SIZE)]
            pcd = pcd[(pcd[:, 1] > -O_SIZE) & (pcd[:, 1] < O_SIZE)]
            pcd = pcd[(pcd[:, 2] > 0.03) & (pcd[:, 2] < 0.45)]

            # print("Processed point cloud: ", pcd.shape)
                
            # if len(self.views) > self.max_views or self.best_grasp_prediction_is_stable():
            #     self.done = True
            # else:
            with Timer("grasp_prediction"):
                self.best_grasp = self.grasp_agent.inference(pcd, 
                                        pcd_from_prompt=None,
                                        shift=self.pcd_shift,
                                        graspness_th=0.6,
                                        vis=True)
            # utilities = gains / np.sum(gains) - costs / np.sum(costs)
            # i = np.argmax(utilities)
            # nbv, gain = views[i], gains[i]

    def best_grasp_prediction_is_stable(self):
        if self.best_grasp is not None:
            # TODO: grasp criteria
            return True
        return False

    def generate_views(self, q):
        thetas = np.deg2rad([15, 30])
        phis = np.arange(8) * np.deg2rad(45)
        view_candidates = []
        for theta, phi in itertools.product(thetas, phis):
            view = self.view_sphere.get_view(theta, phi)
            # TODO: check if the view is reachable
            view_candidates.append(view)
        return view_candidates

    def cost_fn(self, view):
        return 1.0
