import itertools
import numpy as np
from .policy import MultiViewPolicy
from .timer import Timer
from .nbv import get_voxel_at, raycast
from .spatial import SpatialTransform, look_at, ViewHalfSphere
from .bbox import AABBox
from vmf_contact_main.train import main_module, parse_args_from_yaml
from scipy.spatial.transform import Rotation
import os
from .vlm_utils.vlm_agent import VLMAgent 
import os
import numpy as np
import time
from .vlm_utils.vlm_prompts import *
import time
import torch.multiprocessing as multiprocessing
multiprocessing.set_start_method('spawn', force=True)

current_file_folder = os.path.dirname(os.path.abspath(__file__))
O_SIZE = .3

def agent_process(img_queue, depth_queue, pose_queue, pcd_queue, direction_queue):
    agent = VLMAgent()
    while True:
        if img_queue.empty() or depth_queue.empty() or pose_queue.empty() or pcd_queue.empty():
            continue
        print("[VLM]: Processing image in agent process")
        img = img_queue.get()
        depth = depth_queue.get()
        pose = pose_queue.get()
        pcd = pcd_queue.get()
        direction_queue.put(agent(img, depth, pose, pcd))
        print("[VLM]: Done processing image in agent process")


class VLMPolicy(MultiViewPolicy):
    def __init__(self):
        super().__init__()
        self.min_z_dist = .3
        self.max_views = 80
        self.min_gain = 10
        self.downsample = 10
        # spawn agent process
        self.img_queue = multiprocessing.Queue(maxsize=2)
        self.depth_queue = multiprocessing.Queue(maxsize=2)
        self.pose_queue = multiprocessing.Queue(maxsize=2)
        self.pcd_queue = multiprocessing.Queue(maxsize=2)
        self.direction_queue = multiprocessing.Queue(maxsize=2)
        self.agent_process = multiprocessing.Process(target=agent_process, args=(
            self.img_queue, self.depth_queue, self.pose_queue, self.pcd_queue, self.direction_queue), daemon=True)

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
        self.pcd_shift = pcd_shift
        self.block = False
        self.agent_process.start()

    def update_viewsphere(self, box_center):
        box_min = [box_center[0] - O_SIZE, box_center[1] - O_SIZE, box_center[2] - O_SIZE]
        box_max = [box_center[0] + O_SIZE, box_center[1] + O_SIZE, box_center[2] + O_SIZE]
        self.bbox = AABBox(box_min, box_max)
        self.view_sphere = ViewHalfSphere(self.bbox, self.min_z_dist)
            
    def generate_view(self, img, depth, pose, pcd_raw):
        # Extract the current pose

        if self.agent_process.is_alive() and not self.img_queue.full() and not self.depth_queue.full() and not self.pose_queue.full() and not self.pcd_queue.full():
            self.img_queue.put(img)
            self.depth_queue.put(depth)
            self.pose_queue.put(pose)
            self.pcd_queue.put(pcd_raw)
        
        pos_new = pose.translation
        gaze_point_new = [-0.74, 0.1, 0.01]

        self.update_viewsphere(gaze_point_new) # TODO: remove this line, this is a test for gazing at middle of the desk
        
        # Compute the transformation from the current position to the gaze point
        up = np.r_[-1.0, 0.0, 0.0]
        transformation: SpatialTransform = look_at(pos_new, gaze_point_new, up)

        return transformation, 1000
    
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
                nbv, gain = self.generate_view(img, depth, x, pcd_raw)
            # utilities = gains / np.sum(gains) - costs / np.sum(costs)
            # i = np.argmax(utilities)
            # nbv, gain = views[i], gains[i]

            # print("NBV: ", nbv.translation, "Gain: ", gain)

            if gain < self.min_gain and len(self.views) > self.T:
                # self.done = True
                pass # TODO: remove this line, this is a test for gazing at middle of the desk

            self.x_d = nbv
    
    def update_grasp(self, pcd_raw, use_normal_vis=False):
        if not self.done: # and not self.block:
            # TODO: add criteria for grasp execution
            # Process the point cloud
            pcd = (pcd_raw - self.pcd_shift)
            pcd = pcd[(pcd[:, 0] > -O_SIZE) & (pcd[:, 0] < O_SIZE)]
            pcd = pcd[(pcd[:, 1] > -O_SIZE) & (pcd[:, 1] < O_SIZE)]
            pcd = pcd[(pcd[:, 2] > 0.02) & (pcd[:, 2] < 0.45)]

            # print("Processed point cloud: ", pcd.shape)
                
            # if len(self.views) > self.max_views or self.best_grasp_prediction_is_stable():
            #     self.done = True
            # else:
            with Timer("grasp_prediction"):
                time_curr = time.time()
                self.curr_grasp = self.grasp_agent.inference(pcd, 
                                        pcd_from_prompt=None,
                                        shift=self.pcd_shift,
                                        graspness_th=0.5,
                                        vis=True,
                                        use_normal_vis=use_normal_vis)
                print(f"[vMF-Contact] Time taken for grasp inference: {time.time() - time_curr}")
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



