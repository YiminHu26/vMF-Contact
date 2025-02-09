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
from .vlm_utils.vlm_agent import *
from .vlm_utils.mid_perpendicular import *
import time
import torch.multiprocessing as multiprocessing
from functools import partial
multiprocessing.set_start_method('spawn', force=True)

current_file_folder = os.path.dirname(os.path.abspath(__file__))
O_SIZE = .3


class VLMPolicy(MultiViewPolicy):
    def __init__(self):
        super().__init__()
        self.max_views = 80
        self.min_gain = 10
        self.downsample = 10
        self.grasp_agent = main_module(parse_args_from_yaml(current_file_folder + "/../config.yaml"), learning=False)
        self.grasp_buffer = self.grasp_agent.grasp_buffer
        # input queues
        self.img_queue = multiprocessing.Queue(maxsize=1)
        self.depth_queue = multiprocessing.Queue(maxsize=1)
        self.pose_queue = multiprocessing.Queue(maxsize=1)
        self.pcd_queue = multiprocessing.Queue(maxsize=1)
        # output queues
        self.relations_queue = multiprocessing.Queue(maxsize=1)
        self.scene_objects_queue = multiprocessing.Queue(maxsize=1)
        # lock
        self.lock = multiprocessing.Lock()

    def activate(self, bbox, intrinsic, pcd_shift, target_object, min_z_dist):
        self.intrinsic = intrinsic
        self.bbox = bbox
        self.min_z_dist = min_z_dist
        self.view_sphere = ViewHalfSphere(bbox, min_z_dist)
        self.views = []
        self.best_grasp = None
        self.x_d = None
        self.done = False
        self.pcd_shift = pcd_shift
        self.block = False
        self.target_object = target_object
        # initialize the agent process
        self.agent_process = multiprocessing.Process(target=agent_process, args=(
            self.img_queue, 
            self.depth_queue, 
            self.pose_queue, 
            self.pcd_queue, 
            self.relations_queue,
            self.scene_objects_queue,
            target_object), daemon=True)
        self.agent_process.start()

    def update_viewsphere(self, box_center):
        box_min = [box_center[0] - O_SIZE, box_center[1] - O_SIZE, box_center[2] - O_SIZE]
        box_max = [box_center[0] + O_SIZE, box_center[1] + O_SIZE, box_center[2] + O_SIZE]
        self.bbox = AABBox(box_min, box_max)
        self.view_sphere = ViewHalfSphere(self.bbox, self.min_z_dist)
            
    def generate_view(self, img, depth, pose, pcd_raw):
        # Extract the current pose

        self.scene_objects = self.relations = None
        if self.agent_process.is_alive() and self.relations_queue.full() and self.scene_objects_queue.full():
            self.relations = self.relations_queue.get()
            self.scene_objects = self.scene_objects_queue.get()
            
            
            print(self.scene_objects.keys())
            # Find the target object
            target_obj = None
            target_obj_rels = None
            for obj_id, rels in self.relations.items():
                if all(word in obj_id for word in self.target_object.split()):
                    target_obj = self.scene_objects[obj_id]
                    target_obj_rels = self.relations[obj_id]
                    break

            if target_obj is not None:
                print(f"Target object: {target_obj.label} {target_obj.center} is found")
                if len(target_obj_rels):
                    for rel in target_obj_rels:
                        if "below" in rel:
                            obj = rel.split("below ")[1]
                            obj = self.scene_objects[obj]
                            print(f"Relation: {target_obj.label} is below {obj.label}")
                            # Rule 0: Uncovers the object above the target object
                            pcd_grasp = obj.pcd
                            
                        elif "between" in rel:
                            obj1, obj2 = rel.split("between ")[1].split(" and ")
                            obj1, obj2 = self.scene_objects[obj1], self.scene_objects[obj2]

                            # Rule 1: Grasp 1 of the objects in between relation
                            pcd_grasp = np.concatenate((obj1.pcd, obj2.pcd, target_obj.pcd), axis=0)

                        elif "to" in rel:
                            relative_pos = rel[3:].split(" to")[0]
                            obj = rel.split(" to ")[1]
                            obj = self.scene_objects[obj]
                            print(f"Relation: {relative_pos} to {obj.label}")
                            if "low" in relative_pos:

                                # Rule 2: NBV is across the perpendicular plane between the target object 
                                # and the object and towards the target object
                                self.nbv_field = partial(query_tangent_vector, 
                                                    S = self.view_sphere.center, 
                                                    R_s = self.min_z_dist, 
                                                    P1 = obj.center, 
                                                    P2 = target_obj.center)
                                pcd_grasp = target_obj.pcd
                                
                else:
                    print(f"Object {target_obj.label} has no relations")
            else:
                print(f"Target object: {self.target_object} is not found")

            
        
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
            pcd = pcd[(pcd[:, 2] > 0.025) & (pcd[:, 2] < 0.45)]

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



