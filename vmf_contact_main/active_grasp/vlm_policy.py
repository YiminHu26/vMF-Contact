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
import numpy as np
import time
from .vlm_utils.vlm_agent import *
from .vlm_utils.nbv_generator import *
import time
import torch.multiprocessing as multiprocessing
from functools import partial
import open3d as o3d
import random
multiprocessing.set_start_method('spawn', force=True)

current_file_folder = os.path.dirname(os.path.abspath(__file__))
O_SIZE = .3


class VLMPolicy(MultiViewPolicy):
    def __init__(self, target_object):
        super().__init__()
        self.max_views = 80
        self.score_th = 0.5
        self.grasp_agent = main_module(parse_args_from_yaml(current_file_folder + "/../config.yaml"), learning=False)
        self.grasp_buffer = self.grasp_agent.grasp_buffer
        self.target_objects_label = [target_object]
        # input queues
        self.img_queue = multiprocessing.Queue(maxsize=1)
        self.depth_queue = multiprocessing.Queue(maxsize=1)
        self.pose_queue = multiprocessing.Queue(maxsize=1)
        self.pcd_queue = multiprocessing.Queue(maxsize=1)
        self.elevation_angle_queue = multiprocessing.Queue(maxsize=1)
        self.rotation_angle_queue = multiprocessing.Queue(maxsize=1)
        self.vlm_cmd_queue = multiprocessing.Queue(maxsize=1)
        # output queues
        self.vlm_return_queue = multiprocessing.Queue(maxsize=1)
        # lock
        self.lock = multiprocessing.Lock()

    def activate(self, bbox, pcd_shift, target_object, min_z_dist):
        self.nbv_fields = []
        self.vlm_cmd = "scene"
        self.scene_objects = None
        self.bbox = bbox
        self.min_z_dist = min_z_dist
        self.view_sphere = ViewHalfSphere(bbox, min_z_dist)
        self.views = []
        self.best_grasp = None
        self.x_d = None
        self.done = False
        self.pcd_shift = pcd_shift
        self.target_objects = {}
        # initialize the agent process
        self.agent_process = multiprocessing.Process(target=agent_process, args=(
            self.img_queue, 
            self.depth_queue, 
            self.pose_queue, 
            self.pcd_queue, 
            self.vlm_return_queue,
            self.elevation_angle_queue,
            self.rotation_angle_queue,
            self.vlm_cmd_queue,
            target_object,
            ), daemon=True)
        self.agent_process.start()
        # output

    def update_viewsphere(self, box_center):
        box_min = [box_center[0] - O_SIZE, box_center[1] - O_SIZE, box_center[2] - O_SIZE]
        box_max = [box_center[0] + O_SIZE, box_center[1] + O_SIZE, box_center[2] + O_SIZE]
        self.bbox = AABBox(box_min, box_max)
        self.view_sphere = ViewHalfSphere(self.bbox, self.min_z_dist)
            

    def generate_view(self, img, depth, pose, pcd_raw, rotation_angle, elevation_angle):

        while not self.agent_process.is_alive() \
            and self.img_queue.empty() \
            and self.depth_queue.empty() \
            and self.pose_queue.empty() \
            and self.pcd_queue.empty() \
            and self.rotation_angle_queue.empty()\
            and self.vlm_cmd_queue.empty():
            continue
            
        self.img_queue.put(img)
        self.elevation_angle_queue.put(elevation_angle)
        self.rotation_angle_queue.put(rotation_angle)
        self.depth_queue.put(depth)
        self.pose_queue.put(pose)
        self.pcd_queue.put(pcd_raw)
        self.vlm_cmd_queue.put(self.vlm_cmd)    

        while not self.agent_process.is_alive() and self.vlm_return_queue.full():
            continue
            
        vlm_return = self.vlm_return_queue.get()
        if isinstance(vlm_return, dict):
            # update scene objects
            self.scene_objects = vlm_return
        elif isinstance(vlm_return, list):
            # ordered grasp for decluttering
            vlm_return.reverse()
            self.target_objects_label += vlm_return
            self.target_objects += [self.scene_objects[obj_label] for obj_label in vlm_return]
        self.compile_next_view(pose)
        
        # # Compute the transformation from the current position to the gaze point
        # up = np.r_[-1.0, 0.0, 0.0]
        # pos_new = pose.translation # TODO: according to nbv_field to change to new view
        # transformation: SpatialTransform = look_at(pos_new, gaze_point_new, up)

        # return transformation
    
    def compile_next_view(self, pose):
        target_obj_label = self.target_objects_label[-1]
        if self.word_based_matching(target_obj_label) is not None:
            while True:
                # find the target object and its relations
                print(f"[Policy]: Analyzing current target object: {target_obj_label} ...")
                target_obj: SceneObject = self.word_based_matching(target_obj_label)
                target_obj_label = self.compile_related_objects(pose, target_obj)
                if target_obj_label is None:
                    break                
            print(f"[Policy]: Target object to grasp: {self.target_objects_label[-1]} ...")
            self.vlm_cmd = "scene"
        else:
            print(f"[Policy]: Target object {target_obj_label} is not found, start guessing ...")
            self.vlm_cmd = "guess"

        # new gaze point on the target object
        if self.target_objects_label[-1] in self.target_objects.keys():
            gaze_point_new = self.target_objects[self.target_objects_label[-1]].center
        else:
            gaze_point_new = self.view_sphere.center
        self.update_viewsphere(gaze_point_new)
    
    def compile_related_objects(self, pose, target_obj:SceneObject):

        # Decision making based on the relations
        if len(target_obj.relations) > 0:
            for rel in target_obj.relations:
                if "below" in rel:
                    obj = rel.split("below ")[1]
                    obj:SceneObject = self.scene_objects[obj]
                    print(f"[Policy]: Relation: below {obj.label}")

                    # Rule 0: Uncovers the object above the target object
                    self.target_objects_label.append(obj.label)
                    self.target_objects[obj.label] = obj
                    print(f"[Policy]: Before grasping the object: {target_obj.label}, object: {obj.label} needs to be grasped")
                    return obj.label
                    
                elif "between" in rel:
                    current_cam_pos = np.array(pose.translation)
            
                    # extract the objects from the relation
                    obj1, obj2 = rel.split("between ")[1].split(" and ")
                    obj1, obj2 = self.scene_objects[obj1], self.scene_objects[obj2]

                    # distance to the camera
                    cam_obj1_dist = np.linalg.norm(current_cam_pos - obj1.center)
                    cam_obj2_dist = np.linalg.norm(current_cam_pos - obj2.center)
                    cam_target_obj_dist = np.linalg.norm(current_cam_pos - target_obj.center)
                    
                    # whether low to both objects
                    obj_1_low = any(f"high to {obj1.label}" in rel for rel in target_obj.relations)
                    obj_2_low = any(f"high to {obj2.label}" in rel for rel in target_obj.relations)

                    # Rule 1: Uncovers the close object in front of camera if both objects are high
                    if not obj_1_low and not obj_2_low:
                        if cam_obj1_dist < cam_obj2_dist:
                            self.target_objects_label.append(obj1.label)
                            self.target_objects[obj1.label] = obj1
                        else:
                            self.target_objects_label.append(obj2.label)
                            self.target_objects[obj2.label] = obj2
                        print(f"[Policy]: Before grasping the object: {target_obj.label}, object: {obj.label} needs to be grasped")
                        return obj.label

                    # Rule 2: NBV is across the perpendicular plane between the target object and the high object
                    elif not obj_2_low and obj_1_low:
                        if cam_obj2_dist < cam_target_obj_dist:
                            self.nbv_fields.append(
                                partial(
                                query_tangent_vector,
                                S = self.view_sphere.center,
                                R_s = self.min_z_dist,
                                P1 = obj2.center,
                                P2 = target_obj.center
                                ))
                    elif obj_2_low and not obj_1_low:
                        if cam_obj1_dist < cam_target_obj_dist:
                            self.nbv_fields.append(
                                partial(
                                query_tangent_vector,
                                S = self.view_sphere.center,
                                R_s = self.min_z_dist,
                                P1 = obj1.center,
                                P2 = target_obj.center
                                ))
                            
                elif "to" in rel:
                    relative_pos = rel[3:].split(" to")[0]
                    obj = rel.split(" to ")[1]
                    obj = self.scene_objects[obj]
                    print(f"[Policy]: Relation: {relative_pos} to {obj.label}")
                    if not "high" in relative_pos:

                        # Rule 3: NBV is across the perpendicular plane between the target object 
                        # and the object and towards the target object
                        self.nbv_fields.append(
                            partial(
                            query_tangent_vector,
                            S = self.view_sphere.center,
                            R_s = self.min_z_dist,
                            P1 = obj.center,
                            P2 = target_obj.center
                            ))                   
        else:
            print(f"[Policy]: Object {target_obj.label} has no relations")
        return None
    
    def word_based_matching(self, target_obj_label):
        for obj_id in self.scene_objects.keys():
            if all(word in obj_id for word in target_obj_label.split()):
                target_obj = self.scene_objects[obj_id]
                # print(f"[Policy]: Target object: {target_obj.label} at {target_obj.center} is found")
                return target_obj
        print(f"[Policy]: Object {target_obj_label} is not found")
        return None

    def denoise_pcd(self, pcd):
        pcd = pcd - self.pcd_shift
        pcd = pcd[(pcd[:, 0] > -O_SIZE) & (pcd[:, 0] < O_SIZE)]
        pcd = pcd[(pcd[:, 1] > -O_SIZE) & (pcd[:, 1] < O_SIZE)]
        pcd = pcd[(pcd[:, 2] > 0.025) & (pcd[:, 2] < 0.45)]
        # pcd = denoise_point_cloud(pcd, method="statistical", nb_neighbors=20, std_ratio=0.1)
        # pcd_vis = o3d.geometry.PointCloud()
        # pcd_vis.points = o3d.utility.Vector3dVector(pcd)
        # o3d.visualization.draw_geometries([pcd_vis])
        return pcd
    
    
    def update(self, img, depth, pcd_raw, x, rotation_angle, elevation_angle):
        x = self.translate_pose(x)

        if len(self.views) > self.max_views or self.best_grasp_prediction_is_stable():
            self.done = True
        else:
            with Timer("view_generation"):
                self.generate_view(img, depth, x, pcd_raw, rotation_angle, elevation_angle)
    

    def update_grasp(self, pcd_raw, use_normal_vis=False):
        if not self.done:
            pcd = self.denoise_pcd(pcd_raw)

            # print("Processed point cloud: ", pcd.shape)
                
            with Timer("grasp_prediction"):
                time_curr = time.time()
                self.curr_grasp = self.grasp_agent.inference(pcd, 
                                        pcd_from_prompt=None,
                                        shift=self.pcd_shift,
                                        graspness_th=0.5,
                                        vis=True,
                                        use_normal_vis=use_normal_vis)
                # print(f"[vMF-Contact] Time taken for grasp inference: {time.time() - time_curr}")


    def best_grasp_prediction_is_stable(self, sort_by="kappa", sample_num=1):
        if self.target_objects_label[-1] in self.target_objects.keys():
            # get the current pcd of the target object
            pcd_from_prompt=self.target_objects[self.target_objects_label[-1]].pcd
            
            # get the best grasp prediction on the target object
            poses, kappa, graspness = self.grasp_buffer.get_pose_fused(pcd_from_prompt=pcd_from_prompt)

            print(f"[vMF-Contact]: {poses}, {kappa}, {graspness}")
            
            # sort by kappa or graspness
            score = kappa if sort_by == "kappa" else graspness

            # check if the score is empty or below the threshold
            if score.size(0) == 0 or score.max() < self.score_th:
                print(f"[vMF-Contact]: The best grasp prediction is not stable with score:", score.max())
                return False
    
            #sort poses by criterion
            sample_num = min(sample_num, poses.size(0))
            poses_candidates = poses[torch.argsort(score, descending=True)][:sample_num]

            #randomly sample 1 poses
            pose_chosen = poses_candidates[random.randint(0, sample_num-1)].squeeze(0)
            self.best_grasp = pose_chosen

            print(f"[vMF-Contact]: Best grasp prediction: {pose_chosen} on object {self.target_objects_label[-1]}")
            return True
        return False


    def remove_current_target(self):
        self.done = False
        if len(self.target_objects_label) > 1:
            self.target_objects_label.pop()
            self.target_objects.pop(self.target_objects_label[-1])
            return False
        else:
            self.target_objects_label = []
            self.target_objects = {}
            return True
        
    def translate_pose(self, x):
        pos, quat = ([x.position.x, 
                      x.position.y, 
                      x.position.z], 
                      [x.orientation.x, 
                       x.orientation.y, 
                       x.orientation.z, 
                       x.orientation.w])
        x = SpatialTransform.from_translation(pos)
        x.rotation = Rotation.from_quat(quat)
        return x


def denoise_point_cloud(points, method="statistical", nb_neighbors=20, std_ratio=2.0, radius=0.05, min_neighbors=16):
    """
    Denoises a point cloud given as a NumPy array.

    Parameters:
    - points (numpy.ndarray): Nx3 array representing point cloud coordinates.
    - method (str): "statistical" for statistical outlier removal, "radius" for radius outlier removal.
    - nb_neighbors (int): Number of neighbors for statistical outlier removal.
    - std_ratio (float): Standard deviation ratio for statistical outlier removal.
    - radius (float): Radius for radius outlier removal.
    - min_neighbors (int): Minimum number of neighbors for radius outlier removal.

    Returns:
    - numpy.ndarray: Denoised point cloud.
    """
    # Convert NumPy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Apply selected denoising method
    if method == "statistical":
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    elif method == "radius":
        pcd_clean, ind = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    else:
        raise ValueError("Invalid method. Choose 'statistical' or 'radius'.")

    # Convert back to NumPy array
    return np.asarray(pcd_clean.points)


