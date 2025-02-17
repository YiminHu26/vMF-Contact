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
    def __init__(self, target_object, pcd_shift, min_z_dist):
        super().__init__()
        self.max_views = 80
        self.score_th = 0.5
        self.grasp_agent = main_module(parse_args_from_yaml(current_file_folder + "/../config.yaml"), learning=False)
        self.grasp_buffer = self.grasp_agent.grasp_buffer
        self.target_object_final = target_object
        # initialize the agent process
        self.vlm_agent = VLMAgent(target_object=target_object)
        self.pcd_shift = pcd_shift
        self.min_z_dist = min_z_dist


    def activate(self, bbox):
        self.nbv_fields = []
        self.grasp_buffer.clear()
        self.vlm_agent.clear()

        self.bbox = bbox
        self.view_sphere = ViewHalfSphere(bbox, self.min_z_dist)
        self.views = 0
        
        self.best_grasp = None
        self.x_d = None
        
        self.target_object_curr_label = self.target_object_final
        self.target_object_curr = None
        self.done = False


    def generate_view(self, img, pose, pcd_raw, rotation_angle, elevation_angle):
        # start VLM inference
        self.vlm_agent(img, 
                        pose, 
                        pcd_raw, 
                        rotation_angle, 
                        elevation_angle, 
                        self.target_object_curr_label)

        self.compile_next_view(pose)
    

    def compile_next_view(self, pose):
        if self.word_based_matching(self.target_object_curr_label) is not None:
            while True:
                # find the target object and its relations
                print(f"[Policy]: Analyzing current target object: {self.target_object_curr_label} ...")
                # match the target object label with the corresponding scene object
                self.target_object_curr: SceneObject = self.word_based_matching(self.target_object_curr_label)
                if self.compile_related_objects(pose) is None:
                    break                
            print(f"[Policy]: Target object to grasp: {self.target_object_curr.label} ...")
            self.visualize_nbv_fields(pose.translation)
            self.vlm_agent.set_vlm_cmd("scene")
        else:
            print(f"[Policy]: Target object {self.target_object_curr_label} is not found, start guessing ...")
            self.vlm_agent.set_vlm_cmd("guess")

        # new gaze point on the target object
        # if self.target_object_curr is not None:
        #     gaze_point_new = self.target_object_curr.center
        # else:
        #     gaze_point_new = self.view_sphere.center
        # self.update_viewsphere(gaze_point_new)
    

    def compile_related_objects(self, pose):
        self.nbv_fields = []
        self.nbv_occ = []
        current_cam_pos = np.array(pose.translation)
        # Decision making based on the relations
        if len(self.target_object_curr.relations) > 0:
            for rel in self.target_object_curr.relations:
                if "below" in rel:
                    obj = rel.split("below ")[1]
                    obj:SceneObject = self.scene_objects[obj]
                    print(f"[Policy]: Relation: below {obj.label}")

                    # Rule 0: Uncovers the object above the target object
                    print(f"[Policy]: Before grasping the object: {self.target_object_curr.label}, object: {obj.label} needs to be grasped")
                    # update the target object
                    self.target_object_curr = obj
                    self.target_object_curr_label = obj.label
                    return True
                    
                elif "between" in rel:
            
                    # extract the objects from the relation
                    obj1, obj2 = rel.split("between ")[1].split(" and ")
                    obj1, obj2 = self.scene_objects[obj1], self.scene_objects[obj2]
                    print(f"[Policy]: Relation: between {obj1.label} and {obj2.label}")

                    # distance to the camera
                    cam_obj1_dist = np.linalg.norm(current_cam_pos - obj1.center)
                    cam_obj2_dist = np.linalg.norm(current_cam_pos - obj2.center)
                    cam_target_object_dist = np.linalg.norm(current_cam_pos - self.target_object_curr.center)
                    
                    # whether low to both objects
                    obj_1_high = any(f"low to {obj1.label}" in rel for rel in self.target_object_curr.relations)
                    obj_2_high = any(f"low to {obj2.label}" in rel for rel in self.target_object_curr.relations)

                    # Rule 1: Uncovers the close object in front of camera if both objects are high
                    if obj_1_high and obj_2_high:
                        print(f"[Policy]: Before grasping the object: {self.target_object_curr.label}, object: {obj.label} needs to be grasped")
                        # update the target object
                        if cam_obj1_dist < cam_obj2_dist:
                            self.target_object_curr=obj1
                            self.target_object_curr_label=obj1.label
                        else:
                            self.target_object_curr=obj2
                            self.target_object_curr_label=obj2.label
                        return True

                    # Rule 2: NBV is across the perpendicular plane between the target object and the high object
                    elif not obj_2_high and obj_1_high:
                        if cam_obj2_dist < cam_target_object_dist:
                            self.nbv_occ.append(obj2.center)
                            self.nbv_fields.append(
                                query_tangent_vector(
                                S = self.view_sphere.center,
                                R_s = self.min_z_dist,
                                P1 = obj2.center,
                                P2 = self.target_object_curr.center,)
                                )
                    elif not obj_2_high and obj_1_high:
                        if cam_obj1_dist < cam_target_object_dist:
                            self.nbv_occ.append(obj1.center)
                            self.nbv_fields.append(
                                partial(
                                query_tangent_vector,
                                S = self.view_sphere.center,
                                R_s = self.min_z_dist,
                                P1 = obj1.center,
                                P2 = self.target_object_curr.center
                                ))
                            
                elif "to" in rel:
                    relative_pos = rel[3:].split(" to")[0]
                    obj = rel.split(" to ")[1]
                    obj = self.scene_objects[obj]
                    print(f"[Policy]: Relation: {relative_pos} to {obj.label}")
                    if not "high" in relative_pos:

                        # Rule 3: NBV is across the perpendicular plane between the target object 
                        # and the object and towards the target object
                        self.nbv_occ.append(obj.center)
                        self.nbv_fields.append(
                            partial(
                            query_tangent_vector,
                            S = self.view_sphere.center,
                            R_s = self.min_z_dist,
                            P1 = obj.center,
                            P2 = self.target_object_curr.center
                            ))
        else:
            print(f"[Policy]: Object {self.target_object_curr.label} has no relations")

    def visualize_nbv_fields(self, current_cam_pos):
        if True and len(self.nbv_fields) > 0:
            current_cam_pos = np.array(current_cam_pos)
            # visualize the NBV fields
            animate_query_tangent_vector(self.view_sphere.center, 
                                        self.min_z_dist, 
                                        current_cam_pos,
                                        self.target_object_curr.center, 
                                        self.nbv_occ, 
                                        self.nbv_fields)
    

    def word_based_matching(self, target_object_curr_label):
        for obj_id in self.scene_objects.keys():
            if all(word in obj_id for word in target_object_curr_label.split()):
                target_obj = self.scene_objects[obj_id]
                # print(f"[Policy]: Target object: {target_obj.label} at {target_obj.center} is found")
                return target_obj
        print(f"[Policy]: Object {target_object_curr_label} is not found")
        return None
    

    def update(self, img, depth, pcd_raw, x, rotation_angle, elevation_angle):
        x = self.translate_pose(x)

        if self.views > self.max_views or self.best_grasp_prediction_is_stable():
            self.done = True
        else:
            with Timer("view_generation"):
                self.generate_view(img, x, pcd_raw, rotation_angle, elevation_angle)
    

    def update_grasp(self, pcd_raw, use_normal_vis=False):
        if not self.done and pcd_raw.shape[0] > 0:
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
        if self.target_object_curr is not None or len(self.ordered_grasp_list):
            for target_object_curr in self.ordered_grasp_list + [self.target_object_curr]:
                # get the current pcd of the target object
                pcd_from_prompt=target_object_curr.pcd
                
                # get the best grasp prediction on the target object
                poses, kappa, graspness = self.grasp_buffer.get_pose_fused(pcd_from_prompt=pcd_from_prompt)

                if len(poses) > 0:
                    break

                print(f"[vMF-Contact]: No grasp prediction on object: {target_object_curr.label}, even if it's found.")
            
            if len(poses) == 0:
                self.vlm_agent.set_vlm_cmd("ordered_grasp")
                return False
            
            # update the target object which has valid grasp prediction
            self.target_object_curr_label = target_object_curr.label
            self.target_object_curr = target_object_curr
                            
            # sort by kappa or graspness
            score = kappa if sort_by == "kappa" else graspness
    
            #sort poses by criterion
            sample_num = min(sample_num, poses.size(0))
            poses_candidates = poses[torch.argsort(score, descending=True)][:sample_num]

            #randomly sample 1 poses
            pose_chosen = poses_candidates[random.randint(0, sample_num-1)].squeeze(0)
            self.best_grasp = pose_chosen

            print(f"[vMF-Contact]: Best grasp prediction on object: {target_object_curr.label}, identified.")

            return True
        print(f"[vMF-Contact]: No target object found on object: {self.target_object_curr_label}")
        return False
    
    def update_viewsphere(self, box_center):
        box_min = [box_center[0] - O_SIZE, box_center[1] - O_SIZE, box_center[2] - O_SIZE]
        box_max = [box_center[0] + O_SIZE, box_center[1] + O_SIZE, box_center[2] + O_SIZE]
        self.bbox = AABBox(box_min, box_max)
        self.view_sphere = ViewHalfSphere(self.bbox, self.min_z_dist)
    

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
    

    def denoise_pcd(self, pcd):
        pcd = pcd - self.pcd_shift
        pcd = pcd[(pcd[:, 0] > -O_SIZE) & (pcd[:, 0] < O_SIZE)]
        pcd = pcd[(pcd[:, 1] > -O_SIZE) & (pcd[:, 1] < O_SIZE)]
        pcd = pcd[(pcd[:, 2] > 0.025) & (pcd[:, 2] < 0.45)]
        # pcd = denoise_point_cloud(pcd, method="statistical", nb_neighb.35ors=20, std_ratio=0.1)
        # pcd_vis = o3d.geometry.PointCloud()
        # pcd_vis.points = o3d.utility.Vector3dVector(pcd)
        # o3d.visualization.draw_geometries([pcd_vis])
        return pcd
    
    def query_field_fusion_from_list(self, translation):
        assert len(self.nbv_fields) > 0, "No NBV fields to query."
        translation = np.array(translation)
        return query_tangent_vector_sum_from_field_list(S=self.view_sphere.center,
                                                        P_q = translation,
                                                        field_list=self.nbv_fields)
    
    @property
    def scene_objects(self):
        return self.vlm_agent.scene_objects
    
    @property
    def ordered_grasp_list(self) -> list[SceneObject]:
        return self.vlm_agent.ordered_grasp_list


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


