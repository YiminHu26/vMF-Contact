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
    def __init__(self):
        super().__init__()
        self.max_views = 80
        self.score_th = 0.5
        self.grasp_agent = main_module(parse_args_from_yaml(current_file_folder + "/../config.yaml"), learning=False)
        self.grasp_buffer = self.grasp_agent.grasp_buffer
        # input queues
        self.img_queue = multiprocessing.Queue(maxsize=1)
        self.depth_queue = multiprocessing.Queue(maxsize=1)
        self.pose_queue = multiprocessing.Queue(maxsize=1)
        self.pcd_queue = multiprocessing.Queue(maxsize=1)
        self.rotation_angle_queue = multiprocessing.Queue(maxsize=1)
        # output queues
        self.relations_queue = multiprocessing.Queue(maxsize=1)
        self.scene_objects_queue = multiprocessing.Queue(maxsize=1)
        # lock
        self.lock = multiprocessing.Lock()

    def activate(self, bbox, pcd_shift, target_object, min_z_dist):
        self.bbox = bbox
        self.min_z_dist = min_z_dist
        self.view_sphere = ViewHalfSphere(bbox, min_z_dist)
        self.views = []
        self.best_grasp = None
        self.x_d = None
        self.done = False
        self.pcd_shift = pcd_shift
        self.target_objects_label = [target_object]
        self.target_objects = {}
        # initialize the agent process
        self.agent_process = multiprocessing.Process(target=agent_process, args=(
            self.img_queue, 
            self.depth_queue, 
            self.pose_queue, 
            self.pcd_queue, 
            self.relations_queue,
            self.scene_objects_queue,
            self.rotation_angle_queue,
            target_object), daemon=True)
        self.agent_process.start()
        # output
        self.nbv_field = None


    def update_viewsphere(self, box_center):
        box_min = [box_center[0] - O_SIZE, box_center[1] - O_SIZE, box_center[2] - O_SIZE]
        box_max = [box_center[0] + O_SIZE, box_center[1] + O_SIZE, box_center[2] + O_SIZE]
        self.bbox = AABBox(box_min, box_max)
        self.view_sphere = ViewHalfSphere(self.bbox, self.min_z_dist)
            

    def generate_view(self, img, depth, pose, pcd_raw, rotation_angle):
        # Extract the current pose

        self.scene_objects = self.relations = None
        if self.agent_process.is_alive() \
                and self.relations_queue.full() \
                and self.scene_objects_queue.full():
            
            self.relations = self.relations_queue.get()
            self.scene_objects = self.scene_objects_queue.get()
            self.generate_next_view(pose)
        
        if self.agent_process.is_alive() \
                and not self.img_queue.full() \
                and not self.depth_queue.full() \
                and not self.pose_queue.full() \
                and not self.pcd_queue.full() \
                and not self.rotation_angle_queue.full():
            
            self.img_queue.put(img)
            self.rotation_angle_queue.put(rotation_angle)
            self.depth_queue.put(depth)
            self.pose_queue.put(pose)
            self.pcd_queue.put(pcd_raw)            

        # new gaze point on the target object
        if self.target_objects_label[-1] in self.target_objects.keys():
            gaze_point_new = self.target_objects[self.target_objects_label[-1]].center
        else:
            gaze_point_new = self.view_sphere.center
            self.update_viewsphere(gaze_point_new)
        
        # # Compute the transformation from the current position to the gaze point
        # up = np.r_[-1.0, 0.0, 0.0]
        # pos_new = pose.translation # TODO: according to nbv_field to change to new view
        # transformation: SpatialTransform = look_at(pos_new, gaze_point_new, up)

        # return transformation
    

    def generate_next_view(self, pose):
        target_obj_label = self.target_objects_label[-1]
        while target_obj_label is not None:
            print(f"Current target object: {target_obj_label}")
            target_obj_label_old = target_obj_label
            target_obj_label = self.compile_related_objects(pose, target_obj_label)
            if target_obj_label is not None:
                print(f"Before grasping the object: {target_obj_label_old}, object: {target_obj_label} needs to be grasped")

    
    def compile_related_objects(self, pose, target_obj_label):
        # Find the target object
        target_obj = None
        target_obj_rels = None

        # find the target object and its relations
        for obj_id in self.relations.keys():
            if all(word in obj_id for word in target_obj_label.split()):
                target_obj = self.scene_objects[obj_id]
                target_obj_rels = self.relations[obj_id]
                break

        # Decision making based on the relations
        if target_obj is not None:
            current_cam_pos = np.array(pose.translation)
            print(f"Target object: {target_obj.label} {target_obj.center} is found")
            if len(target_obj_rels):
                for rel in target_obj_rels:
                    if "below" in rel:
                        obj = rel.split("below ")[1]
                        obj = self.scene_objects[obj]
                        print(f"Relation: below {obj.label}")

                        # Rule 0: Uncovers the object above the target object
                        self.target_objects_label.append(obj.label)
                        self.target_objects[obj.label] = obj
                        return obj.label
                        
                    elif "between" in rel:
                        obj1, obj2 = rel.split("between ")[1].split(" and ")
                        obj1, obj2 = self.scene_objects[obj1], self.scene_objects[obj2]

                        # distance to the camera
                        cam_obj1_dist = np.linalg.norm(current_cam_pos - obj1.center)
                        cam_obj2_dist = np.linalg.norm(current_cam_pos - obj2.center)
                        cam_target_obj_dist = np.linalg.norm(current_cam_pos - target_obj.center)
                        
                        # whether low to both objects
                        obj_1_low = any(f"high to {obj1.label}" in rel for rel in target_obj_rels)
                        obj_2_low = any(f"high to {obj2.label}" in rel for rel in target_obj_rels)

                        # Rule 1: Uncovers the close object in front of camera if both objects are high
                        if not obj_1_low and not obj_2_low:
                            if cam_obj1_dist < cam_obj2_dist:
                                self.target_objects_label.append(obj1.label)
                                self.target_objects[obj1.label] = obj1
                            else:
                                self.target_objects_label.append(obj2.label)
                                self.target_objects[obj2.label] = obj2
                            return obj.label

                        # Rule 2: NBV is across the perpendicular plane between the target object and the high object
                        elif not obj_2_low and obj_1_low:
                            if cam_obj2_dist < cam_target_obj_dist:
                                self.nbv_field = partial(
                                    query_tangent_vector,
                                    S = self.view_sphere.center,
                                    R_s = self.min_z_dist,
                                    P1 = obj2.center,
                                    P2 = target_obj.center
                                    )
                        elif obj_2_low and not obj_1_low:
                            if cam_obj1_dist < cam_target_obj_dist:
                                self.nbv_field = partial(
                                    query_tangent_vector,
                                    S = self.view_sphere.center,
                                    R_s = self.min_z_dist,
                                    P1 = obj1.center,
                                    P2 = target_obj.center
                                    )
                                
                    elif "to" in rel:
                        relative_pos = rel[3:].split(" to")[0]
                        obj = rel.split(" to ")[1]
                        obj = self.scene_objects[obj]
                        print(f"Relation: {relative_pos} to {obj.label}")
                        if not "high" in relative_pos:

                            # Rule 3: NBV is across the perpendicular plane between the target object 
                            # and the object and towards the target object
                            self.nbv_field = partial(
                                query_tangent_vector,
                                S = self.view_sphere.center,
                                R_s = self.min_z_dist,
                                P1 = obj.center,
                                P2 = target_obj.center
                                )                   
            else:
                print(f"Object {target_obj.label} has no relations")
            return None
        else:
            ###TODO: VLM guess where the object is occluded
            print(f"Target object: {self.target_objects_label[0]} is not found")
    

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
    
    
    def update(self, img, depth, pcd_raw, x, rotation_angle):
        pos, quat = [x.position.x, x.position.y, x.position.z], [x.orientation.x, x.orientation.y, x.orientation.z, x.orientation.w]
        x = SpatialTransform.from_translation(pos)
        x.rotation = Rotation.from_quat(quat)
            
        if len(self.views) > self.max_views or self.best_grasp_prediction_is_stable():
            self.done = True
        else:
            with Timer("view_generation"):
                self.generate_view(img, depth, x, pcd_raw, rotation_angle)
                # nbv = self.generate_view(img, depth, x, pcd_raw, rotation_angle) 
            # self.x_d = nbv
    

    def update_grasp(self, pcd_raw, use_normal_vis=False):
        if not self.done:
            pcd = self.denoise_pcd(pcd_raw)

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


    def best_grasp_prediction_is_stable(self, sort_by="kappa", sample_num=1):
        if self.target_objects_label[-1] in self.target_objects.keys():
            # get the current pcd of the target object
            pcd_from_prompt=self.target_objects[self.target_objects_label[-1]].pcd
            
            # get the best grasp prediction on the target object
            poses, kappa, graspness = self.grasp_buffer.get_pose_fused(pcd_from_prompt=pcd_from_prompt)
            
            # sort by kappa or graspness
            score = kappa if sort_by == "kappa" else graspness

            # check if the score is empty or below the threshold
            if score.size(0) == 0 or score.max() < self.score_th:
                return False
    
            #sort poses by criterion
            sample_num = min(sample_num, poses.size(0))
            poses_candidates = poses[torch.argsort(score, descending=True)][:sample_num]

            #randomly sample 1 poses
            pose_chosen = poses_candidates[random.randint(0, sample_num-1)].squeeze(0)
            self.best_grasp = pose_chosen
            return True
        return False


    def remove_current_target(self):
        if len(self.target_objects_label) > 1:
            self.target_objects_label.pop()
            self.target_objects.pop(self.target_objects_label[-1])
            self.nbv_field = None
            self.done = False
            return False
        else:
            self.target_objects_label = []
            self.target_objects = {}
            self.nbv_field = None
            self.done = False
            return True
    
    def generate_views_from_nbv_field(self):
        if self.nbv_field is not None:
            query_points = generate_random_points_on_sphere(self.view_sphere.center, self.view_sphere.radius, 100)
            query_points = np.array(query_points)
            tangent_vectors = np.zeros_like(query_points)
            for P in self.Ps[1:]:
                tangent_vectors += np.array([self.nbv_field(P_q) for P_q in query_points])
            tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1)[:, None] + 1e-6
            return query_points, tangent_vectors
        return None, None


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


