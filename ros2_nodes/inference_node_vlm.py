import threading
import rclpy
import numpy as np
#import spatialmath as sm
from .utils_node import *

from .inference_node_base import *

from vmf_contact_main.camera_utils import *
from vmf_contact_main.active_grasp.policy import make, registry
from vmf_contact_main.active_grasp.vlm_policy import VLMPolicy
from vmf_contact_main.active_grasp.bbox import AABBox
from vmf_contact_main.active_grasp.spatial import *
from vmf_contact_main.active_grasp.timer import Timer

import argparse
from functools import partial

# from vgn.utils import ros_utils
from pathlib import Path

O_RESOLUTION = 40
O_SIZE = .3
O_VOXEL_SIZE = O_SIZE / O_RESOLUTION
min_z_dist = 0.45
linear_vel = 0.1
angular_vel = 1
control_rate = 60
policy_rate = 4

target_object = "red cup"

class State:
    def __init__(self, tsdf):
        self.tsdf = tsdf

class AIRNodeVLM(AIRNode):

    def __init__(self):
        super().__init__()

        self.camera_ready_pose = list_to_pose_stamped([-0.370, -0.592, 1.224, 0.942, 0.007, -0.005, 0.336], "world")
        self.gaze_point_robot=np.array([-0.86, 0.1, 0.0])
        pcd_center = list_to_pose_stamped(self.gaze_point_robot.tolist() + [0., 0., 0., 1.], "base_link")
        self.publish_new_frame("center", pcd_center)
            
        lower = [pcd_center.pose.position.x - O_SIZE / 2, 
                    pcd_center.pose.position.y - O_SIZE / 2, 
                    pcd_center.pose.position.z]
        upper = [pcd_center.pose.position.x + O_SIZE / 2,
                pcd_center.pose.position.y + O_SIZE / 2,
                pcd_center.pose.position.z + 0.09]
        
        middle = (np.array(lower) + np.array(upper)) / 2
        self.box_center = list_to_pose_stamped(middle.tolist() + [0., 0., 0., 1.], "base_link")
        self.publish_new_frame("box_center", self.box_center)

        self.bbox: AABBox = AABBox(lower, upper)
        
        # Active search setting
        parser = create_parser()
        args = parser.parse_args()
        self.policy:VLMPolicy = make(args.policy, target_object=target_object)
                
        self.user_input_thread = threading.Thread(target=self.handle_user_input)
        self.user_input_thread.start()
        self.set_vel_acc(.2, .1)
  
    
    def handle_user_input(self):

        self.get_camera_info()
        self.to_camera_ready_pose()
        success = clear = False
        while True:
            # Move to the camera ready pose
            self.policy.activate(self.bbox, 
                                 self.gaze_point_robot, 
                                 target_object = target_object,
                                 min_z_dist = min_z_dist)

            # user_input = input("Enter 's' to start next capture and 'q' to quit: ")
            user_input = "s"
            if user_input == "s":
                self.set_eelink("camera_color_optical_frame")
                # Initialize the search policy
                                
                self.change_state_to_servo_ctl()
                self.create_timer(1.0 / control_rate, self.send_vel_cmd)
                self.create_timer(1.0 / control_rate, self.grasp_inference)
                self.rate = self.create_rate(policy_rate)

                while not clear:
                    with Timer("Search time"):
                        self.get_logger().info("Searching for grasp...")
                
                        while not self.policy.done:
                            (pcd, rgb, d, cam_pose, rotation_angle, elevation_angle), identifier = self.process_point_cloud_and_rgbd()
                            if not identifier:
                                self.get_logger().info("No object detected, please try again.")
                                continue
                            self.policy.update(rgb, d, pcd, cam_pose, rotation_angle, elevation_angle)
                            # self.rate.sleep()

                    self.rate.sleep()
                    grasp = self.policy.best_grasp
                    self.change_state_to_cartesian_ctl()
                    self.set_eelink("tcp")

                    if grasp is not None:
                        self.get_logger().info("Search policy done, start grasp execution.")
                        with Timer("Grasp execution"):
                            pose = self.process_grasp(grasp)
                            success = self.execute_grasp(pose)
                    else:
                        self.get_logger().info("Aborted grasp execution.")
                        success = False
                    
                    if success:
                        self.get_logger().info("Grasp successful.")
                        clear = self.policy.remove_current_target()

            elif user_input == "q":
                self.shutdown = True
                self.change_state_to_cartesian_ctl()
                break

    def process_grasp(self, grasp):
        grasp = grasp.cpu().numpy()
        quat = quaternion_from_matrix(grasp)
        translation = translation_from_matrix(grasp)
        pose_chosen = np.concatenate([translation, quat])
        return list_to_pose(pose_chosen)
    
    def send_vel_cmd(self):
        if self.policy.x_d is None or self.policy.done:
            cmd = np.zeros(6)
        else:
            t_robot_2_camera = self.tf_buffer.lookup_transform("base_link", "camera_color_optical_frame", rclpy.time.Time()).transform
            x = SpatialTransform.from_matrix(transform_to_matrix(t_robot_2_camera))
            cmd = self.compute_velocity_cmd(x, linear_vel=linear_vel, angular_vel=angular_vel) 

            # publish the next view
            if len(self.nbv_fields) and self.policy.x_d is not None:
                pose_robot_2_camera_next: Pose = pose_from_spacial_transform(self.policy.x_d)
                self.publish_new_frame("camera_target_view", pose_stamped_from_pose(pose_robot_2_camera_next, "base_link"))

            # publish the view velocity
            pose_robot_2_camera: Pose = transform_to_pose(t_robot_2_camera)
            pose_robot_2_camera_next = apply_transform_to_pose(pose_robot_2_camera, cmd) 
            self.publish_new_frame(f"camera_view_velocity", pose_stamped_from_pose(pose_robot_2_camera_next, "base_link")) 
        self.send_twist_cmd(cmd)

    def compute_velocity_cmd(self, x, linear_vel=0.05, angular_vel=1):
        if len(self.target_objects_label) == 0: 
            view_focus = self.view_sphere.center 
        else: 
            view_focus = self.target_objects[self.target_objects_label[-1]].center
        _, theta, phi = cartesian_to_spherical(x.translation - view_focus)

        if len(self.nbv_fields):
            e_t = np.zeros(3)
            for nbv_field in self.nbv_fields:
                e_t += nbv_field(P_q = x.translation)
        else:
            e_t = self.policy.x_d.translation - x.translation # translation error
        
        r = np.linalg.norm(x.translation - self.gaze_point_robot)
        e_n = (x.translation - view_focus) * (self.view_sphere.r - r) / r # pull the camera towards the sphere

        linear = 1.0 * e_t + 6.0 * (r < self.view_sphere.r) * e_n # weighted sum of the two errors

        scale = np.linalg.norm(linear) + 1e-6
        linear *= np.clip(scale, 0.0, linear_vel) / scale # scale the linear velocity
        angular = self.view_sphere.get_view(theta, phi).rotation * x.rotation.inv() # desired rotation
        angular = angular_vel * angular.as_rotvec()
        return np.r_[linear, angular]
    
    def grasp_inference(self, use_normal_vis=False):
        pcd, identifier = self.process_point_cloud_and_rgbd(pcd_only=True)
        if identifier:
            self.policy.update_grasp(pcd, use_normal_vis)
    
    @property
    def view_sphere(self):
        return self.policy.view_sphere
    
    @property
    def target_objects(self):
        return self.policy.target_objects
    
    @property
    def target_objects_label(self):
        return self.policy.target_objects_label
    
    @property
    def nbv_fields(self):
        return self.policy.nbv_fields
    

def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_listener = AIRNodeVLM()
    try:
        rclpy.spin(pcd_listener)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info("Exiting")

    pcd_listener.destroy_node()
    rclpy.shutdown()

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, choices=registry.keys(), default="vlm")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--wait-for-input", action="store_true")
    parser.add_argument("--logdir", type=Path, default="logs")
    parser.add_argument("--seed", type=int, default=1)
    return parser

if __name__ == "__main__":
    main()

