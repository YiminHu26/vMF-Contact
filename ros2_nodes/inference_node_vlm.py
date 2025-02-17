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

MOVING_BACK_TO_NBV = "moving_back_to_nbv"

O_RESOLUTION = 40
O_SIZE = .3
O_VOXEL_SIZE = O_SIZE / O_RESOLUTION
min_z_dist = .45
linear_vel = .05
angular_vel = 1
control_rate = 30
policy_rate = 4

target_object = "green cup"

class State:
    def __init__(self, tsdf):
        self.tsdf = tsdf

class AIRNodeVLM(AIRNode):

    def __init__(self):
        super().__init__()
        self.camera_ready_pose = list_to_pose_stamped([-0.370, -0.612, 1.224, 0.942, 0.007, -0.005, 0.336], "world")
        self.gaze_point_robot=np.array([-0.73, 0.1, 0.])
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
        self.policy:VLMPolicy = make(args.policy, 
                                     target_object=target_object, 
                                     pcd_shift=self.gaze_point_robot,
                                     min_z_dist=min_z_dist,)
                
        self.user_input_thread = threading.Thread(target=self.handle_user_input)
        self.user_input_thread.start()
        self.set_vel_acc(.2, .1)
        self.nbv_memory_pose = None
  
    
    def handle_user_input(self):

        self.get_camera_info()
        self.to_camera_ready_pose()
        success = clear = False
        # while True:
        # Move to the camera ready pose
        self.policy.activate(self.bbox)

        # user_input = input("Enter 's' to start next capture and 'q' to quit: ")
        # user_input = "s"
        # if user_input == "s":
        
        # Initialize the search policy
                        
        self.create_timer(1.0 / control_rate, self.send_vel_cmd)
        self.create_timer(1.0 / control_rate, self.grasp_inference)
        self.rate = self.create_rate(policy_rate)

        while not clear:
            self.set_eelink("camera_color_optical_frame")
            self.change_state_to_servo_ctl()
            with Timer("Search time"):
                self.get_logger().info("Searching for grasp...")
                self.policy.activate(self.bbox)
        
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
            self.nbv_memory_pose = self.get_current_ee_pose()

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
                clear = all([word in self.policy.target_object_curr_label for word in target_object.split(" ")])
            time.sleep(5)

        # elif user_input == "q":
        #     self.shutdown = True
        #     self.change_state_to_cartesian_ctl()
        #     break

    def process_grasp(self, grasp):
        grasp = grasp.cpu().numpy()
        quat = quaternion_from_matrix(grasp)
        translation = translation_from_matrix(grasp)
        pose_chosen = np.concatenate([translation, quat])
        return list_to_pose(pose_chosen)
    
    def send_vel_cmd(self):
        if len(self.nbv_fields)==0 or self.policy.done:
            cmd = np.zeros(6)
        else:
            t_robot_2_camera = self.tf_buffer.lookup_transform("base_link", "camera_color_optical_frame", rclpy.time.Time()).transform
            x = SpatialTransform.from_matrix(transform_to_matrix(t_robot_2_camera))
            cmd = self.compute_velocity_cmd(x, linear_vel=linear_vel, angular_vel=angular_vel) 

            # publish the view velocity
            pose_robot_2_camera: Pose = transform_to_pose(t_robot_2_camera)
            pose_robot_2_camera_next = apply_transform_to_pose(pose_robot_2_camera, cmd) 
            self.publish_new_frame(f"camera_view_velocity", pose_stamped_from_pose(pose_robot_2_camera_next, "base_link")) 
            # print(f"cmd: {cmd}")
        self.send_twist_cmd(cmd)

    def compute_velocity_cmd(self, x, linear_vel=0.05, angular_vel=1):
        view_focus = self.view_sphere.center
        _, theta, phi = cartesian_to_spherical(x.translation - view_focus)

        e_t = self.policy.query_field_fusion_from_list(x.translation)
        
        r = np.linalg.norm(x.translation - self.gaze_point_robot)
        e_n = (x.translation - view_focus) * (self.view_sphere.r - r) / r # pull the camera towards the sphere

        # self.logger.info(f"e_t: {e_t}, e_n: {e_n}")

        linear = 1.0 * e_t + 2.0 * (r < self.view_sphere.r) * e_n # weighted sum of the two errors

        scale = np.linalg.norm(linear) + 1e-6
        linear *= np.clip(scale, 0.0, linear_vel) / scale # scale the linear velocity
        angular = self.view_sphere.get_view(theta, phi).rotation * x.rotation.inv() # desired rotation
        angular = angular_vel * angular.as_rotvec()
        return np.r_[linear, angular]
    
    def grasp_inference(self, use_normal_vis=False):
        pcd, identifier = self.process_point_cloud_and_rgbd(pcd_only=True)
        if identifier:
            self.policy.update_grasp(pcd, use_normal_vis)
    

    def execute_grasp(self, grasp_pose:Pose, frame = "base_link"):

        self.change_state_to_cartesian_ctl()
        self.get_logger().info("Sending goal now...")

        # transform posestamped from base_link to world using t_world_2_base_link
        grasp_pose = pose_stamped_from_pose(grasp_pose, "base_link")
        t_world_2_base_link = self.tf_buffer.lookup_transform("world", frame, rclpy.time.Time())
        grasp_pose.pose = tf2_geometry_msgs.do_transform_pose(grasp_pose.pose, t_world_2_base_link)
        grasp_pose.header.frame_id = "world"
        self.publish_new_frame("grasp", grasp_pose)

        pregrasp_pose = self.create_pregrasp_pose(copy.deepcopy(grasp_pose))
        self.publish_new_frame("pregrasp", pregrasp_pose)
        
        # ask if the user wants to continue
        user_input = input("Press 'c' to continue or any other key to quit: ")
        if user_input.lower() == 'c':

            self.stop_event.clear()
            self.movement_failed_flag.clear()
            self.movement_finished_flag.clear()
            state_machine_state = IDLE
            
            while True:
                if self.stop_event.is_set():
                    self.stop_event.clear()
                    self.get_logger().info("Goal cancelled")
                    self.to_camera_ready_pose()
                    state_machine_state = MOVING_TO_CAMERA_READY
                    self.get_logger().info("StateMachine switched to MOVING_TO_CAMERA_READY")

                if state_machine_state == IDLE:
                    # Send the goal to move to the pregrasp pose
                    self.send_goal(pregrasp_pose)
                    # Start the state machine
                    state_machine_state = MOVING_TO_PREGRASP
                    self.get_logger().info("StateMachine switched to MOVING_TO_PREGRASP")
                    # Create a thread to handle the input
                    self.cancel_thread = threading.Thread(target=self.get_input)
                    self.cancel_thread.start()

                elif state_machine_state == MOVING_TO_PREGRASP:
                    if self.movement_finished_flag.is_set():
                        self.movement_finished_flag.clear()
                        # Wait for manual grasp evaluation
                        time.sleep(2)
                        self.send_goal(grasp_pose)
                        # Start the state machine
                        state_machine_state = MOVING_TO_GRASP
                        self.get_logger().info("StateMachine switched to MOVING_TO_GRASP")
                        time.sleep(0.1)

                    if self.movement_failed_flag.is_set():
                        self.movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                        
                elif state_machine_state == MOVING_TO_GRASP:
                    # Wait for the action server to finish
                    if self.movement_finished_flag.is_set():
                        self.movement_finished_flag.clear()
                        time.sleep(2)
                        # Close the gripper
                        self.close_gripper()
                        state_machine_state = GRASPING
                        self.get_logger().info("StateMachine switched to GRASPING")
                        
                    if self.movement_failed_flag.is_set():
                        self.movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")

                elif state_machine_state == GRASPING:
                    # Wait for the gripper to finish
                    if self.gripper_movement_finished_flag.is_set():
                        self.gripper_movement_finished_flag.clear()
                        # Before gripper moves away, wait for 2 seconds
                        time.sleep(0.25)
                        # Send the goal to move to the pregrasp pose
                        self.send_goal(pregrasp_pose)
                        state_machine_state = MOVING_TO_PREGRASP_RETURN
                        self.get_logger().info("StateMachine switched to MOVING_TO_PREGRASP_RETURN")
                        time.sleep(1)
                    if self.gripper_movement_failed_flag.is_set():
                        self.gripper_movement_failed_flag.clear()
                        state_machine_state = FAILED
                
                elif state_machine_state == MOVING_TO_PREGRASP_RETURN:
                    # Wait for the action server to finish
                    if self.movement_finished_flag.is_set():
                        self.movement_finished_flag.clear()
                        # Send the goal to move to the drop off pose
                        self.send_goal(self.drop_off_pose)
                        state_machine_state = MOVING_TO_DROP_OFF
                        self.get_logger().info("StateMachine switched to MOVING_TO_DROP_OFF")
                    if self.movement_failed_flag.is_set():
                        self.movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                
                elif state_machine_state == MOVING_TO_DROP_OFF:
                    # Wait for the action server to finish
                    if self.movement_finished_flag.is_set():
                        self.movement_finished_flag.clear()
                        # Open the gripper
                        self.open_gripper()
                        state_machine_state = RELEASEING
                        self.get_logger().info("StateMachine switched to RELEASEING")
                    if self.movement_failed_flag.is_set():
                        self.movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                
                elif state_machine_state == RELEASEING:
                    # Wait for the gripper to finish
                    if self.gripper_movement_finished_flag.is_set():
                        self.gripper_movement_finished_flag.clear()
                        # Send the goal to move to the camera ready pose
                        self.to_nbv_pose()
                        state_machine_state = MOVING_BACK_TO_NBV
                        self.get_logger().info("StateMachine switched to MOVING_BACK_TO_NBV")
                    if self.gripper_movement_failed_flag.is_set():
                        self.gripper_movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                
                elif state_machine_state == MOVING_BACK_TO_NBV:
                    # Wait for the action server to finish
                    if self.movement_finished_flag.is_set():
                        self.movement_finished_flag.clear()
                        state_machine_state = IDLE
                        self.get_logger().info("State machine finished, please enter any key to finish the input thread.")
                        break
                    if self.movement_failed_flag.is_set():
                        self.movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                
                elif state_machine_state == FAILED:
                    self.get_logger().info("State machine failed")
                    return False
            return input("Is the grasp successful? (y/n): ").lower() == "y"
        

    def to_nbv_pose(self):
        self.change_state_to_cartesian_ctl()
        self.set_eelink("tcp")
        self.movement_finished_flag.clear()
        self.send_goal(self.nbv_memory_pose)
        self.open_gripper()
        while not self.movement_finished_flag.is_set():
            continue

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

