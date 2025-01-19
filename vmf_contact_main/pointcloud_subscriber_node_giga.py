import threading
import time
import open3d as o3d
import rclpy
import sensor_msgs.msg as sensor_msgs
from sensor_msgs.msg import PointCloud2
import spatialmath as sm
from rclpy.node import Node
from rclpy.action import ActionClient
from arm_api2_msgs.action import MoveCartesian
from control_msgs.action import GripperCommand
from control_msgs.msg import GripperCommand as GripperCommandMsg
from sensor_msgs_py.point_cloud2 import read_points_numpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import StaticTransformBroadcaster
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud, transform_points
from vmf_contact_main.train import main_module, parse_args_from_yaml
from cv_bridge import CvBridge
import cv2
import os, torch
import numpy as np
import copy
from tf_transformations import quaternion_matrix, quaternion_from_matrix, translation_from_matrix
import tf2_geometry_msgs
from .camera_utils import *
from math import cos, sin
#import spatialmath as sm
from scipy import ndimage
from .utils_node import *
import asyncio

from vgn.detection import VGN
from vgn.detection_implicit import VGNImplicit
from vgn.grasp import *
from vgn.perception import create_tsdf

from .active_grasp.policy import make, registry
from .active_grasp.bbox import AABBox
from .active_grasp.spatial import *
from .active_grasp.timer import Timer

import argparse

# from vgn.utils import ros_utils
from pathlib import Path

current_file_folder = os.path.dirname(os.path.abspath(__file__))


# define state machine states
IDLE = "idle"
MOVING_TO_PREGRASP = "moving_to_pregrasp"
MOVING_TO_GRASP = "moving_to_grasp"
GRASPING = "grasping"
MOVING_TO_DROP_OFF = "moving_to_drop_off"
MOVING_TO_PREGRASP2 = "moving_to_pregrasp2"
RELEASEING = "releasing"
MOVING_TO_PREGRASP_RETURN = "moving_to_pregrasp2"
MOVING_TO_CAMERA_READY = "moving_to_camera_ready"
FAILED = "failed"
MOVING = "moving"

O_RESOLUTION = 40
O_SIZE = .3
O_VOXEL_SIZE = O_SIZE / O_RESOLUTION
min_z_dist = 0.3
linear_vel = 0.1
angular_vel = 1
control_rate = 30
policy_rate = 4
qual_th = 0.8

ACTIVE_GRASP = True

class State:
    def __init__(self, tsdf):
        self.tsdf = tsdf

class PCDListener(Node):

    def __init__(self):
        super().__init__("pcd_subsriber_node")
        
        # Giga
        self.stitched_pointcloud_topic = "/cloud_stitched"  
        self.grasp_result_topic = "/detect_grasps/clustered_grasps" 


        # Set up a subscription to the 'pcd' topic with a callback to the
        # function `listener_callback`
        self.pcd_subscriber = self.create_subscription(
            sensor_msgs.PointCloud2,  # Msg type
            "/camera/depth/points",  # topic
            self.listener_callback_pcd,  # Function to call
            10,  # QoS
        )

        self.img_subscriber = self.create_subscription(
            sensor_msgs.Image,  # Msg type
            "/camera/color/image_raw",  # topic
            self.listener_callback_img,  # Function to call
            10,  # QoS
        )

        self.dpt_subscriber = self.create_subscription(
            sensor_msgs.Image,  # Msg type
            "/camera/depth/image_raw",  # topic
            self.listener_callback_dpt,  # Function to call
            10,  # QoS
        )

        self.camera_info_subscriber = self.create_subscription(
            sensor_msgs.CameraInfo,  # Msg type
            "/camera/color/camera_info",  # topic
            self.listener_callback_caminfo,  # Function to call
            10,  # QoS
        )

        self._robot_action_client = ActionClient(
            self, MoveCartesian, "arm/move_to_pose"
        )
        
        self._gripper_action_client = ActionClient(
            self, GripperCommand, "robotiq_2f_urcap_adapter/gripper_command"
        )


        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        state_machine_state = IDLE

        # Create TF Listener to get the transform
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pcd_shift=np.array([-0.86, 0.1, 0.031])
        self.pcd_resize=np.array(1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.last_point_cloud_msg = None
        self.shutdown = False
        
        self.movement_finished_flag = threading.Event()
        self.movement_failed_flag = threading.Event()
        self.gripper_movement_finished_flag = threading.Event()
        self.gripper_movement_failed_flag = threading.Event()
        self.stop_event = threading.Event()
        self.bridge = CvBridge()

        self.camera_ready_pose = PoseStamped()
        self.camera_ready_pose.header.frame_id = "world"
        

        self.camera_ready_pose = list_to_pose_stamped([-0.435, -0.794, 1.381, 0.995, 0.009, 0.005, 0.100], "world") # long finger
        # self.camera_ready_pose = list_to_pose_stamped([-0.435, -0.572, 1.492, 0.995, 0.009, 0.005, 0.100], "world") # small finger
        self.drop_off_pose: PoseStamped = list_to_pose_stamped([0.15, -0.75, 1.3, 1.0, 0.0, 0.0, 0.0], "world")

        self.pcd_center = list_to_pose_stamped([-0.86, 0.1, 0.03, 0.0, 0.0, 0.0, 1.0], "base_link")
        self.publish_new_frame("center_giga", self.pcd_center)

        if ACTIVE_GRASP:
            
            lower = [self.pcd_center.pose.position.x - O_SIZE / 2, 
                     self.pcd_center.pose.position.y - O_SIZE / 2, 
                     self.pcd_center.pose.position.z]
            upper = [self.pcd_center.pose.position.x + O_SIZE / 2,
                    self.pcd_center.pose.position.y + O_SIZE / 2,
                    self.pcd_center.pose.position.z + 0.09]
            
            middle = (np.array(lower) + np.array(upper)) / 2
            self.box_center = list_to_pose_stamped(middle.tolist() + [0., 0., 0., 1.], "base_link")
            self.publish_new_frame("box_center", self.box_center)

            self.bbox: AABBox = AABBox(lower, upper)
            
            # Active search setting
            parser = create_parser()
            args = parser.parse_args()
            self.policy = make(args.policy)
            self.user_input_thread = threading.Thread(target=self.handle_user_input_active)
        else:
            model_type = "vgn" 
            model_path = "/home/yitian/GIGA/data/models/vgn_packed.pt"
            self.origin_giga = list_to_pose_stamped([-1.0, -0.05, 0.03, 0.0, 0.0, 0.0, 1.0], "base_link")
            self.publish_new_frame("origin_giga", self.origin_giga) 

            if model_type == "vgn":
                self.agent = VGN(model_path, 
                                 model_type=model_type, 
                                 best=True, 
                                 force_detection=True, 
                                 qual_th=qual_th, 
                                 out_th=0.1, 
                                 visualize=False)
            elif model_type == "giga":
                self.agent = VGNImplicit(model_path, 
                                         model_type=model_type, 
                                         best=True, 
                                         force_detection=True, 
                                         qual_th=qual_th, 
                                         resolution=O_RESOLUTION, 
                                         voxel_size=O_VOXEL_SIZE, 
                                         out_th=0.1, 
                                         visualize=False)
            self.user_input_thread = threading.Thread(target=self.handle_user_input)

        self.user_input_thread.start()

    def get_camera_info(self):
        while True:
            try:
                print("Waiting for camera info...")
                self.intrinsics= CameraInfo(
                    width=int(self.image_width), 
                    height=int(self.image_height), 
                    fx=self.camera_matrix[0, 0], 
                    fy=self.camera_matrix[1, 1],
                    cx=self.camera_matrix[0, 2], 
                    cy=self.camera_matrix[1, 2], 
                    scale=1.0
                )
                self.get_logger().info(f"Camera info received: {self.intrinsics}")
                break
            except:
                self.get_logger().info("No camera info received yet.")
                time.sleep(1)

    def listener_callback_pcd(self, msg: sensor_msgs.PointCloud2):
        """Callback function for the subscriber of the point cloud topic."""
        self.last_point_cloud_msg = msg
        if self.shutdown:
            raise SystemExit
    
    def listener_callback_img(self, msg: sensor_msgs.Image):
        """Callback function for the subscriber of the point cloud topic."""
        self.last_image_msg = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if self.shutdown:
            raise SystemExit
    
    def listener_callback_dpt(self, msg: sensor_msgs.Image):
        """Callback function for the subscriber of the point cloud topic."""
        self.last_depth_msg = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1") # / 1000.0
        if self.shutdown:
            raise SystemExit

    def listener_callback_caminfo(self, msg: sensor_msgs.CameraInfo):
        self.distortion = np.array([msg.d[i] for i in range(8)])
        self.camera_matrix = np.array([msg.k[i] for i in range(9)]).reshape(3, 3)
        self.image_width = float(msg.width)
        self.image_height = float(msg.height)
        if self.shutdown:
            raise SystemExit

    def process_point_cloud_and_rgbd(self, save_data=False):
        # TODO: add rgb image processing
        if self.last_point_cloud_msg is None:
            self.get_logger().info("No point cloud message received yet.")
            return [None] * 4, False

        if self.last_image_msg is None:
            self.get_logger().info("No image message received yet.")
            return [None] * 4, False
    
        if self.last_depth_msg is None:
            self.get_logger().info("No depth message received yet.")
            return [None] * 4, False

        if self.last_point_cloud_msg is None:
            self.get_logger().info("No camera info message received yet.")
            return [None] * 4, False
        
        # Get the latest point cloud and image messages
        pcd_msg_camera = self.last_point_cloud_msg
        img = self.last_image_msg

        # transform the point cloud to base_link frame
        camera= CameraInfo(
            width=self.image_width, height=self.image_height, fx=self.camera_matrix[0, 0], fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2], cy=self.camera_matrix[1, 2], scale=1000.0
        )
        pcd_from_depth = create_point_cloud_from_depth_image(self.last_depth_msg, camera, organized=True).reshape(-1, 3)
        
        # Look up for the transformation between base_link and the frame_id of the point cloud
        t_robot_2_camera = self.tf_buffer.lookup_transform(
                "base_link", pcd_msg_camera.header.frame_id, rclpy.time.Time()
            )
        
        # pcd_msg_base_link_1 = do_transform_cloud(pcd_msg_camera, t_robot_2_camera)        
        # pcd_numpy_base_link_1 = read_points_numpy(pcd_msg_base_link_1)
        pcd_numpy_base_link = transform_points(pcd_from_depth, t_robot_2_camera.transform)  
        # self.get_logger().info("First 10 points: ", pcd_numpy_base_link[:10])

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcd_numpy_base_link_1)
        # pcd_new = o3d.geometry.PointCloud()
        # pcd_new.points = o3d.utility.Vector3dVector(pcd_numpy_base_link)
        # o3d.visualization.draw_geometries([pcd, pcd_new])
        
        ## save the depth image with gray scale, normalized to 0-255
        # pcd_from_depth_z = pcd_from_depth[..., 2]
        # pcd_from_depth_z = (pcd_from_depth_z - pcd_from_depth_z.min()) / (pcd_from_depth_z.max() - pcd_from_depth_z.min()) * 255
        # cv2.imwrite(f"{current_file_folder}/depth.jpg", pcd_from_depth_z)

        # remove previous masks
        for file in os.listdir(current_file_folder):
            if file.endswith(".jpg"):
                os.remove(os.path.join(current_file_folder, file))
        
        # transformation to pose
        cam_pose_robot = transform_to_pose(t_robot_2_camera.transform)

        # save the image, depth, point cloud and camera pose as a dictionary of numpy arrays
        # viszualize the rgbd data
        if save_data:
            name = f"/home/yitian/data_active_grasp/{cam_pose_robot.position.x}_{cam_pose_robot.position.y}_{cam_pose_robot.position.z}"
            cv2.imwrite(f"image.jpg", img[..., ::-1])
            dict_rgbd = {
                "image": img,
                "depth": self.last_depth_msg,
                "pcd": pcd_numpy_base_link,
                "cam_pose": [cam_pose_robot.position.x, 
                            cam_pose_robot.position.y, 
                            cam_pose_robot.position.z,
                            cam_pose_robot.orientation.x, 
                            cam_pose_robot.orientation.y, 
                            cam_pose_robot.orientation.z, 
                            cam_pose_robot.orientation.w]
            }
            np.savez(f"{name}.npz", **dict_rgbd)

        return (pcd_numpy_base_link, 
                self.last_image_msg, 
                self.last_depth_msg.astype(np.float32) / 1000.0, 
                cam_pose_robot), True
  
    def handle_user_input(self):
    
        self.get_camera_info()

        while True:
            self.send_goal(self.camera_ready_pose)
            self.open_gripper()
            user_input = input("Enter 's' to start next capture and 'q' to quit: ")
            if user_input == "s":
                (pcd, rgb, d, cam_pose), identifier = self.process_point_cloud_and_rgbd()
                if not identifier:
                    continue
                grasp = self.agent_inference(d) 
                if grasp is not None:
                    self.execute_grasp(grasp, frame="origin_giga")
                else:
                    self.get_logger().info("No grasp pose detected, please try again.")
            elif user_input == "q":
                self.shutdown = True
                break
    
    def handle_user_input_active(self):

        self.get_camera_info()
        success = False
        while True:
            # Move to the camera ready pose
            self.send_goal(self.camera_ready_pose)
            self.open_gripper()

            user_input = input("Enter 's' to start next capture and 'q' to quit: ")
            if user_input == "s":
                # Initialize the search policy
                self.view_sphere = ViewHalfSphere(self.bbox, min_z_dist)
                self.policy.activate(self.bbox, self.view_sphere, self.intrinsics)
                
                # execute = threading.Thread(target=self.send_vel_cmd)
                # execute.start()
                # self.create_timer(1.0 / control_rate, self.send_vel_cmd)

                self.rate = self.create_rate(policy_rate)
                with Timer("Search time"):
                    while not self.policy.done:
                        (pcd, rgb, d, cam_pose), identifier = self.process_point_cloud_and_rgbd()
                        if not identifier:
                            self.get_logger().info("No object detected, please try again.")
                            continue
                        extrinsic = self.get_extrinsics(inverse=True)[0]
                        self.policy.update(d, extrinsic, self.intrinsics)
                        print("Searching for grasp...")
                        # self.rate.sleep()
                        self.send_vel_cmd()
                    self.rate.sleep()
                    grasp = self.policy.best_grasp
                
                self.get_logger().info("Search policy done, start grasp execution.")
                if grasp is not None:
                    with Timer("Grasp execution"):
                        success = self.execute_grasp(grasp.pose)
                else:
                    self.get_logger().info("Aborted grasp execution.")
                    success = False
            elif user_input == "q":
                self.shutdown = True
                break
  
  
    def agent_inference(self, depth_imgs):
        if self.last_depth_msg is None or self.camera_matrix is None:
            self.get_logger().info("Missing depth image or camera info.")
            return None
        
        extrinsics = self.get_extrinsics()
        depth_imgs = np.expand_dims(depth_imgs, axis=0)
        tsdf_volume = create_tsdf(O_SIZE, O_RESOLUTION, depth_imgs, self.intrinsics, extrinsics)

        # visualize the TSDF volume
        pcd = tsdf_volume.get_cloud()
        state = State(tsdf=tsdf_volume)

        # Perform inference
        grasps, scores, toc = self.agent(state)

        grasp_best = grasps[scores.argmax()].pose.to_list()
        self.get_logger().info(f"Best grasp pose: {grasp_best}")

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=grasp_best[4:])
        o3d.visualization.draw_geometries([pcd, frame])

        return grasp_best

    def get_extrinsics(self, inverse=False):
        extrinsics = np.expand_dims(np.eye(4), axis=0)
        # return extrinsics
        if inverse:
            t_robot_2_camera: TransformStamped = self.tf_buffer.lookup_transform(
                "base_link", "camera_color_optical_frame", rclpy.time.Time()
            )
        else:
            t_robot_2_camera: TransformStamped = self.tf_buffer.lookup_transform(
                "camera_color_optical_frame", "origin_giga", rclpy.time.Time()
            )
        translation = [
            t_robot_2_camera.transform.translation.x,
            t_robot_2_camera.transform.translation.y,
            t_robot_2_camera.transform.translation.z,
        ]

        rotation = [
            t_robot_2_camera.transform.rotation.x,
            t_robot_2_camera.transform.rotation.y,
            t_robot_2_camera.transform.rotation.z,
            t_robot_2_camera.transform.rotation.w,
        ]

        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_matrix(rotation)[:3, :3]

        # Construct the transformation matrix
        extrinsics[:, :3, :3] = rotation_matrix
        extrinsics[:, :3, 3] = translation

        # self.get_logger().info(f"Extrinsics: {extrinsics}")

        return  extrinsics
    
    def camera_robot_pose_to_tcp_world_pose(self, pose_robot_2_camera: Pose):
        t_world_2_base_link: TransformStamped = self.tf_buffer.lookup_transform(
                "world", "base_link", rclpy.time.Time()
            )
        t_camera_2_tcp: TransformStamped = self.tf_buffer.lookup_transform(
            "camera_color_optical_frame","tcp", rclpy.time.Time()
        )
        # transform posestamped from base_link to world using t_world_2_base_link
        pose_world_2_camera = tf2_geometry_msgs.do_transform_pose(pose_robot_2_camera, t_world_2_base_link)
        t_world_2_camera = pose_to_transform(pose_world_2_camera, "world")
        pose_camera_2_tcp = transform_to_pose(t_camera_2_tcp.transform)
        pose_world_2_tcp = tf2_geometry_msgs.do_transform_pose(pose_camera_2_tcp, t_world_2_camera)

        pose_stamped = pose_stamped_from_pose(pose_world_2_tcp, "world")
        return pose_stamped, pose_world_2_camera
    
    def change_view(self, pose_robot_2_camera):

        self.get_logger().info("Sending goal now...")

        pose_stamped, pose_world_2_camera = self.camera_robot_pose_to_tcp_world_pose(pose_robot_2_camera)

        self.publish_new_frame(f"camera_target_view_velocity", pose_stamped_from_pose(pose_world_2_camera, "world"))
        
        self.stop_event.clear()
        self.movement_failed_flag.clear()
        self.movement_finished_flag.clear()
        state_machine_state = IDLE
        
        while True:
            if state_machine_state == IDLE:
                # Send the goal to move to the pregrasp pose
                self.send_goal(pose_stamped)
                # Start the state machine
                state_machine_state = MOVING
                self.get_logger().info("StateMachine switched to MOVING")
                # # Create a thread to handle the input
                # self.cancel_thread = threading.Thread(target=self.get_input)
                # self.cancel_thread.start()

            else:
                # Wait for the action server to finish
                if self.movement_finished_flag.is_set():
                    self.movement_finished_flag.clear()
                    state_machine_state = IDLE
                    self.get_logger().info("State machine finished")
                    break

                if self.movement_failed_flag.is_set():
                    self.movement_failed_flag.clear()
                    state_machine_state = FAILED
                    self.get_logger().info("StateMachine switched to FAILED")

                elif state_machine_state == FAILED:
                    self.get_logger().info("State machine failed")
                    return False
        return True

    def execute_grasp(self, pose, frame = "base_link"):

        self.get_logger().info("Sending goal now...")

        if isinstance(pose, np.ndarray):
            pose = list(pose)
            self.get_logger().info(f"Grasp pose: {pose}")
            pose = pose[4:] + pose[:4]
            pose = list_to_pose(pose)
        elif isinstance(pose, SpatialTransform):
            print(pose.translation)
            pose = pose_from_spacial_transform(pose)

        grasp_pose = pose_stamped_from_pose(pose, frame)

        grasp_pose = self.transform_pose_z(grasp_pose, z_offset=0.04) # GIGA is predicting the position of the finger end, so we need to move it a bit in z direction to the tcp

        
        t_world_2_base_link = self.tf_buffer.lookup_transform(
            "world", frame, rclpy.time.Time()
        )

        # transform posestamped from base_link to world using t_world_2_base_link
        grasp_pose.pose = tf2_geometry_msgs.do_transform_pose(grasp_pose.pose, t_world_2_base_link)
        grasp_pose.header.frame_id = "world"
        
        pregrasp_pose = self.create_pregrasp_pose(copy.deepcopy(grasp_pose))

        self.publish_new_frame("grasp", grasp_pose)
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
                    self.send_goal(self.camera_ready_pose)
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
                        self.send_goal(self.camera_ready_pose)
                        state_machine_state = MOVING_TO_CAMERA_READY
                        self.get_logger().info("StateMachine switched to MOVING_TO_CAMERA_READY")
                    if self.gripper_movement_failed_flag.is_set():
                        self.gripper_movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                
                elif state_machine_state == MOVING_TO_CAMERA_READY:
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
            return True


    def get_input(self):
        try:
            user_input = input("Press 'c' to cancel: ")
            self.stop_event.set()
            self.get_logger().info(f"\nYou entered: {user_input}")
        except EOFError:
            # Input terminated unexpectedly
            pass

    def cancel_done(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal successfully canceled')
        else:
            self.get_logger().info('Goal failed to cancel')   
           
    def create_pregrasp_pose(self, grasp_pose: PoseStamped) -> PoseStamped:
        # Create a pregrasp pose by transforming the grasp pose in the z direction
        pregrasp_pose = self.transform_pose_z(grasp_pose, z_offset=-0.1)
        return pregrasp_pose
    
    def transform_pose_z(self, pose_stamped: PoseStamped, z_offset: float) -> PoseStamped:
        # Copy the original pose
        new_pose_stamped = PoseStamped()
        new_pose_stamped.header = pose_stamped.header
        new_pose_stamped.pose = pose_stamped.pose

        # Extract the current pose
        current_position = pose_stamped.pose.position
        current_orientation = pose_stamped.pose.orientation

        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_matrix([current_orientation.x, 
                                             current_orientation.y, 
                                             current_orientation.z, 
                                             current_orientation.w])

        # Translation in the local z direction (12 cm = 0.12 meters)
        translation = [0.0, 0.0, z_offset, 1.0]

        # Apply the translation in the local frame
        transformed_translation = rotation_matrix.dot(translation)

        # Update the position with the transformed translation
        new_pose_stamped.pose.position.x = current_position.x + transformed_translation[0]
        new_pose_stamped.pose.position.y = current_position.y + transformed_translation[1]
        new_pose_stamped.pose.position.z = current_position.z + transformed_translation[2]  

        return new_pose_stamped
    
    def publish_new_frame(self, name, pose: PoseStamped):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = pose.header.frame_id
        t.child_frame_id = name
        t.transform.translation.x = pose.pose.position.x
        t.transform.translation.y = pose.pose.position.y
        t.transform.translation.z = pose.pose.position.z
        t.transform.rotation.x = pose.pose.orientation.x
        t.transform.rotation.y = pose.pose.orientation.y
        t.transform.rotation.z = pose.pose.orientation.z
        t.transform.rotation.w = pose.pose.orientation.w

        self.tf_static_broadcaster.sendTransform(t)

    def send_goal(self, goal):
        goal_msg = MoveCartesian.Goal()
        goal_msg.goal = goal

        self.get_logger().info("Waiting for action server...")

        self._robot_action_client.wait_for_server()

        self.get_logger().info("Sending goal request...")

        self._send_goal_future = self._robot_action_client.send_goal_async(
            goal_msg, feedback_callback=self.robot_feedback_callback
        )
        self._send_goal_future.add_done_callback(self.robot_goal_response_callback)

    def robot_goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return
        self.get_logger().info("Goal accepted")

        self._get_result_future = self.goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.robot_get_result_callback)

    def robot_get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result}")
        if result.success:
            self.movement_finished_flag.set()
        else:
            self.movement_failed_flag.set()
    
    def robot_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info("Feedback: {0}".format(feedback.status))
    
    def send_gripper_command(self, command: GripperCommandMsg):
        goal_msg = GripperCommand.Goal()
        goal_msg.command = command

        self.get_logger().info("Waiting for gripper action server...")

        self._gripper_action_client.wait_for_server()

        self.get_logger().info("Sending gripper goal request...")

        self._send_goal_future = self._gripper_action_client.send_goal_async(
            goal_msg, feedback_callback=self.gripper_feedback_callback
        )

        self._send_goal_future.add_done_callback(self.gripper_goal_response_callback)

    def gripper_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Gripper goal rejected")
            return

        self.get_logger().info("Gripper goal accepted")

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.gripper_get_result_callback)

    def gripper_get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Gripper result: {result}")
        #if result.success:
        self.gripper_movement_finished_flag.set()
        #else:
         #   self.movement_failed_flag.set()

    def gripper_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Gripper Feedback: {feedback}")
    
    def open_gripper(self):
        self.get_logger().info("Opening gripper...")
        command = GripperCommandMsg()
        command.position = 0.0  # rad
        command.max_effort = 60.0  # N
        self.send_gripper_command(command)

    def close_gripper(self):
        self.get_logger().info("Closing gripper...")
        command = GripperCommandMsg()
        command.position = 0.8  # 0.0 (open) - 0.085 (close) rad
        command.max_effort = 150.0  # 20 N - 235 N
        self.send_gripper_command(command)

    def send_vel_cmd(self):
        if self.policy.x_d is None or self.policy.done:
            cmd = np.zeros(6)
        else:
            t_robot_2_camera = self.tf_buffer.lookup_transform("base_link", "camera_color_optical_frame", rclpy.time.Time()).transform
            x = SpatialTransform.from_matrix(transform_to_matrix(t_robot_2_camera))
            cmd = self.compute_velocity_cmd(self.policy.x_d, x, linear_vel=linear_vel, angular_vel=angular_vel)  

        if cmd is not None and not all([cmd[i] == 0 for i in range(6)]):

            pose_robot_2_camera: Pose = transform_to_pose(t_robot_2_camera)
            print("Before move: ", pose_robot_2_camera.position.x, pose_robot_2_camera.position.y, pose_robot_2_camera.position.z)

            pose_robot_2_camera_next: Pose = pose_from_spacial_transform(self.policy.x_d)
            self.publish_new_frame("camera_target_view", pose_stamped_from_pose(pose_robot_2_camera_next, "base_link"))
            # pose_robot_2_camera_next = apply_transform_to_pose(pose_robot_2_camera, cmd)  
            print("After move: ", pose_robot_2_camera_next.position.x, pose_robot_2_camera_next.position.y, pose_robot_2_camera_next.position.z)     

            try:
                assert self.change_view(pose_robot_2_camera_next)
            except:
                self.get_logger().info("Failed to move the robot to the next view, keep searching...")
                return
            
        # send the velocity command to the robot

    def compute_velocity_cmd(self, x_d, x, linear_vel=0.05, angular_vel=1):
        r, theta, phi = cartesian_to_spherical(x.translation - self.view_sphere.center)
        e_t = x_d.translation - x.translation # translation error
        e_n = (x.translation - self.view_sphere.center) * (self.view_sphere.r - r) / r # pull the camera towards the sphere
        linear = 1.0 * e_t + 6.0 * (r < self.view_sphere.r) * e_n # weighted sum of the two errors
        scale = np.linalg.norm(linear) + 1e-6
        linear *= np.clip(scale, 0.0, linear_vel) / scale # scale the linear velocity
        angular = self.view_sphere.get_view(theta, phi).rotation * x.rotation.inv() # desired rotation
        angular = angular_vel * angular.as_rotvec()
        return np.r_[linear, angular]

def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_listener = PCDListener()
    try:
        rclpy.spin(pcd_listener)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info("Exiting")

    pcd_listener.destroy_node()
    rclpy.shutdown()

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, choices=registry.keys(), default="nbv")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--wait-for-input", action="store_true")
    parser.add_argument("--logdir", type=Path, default="logs")
    parser.add_argument("--seed", type=int, default=1)
    return parser

if __name__ == "__main__":
    main()

