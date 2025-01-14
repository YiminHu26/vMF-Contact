import threading
import time
import open3d as o3d
import rclpy
import sensor_msgs.msg as sensor_msgs
from sensor_msgs.msg import PointCloud2
from rclpy.node import Node
from rclpy.action import ActionClient
from arm_api2_msgs.action import MoveCartesian
from control_msgs.action import GripperCommand
from control_msgs.msg import GripperCommand as GripperCommandMsg
from sensor_msgs_py.point_cloud2 import read_points_numpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose, Transform
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import StaticTransformBroadcaster
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud, transform_points
from vmf_contact_main.train import main_module, parse_args_from_yaml
from cv_bridge import CvBridge
import os, torch
import numpy as np
import copy
from tf_transformations import quaternion_matrix, quaternion_from_matrix, translation_from_matrix
import tf2_geometry_msgs
from .camera_utils import *
from math import cos, sin
from scipy import ndimage
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
from vgn.networks import load_network
from vgn.utils import ros_utils
from pathlib import Path

import enum

import sys
print(f"ROS 2 is using Python interpreter: {sys.executable}")


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




class PCDListener(Node):

    def __init__(self, model_type, model_dir):
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
            "/camera/depth/camera_info",  # topic

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

        self.last_point_cloud_msg = None
        self.shutdown = False
        #self.agent = main_module(parse_args_from_yaml(current_file_folder + "/config.yaml"), learning=False)
        model_type = "giga_aff_packed.pt" 
        model_dir = "/home/sheng/GIGA/data/models" 
        self.agent.net = self.load_model(model_type,model_dir)


        self.movement_finished_flag = threading.Event()
        self.movement_failed_flag = threading.Event()
        self.gripper_movement_finished_flag = threading.Event()
        self.gripper_movement_failed_flag = threading.Event()
        self.stop_event = threading.Event()
        self.bridge = CvBridge()

        self.camera_ready_pose = PoseStamped()
        self.camera_ready_pose.header.frame_id = "world"
        
        # small finger
        # self.camera_ready_pose.pose.position.x = -0.435
        # self.camera_ready_pose.pose.position.y = -0.572
        # self.camera_ready_pose.pose.position.z = 1.492

        self.camera_ready_pose.pose.orientation.x = 0.995
        self.camera_ready_pose.pose.orientation.y = 0.009
        self.camera_ready_pose.pose.orientation.z = 0.005
        self.camera_ready_pose.pose.orientation.w = 0.100
        
        # long finger
        self.camera_ready_pose.pose.position.x = -0.435
        self.camera_ready_pose.pose.position.y = -0.794
        self.camera_ready_pose.pose.position.z = 1.381

        self.drop_off_pose = PoseStamped()
        self.drop_off_pose.header.frame_id = "world"
        self.drop_off_pose.pose.position.x = 0.15
        self.drop_off_pose.pose.position.y = -0.75
        self.drop_off_pose.pose.position.z = 1.3
        self.drop_off_pose.pose.orientation.x = 1.0
        self.drop_off_pose.pose.orientation.y = 0.0
        self.drop_off_pose.pose.orientation.z = 0.0
        self.drop_off_pose.pose.orientation.w = 0.0

        self.user_input_thread = threading.Thread(target=self.handle_user_input)
        self.user_input_thread.start()
    
    def load_model(self, model_type, model_dir):

        model_name = f"giga_{model_type}.pt"
        model_path = Path(model_dir) / model_name

        
        self.get_logger().info(f"Load_model: {model_path}")
        return VGNImplicit(
            model_path=model_path,
            model_type="giga",
            best=True,
            qual_th=0.75,
            force_detection=True,
            out_th=0.1,
            select_top=False,
        )

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

    def process_point_cloud(self):
        if self.last_point_cloud_msg is None:
            print("No point cloud message received yet.")
            return None, False

        if self.last_image_msg is None:
            print("No image message received yet.")
            return None, False
    
        if self.last_depth_msg is None:
            print("No depth message received yet.")
            return None, False

        if self.last_point_cloud_msg is None:
            print("No camera info message received yet.")
            return None, False
        
        # Get the latest point cloud and image messages
        pcd_msg_camera = self.last_point_cloud_msg
        img = self.last_image_msg

        # transform the point cloud to base_link frame
        camera= CameraInfo(
            width=self.image_width, 
            height=self.image_height, 
            fx=self.camera_matrix[0, 0], 
            fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2], 
            cy=self.camera_matrix[1, 2], 
            scale=1000.0
        )
        pcd_from_depth = create_point_cloud_from_depth_image(self.last_depth_msg, camera, organized=True).reshape(-1, 3)
        
        # Look up for the transformation between base_link and the frame_id of the point cloud
        from_frame_rel = "base_link"
        to_frame_rel = pcd_msg_camera.header.frame_id
        t_base_link_2_camera = self.tf_buffer.lookup_transform(
                from_frame_rel, to_frame_rel, rclpy.time.Time()
            )
        
        pcd_numpy_base_link = transform_points(pcd_from_depth, t_base_link_2_camera.transform)  
        
        return pcd_numpy_base_link, True  

    def handle_user_input(self):

        while True:
            self.send_goal(self.camera_ready_pose)
            print("Camera ready pose sent: ", self.camera_ready_pose)
            self.open_gripper()

            user_input = input("Enter 's' to start next capture and 'q' to quit: ")
            if user_input == "s":

                pcd, identifier = self.process_point_cloud()
                if not identifier:
                    print("No object detected, please try again.")
                    continue

                grasp = self.agent_inference() 
                if grasp is not None:
                    self.execute_grasp_vgn(grasp)
                else:
                    print("No grasp pose detected, please try again.")
            elif user_input == "q":
                self.shutdown = True
                break
    
  
    def agent_inference(self):
        if self.last_depth_msg is None or self.camera_matrix is None:
            print("Missing depth image or camera info.")
            return None
        
        O_RESOLUTION = 40
        O_SIZE = 0.3
        O_VOXEL_SIZE = O_SIZE / O_RESOLUTION

        camera = CameraInfo(
            width=self.image_width, 
            height=self.image_height, 
            fx=self.camera_matrix[0, 0], 
            fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2], 
            cy=self.camera_matrix[1, 2], 
            scale=1000.0
        )

        depth_imgs = np.expand_dims(self.last_depth_msg, axis=0)
        extrinsics = np.expand_dims(np.eye(4), axis=0)
        print(f"Extrinsics: {extrinsics}")

        tsdf_volume = create_tsdf(O_SIZE, O_RESOLUTION, depth_imgs, camera, extrinsics)
        tsdf_grid = tsdf_volume.get_grid()

        qual_vol, rot_vol, width_vol = predict(tsdf_grid, self.agent.net, self.agent.device)

        grasps, scores = select(qual_vol.copy(), rot_vol, width_vol, threshold=self.agent.qual_th)
        if len(grasps) == 0:
            print("No grasp found by VGN.")
            return None
        chosen_grasp = grasps[0]
        print("Chosen grasp:", chosen_grasp)
        return chosen_grasp

    
    def change_view(self, pose_base_link_2_camera, elevation=0, azimuth=0):

        self.get_logger().info("Sending goal now...")

        try:
            t_world_2_base_link: TransformStamped = self.tf_buffer.lookup_transform(
                "world", "base_link", rclpy.time.Time()
            )
            t_camera_2_tcp: TransformStamped = self.tf_buffer.lookup_transform(
                "camera_color_optical_frame","tcp", rclpy.time.Time()
            )
            # transform posestamped from base_link to world using t_world_2_base_link
            pose_world_2_camera = tf2_geometry_msgs.do_transform_pose(pose_base_link_2_camera, t_world_2_base_link)
            t_world_2_camera = pose_to_transform(pose_world_2_camera, "world")
            pose_camera_2_tcp = transform_to_pose(t_camera_2_tcp.transform)
            pose_world_2_tcp = tf2_geometry_msgs.do_transform_pose(pose_camera_2_tcp, t_world_2_camera)

            pose_stamped = pose_stamped_from_pose(pose_world_2_tcp, "world")
            self.publish_new_frame(f"view_ele_{elevation}_azi_{azimuth}", pose_stamped_from_pose(pose_world_2_camera, "world"))

            print("View pose in world frame: ", pose_stamped.pose.position)
            print("View orientation in world frame: ", pose_stamped.pose.orientation)

        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform pose from base_link to world: {ex}"
            )
            return
        
        self.stop_event.clear()
        self.movement_failed_flag.clear()
        self.movement_finished_flag.clear()
        state_machine_state = IDLE
        
        while True:
            if state_machine_state == IDLE:
                # Send the goal to move to the pregrasp pose
                self.send_goal(pose_stamped)
                # Start the state machine
                state_machine_state = MOVING_TO_PREGRASP
                self.get_logger().info("StateMachine switched to MOVING_TO_PREGRASP")
                # Create a thread to handle the input
                self.cancel_thread = threading.Thread(target=self.get_input)
                self.cancel_thread.start()

            elif state_machine_state == MOVING_TO_PREGRASP:
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

    def execute_grasp_vgn(self, best_grasp: Grasp):
        rot = best_grasp.pose.orientation
        trans = best_grasp.pose.translation

        qx, qy, qz, qw = rot.as_quat()
        grasp_pose_stamped = PoseStamped()
        grasp_pose_stamped.header.frame_id = "world"  
        grasp_pose_stamped.pose.position.x = float(trans[0])
        grasp_pose_stamped.pose.position.y = float(trans[1])
        grasp_pose_stamped.pose.position.z = float(trans[2])
        grasp_pose_stamped.pose.orientation.x = float(qx)
        grasp_pose_stamped.pose.orientation.y = float(qy)
        grasp_pose_stamped.pose.orientation.z = float(qz)
        grasp_pose_stamped.pose.orientation.w = float(qw)

        self.publish_new_frame("vgn_grasp_pose_raw", grasp_pose_stamped)
        self.get_logger().info(f"[VGN] Got grasp pose in 'world': "
                           f"{grasp_pose_stamped.pose.position}, "
                           f"quaternion=({qx}, {qy}, {qz}, {qw})")
        
        try:
            t_world_to_base = self.tf_buffer.lookup_transform(
            "base_link",
            "world",     
            rclpy.time.Time()
        )
            grasp_pose_in_base = tf2_geometry_msgs.do_transform_pose(
            grasp_pose_stamped, t_world_to_base
        )
            grasp_pose_in_base.header.frame_id = "base_link"
        except TransformException as ex:
            self.get_logger().error(f"Failed to transform world->base_link: {ex}")
            return
        
        self.publish_new_frame("vgn_grasp_pose_in_base", grasp_pose_in_base)
        pre_grasp_pose_in_base = self.create_pregrasp_pose(grasp_pose_in_base)
        self.publish_new_frame("vgn_pregrasp_pose_in_base", pre_grasp_pose_in_base)

        user_input = input("Press 'c' to continue with VGN grasp or any other key to quit: ")
        if user_input.lower() != 'c':
            self.get_logger().info("User aborted VGN grasp.")
            return
        self.stop_event.clear()
        self.movement_failed_flag.clear()
        self.movement_finished_flag.clear()
        state_machine_state = IDLE
        while True:
            if state_machine_state == IDLE:
                self.send_goal(pre_grasp_pose_in_base)
                state_machine_state = MOVING_TO_PREGRASP
                self.get_logger().info("[VGN] State -> MOVING_TO_PREGRASP")
            
            elif state_machine_state == MOVING_TO_PREGRASP:
                if self.movement_finished_flag.is_set():
                    self.movement_finished_flag.clear()

                    self.send_goal(grasp_pose_in_base)
                    state_machine_state = MOVING_TO_GRASP
                    self.get_logger().info("[VGN] State -> MOVING_TO_GRASP")
                elif self.movement_failed_flag.is_set():
                    self.movement_failed_flag.clear()
                    state_machine_state = FAILED
                    self.get_logger().info("[VGN] State -> FAILED")

            elif state_machine_state == MOVING_TO_GRASP:
                if self.movement_finished_flag.is_set():
                    self.movement_finished_flag.clear()
                    self.close_gripper()
                    state_machine_state = GRASPING
                    self.get_logger().info("[VGN] State -> GRASPING")
                elif self.movement_failed_flag.is_set():
                    self.movement_failed_flag.clear()
                    state_machine_state = FAILED
                    self.get_logger().info("[VGN] State -> FAILED")

            elif state_machine_state == GRASPING:
                if self.gripper_movement_finished_flag.is_set():
                    self.gripper_movement_finished_flag.clear()

                    lifted_pose_in_base = self.transform_pose_z(
                        copy.deepcopy(grasp_pose_in_base), z_offset=-0.15
                    )
                    self.send_goal(lifted_pose_in_base)
                    state_machine_state = MOVING_TO_PREGRASP_RETURN
                    self.get_logger().info("[VGN] State -> MOVING_TO_PREGRASP_RETURN")
                elif self.gripper_movement_failed_flag.is_set():
                    self.gripper_movement_failed_flag.clear()
                    state_machine_state = FAILED
                    self.get_logger().info("[VGN] State -> FAILED")
            
            elif state_machine_state == MOVING_TO_PREGRASP_RETURN:
                if self.movement_finished_flag.is_set():
                    self.movement_finished_flag.clear()

                    self.send_goal(self.drop_off_pose) 
                    state_machine_state = MOVING_TO_DROP_OFF
                    self.get_logger().info("[VGN] State -> MOVING_TO_DROP_OFF")
                elif self.movement_failed_flag.is_set():
                    self.movement_failed_flag.clear()
                    state_machine_state = FAILED
                    self.get_logger().info("[VGN] State -> FAILED")

            elif state_machine_state == MOVING_TO_DROP_OFF:
                if self.movement_finished_flag.is_set():
                    self.movement_finished_flag.clear()

                    self.open_gripper()
                    state_machine_state = RELEASEING
                    self.get_logger().info("[VGN] State -> RELEASING")
                elif self.movement_failed_flag.is_set():
                    self.movement_failed_flag.clear()
                    state_machine_state = FAILED
                    self.get_logger().info("[VGN] State -> FAILED")

            elif state_machine_state == RELEASEING:
                if self.gripper_movement_finished_flag.is_set():
                    self.gripper_movement_finished_flag.clear()
                    self.send_goal(self.camera_ready_pose)
                    state_machine_state = MOVING_TO_CAMERA_READY
                    self.get_logger().info("[VGN] State -> MOVING_TO_CAMERA_READY")
                elif self.gripper_movement_failed_flag.is_set():
                    self.gripper_movement_failed_flag.clear()
                    state_machine_state = FAILED
                    self.get_logger().info("[VGN] State -> FAILED")

            elif state_machine_state == MOVING_TO_CAMERA_READY:
                if self.movement_finished_flag.is_set():
                    self.movement_finished_flag.clear()

                    state_machine_state = IDLE
                    self.get_logger().info("[VGN] Grasp cycle finished!")
                    break
                elif self.movement_failed_flag.is_set():
                    self.movement_failed_flag.clear()
                    state_machine_state = FAILED
                    self.get_logger().info("[VGN] State -> FAILED")

            elif state_machine_state == FAILED:
                self.get_logger().warning("[VGN] State machine failed. Aborting.")
                break

            if self.stop_event.is_set():
                self.get_logger().info("[VGN] Grasp canceled by user/event.")
            
                self.stop_event.clear()
                break

    def get_input(self):
        try:
            user_input = input("Press 'c' to cancel: ")
            self.stop_event.set()
            print(f"\nYou entered: {user_input}")
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
        self.get_logger().info("Result: {0}".format(result))
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

        self.get_logger().info("Sending goal request...")

        self._send_goal_future = self._gripper_action_client.send_goal_async(
            goal_msg, feedback_callback=self.gripper_feedback_callback
        )

        self._send_goal_future.add_done_callback(self.gripper_goal_response_callback)

    def gripper_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected :(")
            return

        self.get_logger().info("Goal accepted :)")

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.gripper_get_result_callback)

    def gripper_get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: {result}")
        #if result.success:
        self.gripper_movement_finished_flag.set()
        #else:
         #   self.movement_failed_flag.set()

    def gripper_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Feedback: {feedback}")
    
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

    def send_and_wait_for_grasps(self, state):

        points = np.asarray(state.pc.points)
        msg = ros_utils.to_cloud_msg(points, frame="task")
        self.cloud_pub.publish(msg)

        self._msg_event.clear()
        self._current_result = None

        start_time = time.time()
        while rclpy.ok() and not self._msg_event.is_set():
            rclpy.spin_once(self, timeout_sec=0.1) 

        elapsed = time.time() - start_time

        if self._current_result is None:
            grasps, scores = [], []
        else:
            grasps, scores = self.to_grasp_list(self._current_result)

        return grasps, scores, elapsed

    
class TSDFVolume():
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

    def integrate(self, depth_img, camera: CameraInfo, extrinsic):
        """
        Args:
            depth_img: The depth image.
            intrinsic: The intrinsic parameters of a pinhole camera model.
            extrinsics: The transform from the TSDF to camera coordinates, T_eye_task.
        """
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False,
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(camera.width), 
            int(camera.height), 
            float(camera.fx),   
            float(camera.fy),   
            float(camera.cx),   
            float(camera.cy),
        )
        if not isinstance(extrinsic, Transform):
            raise TypeError(f"Expected extrinsic to be Transform, got {type(extrinsic)}")
    

        extrinsic = extrinsic.as_matrix()
        #print(f"Integrating with extrinsic: Rotation:\n{extrinsic.rotation.as_matrix()}, Translation: {extrinsic.translation}")

        self._volume.integrate(rgbd, intrinsic, extrinsic)

    def get_grid(self):
        shape = (1, self.resolution, self.resolution, self.resolution)
        tsdf_grid = np.zeros(shape, dtype=np.float32)
        tsdf_vol = np.asarray(self._volume.extract_volume_tsdf()).astype(np.float32)
        tsdf_vals = np.copy(tsdf_vol[:, 0]).reshape(shape)
        tsdf_weights = np.copy(tsdf_vol[:, 1]).reshape(shape)
        tsdf_grid = np.where(
            (tsdf_weights != 0.0) & (tsdf_vals < 0.98) & (tsdf_vals >= -0.98),
            (tsdf_vals + 1.0) * 0.5,
            tsdf_grid,
        )
        return tsdf_grid

    def get_cloud(self):
        return self._volume.extract_point_cloud()


def create_tsdf(size, resolution, depth_imgs, camera, extrinsics):
    print(f"Extrinsics: {extrinsics}")
    tsdf = TSDFVolume(size, resolution)

    for i, depth_img in enumerate(depth_imgs):
        extrinsic = extrinsics[i]
        print(f"Original extrinsic[{i}]:\n{extrinsic}")

        if isinstance(extrinsic, np.ndarray) and extrinsic.shape == (4, 4):
            rotation = extrinsic[:3, :3]
            translation = extrinsic[:3, 3]
            extrinsic = Transform(Rotation.from_matrix(rotation), translation)
        else:
            raise ValueError(f"Invalid extrinsic format: {type(extrinsic)}, expected 4x4 numpy array.")

        print(f"Transformed extrinsic[{i}]: Rotation:\n{extrinsic.rotation.as_matrix()}, Translation: {extrinsic.translation}")

        tsdf.integrate(depth_img, camera, extrinsic)
    return tsdf

def camera_on_sphere(origin, radius, theta, phi):
    eye = np.r_[
        radius * sin(theta) * cos(phi),
        radius * sin(theta) * sin(phi),
        radius * cos(theta),
    ]
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])  # this breaks when looking straight down
    return Transform.look_at(eye, target, up) * origin.inverse()


class VGN():
    """Grasp reasoning from tsdf using VGN"""
    def __init__(self, model_path, model_type="giga", best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            current_file_folder = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_file_folder, "models")
            model_name = "giga_aff_packed.pt"
            model_path = os.path.join(model_dir, model_name)
        
        self.net = load_network(model_path, self.device, model_type=model_type)
        self.net.eval()
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        #self.visualize = visualize


    def __call__(self, state, scene_mesh=None, aff_kwargs={}):
        if isinstance(state.tsdf, np.ndarray):
            tsdf_vol = state.tsdf
            voxel_size = 0.3 / 40
            size = 0.3
        else:
            tsdf_vol = state.tsdf.get_grid()
            voxel_size = state.tsdf.voxel_size
            size = state.tsdf.size

        tic = time.time()
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)


        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        qual_vol = bound(qual_vol, voxel_size)

        #if self.visualize:
        #    colored_scene_mesh = visual.affordance_visual(
        #       qual_vol, rot_vol.transpose(1, 2, 3, 0),
        #        scene_mesh, size, 40, **aff_kwargs)
                
        grasps, scores = select(qual_vol.copy(), rot_vol, width_vol, threshold=self.qual_th, force_detection=self.force_detection, max_filter_size=8)
        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)

        if len(grasps) > 0:
            if self.best:
                p = np.arange(len(grasps))
            else:
                p = np.random.permutation(len(grasps))

            grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]
            scores = scores[p]

        
        return grasps, scores, toc

class VGNImplicit():
    """Similar to class VGN, but allows implicit network to learn coordinates in a more fine-grained manner"""
    def __init__(self, model_path, model_type="giga", best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False, resolution=40, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            current_file_folder = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_file_folder, "models")
            model_name = "giga_aff_packed.pt"
            model_path = os.path.join(model_dir, model_name)
        
        self.net = load_network(model_path, model_type)
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        #self.visualize = visualize
        
        self.resolution = resolution
        x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution))
        # 1, self.resolution, self.resolution, self.resolution, 3
        pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)
        self.pos = pos.view(1, self.resolution * self.resolution * self.resolution, 3)

    def __call__(self, state, scene_mesh=None, aff_kwargs={}):
        if hasattr(state, 'tsdf_process'):
            tsdf_process = state.tsdf_process
        else:
            tsdf_process = state.tsdf

        if isinstance(state.tsdf, np.ndarray):
            tsdf_vol = state.tsdf
            voxel_size = 0.3 / self.resolution
            size = 0.3
        else:
            tsdf_vol = state.tsdf.get_grid()
            voxel_size = tsdf_process.voxel_size
            tsdf_process = tsdf_process.get_grid()
            size = state.tsdf.size

        tic = time.time()
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.pos, self.net, self.device)
        qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
        width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        qual_vol, rot_vol, width_vol = process(tsdf_process, qual_vol, rot_vol, width_vol, out_th=self.out_th)
        qual_vol = bound(qual_vol, voxel_size)
        #if self.visualize:
        #    colored_scene_mesh = visual.affordance_visual(qual_vol, rot_vol, scene_mesh, size, self.resolution, **aff_kwargs)
        grasps, scores = select(qual_vol.copy(), self.pos.view(self.resolution, self.resolution, self.resolution, 3).cpu(), rot_vol, width_vol, threshold=self.qual_th, force_detection=self.force_detection, max_filter_size=8)
        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)

        new_grasps = []
        if len(grasps) > 0:
            if self.best:
                p = np.arange(len(grasps))
            else:
                p = np.random.permutation(len(grasps))
            for g in grasps[p]:
                pose = g.pose
                pose.translation = (pose.translation + 0.5) * size
                width = g.width * size
                new_grasps.append(Grasp(pose, width))
            scores = scores[p]
        grasps = new_grasps

        return grasps, scores, toc

def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
    # avoid grasp out of bound [0.02  0.02  0.055]
    x_lim = int(limit[0] / voxel_size)
    y_lim = int(limit[1] / voxel_size)
    z_lim = int(limit[2] / voxel_size)
    qual_vol[:x_lim] = 0.0
    qual_vol[-x_lim:] = 0.0
    qual_vol[:, :y_lim] = 0.0
    qual_vol[:, -y_lim:] = 0.0
    qual_vol[:, :, :z_lim] = 0.0
    return qual_vol

def predict(tsdf_vol, net, device):
    assert tsdf_vol.shape == (1, 40, 40, 40)

    # move input to the GPU
    tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(tsdf_vol)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol

def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=1.33,
    max_width=9.33,
    out_th=0.5
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0



    return qual_vol, rot_vol, width_vol


LOW_TH = 0.5

def select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False):
    best_only = False
    qual_vol[qual_vol < LOW_TH] = 0.0
    if force_detection and (qual_vol >= threshold).sum() == 0:
        best_only = True
    else:
        # threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]

    if best_only and len(sorted_grasps) > 0:
        sorted_grasps = [sorted_grasps[0]]
        sorted_scores = [sorted_scores[0]]

    return sorted_grasps, sorted_scores

def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score


class Label(enum.IntEnum):
    FAILURE = 0  # grasp execution failed due to collision or slippage
    SUCCESS = 1  # object was successfully removed


class Grasp():
    """Grasp parameterized as pose of a 2-finger robot hand.
    
    TODO(mbreyer): clarify definition of grasp frame
    """

    def __init__(self, pose, width):
        self.pose = pose
        self.width = width


def to_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation /= voxel_size
    width = grasp.width / voxel_size
    return Grasp(pose, width)


def from_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation *= voxel_size
    width = grasp.width * voxel_size
    return Grasp(pose, width)

def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    model_type = "giga_aff_packed.pt" 
    model_dir = "/home/sheng/GIGA/data/models" 
    pcd_listener = PCDListener(model_type, model_dir)
    try:
        rclpy.spin(pcd_listener)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info("Exiting")

    pcd_listener.destroy_node()
    rclpy.shutdown()
    
def look_at_transformation(gaze_point, robot_position):
    """
    Compute a transformation matrix that aligns the robot's orientation to look at a gaze point.
    
    :param gaze_point: (x, y, z) coordinates of the gaze target in the world frame
    :param robot_position: (x, y, z) coordinates of the robot's reference point (e.g., end-effector or camera)
    :return: (position, quaternion) representing the pose
    """
    gaze_point = np.array(gaze_point)
    robot_position = np.array(robot_position)

    # Compute direction vector from robot to gaze point
    direction = gaze_point - robot_position
    direction /= np.linalg.norm(direction)  # Normalize

    # Define a reference up vector (assuming Z-up world frame)
    left_vector = np.array([0, -1, 0])

    # Compute right vector (cross product of up and direction)
    up_vector = np.cross(left_vector, direction)
    up_vector /= np.linalg.norm(up_vector)

    # Compute new up vector (orthogonal to both direction and right)
    right_vector = np.cross(up_vector, direction)

    # Construct rotation matrix
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, 0] = right_vector
    rotation_matrix[:3, 1] = up_vector
    rotation_matrix[:3, 2] = direction
    rotation_matrix[:3, 3] = robot_position  # Set translation

    # Convert rotation matrix to quaternion
    quaternion = quaternion_from_matrix(rotation_matrix)

    return list(quaternion)

def pose_to_transform(pose: Pose, header = None) -> TransformStamped:
    tf = TransformStamped()
    tf.header.frame_id = header
    tf.transform.translation.x = pose.position.x
    tf.transform.translation.y = pose.position.y
    tf.transform.translation.z = pose.position.z
    tf.transform.rotation.x = pose.orientation.x
    tf.transform.rotation.y = pose.orientation.y
    tf.transform.rotation.z = pose.orientation.z
    tf.transform.rotation.w = pose.orientation.w
    return tf

def transform_to_pose(tf: Transform) -> Pose:
    pose = Pose()
    pose.position.x = tf.translation.x
    pose.position.y = tf.translation.y
    pose.position.z = tf.translation.z
    pose.orientation.x = tf.rotation.x
    pose.orientation.y = tf.rotation.y
    pose.orientation.z = tf.rotation.z
    pose.orientation.w = tf.rotation.w
    return pose

def pose_stamped_from_pose(pose: Pose, frame_id: str) -> PoseStamped:
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = frame_id
    pose_stamped.pose = pose
    return pose_stamped

def azi_to_pos(azimuth, elevation, distance):
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    x = distance * np.cos(azimuth) * np.cos(elevation)
    y = distance * np.sin(azimuth) * np.cos(elevation)
    z = distance * np.sin(elevation)
    return [x, y, z]

if __name__ == "__main__":
    main()

