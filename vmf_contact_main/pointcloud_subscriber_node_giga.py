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
from vgn.networks import load_network
import copy
from tf_transformations import quaternion_matrix, quaternion_from_matrix, translation_from_matrix
import tf2_geometry_msgs
from .camera_utils import *
from math import cos, sin
#import spatialmath as sm
from scipy import ndimage
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
# from vgn.utils import ros_utils
from pathlib import Path
import enum


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

O_RESOLUTION = 600
#O_RESOLUTION = 256
#O_RESOLUTION = 100
O_SIZE = 1.0
O_VOXEL_SIZE = O_SIZE / O_RESOLUTION

class State:
    def __init__(self, tsdf):
        self.tsdf = tsdf

class PCDListener(Node):

    def __init__(self, model_type, model_path):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.last_point_cloud_msg = None
        self.shutdown = False
        #self.agent = main_module(parse_args_from_yaml(current_file_folder + "/config.yaml"), learning=False)
        model_type = "giga" 
        model_path = "/home/sheng/GIGA/data/models/giga_packed.pt" 
        self.agent = load_network(model_path, self.device, model_type=model_type)
        #print(f"self.agent: {self.agent}")



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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                    self.execute_grasp(grasp)
                else:
                    print("No grasp pose detected, please try again.")
            elif user_input == "q":
                self.shutdown = True
                break
    
    def predict(self, depth: np.ndarray, camTtask_np: np.ndarray, reconstruction: bool = False):
        assert camTtask_np.shape == (4, 4)

        start_time = time.time()
        tsdf_volume = create_tsdf(
            size=O_SIZE,
            resolution=O_RESOLUTION,
            depth_imgs=np.expand_dims(depth, axis=0),
            intrinsic=self.camera_intrinsic,
            extrinsics=np.expand_dims(camTtask_np, axis=0),
        )

        state = State(tsdf=tsdf_volume)
        grasps, scores, toc = self.giga_model(state)
        inference_time = time.time() - start_time

        if reconstruction:
            pc_torch = torch.tensor(tsdf_volume.get_grid())
            pred_mesh, _ = self.generator.generate_mesh({"inputs": pc_torch})
        else:
            pred_mesh = None
        return grasps, scores, inference_time, tsdf_volume.get_cloud(), pred_mesh
  
    def agent_inference(self):
        if self.last_depth_msg is None or self.camera_matrix is None:
            print("Missing depth image or camera info.")
            return None
        
        print("Depth message stats:", np.min(self.last_depth_msg), np.max(self.last_depth_msg))

        

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

        extrinsics = self.get_extrinsics()
        #print(f"Extrinsics: {extrinsics}")
        print("Camera matrix:", self.camera_matrix)
        print("Extrinsics:", extrinsics)

        tsdf_volume = create_tsdf(O_SIZE, O_RESOLUTION, depth_imgs, camera, extrinsics)

        tsdf_grid = tsdf_volume.get_grid()
        print(tsdf_grid.max())
        pcd = tsdf_volume.get_cloud()

        o3d.visualization.draw_geometries([pcd])

        print(self.agent)

        # Perform inference
        grasps, scores, inference_time, tsdf_pc, pred_mesh = self.predict(
            self.depth, self.camTtask.A, reconstruction=False
        )
        #best_grasp = sm.SE3.Rt(R=grasps[0].pose.rotation.as_matrix(), t=grasps[0].pose.translation)
        #wTgrasp = self.wTtask * best_grasp

        print(f"Number of grasps: {len(grasps)}")
        print(f"Top grasp score: {max(scores) if scores else 'N/A'}")

        return grasps, scores
    
    def print_grasp_details(self, grasp):
        print("Grasp Detailed Info:")
        print("  Width: ", grasp.width)
        if hasattr(grasp.pose, 'translation'):
            print("  Translation: ", grasp.pose.translation)
        else:
            print("  Translation: N/A")
        if hasattr(grasp.pose, 'rotation'):
            print("  Rotation: ", grasp.pose.rotation)
        else:
            print("  Rotation: N/A")


    def get_extrinsics(self):
        extrinsics = np.expand_dims(np.eye(4), axis=0)

        t_robot_2_camera: TransformStamped = self.tf_buffer.lookup_transform(
            "camera_color_optical_frame","base_link", rclpy.time.Time()
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

        return  extrinsics

    
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
    
    def execute_grasp(self, pose):

        self.get_logger().info("Sending goal now...")
        grasp_pose = pose_stamped_from_pose(pose, "base_link")

        self.publish_new_frame("grasp_before", grasp_pose)

        grasp_pose2 = copy.deepcopy(grasp_pose)

        try:
            t_world_2_base_link = self.tf_buffer.lookup_transform(
                "world", "base_link", rclpy.time.Time()
            )

            # transform posestamped from base_link to world using t_world_2_base_link
            grasp_pose2.pose = tf2_geometry_msgs.do_transform_pose(grasp_pose.pose, t_world_2_base_link)
            grasp_pose2.header.frame_id = "world"

            print("Grasp pose in world frame: ", grasp_pose2.pose.position)
            print("Grasp orientation in world frame: ", grasp_pose2.pose.orientation)

        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform pose from base_link to world: {ex}"
            )
            return
        
        pregrasp_pose = self.create_pregrasp_pose(copy.deepcopy(grasp_pose2))

        self.publish_new_frame("grasp_after", grasp_pose2)
        self.publish_new_frame("pregrasp", pregrasp_pose)

        print("Grasp pose frame ", grasp_pose2.header.frame_id)
        print("Pregrasp frame ", pregrasp_pose.header.frame_id)
        
        # ask if the user wants to continue
        user_input = input("Press 'c' to continue or any other key to quit: ")

        if user_input.lower() == 'c':

            self.stop_event.clear()
            self.movement_failed_flag.clear()
            self.movement_finished_flag.clear()
            state_machine_state = IDLE
            
            while True:
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
                    if self.stop_event.is_set():
                        self.stop_event.clear()
                        self.get_logger().info("Goal cancelled")
                        self.send_goal(self.camera_ready_pose)
                        state_machine_state = MOVING_TO_CAMERA_READY
                        self.get_logger().info("StateMachine switched to MOVING_TO_CAMERA_READY")
                    # Wait for the action server to finish
                    elif self.movement_finished_flag.is_set():
                        self.movement_finished_flag.clear()
                        # Wait for manual grasp evaluation
                        time.sleep(2)
                        if not self.stop_event.is_set():
                            self.send_goal(grasp_pose2)
                            # Start the state machine
                            state_machine_state = MOVING_TO_GRASP
                            self.get_logger().info("StateMachine switched to MOVING_TO_GRASP")
                            time.sleep(0.1)
                    elif self.movement_failed_flag.is_set():
                        self.movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                        
                elif state_machine_state == MOVING_TO_GRASP:
                    # Wait for the action server to finish
                    if self.movement_finished_flag.is_set():
                        self.movement_finished_flag.clear()
                        time.sleep(2)
                        if not self.stop_event.is_set():
                            # Close the gripper
                            self.close_gripper()
                            state_machine_state = GRASPING
                            self.get_logger().info("StateMachine switched to GRASPING")
                        else:
                            self.stop_event.clear()
                            self.get_logger().info("Goal cancelled")
                            self.send_goal(self.camera_ready_pose)
                            state_machine_state = MOVING_TO_CAMERA_READY
                            self.get_logger().info("StateMachine switched to MOVING_TO_CAMERA_READY")
                    elif self.movement_failed_flag.is_set():
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
                        self.send_goal(self.transform_pose_z(copy.deepcopy(grasp_pose2), z_offset=-0.15))
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
                        self.get_logger().info("State machine finished")
                        break
                    if self.movement_failed_flag.is_set():
                        self.movement_failed_flag.clear()
                        state_machine_state = FAILED
                        self.get_logger().info("StateMachine switched to FAILED")
                
                elif state_machine_state == FAILED:
                    self.get_logger().info("State machine failed")
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

'''
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
'''
    
class TSDFVolume():
    """Integration of multiple depth images using a TSDF."""

    def __init__(self, size, resolution,dynamic_center=False):
        self.size = size
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = 4 * self.voxel_size
       # self.dynamic_center = dynamic_center
        print(f"Voxel size: {self.voxel_size} meters")
        print(f"SDF truncation distance: {self.sdf_trunc} meters")


        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )
    
    def adjust_tsdf_center(self, translation):
        if self.dynamic_center:
            print(f"Adjusting TSDF center to camera position: {translation}")
            self._volume = o3d.pipelines.integration.UniformTSDFVolume(
                length=self.size,
                resolution=self.resolution,
                sdf_trunc=self.sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
            )

    def integrate(self, depth_img, camera: CameraInfo, extrinsic):
        depth_img = np.float32(depth_img)

        print(f"Depth image stats before processing: min={np.min(depth_img)}, max={np.max(depth_img)}")
        depth_img[depth_img <= 0] = 0.001 
        print(f"Depth image stats after replacing invalid values: min={np.min(depth_img)}, max={np.max(depth_img)}")

        depth_img = np.clip(depth_img, 0.001, 900.0) 
        print(f"Processed depth image stats: min={np.min(depth_img)}, max={np.max(depth_img)}")

        color_image = np.zeros_like(depth_img, dtype=np.uint8) 
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_img),
            depth_scale=1000.0,
            depth_trunc=1.0,
            convert_rgb_to_intensity=False,
        )
        print(f"RGBD depth image stats: min={np.min(np.asarray(rgbd.depth))}, max={np.max(np.asarray(rgbd.depth))}")


        tsdf_min = -self.size / 2
        tsdf_max = self.size / 2
        print(f"TSDF range: x={tsdf_min} to {tsdf_max}, y={tsdf_min} to {tsdf_max}, z={tsdf_min} to {tsdf_max}")



        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(camera.width),
            int(camera.height), 
            float(camera.fx), 
            float(camera.fy),   
            float(camera.cx),   
            float(camera.cy),
        )

        if not isinstance(extrinsic, Transform):
            raise TypeError(f"Expected extrinsics to be Transform, got {type(extrinsic)}")
    
        extrinsics = extrinsic.as_matrix()
        extrinsics = np.linalg.inv(extrinsics)


        print(f"Extrinsics translation: {extrinsics[:3, 3]}")
        
        #extrinsic = np.eye(4)
        print(f"Transform applied: {extrinsic}")

        self._volume.integrate(rgbd, intrinsic, extrinsics)
        print(f"TSDF stats after integration: {self._volume.extract_point_cloud().points}")

    def get_grid(self):
        """Extract the TSDF grid for further processing."""
        tsdf_grid = np.zeros((1, self.resolution, self.resolution, self.resolution), dtype=np.float32)

        try:
            tsdf_vol = np.asarray(self._volume.extract_volume_tsdf()).astype(np.float32)
            tsdf_vals = np.copy(tsdf_vol[:, 0]).reshape((1, self.resolution, self.resolution, self.resolution))
            tsdf_weights = np.copy(tsdf_vol[:, 1]).reshape((1, self.resolution, self.resolution, self.resolution))

            tsdf_grid = np.where(
                (tsdf_weights != 0.0) & (tsdf_vals < 0.98) & (tsdf_vals >= -0.98),
                (tsdf_vals + 1.0) * 0.5,
                tsdf_grid,
            )
            print("TSDF grid stats: min =", tsdf_grid.min(), ", max =", tsdf_grid.max())

        except Exception as e:
            print(f"Error extracting TSDF grid: {e}")

        return tsdf_grid

    def get_cloud(self):
        return self._volume.extract_point_cloud()

def create_tsdf(size, resolution, depth_imgs, camera, extrinsics):
    #print(f"Extrinsics: {extrinsics}")
    tsdf = TSDFVolume(size, resolution)

    for i, depth_img in enumerate(depth_imgs):
     
        #normalized_depth = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        #normalized_depth = normalized_depth.astype(np.uint8) 

        #cv2.imshow("Depth Image", normalized_depth)
        #cv2.waitKey(0)  
        #cv2.destroyAllWindows()
        extrinsic = extrinsics[i]
        #print(f"Original extrinsic[{i}]:\n{extrinsic}")

        if isinstance(extrinsic, np.ndarray) and extrinsic.shape == (4, 4):
            rotation = extrinsic[:3, :3]
            translation = extrinsic[:3, 3]
            extrinsic = Transform(Rotation.from_matrix(rotation), translation)
        else:
            raise ValueError(f"Invalid extrinsic format: {type(extrinsic)}, expected 4x4 numpy array.")

        # print(f"Transformed extrinsic[{i}]: Rotation:\n{extrinsic.rotation.as_matrix()}, Translation: {extrinsic.translation}")

        tsdf.integrate(depth_img, camera, extrinsic)
    print(f"Processing depth image {i}, min: {np.min(depth_img)}, max: {np.max(depth_img)}")

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

        if model_path is None:
            current_file_folder = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_file_folder, "models")
            model_name = "giga_packed.pt"
            model_path = os.path.join(model_dir, model_name)
        
        #self.net = load_network(model_path, self.device, model_type=model_type)
        self.net.eval()
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        #self.visualize = visualize


    def __call__(self, state, scene_mesh=None, aff_kwargs={}):
        if isinstance(state.tsdf, np.ndarray):
            tsdf_vol = state.tsdf
            voxel_size = 0.3 / self.resolution
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

class VGNImplicit(PCDListener):
    """Similar to class VGN, but allows implicit network to learn coordinates in a more fine-grained manner"""
    def __init__(self, model_path, model_type="giga", best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False, resolution=100, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.net = load_network(model_path, self.device, model_type=model_type)

        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.tsdf = None
        #self.visualize = visualize
        
        self.resolution = resolution
        x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution))
        # 1, self.resolution, self.resolution, self.resolution, 3
        pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)
        self.pos = pos.view(1, self.resolution * self.resolution * self.resolution, 3)

    def __call__(self, tsdf_volume, scene_mesh=None, aff_kwargs={}):
        if isinstance(tsdf_volume, np.ndarray):

            tsdf_vol = tsdf_volume
            voxel_size = 0.3 / self.resolution 
            size = 0.3 
        else:
            tsdf_vol = tsdf_volume.get_grid()  
            voxel_size = tsdf_volume.voxel_size 
            size = tsdf_volume.size

        tic = time.time()
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.pos, self.net, self.device)
        qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        rot_vol = rot_vol.reshape((self.resolution, self.resolution, self.resolution, 4))
        width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        #qual_vol, rot_vol, width_vol = process(tsdf_process, qual_vol, rot_vol, width_vol, out_th=self.out_th)
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

        return grasps, scores

def predict(tsdf_vol, pos, net, device):
    assert tsdf_vol.shape == (1, 100, 100, 100)

    # move input to the GPU
    tsdf_vol = torch.from_numpy(tsdf_vol).to(device)

    # forward pass
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(tsdf_vol, pos)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol

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




"""
def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=0.033,
    max_width=0.233,
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
"""

LOW_TH = 0.5

def select(qual_vol, center_vol,rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False):
    """
    print("Size before reshape:")
    print("qual_vol shape:", qual_vol.shape)
    print("rot_vol shape:", rot_vol.shape)
    print("width_vol shape:", width_vol.shape)

    grid_size = int(round(qual_vol.size ** (1 / 3)))
    if grid_size ** 3 != qual_vol.size:
        raise ValueError("Size of qual_vol: {})".format(qual_vol.size))
    
    qual_vol = qual_vol.reshape((grid_size, grid_size, grid_size))
    width_vol = width_vol.reshape((grid_size, grid_size, grid_size))

    rot_vol = rot_vol.reshape((grid_size, grid_size, grid_size, 4))

    print("Size after reshape:")
    print("qual_vol shape:", qual_vol.shape)
    print("rot_vol shape:", rot_vol.shape)
    print("width_vol shape:", width_vol.shape)
    """

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
    print("mask shape:", mask.shape)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        print("Index:", index)
        grasp, score = select_index(qual_vol,center_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]

    if best_only and len(sorted_grasps) > 0:
        sorted_grasps = [sorted_grasps[0]]
        sorted_scores = [sorted_scores[0]]

    return sorted_grasps, sorted_scores

def select_index(qual_vol,center_vol, rot_vol, width_vol, index):
    #print("index:", index)
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[i, j, k])
    pos = center_vol[i, j, k].numpy()
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
    model_type = "giga_packed.pt" 
    model_path = "/home/sheng/GIGA/data/models" 
    pcd_listener = PCDListener(model_type, model_path)
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

def pose_stamped_from_pose(pose_in: Pose, frame_id: str) -> PoseStamped:

    if not isinstance(pose_in, Pose):
        pose = transform_to_pose(pose_in)
    else:
        pose = pose_in
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

