import threading
import time
import open3d as o3d
import rclpy
import sensor_msgs.msg as sensor_msgs
from rclpy.node import Node
from rclpy.action import ActionClient
import cv2
from arm_api2_msgs.action import MoveCartesianPath
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
from PIL import Image
from lang_sam import LangSAM
from .camera_utils import *
import time
import subprocess
from scipy.spatial.transform import Rotation as R

use_langsam = False

langsam_model = LangSAM() if use_langsam else None
#image_pil = Image.open("./assets/car.jpeg").convert("RGB")
#text_prompt = "wheel."
#results = model.predict([image_pil], [text_prompt])


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

    def __init__(self):
        super().__init__("pcd_subsriber_node")

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

        self._robot_action_client = ActionClient(self, MoveCartesian, "arm/move_to_pose")
        self._gripper_action_client = ActionClient(
            self, GripperCommand, "robotiq_2f_urcap_adapter/gripper_command"
        )

        self._robot_action_client_traj = ActionClient(
            self, MoveCartesianPath, "arm/move_to_pose_path"
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
        self.agent = main_module(parse_args_from_yaml(current_file_folder + "/config.yaml"), learning=False)
        self.movement_finished_flag = threading.Event()
        self.movement_failed_flag = threading.Event()
        self.gripper_movement_finished_flag = threading.Event()
        self.gripper_movement_failed_flag = threading.Event()
        self.stop_event = threading.Event()
        self.send_traj_flag = threading.Event()
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
            width=self.image_width, height=self.image_height, fx=self.camera_matrix[0, 0], fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2], cy=self.camera_matrix[1, 2], scale=1000.0
        )
        pcd_from_depth = create_point_cloud_from_depth_image(self.last_depth_msg, camera, organized=True).reshape(-1, 3)
        
        # Look up for the transformation between base_link and the frame_id of the point cloud
        from_frame_rel = "base_link"
        to_frame_rel = pcd_msg_camera.header.frame_id
        t_base_link_2_camera = self.tf_buffer.lookup_transform(
                from_frame_rel, to_frame_rel, rclpy.time.Time()
            )
        
        # pcd_msg_base_link_1 = do_transform_cloud(pcd_msg_camera, t_base_link_2_camera)        
        # pcd_numpy_base_link_1 = read_points_numpy(pcd_msg_base_link_1)
        pcd_numpy_base_link = transform_points(pcd_from_depth, t_base_link_2_camera.transform)  
        # print("First 10 points: ", pcd_numpy_base_link[:10])

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
        
        if langsam_model is not None:
            time_curr = time.time()
            prompt_input = ""
            while prompt_input == "":
                prompt_input = input("Please enter what you would like to grasp: ")
            prompt_input = prompt_input.split(".")
            print("Prompt: ", prompt_input)
        
            self.masked_pcd_dict = {}
            # predict masks with lang_sam
            results = langsam_model.predict([Image.fromarray(img)], [". ".join(prompt_input)])

            print(f"Time taken for inference: {time.time() - time_curr}")

            print(f"save images to {current_file_folder}")
            cv2.imwrite(f"{current_file_folder}/image.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # check if there are labels detected
            labels = results[0]["labels"]
            if len(labels) == 0:
                print("No labels detected.")
                return pcd_numpy_base_link, False
            # check duplicates in labels, if there are duplicates, mark them with a number
            labels = mark_duplicates(labels)
            print("Results: ", labels)
            print("Scores: ", results[0]["scores"])

            # mask point cloud and image
            for i, text in enumerate(labels):
                mask = results[0]["masks"][i].astype(np.uint8)[:, :, None]
                # mask image and point cloud
                pcd_masked = pcd_numpy_base_link[mask.reshape(-1) == 1]
                self.masked_pcd_dict[text] = pcd_masked
                
                # save masked image
                cv2.imwrite(f"{current_file_folder}/{text}.jpg", img[..., ::-1] * mask)
                
                # # visualize the masked point cloud
                # pcd_masked_color = img.reshape(-1, 3)[mask.reshape(-1) == 1]
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pcd_masked)
                # pcd.colors = o3d.utility.Vector3dVector(pcd_masked_color[:, ::-1] / 255)
                # o3d.visualization.draw_geometries([pcd], window_name=text)

        return pcd_numpy_base_link, True
    

    def process_point_cloud_and_rgbd(self, save_data=False):
        # TODO: add rgb image processing
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
            width=self.image_width, height=self.image_height, fx=self.camera_matrix[0, 0], fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2], cy=self.camera_matrix[1, 2], scale=1000.0
        )
        pcd_from_depth = create_point_cloud_from_depth_image(self.last_depth_msg, camera, organized=True).reshape(-1, 3)
        
        # Look up for the transformation between base_link and the frame_id of the point cloud
        from_frame_rel = "base_link"
        to_frame_rel = pcd_msg_camera.header.frame_id
        t_base_link_2_camera = self.tf_buffer.lookup_transform(
                from_frame_rel, to_frame_rel, rclpy.time.Time()
            )
        
        # pcd_msg_base_link_1 = do_transform_cloud(pcd_msg_camera, t_base_link_2_camera)        
        # pcd_numpy_base_link_1 = read_points_numpy(pcd_msg_base_link_1)
        pcd_numpy_base_link = transform_points(pcd_from_depth, t_base_link_2_camera.transform)  
        # print("First 10 points: ", pcd_numpy_base_link[:10])

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

        if langsam_model is not None:
            time_curr = time.time()
            prompt_input = ""
            while prompt_input == "":
                prompt_input = input("Please enter what you would like to grasp: ")
            prompt_input = prompt_input.split(".")
            print("Prompt: ", prompt_input)
        
            self.masked_pcd_dict = {}
            # predict masks with lang_sam
            results = langsam_model.predict([Image.fromarray(img)], [". ".join(prompt_input)])

            print(f"Time taken for inference: {time.time() - time_curr}")

            print(f"save images to {current_file_folder}")
            cv2.imwrite(f"{current_file_folder}/image.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # check if there are labels detected
            labels = results[0]["labels"]
            if len(labels) == 0:
                print("No labels detected.")
                return pcd_numpy_base_link, False
            # check duplicates in labels, if there are duplicates, mark them with a number
            labels = mark_duplicates(labels)
            print("Results: ", labels)
            print("Scores: ", results[0]["scores"])

            # mask point cloud and image
            for i, text in enumerate(labels):
                mask = results[0]["masks"][i].astype(np.uint8)[:, :, None]
                # mask image and point cloud
                pcd_masked = pcd_numpy_base_link[mask.reshape(-1) == 1]
                self.masked_pcd_dict[text] = pcd_masked
                
                # save masked image
                cv2.imwrite(f"{current_file_folder}/{text}.jpg", img[..., ::-1] * mask)
                
                # # visualize the masked point cloud
                # pcd_masked_color = img.reshape(-1, 3)[mask.reshape(-1) == 1]
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pcd_masked)
                # pcd.colors = o3d.utility.Vector3dVector(pcd_masked_color[:, ::-1] / 255)
                # o3d.visualization.draw_geometries([pcd], window_name=text)
        
        # transformation to pose
        cam_pose_robot = transform_to_pose(t_base_link_2_camera.transform)

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

        return (pcd_numpy_base_link, self.last_image_msg, self.last_depth_msg, cam_pose_robot), True


    def generate_trajectory(self):
        traj = []
        gaze_point_robot = [-0.74, 0.1, 0.031] # TODO: remove this line, this is a test for gazing at middle of the desk
        dist = .47 # np.linalg.norm(np.array(gaze_point_robot) - np.array(pos))
        azi_ele_groups = [
            [(-90, -50), (30, 155)], 
            [(-50, 0), (50, 135)],
            [(0, 50), (50, 135)],
            [(50, 90), (30, 155)]
            ]
        
        round = 0
        for azi_ele in azi_ele_groups:
            for azimuth in range(*azi_ele[0], 5):
                t = []            
                ele_range = range(*azi_ele[1]) if round % 2 == 0 else range(*azi_ele[1])[::-1]
                for elevation in ele_range:
                    pos = azi_to_pos(azimuth, elevation, dist)
                    pos =[pos[i] + gaze_point_robot[i] for i in range(3)]
                    quaternion = look_at_transformation(gaze_point_robot, pos)
                    t.append(pos + quaternion)
                round += 1
                traj.append(self.create_trajectory(t))

        for t in traj:
            while not self.movement_finished_flag.is_set():
                camera_data, is_data = self.process_point_cloud_and_rgbd(save_data=True)
                # if is_data:
                #     pcd = camera_data[0]
                #     grasp, grasp_criterien = self.agent_inference(pcd)
                pass
            print("Sending trajectory")
            self.movement_finished_flag.clear()
            self.send_goal_traj(t)            
    
    def handle_user_input(self):

        while True:
            self.send_goal(self.camera_ready_pose)
            print("Camera ready pose sent: ", self.camera_ready_pose)
            self.open_gripper()
            user_input = input("Enter 's' to start next capture and 'q' to quit: ")
            if user_input == "s":

                ###select best view until grasp criterien is met
                (pcd, rgb, d, cam_pose), identifier = self.process_point_cloud_and_rgbd()
                if not identifier:
                    print("No object detected, please try again.")
                    continue
                # grasp, grasp_criterien = self.agent_inference(pcd)
                grasp_criterien = False # TODO: use agent to give grasp criterien instead of False, 
                                        # this is a test for view selection
                
                self.generate_trajectory()
                
                while not grasp_criterien:
                    cam_pose = self.VLM_inference(cam_pose, rgb, d) # Including process point cloud, llm inference: rgbd -> pose
                    print("VLM pose: ", cam_pose)
                    self.change_view(cam_pose) # Move robot to the pose
                    print("Changed view")
                    (pcd, rgb, d, cam_pose), identifier = self.process_point_cloud_and_rgbd()
                    if not identifier:
                        print("No object detected, please try again.")
                        continue
                    grasp, grasp_criterien = self.agent_inference(pcd) # TODO: see above

                ### TODO: no need any more
                pcd, identifier = self.process_point_cloud()
                if not identifier:
                    print("No object detected, please try again.")
                    continue
                grasp, grasp_criterien = self.agent_inference(pcd)
                ###

                if grasp is not None:
                    self.execute_grasp(grasp)
                else:
                    print("No grasp pose detected, please try again.")
            elif user_input == "q":
                self.shutdown = True
                break

    def VLM_inference(self, pose: Pose, rgb, d):
        # Extract the current pose
        current_position = pose.position
        current_orientation = pose.orientation

        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_matrix([current_orientation.x, 
                                             current_orientation.y, 
                                             current_orientation.z, 
                                             current_orientation.w])
        pos = [current_position.x, current_position.y, current_position.z]

        # TODO: add vlm inference
        # camera_pos_increment, gaze_point = self.vlm_agent(rgb, d)
        # camera_pos = camera_pos + camera_pos_increment * 0.1
        gaze_point_robot = [-0.74, 0.1, 0.031] # TODO: remove this line, this is a test for gazing at middle of the desk

        quaternion = look_at_transformation(gaze_point_robot, pos)
        
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        return pose
    
    def agent_inference(self, pcd_raw):
        # TODO: add criteria for grasp execution
        # Process the point cloud
        self.pcd_shift = pcd_raw.mean(axis=0)
        self.pcd_shift[2] = 0.0
        pcd = (pcd_raw - self.pcd_shift) / self.pcd_resize
        pcd = pcd[(pcd[:, 0] > -0.3 / self.pcd_resize) & (pcd[:, 0] < 0.3 / self.pcd_resize)]
        pcd = pcd[(pcd[:, 1] > -0.3 / self.pcd_resize) & (pcd[:, 1] < 0.3 / self.pcd_resize)]
        pcd = pcd[(pcd[:, 2] > 0.03) & (pcd[:, 2] < 0.45)]

        print("Processed point cloud: ", pcd.shape)

        # # Viszualize the point cloud
        # o3d_pcd = o3d.geometry.PointCloud(
        #     o3d.utility.Vector3dVector(pcd)
        # )
        # # draw  the origin as a red sphere
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.1, origin=[0, 0, 0]
        # )
        # o3d.visualization.draw_geometries([o3d_pcd, mesh_frame])

        # Process the prompt point cloud
        if langsam_model is not None:
            pcd_from_prompt = []
            for label in self.masked_pcd_dict:
                pcd_from_prompt.append(self.masked_pcd_dict[label])
                break
            pcd_from_prompt = np.concatenate(pcd_from_prompt, axis=0)
            pcd_from_prompt = (pcd_from_prompt - self.pcd_shift) / self.pcd_resize
            print("Prompt point cloud: ", pcd_from_prompt.shape)
        else:
            pcd_from_prompt = None

        # draw the point cloud and prompt point cloud
        # o3d_pcd = o3d.geometry.PointCloud(
        #     o3d.utility.Vector3dVector(pcd)
        # )
        # o3d_pcd_from_prompt = o3d.geometry.PointCloud(
        #     o3d.utility.Vector3dVector(pcd_from_prompt)
        # )
        # # draw  the origin as a red sphere
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.1, origin=[0, 0, 0]
        # )
        # o3d.visualization.draw_geometries([o3d_pcd_from_prompt, mesh_frame])


        # inference
        pose_chosen = self.agent.inference(pcd, 
                                        pcd_from_prompt=pcd_from_prompt,
                                        shift=self.pcd_shift,
                                        graspness_th=0.6,
                                        resize=self.pcd_resize)
        # Add the new geometry for the current frame

        if pose_chosen is None:
            return None, False
        # translate back to the original coordinate system
        print("Chosen pose: ", pose_chosen)
        # pose_chosen[:3, 3] = pose_chosen[:3, 3] * self.pcd_resize + self.pcd_shift
        # Visualize the poses
        # frames = []

        # for i in range(poses.shape[0]):
        #     # Extract the transformation matrix for the ith pose
        #     pose_matrix = poses[i]

        #     # Create a coordinate frame at this pose
        #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)  # Size can be adjusted
        #     frame.transform(pose_matrix)  # Apply the transformation matrix to the frame

        #     # Add the frame to the list of frames
        #     frames.append(frame)
        #     break

        # pcd_raw = o3d.geometry.PointCloud(
        #     o3d.utility.Vector3dVector(pcd_raw)
        # )stop_eventation matrix: ", pose_chosen[:3, :3])

        # Convert the rotation matrix to a quaternion
        quat = quaternion_from_matrix(pose_chosen)
        translation = translation_from_matrix(pose_chosen)
        # print("Chosen quaternion: ", quat)

        pose_chosen = np.concatenate([translation, quat])
        print("Chosen pose: ", pose_chosen)

        return pose_chosen, False
    

    def camera_robot_pose_to_tcp_world_pose(self, pose_base_link_2_camera: Pose):
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
        return pose_stamped, pose_world_2_camera
    

    def change_view(self, pose_base_link_2_camera):

        self.get_logger().info("Sending goal now...")

        try:
            pose_stamped, pose_world_2_camera = self.camera_robot_pose_to_tcp_world_pose(pose_base_link_2_camera)

            self.publish_new_frame(f"camera_target_view", pose_stamped_from_pose(pose_world_2_camera, "world"))

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


    def send_goal_traj(self, goal_path):
        goal_msg = MoveCartesianPath.Goal()
        goal_msg.poses = goal_path

        self.get_logger().info("Waiting for action server...")

        self._robot_action_client_traj.wait_for_server()

        self.get_logger().info("Sending goal request...")

        self._send_goal_future = self._robot_action_client_traj.send_goal_async(
            goal_msg, feedback_callback=self.robot_feedback_callback 
        )

        self._send_goal_future.add_done_callback(self.robot_goal_response_callback)


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

    def create_trajectory(self, data):
        path = []
        for p in data:
            rosp = PoseStamped()
            cam_pose_to_robot = series_to_pose(p)
            tcp_pose_to_world, _ = self.camera_robot_pose_to_tcp_world_pose(cam_pose_to_robot)
            rosp.header.frame_id = "world"
            path.append(tcp_pose_to_world)
        return path


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


# Function to mark duplicates with a number
def mark_duplicates(labels):
    label_count = {}
    result = []
    
    for label in labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
        result.append(f"{label}_{label_count[label]}")
    
    return result
    
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

def series_to_pose(series):
    pose = Pose()
    pose.position.x = series[0]
    pose.position.y = series[1]
    pose.position.z = series[2]
    pose.orientation.x = series[3]
    pose.orientation.y = series[4]
    pose.orientation.z = series[5]
    pose.orientation.w = series[6]
    return pose

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



