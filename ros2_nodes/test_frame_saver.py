import threading
from .inference_node_base import *
import time
import rclpy
import cv2
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose, Transform
# from vmf_contact_main.train import main_module, parse_args_from_yaml
from cv_bridge import CvBridge
import os
import torch
import numpy as np
import copy
from tf_transformations import quaternion_from_matrix, translation_from_matrix
from PIL import Image
from ros2_nodes.camera_utils import *
import time
# import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
from pathlib import Path

class FrameSaver(AIRNode):
    def __init__(self):
        super().__init__("frame_saver")
        # Initialize visualization-specific components here
        self.get_logger().info("Frame Saver Node Initialized")
        
        # Create subscriber for frame capture
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth/image_raw',
            self.timer_callback,
            10)

    def timer_callback(self):
        if self.last_point_cloud_msg is None:
            self.get_logger().info("No point cloud message received yet.")
            return [None] * 4, False

        if self.last_image_msg is None:
            self.get_logger().info("No image message received yet.")
            return [None] * 4, False
    
        if self.last_depth_msg is None:
            self.get_logger().info("No depth message received yet.")
            return [None] * 4, False
        
        pcd_msg_camera = self.last_depth_msg

        camera= CameraInfo(
            width=self.image_width, height=self.image_height, fx=self.camera_matrix[0, 0], fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2], cy=self.camera_matrix[1, 2], scale=1000.0
        )
        pcd_from_depth = create_point_cloud_from_depth_image(self.last_depth_msg, camera, organized=False).reshape(-1, 3)
        print("Final cloud shape:", pcd_from_depth.shape)

        sec = pcd_msg_camera.header.stamp.sec
        nanosec = pcd_msg_camera.header.stamp.nanosec
        timestamp = f"{sec}_{nanosec:09d}"

        FILE_DIR = Path("..")
        FILE_NAME = f"vmf_input_{timestamp}.pt"
        FILE_SAVE_DIR = FILE_DIR / FILE_NAME
        self.get_logger().info(f"Saving point cloud {FILE_NAME} to {FILE_SAVE_DIR}")
        tensor = torch.from_numpy(pcd_from_depth) 
        torch.save(tensor, FILE_SAVE_DIR)
        self.get_logger().info(f"Saved {FILE_NAME}")
        rclpy.shutdown()     

def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_listener = FrameSaver()
    try:
        rclpy.spin(pcd_listener)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info("Exiting")

    pcd_listener.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()