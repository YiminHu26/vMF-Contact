import threading
from .inference_node_base import *
import time
import rclpy
import cv2
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose, Transform
# from vmf_contact_main.train import main_module, parse_args_from_yaml
from cv_bridge import CvBridge
import os
# import torch
import numpy as np
import copy
# from tf_transformations import quaternion_from_matrix, translation_from_matrix
from PIL import Image
from ros2_nodes.utils_camera import *
import time
# import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct

class VisualizationNode(AIRNode):
    def __init__(self):
        super().__init__("visualization_node")
        # Initialize visualization-specific components here
        self.get_logger().info("Visualization Node Initialized")
        
        # Create publishers for RViz visualization
        self.pcd_publisher = self.create_publisher(PointCloud2, '/visualization/point_cloud', 10)
        
        self.create_timer(0.1, self.timer_callback) 
        self.counter_ = 0

    def timer_callback(self):
        # self.visualize_data()
        self.process_point_cloud_to_base_link()
        self.get_logger().info("Visualization " + str(self.counter_))
        self.counter_ += 1

    def process_point_cloud_to_base_link(self, pcd_only=False):
        # TODO: add rgb image processing
        if self.last_point_cloud_msg is None:
            self.get_logger().info("No point cloud message received yet.")
            if pcd_only:
                return None, False
            return [None] * 4, False

        if self.last_image_msg is None:
            self.get_logger().info("No image message received yet.")
            return [None] * 4, False
    
        if self.last_depth_msg is None:
            self.get_logger().info("No depth message received yet.")
            return [None] * 4, False
        
        # Get the latest point cloud and image messages
        pcd_msg_camera = self.last_point_cloud_msg

        # transform the point cloud to base_link frame
        camera= CameraInfo(
            width=self.image_width, height=self.image_height, fx=self.camera_matrix[0, 0], fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2], cy=self.camera_matrix[1, 2], scale=1000.0
        )
        pcd_from_depth = create_point_cloud_from_depth_image(self.last_depth_msg, camera, organized=False).reshape(-1, 3)
        
        # Look up for the transformation between base_link and the frame_id of the point cloud
        t_robot_2_camera = self.tf_buffer.lookup_transform(
                "base_link", "orbbec_femto_mega_link", rclpy.time.Time()
            )
        print("depth_msg_stamp: ", pcd_msg_camera.header.stamp)
        print("transform_msg_stamp: ", t_robot_2_camera.header.stamp)
        # pcd_numpy_base_link = transform_points(pcd_msg_camera, t_robot_2_camera.transform) 
        pcd_numpy_base_link = transform_points(pcd_from_depth, t_robot_2_camera.transform) 

        pc2_msg = self.numpy_to_pointcloud2(pcd_numpy_base_link)
        self.pcd_publisher.publish(pc2_msg)
        
    def numpy_to_pointcloud2(self, points):
        """
        Convert numpy array of points (N x 3) to PointCloud2 message.

        :param points: Nx3 numpy array of point coordinates (from utils_node.transform.points() method)
        :return: PointCloud2 message
        """
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'  # Adjust frame_id as needed
        
        # Create PointCloud2 message
        pc2 = PointCloud2()
        pc2.header = header
        pc2.height = 1
        pc2.width = len(points)
        
        # Define point fields
        pc2.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        pc2.is_bigendian = False
        pc2.point_step = 12  # 3 floats * 4 bytes
        pc2.row_step = pc2.point_step * pc2.width
        
        # Pack point data
        pc2.data = b''.join([struct.pack('fff', *point) for point in points])
        
        return pc2
    


def main(args=None):
    # Boilerplate code.
    rclpy.init(args=args)
    pcd_listener = VisualizationNode()
    try:
        rclpy.spin(pcd_listener)
    except SystemExit:
        rclpy.logging.get_logger("Quitting").info("Exiting")

    pcd_listener.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()