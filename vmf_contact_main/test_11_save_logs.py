import argparse
from datetime import datetime
import logging
import os
from typing import Optional
import time
from zoneinfo import ZoneInfo
import open3d as o3d
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import sys, os
import rclpy
from rclpy.duration import Duration as RclpyDuration
from geometry_msgs.msg import PoseStamped
import numpy as np
# Compatibility shim for deps that still reference np.float (removed in NumPy 1.24).
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
    
from tf_transformations import quaternion_from_matrix, translation_from_matrix, quaternion_matrix
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from vmf_contact import vmfContactModule
from vmf_contact import DATASET_REGISTRY
from openpoints.utils import EasyConfig
import glob
import warnings

logger = logging.getLogger("vmf_contact")

def resolve_data_path() -> str:
    data_path = os.environ.get("LSDFPROJECTS")
    if data_path is None or not os.path.exists(data_path):
        data_path = ".."
    assert os.path.exists(data_path), f"Data path {data_path} does not exist. Please set it."
    return data_path

def suppress_pytorch_lightning_logs():
    """
    Suppresses annoying PyTorch Lightning logs.
    """
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers`.*")
    warnings.filterwarnings("ignore", ".*this may lead to large memory footprint.*")
    warnings.filterwarnings("ignore", ".*DataModule.setup has already been called.*")
    warnings.filterwarnings("ignore", ".*DataModule.teardown has already been called.*")
    warnings.filterwarnings("ignore", ".*Set the gpus flag in your trainer.*")
    warnings.filterwarnings("ignore", ".*It is recommended to use.*")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)




def get_args_parser(
    description: Optional[str] = None,
    add_help: bool = True,
    data_path: Optional[str] = None,
):
    if data_path is None:
        data_path = resolve_data_path()

    parser = argparse.ArgumentParser(
        description=description,
        add_help=add_help,
    )
    # Training
    parser.add_argument(
        "--devices",
        type=int,
        default=4,
        help="Distributed training device number",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--learning_rate_score",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--learning_rate_decay",
        type=float,
        default=5e-4,
        help="Learning rate decay",
    )
    parser.add_argument(
        "--run-finetuning",
        action="store_true",
        help="Whether to run finetuning",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=20000,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--flow_finetune",
        type=int,
        default=0,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--camera-num",
        type=int,
        help="Number of cameras",
        default=2,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Whether to evaluate the model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint to validate, if None then training",
    )
    # Data module
    parser.add_argument(
        "--data-root-dir",
        type=str,
        help="Root directory of the data",
        default="dataset/vmf_data",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Embedding dimension vmfContact",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(480, 640),
        help="Image size",
    )
    parser.add_argument(
        "--pcd_with_rgb",
        action="store_true",
        help="Whether to use RGB with PCD",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7 / 8,
        help="Image size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mgn",
        help="Dataset name",
    )
    ## Uncertainty
    parser.add_argument(
        "--prob_baseline",
        type=str,
        default=None,
        choices=["post", "lh", None],
        help="Baseline vector modeled as a constant or a learnable parameter",
    )
    parser.add_argument(
        "--certainty_budget",
        type=str,
        default="constant",
        help="Certainty budget",
    )
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=1e-6,
        help="The weight for the entropy regularizer.",
    )

    parser.add_argument(
        "--flow_layers",
        type=int,
        default=4,
        help="Number of flow layers",
    )

    parser.add_argument(
        "--point_backbone",
        type=str,
        default=None,
        choices=["pointnet++", "pointnext-s", "pointnext-b", "pointnext-l","spotr", "dgcnn"],
    )
    # Flow
    parser.add_argument(
        "--hidden_feat_flow",
        type=int,
        default=512,
        help="Hidden feature size for the flow",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=256,
        help="Embedding dimension vmfContact",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="clip",
        choices=["clip", "vits", "vitb", "vitl", "vitg", "resnet"],
    )

    parser.add_argument(
        "--flow_type",
        type=str,
        default="resflow",
        choices=["resflow", "glow"],
    )

    parser.set_defaults(
        epochs=10,
        num_workers=10,
        epoch_length=1250,
        learning_rates=[
            1e-5,
            2e-5,
            5e-5,
            1e-4,
            2e-4,
            5e-4,
            1e-3,
            2e-3,
            5e-3,
            1e-2,
            2e-2,
            5e-2,
            0.1,
        ],
        data_root_dir=[f"{data_path}/vmf_data"],
    )
    return parser


import yaml

def parse_args_from_yaml(config_path: str, data_path: Optional[str] = None):
    # Load default configurations from YAML
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Create the argument parser
    parser = get_args_parser(data_path=data_path)

    # Set defaults from YAML file
    parser.set_defaults(**yaml_config)

    # Parse command-line arguments (they will override YAML defaults)
    args = parser.parse_args()

    print(args)

    return args
    

import torch
from ros2_nodes.inference_node_base import *
from ros2_nodes.utils_camera import *
from ros2_nodes.utils_node import *


class InferenceTest2(AIRNode):
    """
    ROS2 node that waits for depth + camera info, computes a base_link point cloud,
    runs inference on demand, and publishes the pose.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        output_name: Optional[str] = None,
    ):
        super().__init__()
        self.args = args
        self.model = model
        self.output_name = output_name
        self.last_depth_msg = None
        self.camera_matrix = None
        self.image_width = None
        self.image_height = None
        self.grasp_pose_publisher = self.create_publisher(PoseStamped, "/arm_vmf/pose_chosen", 10)
        # self.place_pose_publisher = self.create_publisher(PoseStamped, "/arm_vmf/place_pose", 10)
        self.inference_done = False
        self.inference_count = 0
        self.output_name_base = output_name or time.strftime("vmf_%Y%m%d_%H%M%S")
        self.current_output_name = self.output_name_base
        self._tf_wait_logged = False
        self.get_logger().info("Waiting for depth + camera info...")
        self.timer = None

    def start_inference_timer(self) -> None:
        if self.timer is None:
            self.timer = self.create_timer(0.1, self._tick)

    def wait_for_initial_inputs(self, timeout_sec: Optional[float] = 2.0) -> bool:
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        while (
            rclpy.ok()
            and not self._inputs_ready()
            and (deadline is None or time.monotonic() < deadline)
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self._inputs_ready():
            if timeout_sec is None:
                self.get_logger().warning("Stopped before depth/camera info was ready.")
            else:
                self.get_logger().warning(
                    f"No depth/camera info after {timeout_sec:.1f}s fallback wait; "
                    "continuing to wait in timer."
                )
            return False

        self.get_logger().info("Depth + camera info received.")
        return True

    def _prepare_next_inference(self) -> None:
        self.inference_count += 1
        self.current_output_name = f"{self.output_name_base}_{self.inference_count:03d}"
        self.inference_done = False
        self.last_depth_msg = None
        self.get_logger().info(
            f"Starting inference cycle {self.inference_count}; output prefix: "
            f"{self.current_output_name}"
        )
        self.get_logger().info("Waiting for a fresh depth frame...")

    def _wait_for_tf(self) -> bool:
        while rclpy.ok() and not self._tf_ready():
            rclpy.spin_once(self, timeout_sec=0.1)
        return rclpy.ok()

    def run_inference_cycle(self) -> None:
        if not self.wait_for_initial_inputs(timeout_sec=None):
            return
        if not self._wait_for_tf():
            return

        pcd_from_saver = self.compute_pcd_base(target_points=100000)
        self.run_inference_once(pcd_from_saver)
        self.inference_done = True

    def run_interactive_loop(self) -> None:
        while rclpy.ok():
            try:
                input(
                    f"\n[{self.inference_count + 1}] Press Enter to receive a "
                    "point cloud and run inference (Ctrl+C to exit)..."
                )
            except EOFError:
                self.get_logger().warning("stdin closed; stopping inference loop.")
                break

            self._prepare_next_inference()
            try:
                self.run_inference_cycle()
            except Exception as exc:
                self.get_logger().error(f"Inference failed: {exc}")
            finally:
                self.inference_done = True
                self.get_logger().info(
                    "Inference cycle complete. Press Enter for the next cycle."
                )

    def _inputs_ready(self) -> bool:
        return (
            getattr(self, "last_depth_msg", None) is not None
            and getattr(self, "camera_matrix", None) is not None
        )

    def _tf_ready(self) -> bool:
        try:
            ready = self.tf_buffer.can_transform(
                "base_link",
                "orbbec_femto_mega_link",
                rclpy.time.Time(),
                timeout=RclpyDuration(seconds=0.2),
            )
        except Exception as exc:
            if not self._tf_wait_logged:
                self.get_logger().warning(f"TF not ready yet: {exc}")
                self._tf_wait_logged = True
            return False

        if not ready and not self._tf_wait_logged:
            self.get_logger().info(
                "Waiting for TF: base_link -> orbbec_femto_mega_link"
            )
            self._tf_wait_logged = True
        if ready:
            self._tf_wait_logged = False
        return ready

    def _tick(self) -> None:
        if self.inference_done:
            self.get_logger().info("Inference already done, skipping.")
            return
        if not self._inputs_ready():
            self.get_logger().warning("Inputs not ready; waiting for depth + camera info.")
            return
        if not self._tf_ready():
            self.get_logger().warning("TF not ready.")
            return

        try:
            pcd_from_saver = self.compute_pcd_base(target_points=100000)
            self.run_inference_once(pcd_from_saver)
        except Exception as exc:
            self.get_logger().error(f"Inference failed: {exc}")
        finally:
            self.inference_done = True
            self.get_logger().info("Inference complete.")

    def compute_pcd_base(self, target_points: int = 40000) -> np.ndarray:
        if self.last_depth_msg is None or self.camera_matrix is None:
            raise RuntimeError("Depth or camera info not ready.")

        depth_image = self.last_depth_msg
        self.get_logger().info(f"Shape of the depth image: {depth_image.shape}")

        camera = CameraInfo(
            width=self.image_width,
            height=self.image_height,
            fx=self.camera_matrix[0, 0],
            fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2],
            cy=self.camera_matrix[1, 2],
            scale=1000.0,
        )

        pcd = create_point_cloud_from_depth_image(
            depth_image, camera, organized=False
        )
        pcd = self._resample_pointcloud(pcd, target_points=target_points)
        self.get_logger().info(f"Shape of pcd: {pcd.shape}")

        t = self.tf_buffer.lookup_transform(
            "base_link",
            "orbbec_femto_mega_link",
            rclpy.time.Time(),
            timeout=RclpyDuration(seconds=0.2)
        )
        self.get_logger().info(f"Transform:\n{t}")

        pcd_base = transform_points(pcd, t.transform)
        self.get_logger().info(f"Shape of pcd_base: {pcd_base.shape}")
        return pcd_base

    def compute_obb_pose_with_world_z(
        self,
        obb: o3d.geometry.OrientedBoundingBox,
        points_np: np.ndarray,
        world_z: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> tuple[np.ndarray, np.ndarray]:
        ''' Compute a pose for the Oriented Bounding Box (OBB) of a point cloud,

        with the constraint that the Z-axis of the OBB must be aligned with the given world Z direction,
        and the x-axis should be oriented such that the positive end of x-axis points towards the side
        with higher average height (z value) in the point cloud.
        '''
        center = np.asarray(obb.center)
        rot = np.asarray(obb.R)
        extent = np.asarray(obb.extent)

        axis_order = np.argsort(extent)[::-1]
        world_z = world_z / np.linalg.norm(world_z)

        x_axis = None
        for axis_idx in axis_order:
            candidate = rot[:, axis_idx]
            candidate = candidate - np.dot(candidate, world_z) * world_z
            candidate_norm = np.linalg.norm(candidate)
            if candidate_norm > 1e-8:
                x_axis = candidate / candidate_norm
                break

        if x_axis is None:
            x_axis = np.array([1.0, 0.0, 0.0])

        centered_points = points_np - center
        points_along_x = centered_points @ x_axis
        span_along_x = np.max(np.abs(points_along_x))

        if span_along_x > 1e-8:
            end_band_threshold = 0.6 * span_along_x
            pos_end_points = points_np[points_along_x >= end_band_threshold]
            neg_end_points = points_np[points_along_x <= -end_band_threshold]

            if len(pos_end_points) > 0 and len(neg_end_points) > 0:
                pos_mean_height = np.mean(pos_end_points[:, 2])
                neg_mean_height = np.mean(neg_end_points[:, 2])
                if neg_mean_height > pos_mean_height:
                    x_axis = -x_axis

        z_axis = world_z

        y_axis = np.cross(z_axis, x_axis)
        y_axis_norm = np.linalg.norm(y_axis)
        if y_axis_norm <= 1e-8:
            y_axis = np.array([0.0, 1.0, 0.0])
        else:
            y_axis /= y_axis_norm

        constrained_rot = np.column_stack((x_axis, y_axis, z_axis))
        return center, constrained_rot

    def _resample_pointcloud(self, pcd: np.ndarray, target_points: int = 40000) -> np.ndarray:
        num_points = pcd.shape[0]
        if num_points == target_points:
            return pcd

        replace = num_points < target_points
        sampled_indices = np.random.choice(num_points, size=target_points, replace=replace)
        return pcd[sampled_indices]


    # def _lineset_between_points(self, a: np.ndarray, b: np.ndarray, color: np.ndarray) -> o3d.geometry.LineSet:
    #     line = o3d.geometry.LineSet()
    #     line.points = o3d.utility.Vector3dVector([a, b])
    #     line.lines = o3d.utility.Vector2iVector([[0, 1]])
    #     line.colors = o3d.utility.Vector3dVector(np.tile(color, (1, 1)))
    #     return line

    def _pose_axes_as_point_cloud(
        self,
        pos: np.ndarray,
        rot: np.ndarray,
        axis_length: float,
        points_per_axis: int = 100,
    ) -> o3d.geometry.PointCloud:
        axis_points = []
        axis_colors = []
        for axis_idx, color in enumerate(
            (
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            )
        ):
            t_values = np.linspace(0.0, axis_length, points_per_axis)
            points = pos + np.outer(t_values, rot[:, axis_idx])
            axis_points.append(points)
            axis_colors.append(np.tile(color, (points_per_axis, 1)))

        pose_axes = o3d.geometry.PointCloud()
        pose_axes.points = o3d.utility.Vector3dVector(np.vstack(axis_points))
        pose_axes.colors = o3d.utility.Vector3dVector(np.vstack(axis_colors))
        return pose_axes

    def _visualize_pose_and_pcd(
        self,
        pcd_np: np.ndarray,
        pose_msg: PoseStamped,
        save_pcd: Optional[bool] = None,
        save_path: Optional[str] = None,
        axis_length: float = 0.05,
        grasp_x_offset: float = 0.0,
        grasp_y_offset: float = 0.0,
        grasp_z_offset: float = 0.0,
    ) -> None:
        ''' Visualize the point cloud and the predicted grasp pose in open3d, 
        
        and optionally save the combined point cloud with grasp pose as a ply file.
        (The grasp pose would be saved as lines visually but actually points in the ply file)
        '''
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(pcd_np)

        # grasp_x_offset = -0.70
        # grasp_y_offset = 0.0
        # grasp_z_offset = 0.131

        pos = np.array(
            [
                pose_msg.pose.position.x - grasp_x_offset,
                pose_msg.pose.position.y - grasp_y_offset,
                pose_msg.pose.position.z - grasp_z_offset,
            ],
            dtype=np.float32,
        )
        quat = np.array(
            [
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w,
            ],
            dtype=np.float32,
        )
        rot = quaternion_matrix(quat)[:3, :3]

        vis_list = [pcd_vis]

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis_list.append(axis)
        
        grasp_pose_as_points = self._pose_axes_as_point_cloud(pos, rot, axis_length)
        pcd_with_grasp = o3d.geometry.PointCloud()
        pcd_with_grasp.points = o3d.utility.Vector3dVector(np.vstack((np.asarray(pcd_vis.points), np.asarray(grasp_pose_as_points.points))))
        pcd_colors = np.asarray(pcd_vis.colors)
        if len(pcd_colors) != len(pcd_vis.points):
            pcd_colors = np.tile(np.array([0.7, 0.7, 0.7]), (len(pcd_vis.points), 1))
        pcd_with_grasp.colors = o3d.utility.Vector3dVector(np.vstack((pcd_colors, np.asarray(grasp_pose_as_points.colors))))
        
        vis_list.append(grasp_pose_as_points)
        o3d.visualization.draw_geometries(vis_list)

        if save_pcd is not None and save_path is not None:
            o3d.io.write_point_cloud(save_path, pcd_with_grasp)
            self.get_logger().info(f"Saved point cloud with grasp pose to {save_path}")
    

    def run_inference_once(self, pcd_from_saver: np.ndarray) -> None:
        t = time.time()
        
        # default_save_dir = "/home/jetson/wbk_ur10_ws/src/vmf"
        # os.makedirs(default_save_dir, exist_ok=True)

        default_save_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(default_save_dir, "logs"), exist_ok=True)
        output_name = self.current_output_name or self.output_name_base
        grasp_pose_ply_path = os.path.join(default_save_dir, "logs", f"{output_name}_grasp.ply")
        pcd_no_pose_ply_path = os.path.join(default_save_dir, "logs", f"{output_name}_no_pose.ply")
        
        # The point cloud pcd_from_saver is already in base_link frame and has shape (N, 3)
        pcd = torch.from_numpy(pcd_from_saver).float()

        

        '''
        Shift and scale the point cloud, so that the center of the region of 
        interest is at the agv_table_link and the whole area fits in a unit cube. 
        '''
        
        # agv_table_center_link [-0.7, 0, 0.101] in base_link
        # pcd_bounds = torch.tensor([[-1.2, -0.5, -0.399], [-0.2, 0.5, 0.601]], dtype=torch.float32)  # 40000 front distant high 20260402
        
        # agv_table_center_link [-0.7, 0, 0.131] in base_link
        pcd_bounds = torch.tensor([[-1.2, -0.5, -0.369], [-0.2, 0.5, 0.631]], dtype=torch.float32)  # 40000 front distant high foam 20260519
        pcd_shift = (pcd_bounds[0] + pcd_bounds[1]) / 2 
        pcd_resize = pcd_bounds[1] - pcd_bounds[0] 

        pcd = (pcd.view(-1, 3) - pcd_shift) / pcd_resize

        pcd = pcd[(pcd[:, 0] > -0.3) & (pcd[:, 0] < 0.4)]
        pcd = pcd[(pcd[:, 1] > -0.4) & (pcd[:, 1] < 0.4)]
        pcd = pcd[(pcd[:, 2] > -0.01) & (pcd[:, 2] < 0.3)] # 40000 front distant high foam reversed 20260519

        pcd_np = pcd.detach().cpu().numpy()
        if pcd_np.shape[0] < 4:
            self.get_logger().error(
                f"Not enough cropped points for OBB computation: {pcd_np.shape[0]}"
            )

        pcd = torch.from_numpy(pcd_np).float()

        # ==========================================================================
        # Calculate the point cloud and its bounding box in open3d
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(pcd_np),
            robust=False,
        )
        obb.color = (1, 0, 0)

        obb_center, obb_rot = self.compute_obb_pose_with_world_z(
            obb,
            pcd_np,
        )

        obb_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])
        obb_pose.rotate(obb_rot, center=(0, 0, 0))
        obb_pose.translate(obb_center)
        self.get_logger().info(f"Bounding Box center: {obb_center}")

        # # Visual marker for bounding box center
        # obb_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # obb_marker.paint_uniform_color((0.5, 0.5, 1.0))  # Purple-ish
        # obb_marker.translate(obb_center)

        # Center of gravity (centroid) of the point cloud (which may have some density bias)
        cog_pcd = np.mean(pcd_np, axis=0)
        self.get_logger().info(f"Point-cloud center of gravity: {cog_pcd}")

        # # Visual marker for center of gravity
        # cog_pcd_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # cog_pcd_marker.paint_uniform_color((0.5, 1.0, 0.5))  # GREEN
        # cog_pcd_marker.translate(cog_pcd)

        # The mean of the bounding box center and the point cloud center of gravity, which can be a more balanced estimate of the object center.
        cog_mean = 0.2 * obb_center + 0.8 * cog_pcd

        # Visual marker for center of gravity
        cog_mean_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        cog_mean_marker.paint_uniform_color((0.0, 0.0, 0.0))  # BLACK
        cog_mean_marker.translate(cog_mean)

        cog_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=[0, 0, 0])
        cog_axis.rotate(obb_rot, center=(0, 0, 0))
        cog_axis.translate(cog_mean)

        # ====================================================================
        grasp_x_offset = -0.70
        grasp_y_offset = 0.0
        grasp_z_offset = 0.131

        prediction = self.model.inference(
            pcd.to("cuda"),
            graspness_th=0.2,
            grasp_height_th=0.0020,
            vis=True,
            integrate=False,
            fused_pose=False,
            interactive_vis=True,
            grasp_cog_max_dist_th=0.025,
            grasp_cog_min_dist_th=None,
            cog=cog_mean,
            cog_axis=obb_rot,
            # ====CoG-based filtering parameters====
            use_cog_filter=False,
            cog_axis_projection_th=0.5,
            # ====Reachable grasp filtering parameters====
            use_reachable_grasp_filter=True,
            world_y_axis=np.array([-1.0, 0.0, 0.0]), # which is the frontal direction of the robot
            grasp_axis_projection_th=0.0,
            grasp_z_negative_world_z_angle_th=np.deg2rad(10.0),
        )
        self.get_logger().info(f"Inference time: {time.time() - t:.3f}s")

        if prediction is None:
            self.get_logger().warning("No prediction returned.")
            pcd_no_prediction = o3d.geometry.PointCloud()
            pcd_no_prediction.points = o3d.utility.Vector3dVector(pcd_np)

            # Create a coordinate frame (agv_table_center_link) for better orientation in the visualization
            agv_center_link_in_base_link = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd_no_prediction, agv_center_link_in_base_link, obb, cog_mean_marker, cog_axis, obb_pose])  

            if pcd_no_pose_ply_path is not None:
                o3d.io.write_point_cloud(pcd_no_pose_ply_path, pcd_no_prediction)
                self.get_logger().info(f"Saved point cloud (no pose) to {pcd_no_pose_ply_path}")
            return

        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        prediction = np.asarray(prediction)
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.shape == (3, 4):
            prediction = np.vstack((prediction, np.array([0.0, 0.0, 0.0, 1.0], dtype=prediction.dtype)))
        if prediction.shape != (4, 4):
            self.get_logger().error(
                f"Model inference returned invalid pose matrix shape {prediction.shape}, expected (4, 4)."
            )

    
        grasp_quat = np.asarray(quaternion_from_matrix(prediction), dtype=float)
        grasp_translation = np.asarray(translation_from_matrix(prediction), dtype=float)
        grasp_translation[0] += grasp_x_offset
        grasp_translation[1] += grasp_y_offset
        grasp_translation[2] += grasp_z_offset
        
        self.get_logger().info(f"Predicted grasp pose (in base_link): {grasp_translation}, quaternion: {grasp_quat}")

        grasp_msg = PoseStamped()
        grasp_msg.header.stamp = self.get_clock().now().to_msg()
        grasp_msg.header.frame_id = "base_link"
        grasp_msg.pose.position.x = float(grasp_translation[0])
        grasp_msg.pose.position.y = float(grasp_translation[1])
        grasp_msg.pose.position.z = float(grasp_translation[2])
        grasp_msg.pose.orientation.x = float(grasp_quat[0])
        grasp_msg.pose.orientation.y = float(grasp_quat[1])
        grasp_msg.pose.orientation.z = float(grasp_quat[2])
        grasp_msg.pose.orientation.w = float(grasp_quat[3])
        self.grasp_pose_publisher.publish(grasp_msg)
        self.get_logger().info("Grasp pose published.")

        metadata = {
            "inference_time_s": time.time() - t,
            "obb_center": obb_center.tolist(),
            "cog_pcd": cog_pcd.tolist(),
            "cog_mean": cog_mean.tolist(),
            "grasp_translation": grasp_translation.tolist(),
            "grasp_quaternion": grasp_quat.tolist(),
            "grasp_x_offset": grasp_x_offset,
            "grasp_y_offset": grasp_y_offset,
            "grasp_z_offset": grasp_z_offset,
        }
        with open(os.path.join(default_save_dir, "logs", f"{output_name}_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        self._visualize_pose_and_pcd(pcd_np, 
                                     grasp_msg,
                                     save_pcd=True,
                                     save_path=grasp_pose_ply_path, 
                                     grasp_x_offset=grasp_x_offset, 
                                     grasp_y_offset=grasp_y_offset, 
                                     grasp_z_offset=grasp_z_offset)
def main_module(
    args: argparse.Namespace,
    learning: bool = True,
):
    logging.getLogger("vmf_contact").setLevel(logging.INFO)
    suppress_pytorch_lightning_logs()

    # Fix randomness
    pl.seed_everything(args.seed)
    logger.info("Using seed %s.", os.getenv("PL_GLOBAL_SEED"))
    point_backbone_cfgs = EasyConfig()
    current_file_path = os.path.abspath(__file__)
    directory_path = os.path.dirname(current_file_path)
    print(f"{directory_path}/../vmf_contact_main/cfgs/vmfcontact/*.yaml")
    cfgs = glob.glob(f"{directory_path}/../vmf_contact_main/cfgs/vmfcontact/*.yaml")
    for cfg in cfgs:
        if args.point_backbone in cfg:
            print(f"Loading {cfg}")
            setattr(args, "point_backbone_cfgs", cfg)
            break
    else:
        raise ValueError(f"Point backbone {args.point_backbone} not found.")
    point_backbone_cfgs.load(args.point_backbone_cfgs, recursive=True)
    args.point_backbone_cfgs = point_backbone_cfgs

    if args.debug:
        args.batch_size = 1

    # Initialize logger if needed
    if args.experiment is not None and not args.debug:
        remote_logger = WandbLogger(name=args.experiment, project="vmf_contact")
        try:
            remote_logger.experiment.config.update(
                {
                    "seed": os.getenv("PL_GLOBAL_SEED"),
                    "dataset": args.dataset,
                    "flow_type": "residual",
                    "flow_layers": args.flow_layers,
                    "certainty_budget": args.certainty_budget,
                    "learning_rate": args.learning_rate,
                    "learning_rate_decay": args.learning_rate_decay,
                    "max_epochs": args.max_epochs,
                    "entropy_weight": args.entropy_weight,
                    "flow_finetune": args.flow_finetune,
                    "run_finetuning": args.run_finetuning,
                }
            )
        except:
            print("Dummy remote logger")
            remote_logger = None
    else:
        remote_logger = None

    dm = DATASET_REGISTRY[args.dataset](
        args, seed=int(os.getenv("PL_GLOBAL_SEED") or 0)
    )

    estimator = vmfContactModule(
        args,
        finetune=args.run_finetuning,
        user_params=dict(
            max_epochs=args.max_epochs,
            logger=remote_logger,
            accelerator="gpu",
            default_root_dir="logs",
            devices= 1 if args.debug or args.eval else args.devices,
            strategy="ddp_find_unused_parameters_true" if not args.eval else "auto"
        ),
    )
    # Clear GPU cache before loading checkpoint to avoid memory allocation errors
    torch.cuda.empty_cache()
    model, ckpt_loaded = estimator.module_loader(args.ckpt)
    model = model.to("cuda")

    return model

from pathlib import Path
import json


def main(args=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = current_dir + "/logs"
    os.makedirs(log_dir, exist_ok=True)

    output_name = input(
        'Enter a name for the visualization: \n'
        'Name convention is "<input>_001_grasp.ply" or "<input>_001_no_pose.ply" \n '
    ).strip()
    if not output_name:
        output_name = time.strftime("vmf_%Y%m%d_%H%M%S")

    # log_path = os.path.join(log_dir, f"{output_name}.log")
    # logger.info(f"Logging to {log_path}")

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s, +0200 | %(levelname)s | %(name)s | %(message)s",
    #     handlers=[
    #         logging.FileHandler(log_path),
    #         logging.StreamHandler(),
    #     ],
    #     force=True,
    # )

    data_path = resolve_data_path()
    logger.info("Logging to %s", log_dir)
    logger.info(torch.__version__)
    logger.info("Cuda available: %s", torch.cuda.is_available())
    logger.info("Cuda device number: %s", torch.cuda.device_count())
    # logger.info("1.Current data path: %s", data_path)
    # logger.info("2.%s/dataset/vmf_data/data*", data_path)

    parsed_args = parse_args_from_yaml(current_dir + "/config.yaml", data_path=data_path)
    model = main_module(parsed_args)
    # try:
    #     package_dir = get_package_share_directory("robot_grasping")
    #     config_file_dir = os.path.join(package_dir, "vmf_contact_main", "config.yaml")
    #     parsed_args = parse_args_from_yaml(config_file_dir)
    # except PackageNotFoundError:
    #     print("Package 'robot_grasping' not found. Using default config path.")
    #     parsed_args = parse_args_from_yaml(current_file_folder + "/config.yaml")
    # model = main_module(parsed_args)

    rclpy.init(args=args)
    node = InferenceTest2(parsed_args, 
                          model,                          
                          output_name=output_name)
    try:
        node.run_interactive_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
