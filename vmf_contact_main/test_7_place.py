import argparse
import logging
import os
from typing import Optional
import time
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

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '.'))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
print(torch.__version__)
print("Cuda available: ", torch.cuda.is_available())
print("Cuda device number: ", torch.cuda.device_count())


data_path = os.environ.get("LSDFPROJECTS")
if data_path is None or not os.path.exists(data_path):
    data_path = ".."
assert os.path.exists(data_path), f"Data path {data_path} does not exist. Please set it."
print(f"Current data path: {data_path}")

from vmf_contact import vmfContactModule
from vmf_contact import DATASET_REGISTRY
from openpoints.utils import EasyConfig
import glob
import warnings


def denoise_point_cloud(
    points_np: np.ndarray,
    nb_neighbors: int = 30,
    std_ratio: float = 1.0,
    radius: float = 0.03,
    min_points: int = 12,
) -> tuple[np.ndarray, o3d.geometry.PointCloud]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    pcd_stat, stat_inliers = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    pcd_radius, radius_inliers = pcd_stat.remove_radius_outlier(
        nb_points=min_points,
        radius=radius,
    )

    print(
        f"Denoise: raw={len(points_np)}, "
        f"after_stat={len(stat_inliers)}, after_radius={len(radius_inliers)}"
    )
    return np.asarray(pcd_radius.points), pcd_radius


def compute_obb_pose_with_world_z(
    obb: o3d.geometry.OrientedBoundingBox,
    points_np: np.ndarray,
    world_z: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> tuple[np.ndarray, np.ndarray]:
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

    y_axis = np.cross(world_z, x_axis)
    y_axis_norm = np.linalg.norm(y_axis)
    if y_axis_norm <= 1e-8:
        y_axis = np.array([0.0, 1.0, 0.0])
    else:
        y_axis /= y_axis_norm
    z_axis = world_z

    constrained_rot = np.column_stack((x_axis, y_axis, z_axis))
    return center, constrained_rot

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

logger = logging.getLogger(__name__)


def get_args_parser(
    description: Optional[str] = None,
    add_help: bool = True,
):
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
print(f"{data_path}/dataset/vmf_data/data*")

def parse_args_from_yaml(config_path: str = current_dir + "/config.yaml"):
    # Load default configurations from YAML
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Create the argument parser
    parser = get_args_parser()

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
    runs inference once, publishes the pose, then shuts down.
    """

    def __init__(self, args: argparse.Namespace, model: torch.nn.Module):
        super().__init__()
        self.args = args
        self.model = model
        self.grasp_pose_publisher = self.create_publisher(PoseStamped, "/arm_vmf/pose_chosen", 10)
        self.place_pose_publisher = self.create_publisher(PoseStamped, "/arm_vmf/place_pose", 10)
        self.inference_done = False
        self._tf_wait_logged = False
        self.get_logger().info("Waiting for depth + camera info...")
        self.timer = self.create_timer(0.1, self._tick)

    def _inputs_ready(self) -> bool:
        return self.last_depth_msg is not None and self.camera_matrix is not None

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
            self.get_logger().warning("Inputs not ready.")
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
            self.get_logger().info("Inference complete. Shutting down.")
            rclpy.shutdown()

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

    def _resample_pointcloud(self, pcd: np.ndarray, target_points: int = 40000) -> np.ndarray:
        num_points = pcd.shape[0]
        if num_points == target_points:
            return pcd

        replace = num_points < target_points
        sampled_indices = np.random.choice(num_points, size=target_points, replace=replace)
        return pcd[sampled_indices]


    def _lineset_between_points(self, a: np.ndarray, b: np.ndarray, color: np.ndarray) -> o3d.geometry.LineSet:
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([a, b])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector(np.tile(color, (1, 1)))
        return line

    def _visualize_pose_and_pcd(self, pcd_np: np.ndarray, pose_msg: PoseStamped, axis_length: float = 0.05) -> None:
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(pcd_np)

        grasp_x_offset = -0.70
        grasp_y_offset = 0.0
        grasp_z_offset = 0.101

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
        vis_list.append(self._lineset_between_points(pos, pos + rot[:, 0] * axis_length, np.array([1.0, 0.0, 0.0])))
        vis_list.append(self._lineset_between_points(pos, pos + rot[:, 1] * axis_length, np.array([0.0, 1.0, 0.0])))
        vis_list.append(self._lineset_between_points(pos, pos + rot[:, 2] * axis_length, np.array([0.0, 0.0, 1.0])))

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis_list.append(axis)

        o3d.visualization.draw_geometries(vis_list)


    def run_inference_once(self, pcd_from_saver: np.ndarray) -> None:
        
        # The point cloud pcd_from_saver is already in base_link frame and has shape (N, 3)
        pcd = torch.from_numpy(pcd_from_saver).float()

        t = time.time()

        # Shift and scale the point cloud, so that the center of the region of interest is at the agv_table_link ([-0.45, 0, 0.101] in base_link)
        # and the whole area fits in a unit cube. This can help with model generalization and convergence.
        # pcd_bounds=torch.tensor([[-0.95, -0.5, -0.399], [0.05, 0.5, 0.601]], dtype=torch.float32) # 40000 front high new 1 2 3
        
        # agv_table_center_link [-0.7, 0, 0.101] in base_link
        pcd_bounds = torch.tensor([[-1.2, -0.5, -0.399], [-0.2, 0.5, 0.601]], dtype=torch.float32)  # 40000 front distant high 20260402
        pcd_shift = (pcd_bounds[0] + pcd_bounds[1]) / 2 
        pcd_resize = pcd_bounds[1] - pcd_bounds[0] 

        pcd = (pcd.view(-1, 3) - pcd_shift) / pcd_resize  # with bounds

        # pcd = pcd[(pcd[:, 0] > -1.0) & (pcd[:, 0] < 0.5)]
        # pcd = pcd[(pcd[:, 1] > -0.5) & (pcd[:, 1] < 0.5)]
        # pcd = pcd[(pcd[:, 2] > 0.09) & (pcd[:, 2] < 0.3)]  # 40000 & 240000 front high new 1 2 3

        # pcd = pcd[(pcd[:, 0] > -0.15) & (pcd[:, 0] < 0.5)]
        # pcd = pcd[(pcd[:, 1] > -0.2) & (pcd[:, 1] < 0.3)]
        # pcd = pcd[(pcd[:, 2] > 0.0) & (pcd[:, 2] < 0.3)] # 40000 front distant high 20260402

        # pcd = pcd[(pcd[:, 0] > -0.15) & (pcd[:, 0] < 0.5)]
        # pcd = pcd[(pcd[:, 1] > -0.3) & (pcd[:, 1] < 0.3)]
        # pcd = pcd[(pcd[:, 2] > 0.01) & (pcd[:, 2] < 0.3)] # 40000 front distant high foam 20260409

        pcd = pcd[(pcd[:, 0] > -0.3) & (pcd[:, 0] < 0.4)]
        pcd = pcd[(pcd[:, 1] > -0.4) & (pcd[:, 1] < 0.3)]
        pcd = pcd[(pcd[:, 2] > 0.035) & (pcd[:, 2] < 0.3)] # 40000 front distant high foam reversed 20260423

        pcd_np = pcd.detach().cpu().numpy()
        if pcd_np.shape[0] < 4:
            raise RuntimeError(
                f"Not enough cropped points for OBB computation: {pcd_np.shape[0]}"
            )

        # pcd_np, _ = denoise_point_cloud(
        #     pcd_np,
        #     nb_neighbors=30,
        #     std_ratio=1.0,
        #     radius=0.03,
        #     min_points=12,
        # )
        if pcd_np.shape[0] < 4:
            raise RuntimeError(
                f"Not enough denoised points for OBB computation: {pcd_np.shape[0]}"
            )

        pcd = torch.from_numpy(pcd_np).float()

        # ==========================================================================
        # Calculate the point cloud and its bounding box in open3d
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(pcd_np),
            robust=False,
        )
        obb.color = (1, 0, 0)

        obb_pose_center, obb_pose_rot = compute_obb_pose_with_world_z(
            obb,
            pcd_np,
        )

        obb_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])
        obb_pose.rotate(obb_pose_rot, center=(0, 0, 0))
        obb_pose.translate(obb_pose_center)
        print(f"Bounding Box center: {obb_pose_center}")

        # # Visual marker for bounding box center
        # obb_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # obb_marker.paint_uniform_color((0.5, 0.5, 1.0))  # Purple-ish
        # obb_marker.translate(obb_pose_center)

        # Center of gravity (centroid) of the point cloud (which may have some density bias)
        cog_pcd = np.mean(pcd_np, axis=0)
        print(f"Point-cloud center of gravity: {cog_pcd}")

        # # Visual marker for center of gravity
        # cog_pcd_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # cog_pcd_marker.paint_uniform_color((0.5, 1.0, 0.5))  # GREEN
        # cog_pcd_marker.translate(cog_pcd)

        # The mean of the bounding box center and the point cloud center of gravity, which can be a more balanced estimate of the object center.
        cog_mean = (obb_pose_center + cog_pcd) / 2

        # Visual marker for center of gravity
        cog_mean_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        cog_mean_marker.paint_uniform_color((0.0, 0.0, 0.0))  # BLACK
        cog_mean_marker.translate(cog_mean)

        cog_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=[0, 0, 0])
        cog_axis.rotate(obb_pose_rot, center=(0, 0, 0))
        cog_axis.translate(cog_mean)

        # ====================================================================
        grasp_x_offset = -0.70
        grasp_y_offset = 0.0
        grasp_z_offset = 0.101

        prediction = self.model.inference(
            pcd.to("cuda"),
            graspness_th=0.6,
            grasp_height_th=0.025,
            grasp_cog_dist_th=0.05,
            vis=True,
            integrate=False,
            fused_pose=False,
            interactive_vis=True,
            cog=cog_mean,
            cog_axis=obb_pose_rot,
            cog_axis_projection_th=0.7,
        )
        self.get_logger().info(f"Inference time: {time.time() - t:.3f}s")

        if prediction is None:
            self.get_logger().info("No prediction returned.")
            # ==========================================
            pcd_vis_test = o3d.geometry.PointCloud()
            pcd_vis_test.points = o3d.utility.Vector3dVector(pcd_np)

            # Create a coordinate frame for better orientation in the visualization
            axis_test = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd_vis_test, axis_test, obb, cog_mean_marker, cog_axis, obb_pose])  
            # ===============================================
            return

        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        prediction = np.asarray(prediction)
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.shape == (3, 4):
            prediction = np.vstack((prediction, np.array([0.0, 0.0, 0.0, 1.0], dtype=prediction.dtype)))
        if prediction.shape != (4, 4):
            raise RuntimeError(
                f"Model inference returned invalid pose matrix shape {prediction.shape}, expected (4, 4)."
            )

        grasp_quat = quaternion_from_matrix(prediction)
        grasp_translation = translation_from_matrix(prediction)
        grasp_translation[0] += grasp_x_offset
        grasp_translation[1] += grasp_y_offset
        grasp_translation[2] += grasp_z_offset
        
        self.get_logger().info(f"Predicted translation: {grasp_translation}, quaternion: {grasp_quat}")

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
        

        agv_T_cog = np.eye(4)
        agv_T_cog[:3, :3] = obb_pose_rot
        agv_T_cog[:3, 3] = cog_mean

        cog_T_agv = np.linalg.inv(agv_T_cog)

        # agv_T_grasp = prediction
        base_T_grasp = prediction
        agv_T_base = np.eye(4)
        agv_T_base[:3, 3] = np.array([-grasp_x_offset, -grasp_y_offset, -grasp_z_offset])  # agv_table_center_link in base_link

        # cog_T_grasp = cog_T_agv @ agv_T_base @ base_T_grasp
        # cog_T_grasp = cog_T_agv @ agv_T_grasp
        cog_T_grasp = cog_T_agv @ agv_T_base @ base_T_grasp

        print(f"cog_T_grasp:\n{cog_T_grasp}")
        
        # ============== agv_T_place =======================================
        # # agv_T_placement_center = self.tf_buffer.lookup_transform(
        # #     "placement_link",
        # #     "agv_table_center_link",
        # #     rclpy.time.Time(),
        # #     timeout=RclpyDuration(seconds=0.2)
        # # )

        # agv_to_placement_tf = self.tf_buffer.lookup_transform(
        #     "agv_table_center_link",
        #     "placement_link",            
        #     rclpy.time.Time(),
        #     timeout=RclpyDuration(seconds=0.2)
        # )
        # agv_T_placement_center = transform_to_matrix(agv_to_placement_tf.transform)

        # print(f"agv_T_placement_center:\n{agv_T_placement_center}")

        # # agv_T_place = agv_T_placement_center @ placement_center_T_place
        # #             = agv_T_placement_center @ cog_T_grasp

        # agv_T_place = agv_T_placement_center @ cog_T_grasp
        # print(f"agv_T_place:\n{agv_T_place}")

        # place_quat = quaternion_from_matrix(agv_T_place)
        # place_translation = translation_from_matrix(agv_T_place)
        # place_translation[0] += grasp_x_offset
        # place_translation[1] += grasp_y_offset
        # place_translation[2] += grasp_z_offset
        
        # self.get_logger().info(f"Placement pose translation: {place_translation}, quaternion: {place_quat}")

        # place_msg = PoseStamped()
        # place_msg.header.stamp = self.get_clock().now().to_msg()
        # place_msg.header.frame_id = "base_link"
        # place_msg.pose.position.x = float(place_translation[0])
        # place_msg.pose.position.y = float(place_translation[1])
        # place_msg.pose.position.z = float(place_translation[2])
        # place_msg.pose.orientation.x = float(place_quat[0])
        # place_msg.pose.orientation.y = float(place_quat[1])
        # place_msg.pose.orientation.z = float(place_quat[2])
        # place_msg.pose.orientation.w = float(place_quat[3])
        # self.place_pose_publisher.publish(place_msg)
        # ========================================================

        # ========placement_center_T_place = cog_T_grasp =====================
        placement_center_T_place = cog_T_grasp

        place_quat = quaternion_from_matrix(placement_center_T_place)
        place_translation = translation_from_matrix(placement_center_T_place)
        self.get_logger().info(f"Placement pose translation: {place_translation}, quaternion: {place_quat}")

        place_msg = PoseStamped()
        place_msg.header.stamp = self.get_clock().now().to_msg()
        place_msg.header.frame_id = "placement_link"
        place_msg.pose.position.x = float(place_translation[0])
        place_msg.pose.position.y = float(place_translation[1])
        place_msg.pose.position.z = float(place_translation[2])
        place_msg.pose.orientation.x = float(place_quat[0])
        place_msg.pose.orientation.y = float(place_quat[1])
        place_msg.pose.orientation.z = float(place_quat[2])
        place_msg.pose.orientation.w = float(place_quat[3])
        self.place_pose_publisher.publish(place_msg)
        self.get_logger().info("Place pose published.")
        # ========================================================

        self._visualize_pose_and_pcd(pcd_np, grasp_msg)
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

def main(args=None):
    current_file_folder = os.path.dirname(os.path.abspath(__file__))
    parsed_args = parse_args_from_yaml(current_file_folder + "/config.yaml")
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
    node = InferenceTest2(parsed_args, model)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
