import glob
import os

import cv2
import einops
import numpy as np
import open3d as o3d
import torch
import torchvision.transforms.v2 as v2  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
from .pointcloud_process import (
    compute_normal_map,
    denoise_point_cloud,
    over_or_re_sample,
    save_image_tensor,
)

input_arguments = [
    "rgb",
    # "depth",
    "normals",
    # "instance",
    # "grasp_pos_img",
    # "grasp_pose",
    "pcd_img",
    "pcd",
    "camera_pose",
    "pcd_gt",
]


class MGNDataset(Dataset):

    def __init__(
        self,
        root_dir,
        image_size,
        num_points=20000,
        input_mean=(0.485, 0.456, 0.406),
        input_std=(0.229, 0.224, 0.225),
        pcd_with_rgb=False,
        num_cameras=1,
    ):
        """
        Args:
            root_dir (string): Root directory containing subdirectories with .npz files.
        """
        super(MGNDataset, self).__init__()

        pcd_bounds=torch.tensor(
            [[0.2, -0.5, -0.5], [1.2, 0.5, 0.5]], dtype=torch.float32
        )

        if isinstance(root_dir, list):
            self.data = []
            for root in root_dir:
                self.data += glob.glob(os.path.join(root, "*_data.pt"))
        else:
            self.data = glob.glob(os.path.join(root_dir, "*_data.pt"))

        pcd_bounds = pcd_bounds
        self.num_points = num_points
        self.num_cameras = num_cameras
        self.pcd_with_rgb = pcd_with_rgb

        self.pcd_shift = (pcd_bounds[0] + pcd_bounds[1]) / 2
        self.pcd_resize = pcd_bounds[1] - pcd_bounds[0]
        self.preprocess_pcd = v2.Compose(
            [
                v2.Normalize(mean=self.pcd_shift, std=self.pcd_resize),
                v2.Resize(
                    size=(image_size[0] // 2, image_size[1] // 2),
                    interpolation=v2.InterpolationMode.BILINEAR,
                ),
            ]
        )
        self.preprocess_rgb = v2.Compose(
            [
                v2.Normalize(mean=input_mean, std=input_std),
                v2.Resize(
                    size=(image_size[0] // 2, image_size[1] // 2),
                    interpolation=v2.InterpolationMode.BILINEAR,
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pth_path = self.data[idx]
        data = torch.load(pth_path, map_location="cpu", weights_only=False)
        parts = pth_path.split("/")[-1].split("_")

        # Extract the env, epi, and step values
        env_id = int(parts[1])
        epi_id = int(parts[3])
        step_id = int(parts[5])

        rgbs = []
        depths = []
        normals = []
        colors = []
        instances = []
        grasp_pos_imgs = []
        grasp_poses = []
        pcd_imgs = []
        pcds = []

        input_sample = {
            "id": torch.tensor([env_id, epi_id, step_id]),
            "file_name": pth_path,
        }

        # random camera id
        cam_ids = np.random.choice(
            [cam_id for cam_id in data.keys() if "camera" in cam_id], self.num_cameras
        )

        for cam_id in cam_ids:

            # Extract your specific data fields
            rgb = data[cam_id].get("rgb") if "rgb" in input_arguments else None
            grasp_pose = data[cam_id].get("grasp_pose") if "grasp_pose" in input_arguments else None
            depth = data[cam_id].get("depth") if "depth" in input_arguments else None
            normal = data[cam_id].get("normals") if "normals" in input_arguments else None
            instance = data[cam_id].get("instance") if "instance" in input_arguments else None
            grasp_pos_img = data[cam_id].get("grasp_pos_img")  if "grasp_pos_img" in input_arguments else None
            id_to_label = data[cam_id].get("id_to_labels") if "instance" in input_arguments else None
            pcd_img = data[cam_id].get("pcd") / 1000 if "pcd_img" in input_arguments else None
            # cam_pos = data[cam_id].get("camera_pose")[:3].numpy() if "camera_pose" in input_arguments else None

            # Convert data to tensors
            if rgb is not None:
                rgb = rgb.permute(2, 0, 1)
                rgb = self.preprocess_rgb(rgb/255.)
                rgbs.append(rgb)
            
            # depths
            if depth is not None:
                depths.append(depth.squeeze(-1))

            # point cloud
            if pcd_img is not None:
                pcd_img = pcd_img.permute(2, 0, 1)
                pcd_imgs.append(pcd_img)

            # crop the point cloud to the table region by pcd_bounds
            pcd = self.preprocess_pcd(pcd_img)

            if normal is None:
                # compute the normal map
                normal = compute_normal_map(pcd)
                # visualize the normal map
                # normal = normal.permute(1, 2, 0).cpu().numpy()
                # normal = cv2.normalize(normal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # cv2.imwrite("normal.jpg", normal)
                
            pcd = torch.cat([pcd, normal], dim=0)

            if self.pcd_with_rgb:
                pcd = torch.cat([pcd, rgb], dim=0)

            # crop the point cloud to the table region by pcd_bounds
            pcd = einops.rearrange(pcd, "c h w -> (h w) c")
            pcd = pcd[(pcd[:, 0] > -0.5) & (pcd[:, 0] < 0.5)]
            pcd = pcd[(pcd[:, 1] > -0.5) & (pcd[:, 1] < 0.5)]
            pcd = pcd[pcd[:, 2] > -0.02]
            pcd = denoise_point_cloud(pcd, method="statistical", nb_neighbors=20, std_ratio=2.0, radius=0.05, min_neighbors=16)
            pcd = over_or_re_sample(pcd, self.num_points // self.num_cameras)

            pcds.append(pcd[:, :3])
            normals.append(pcd[:, 3:6])
            if self.pcd_with_rgb:
                colors.append(pcd[:, 6:])

            # pcd.colors = o3d.utility.Vector3dVector(samples[:,3:])
            # vis_list=[pcd]
            # o3d.visualization.draw_geometries(vis_list)

            # Visualize the sampled points
            # samples = pcd.cpu().numpy()
            #pcd_o3d = o3d.geometry.PointCloud()
            #pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:,:3].cpu().numpy())
            #pcd_o3d.normals = o3d.utility.Vector3dVector(normal.cpu().numpy())
            #pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=5000))
            # pcd_o3d.orient_normals_towards_camera_location(camera_location=cam_pos)
            # draw camera view point
            # camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=cam_pos)
            #o3d.visualization.draw_geometries([pcd_o3d], point_show_normal=True)
            #normal = np.asarray(pcd_o3d.normals)
            #normal = torch.tensor(normal, dtype=torch.float32, device=pcd.device)
            
            if id_to_label is not None:
                for id, label in id_to_label.items():
                    instance[instance == id] = int(label)
                instances.append(instance)

            if grasp_pose is not None:
                grasp_pos_imgs.append(grasp_pos_img)
                grasp_poses.append(grasp_pose)

        if len(rgbs) > 0:
            input_sample["rgb"] = torch.stack(rgbs)
        if len(depths) > 0:
            input_sample["depth"] = torch.stack(depths)
        if len(normals) > 0:
            input_sample["normals"] = torch.cat(normals)
        if len(instances) > 0:
            input_sample["instance"] = torch.stack(instances)
        if len(grasp_pos_imgs) > 0:
            input_sample["grasp_pos_img"] = torch.stack(grasp_pos_imgs)
        if len(grasp_poses) > 0:
            input_sample["grasp_pose"] = torch.stack(grasp_poses)
        if len(pcd_imgs) > 0:
            input_sample["pcd_img"] = torch.stack(pcd_imgs)
        if len(pcds) > 0:
            input_sample["pcd"] = torch.cat(pcds, dim=0)
        if len(colors) > 0:
            input_sample["pcd_color"] = torch.cat(colors, dim=0)
        if "pcd_gt" in input_arguments:
            pcds_gt = data.get("pcd_gt").to(torch.float32) / 1000
            pcds_gt = over_or_re_sample(pcds_gt, self.num_points)
            pcds_gt = (pcds_gt - self.pcd_shift) / self.pcd_resize
            input_sample["pcd_gt"] = pcds_gt

        gt_sample = {}
        gt_width = data["non_colliding_parallel_contact_width"] / 1e2
        gt_filter = gt_width > 0.0

        gt_width = gt_width[gt_filter].to(torch.float32)
        gt_poses = data["non_colliding_parallel_contact_poses"]
        gt_poses = gt_poses[gt_filter]
        gt_poses[:, :3, -1] /= 100

        gt_poses[:, :3, -1] = (gt_poses[:, :3, -1] - self.pcd_shift.to(gt_poses.device)) / self.pcd_resize.to(gt_poses.device)
        # gt_poses[:, 0, -1] += 0.655

        gt_scores = data["non_colliding_parallel_analytical_score"]
        gt_scores = gt_scores[gt_filter]

        gt_pt = gt_poses[:, :3, -1]  # Contact point coordinates convert to meters
        gt_pt_extended = gt_pt + gt_poses[:, :3, 0] * gt_width.unsqueeze(
            -1
        )  # Extended contact points with another side of the contact
        gt_pt = torch.cat([gt_pt, gt_pt_extended], dim=0)


        # #visualize the gt pt
        # pcd_o3d = o3d.geometry.PointCloud()
        # pcd_o3d.points = o3d.utility.Vector3dVector(gt_pt.cpu().numpy())
        # pcd2_o3d = o3d.geometry.PointCloud()
        # pcd2_o3d.points = o3d.utility.Vector3dVector(torch.cat(pcds, dim=0).cpu().numpy())
        # #pcd3_o3d = o3d.geometry.PointCloud()
        # #pcd3_o3d.points = o3d.utility.Vector3dVector(pcds_gt[:, :3].cpu().numpy())
        # o3d.visualization.draw_geometries([pcd_o3d, pcd2_o3d])
        

        gt_sample["gt_width"] = torch.cat([gt_width, gt_width], dim=0)
        gt_sample["gt_baseline"] = torch.cat(
            [gt_poses[:, :3, 0], -gt_poses[:, :3, 0]], dim=0
        )  # Baseline vectors
        gt_sample["gt_pt"] = gt_pt  # Contact points
        gt_sample["gt_approach"] = torch.cat(
            [gt_poses[:, :3, 2], gt_poses[:, :3, 2]]
        )  # Approaching vectors
        gt_sample["gt_scores"] = torch.cat(
            [gt_scores, gt_scores], dim=0
        )  # Grasp scores

        return input_sample, gt_sample


preview_image = False


def custom_collate_fn(batch):
    input_batch = [input[0] for input in batch]
    try:
        input_batch = default_collate(input_batch)
    except Exception as e:
        print(e)
    gt_batch = [input[1] for input in batch]

    if preview_image:
        save_image_tensor(
            input_batch["rgb"][0], input_batch["file_name"][0].split("/")[-1]
        )
        save_image_tensor(
            input_batch["depth"][0],
            input_batch["file_name"][0].split("/")[-1],
            mode="depth",
        )
        save_image_tensor(
            input_batch["pcd_img"][0][-1, ...],
            input_batch["file_name"][0].split("/")[-1],
            mode="pcd_img",
        )

    return input_batch, gt_batch

if __name__ == "__main__":
    # Create dataset and dataloader
    data_root_dir = "data_all"
    npz_dataset = MGNDataset(data_root_dir)
    # Example of iterating through the dataloader
    npz_dataloader = DataLoader(
        npz_dataset, collate_fn=custom_collate_fn, batch_size=2, shuffle=False
    )
    for i_batch, sample_batched in enumerate(npz_dataloader):
        print(f'id: {sample_batched["id"]}')
        print(sample_batched["rgb"].size(), sample_batched["grasp_pose"].size())
