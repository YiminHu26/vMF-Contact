import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import cv2

def denoise_point_cloud(points, method="statistical", nb_neighbors=30, std_ratio=2.0, radius=0.05, min_neighbors=16):
    """
    Denoises a point cloud given as a NumPy array.

    Parameters:
    - points (numpy.ndarray): Nx6 array representing point cloud coordinates with color.
    - method (str): "statistical" for statistical outlier removal, "radius" for radius outlier removal.
    - nb_neighbors (int): Number of neighbors for statistical outlier removal.
    - std_ratio (float): Standard deviation ratio for statistical outlier removal.
    - radius (float): Radius for radius outlier removal.
    - min_neighbors (int): Minimum number of neighbors for radius outlier removal.

    Returns:
    - numpy.ndarray: Denoised point cloud.
    """
    # Convert NumPy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].cpu().numpy())

    # Apply selected denoising method
    if method == "statistical":
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    elif method == "radius":
        pcd_clean, ind = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    else:
        raise ValueError("Invalid method. Choose 'statistical' or 'radius'.")
    points = points[ind]
    points = torch.tensor(points, dtype=torch.float32, device=points.device)
    return points

def save_image_tensor(image_tensor, file_name, mode="rgb"):
    """
    Save an image tensor to a file using OpenCV.

    Parameters:
    image_tensor (numpy.ndarray): Image tensor with shape (3, h, w).
    file_name (str): Name of the file to save the image.
    """
    # Ensure the image tensor is in the correct shape (h, w, 3) for imwrite
    # Convert image tensor to a suitable format (e.g., uint8)
    if mode == "depth" or mode == "pcd_img":
        image_tensor = cv2.normalize(
            image_tensor.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        image_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_GRAY2RGB)
    elif mode == "rgb":
        image_tensor = np.transpose(image_tensor.cpu().numpy(), (1, 2, 0))
        image_tensor = (image_tensor).astype(np.uint8)
        image_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_BGR2RGB)

    # Write the image to a file using imwrite
    cv2.imwrite("pic_debug" + file_name.split(".")[0] + f"_{mode}.png", image_tensor)


def over_or_re_sample(pcd, num_points):
    c = pcd.shape[-1]
    # Determine the maximum size
    if pcd.shape[0] < num_points:
        # Oversample the point cloud
        pad_size = num_points - pcd.shape[0]
        pcd_rest_ind = torch.randint(0, pcd.shape[0], (pad_size,))
        pcd = torch.cat([pcd, pcd[pcd_rest_ind]], dim=0)
    else:
        # Resample the point cloud
        indices = torch.randint(0, pcd.shape[0], (num_points,))
        pcd = torch.gather(pcd, 0, indices.unsqueeze(-1).expand(-1, c))
    return pcd



def compute_normal_map(pcd_img: torch.Tensor) -> torch.Tensor:
    """
    Computes the normal map from a point cloud image tensor in PyTorch.
    
    Args:
        pcd_img (torch.Tensor): Input tensor of shape (3, 480, 640), representing (X, Y, Z) coordinates.

    Returns:
        torch.Tensor: Normal map of shape (3, 480, 640), representing the normal vectors.
    """
    # Ensure the input is of correct shape (3, 480, 640)
    assert pcd_img.shape[0] == 3, "Input tensor should have 3 channels (X, Y, Z)."
    
    # Compute gradients in x and y directions for each channel (X, Y, Z)
    dzdx = F.pad(pcd_img[:, :, 1:] - pcd_img[:, :, :-1], (1, 0), mode='replicate')
    dzdy = F.pad(pcd_img[:, 1:, :] - pcd_img[:, :-1, :], (0, 0, 1, 0), mode='replicate')

    # dzdx and dzdy are the derivatives of the point cloud along x and y directions

    # Now, compute the cross product between dzdx and dzdy to get the normals
    normal_x = dzdy[1] * dzdx[2] - dzdy[2] * dzdx[1]
    normal_y = dzdy[2] * dzdx[0] - dzdy[0] * dzdx[2]
    normal_z = dzdy[0] * dzdx[1] - dzdy[1] * dzdx[0]

    # Stack the normal components back into a (3, 480, 640) tensor
    normal_map = torch.stack((normal_x, normal_y, normal_z), dim=0)

    # Normalize the normal vectors to unit length
    normal_map = F.normalize(normal_map, dim=0)

    return normal_map