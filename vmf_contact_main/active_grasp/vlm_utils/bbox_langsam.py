import numpy as np
import open3d as o3d
import math

class SceneObject:
    def __init__(self, pcd, center, bbox_dims, bbox_3d, label, adjectives):
        """
        :param center: 物体中心点坐标，例如 (x, y, z)
        :param bbox_dims: bbox 的尺寸 (长, 宽, 高)
        :param bbox_3d: 3D bounding box，8 个角点列表
        :param label: 物体类别标签
        :param adjectives: 描述物体的形容词列表（例如颜色、大小、特性等）
        """
        self.center = center
        self.pcd = pcd
        self.bbox_dims = bbox_dims
        self.bbox_3d = bbox_3d
        self.label = label
        self.adjectives = adjectives
        # 生成一个唯一的标识符，用 label 加上第一个形容词（例如颜色），方便区分同一类别中不同属性的实例
        # 如果没有形容词，也可直接用 label
        self.instance_id = f"{label}_{adjectives[0]}" if adjectives else label
    def __str__(self):
        return (
                # f"Instance ID: {self.instance_id}\n"
                f"Label: {self.label}\n"
                f"Center: {self.center}\n"
                f"BBox size: (length, width, height): {self.bbox_dims}\n"
                f"3D BBox: {self.bbox_3d}\n"
                f"Adjectives: {self.adjectives}\n")
    def print_adj(self):
        print(f"Adjectives: {self.adjectives}\n")


def compute_oriented_bounding_box(pcd, label="Unknown", adjectives=[], lower_percentile=2, upper_percentile=98):
    """
    Computes a SceneObject with an Oriented Bounding Box (OBB) from a point cloud.

    Args:
        pcd (np.ndarray): A numpy array of shape (N, 3) representing the point cloud.
        label (str): The label for the object.
        adjectives (list): List of descriptive adjectives (e.g., color, size, properties).
        lower_percentile (float): The lower percentile to clip outliers.
        upper_percentile (float): The upper percentile to clip outliers.

    Returns:
        SceneObject: A SceneObject instance containing the OBB details.
    """
    # Step 1: Remove outliers using percentile clipping
    lower_bound = np.percentile(pcd, lower_percentile, axis=0)
    upper_bound = np.percentile(pcd, upper_percentile, axis=0)
    
    mask = np.all((pcd >= lower_bound) & (pcd <= upper_bound), axis=1)
    filtered_points = pcd[mask]  # Keep only inliers

    # Step 2: Compute PCA (Covariance matrix)
    mean = np.mean(filtered_points, axis=0)
    centered_points = filtered_points - mean
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # Step 3: Eigen decomposition (PCA axes)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)  # Eigen decomposition
    rotation_matrix = eigenvectors  # The eigenvectors form the rotation matrix

    # Step 4: Transform points to PCA coordinate space
    transformed_points = centered_points @ rotation_matrix

    # Step 5: Get min/max in the rotated frame
    min_corner = np.min(transformed_points, axis=0)
    max_corner = np.max(transformed_points, axis=0)

    # Step 6: Compute OBB center and dimensions
    obb_center = (min_corner + max_corner) / 2
    obb_center_world = obb_center @ rotation_matrix.T + mean  # Transform back
    bbox_dims = max_corner - min_corner  # Dimensions: (length, width, height)

    # Step 7: Compute the 8 corners of the bounding box in the PCA space
    corner_offsets = np.array([
        [min_corner[0], min_corner[1], min_corner[2]],
        [min_corner[0], min_corner[1], max_corner[2]],
        [min_corner[0], max_corner[1], min_corner[2]],
        [min_corner[0], max_corner[1], max_corner[2]],
        [max_corner[0], min_corner[1], min_corner[2]],
        [max_corner[0], min_corner[1], max_corner[2]],
        [max_corner[0], max_corner[1], min_corner[2]],
        [max_corner[0], max_corner[1], max_corner[2]],
    ])
    
    # Transform corners back to world space
    obb_corners = (corner_offsets @ rotation_matrix.T) + mean

    # Create and return a SceneObject instance
    return SceneObject(pcd=pcd, center=obb_center_world, bbox_dims=bbox_dims, bbox_3d=obb_corners, label=label, adjectives=adjectives)


def visualize_pcd_with_obb(pcd, obb_corners):
    """
    Visualize the point cloud and its Oriented Bounding Box (OBB) in Open3D.
    
    Args:
        pcd (np.ndarray): (N, 3) numpy array containing point cloud data.
        obb_corners (np.ndarray): (8, 3) numpy array containing the 8 OBB corners.
    """
    # obb_corners = expand_obb(obb_corners, 0.02)  # Expand OBB for better visualization

    # Create Open3D Point Cloud
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pcd)
    
    # Create Open3D Line Set for OBB
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
    ]
    colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for bounding box

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(obb_corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return [pcd_vis, line_set]

####################################################################################################

def get_edges(corners):
    """Compute edge vectors from 8 corner points of an OBB."""
    edges = [
        corners[1] - corners[0],  # Edge along width
        corners[3] - corners[0],  # Edge along height
        corners[4] - corners[0],  # Edge along depth
    ]
    return np.array(edges)

def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def project_obb(corners, axis):
    """Project the 8 corners of an OBB onto a given axis and return min/max projection."""
    projections = np.dot(corners, axis)
    return np.min(projections), np.max(projections)

def overlap(min1, max1, min2, max2):
    """Check if two 1D projections overlap."""
    return max1 >= min2 and max2 >= min1

def obb_collision(obb1, obb2):
    """
    Check if two 3D OBBs collide using the Separating Axis Theorem.

    Parameters:
    obb1, obb2: numpy arrays of shape (8,3), representing the 8 corner points of each OBB.

    Returns:
    True if OBBs collide, False otherwise.
    """
    edges1 = get_edges(obb1)
    edges2 = get_edges(obb2)

    # Compute 15 possible separating axes
    axes = []
    axes.extend([normalize(e) for e in edges1])  # 3 axes of OBB1
    axes.extend([normalize(e) for e in edges2])  # 3 axes of OBB2
    for e1 in edges1:
        for e2 in edges2:
            cross_axis = np.cross(e1, e2)
            if np.linalg.norm(cross_axis) > 1e-6:  # Avoid zero vector
                axes.append(normalize(cross_axis))

    # Check for separation on any axis
    for axis in axes:
        min1, max1 = project_obb(obb1, axis)
        min2, max2 = project_obb(obb2, axis)
        if not overlap(min1, max1, min2, max2):
            return False  # Separating axis found → No collision

    return True  # No separating axis found → Collision


def expand_obb(corners, expansion_distance):
    """
    Expand an OBB by a given distance in all directions.
    
    Parameters:
    corners (numpy array): (8,3) array representing the 8 corner points of the OBB.
    expansion_distance (float): Distance to expand the OBB.

    Returns:
    numpy array: (8,3) expanded OBB corner points.
    """
    # Compute OBB center
    center = np.mean(corners, axis=0)

    # Compute the primary axes (edge vectors)
    edges = get_edges(corners)
    axes = np.array([normalize(edge) for edge in edges])  # Normalize for unit directions

    # Expand along each axis
    expanded_corners = []
    for corner in corners:
        expanded_corner = corner.copy()
        for axis in axes:
            expanded_corner += axis * expansion_distance if np.dot(corner - center, axis) > 0 else -axis * expansion_distance
        expanded_corners.append(expanded_corner)

    return np.array(expanded_corners)

def obb_collision_expanded(obb1:SceneObject, obb2:SceneObject, expansion_distance=0.0):
    """
    Check if two 3D OBBs collide using the Separating Axis Theorem with expanded OBBs.
    
    Parameters:
    obb1, obb2: numpy arrays of shape (8,3), representing the 8 corner points of each OBB.
    expansion_distance: float, distance to expand the OBBs.

    Returns:
    True if OBBs collide, False otherwise.
    """
    obb1 = expand_obb(obb1, expansion_distance)
    obb2 = expand_obb(obb2, expansion_distance)
    return obb_collision(obb1, obb2)

