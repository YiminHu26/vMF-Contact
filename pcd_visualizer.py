import argparse
import open3d as o3d

parser = argparse.ArgumentParser(
    description="Visualize a point cloud file"
)

parser.add_argument(
    "file",
    help="Path to point cloud file"
)

args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.file)

o3d.visualization.draw_geometries([pcd])