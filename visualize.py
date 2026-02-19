import open3d as o3d
import numpy as np

# Load once
pcd = o3d.io.read_point_cloud("Sample1zivid.ply")
points = np.asarray(pcd.points)

# Print info about original
print(f"Number of points: {len(pcd.points)}")
print(f"Has colors: {pcd.has_colors()}")
print(f"Has normals: {pcd.has_normals()}")
print(f"X range: {points[:,0].min():.3f} to {points[:,0].max():.3f}")
print(f"Y range: {points[:,1].min():.3f} to {points[:,1].max():.3f}")
print(f"Z range: {points[:,2].min():.3f} to {points[:,2].max():.3f}")
print(f"Total bounding box size: {points.max(axis=0) - points.min(axis=0)}")

# Visualize original
print("\nShowing original point cloud...")
o3d.visualization.draw_geometries([pcd])

# Downsample
print(f"\nOriginal points: {len(pcd.points)}")
downsampled = pcd.voxel_down_sample(voxel_size=10.0)
print(f"After downsampling: {len(downsampled.points)}")

# Visualize downsampled
print("Showing downsampled point cloud...")
o3d.visualization.draw_geometries([downsampled])