"""
batch_icp.py
─────────────────────────────────────────────────────────────────────────────
Runs ICP (Iterative Closest Point) on all consecutive .ply frame pairs in a
folder and outputs a motion timeline. Designed for linear/rigid motion tracking
for SPECT patient motion correction.

Usage:
    python batch_icp.py --folder point_clouds/point_clouds/

Optional:
    --voxel       Voxel size for downsampling (default: 5.0 mm)
    --threshold   ICP max correspondence distance (default: 10.0 mm)
    --output      Save results to a CSV file (default: icp_results.csv)
    --visualize   Show visualization of first frame pair
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import open3d as o3d


# ─── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch ICP motion estimation across consecutive point cloud frames."
    )
    parser.add_argument("--folder", required=True, help="Folder containing .ply files")
    parser.add_argument("--voxel", type=float, default=5.0,
                        help="Voxel size for downsampling in mm (default: 5.0)")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="ICP max correspondence distance in mm (default: 10.0)")
    parser.add_argument("--output", default="icp_results.csv",
                        help="Output CSV file (default: icp_results.csv)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the first frame pair alignment")
    return parser.parse_args()


# ─── Load and preprocess ─────────────────────────────────────────────────────

def load_pcd(filepath: Path, voxel_size: float) -> o3d.geometry.PointCloud:
    """Load a PLY file, remove outliers, and downsample."""
    pcd = o3d.io.read_point_cloud(str(filepath))

    if len(pcd.points) == 0:
        print(f"  [WARN] Empty point cloud: {filepath.name}")
        return None

    # Downsample
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Estimate normals (required for point-to-plane ICP)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    return pcd


# ─── ICP ─────────────────────────────────────────────────────────────────────

def run_icp(source: o3d.geometry.PointCloud,
            target: o3d.geometry.PointCloud,
            threshold: float):
    """Run point-to-plane ICP between source and target point clouds."""
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.eye(4),  # initial transformation (identity)
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    return result


def extract_motion(transformation: np.ndarray):
    """Extract translation (mm) and rotation (degrees) from 4x4 transform matrix."""
    translation = transformation[:3, 3]
    displacement = np.linalg.norm(translation)

    # Extract rotation angle from rotation matrix
    rotation_matrix = transformation[:3, :3]
    trace = np.trace(rotation_matrix)
    cos_angle = (trace - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)
    rotation_angle = np.degrees(np.arccos(cos_angle))

    return translation, displacement, rotation_angle


# ─── Visualization ───────────────────────────────────────────────────────────

def visualize_alignment(source, target, transformation):
    """Visualize ICP alignment — source in blue, target in red, aligned in green."""
    source_aligned = source.transform(transformation)

    source_vis = o3d.geometry.PointCloud(target)
    source_vis.paint_uniform_color([1.0, 0.2, 0.2])  # red = target

    target_vis = o3d.geometry.PointCloud(source)
    target_vis.paint_uniform_color([0.2, 0.4, 1.0])  # blue = source

    aligned_vis = o3d.geometry.PointCloud(source_aligned)
    aligned_vis.paint_uniform_color([0.2, 0.8, 0.2])  # green = aligned

    o3d.visualization.draw_geometries(
        [source_vis, target_vis, aligned_vis],
        window_name="ICP Alignment — Red: target, Blue: source, Green: aligned",
        width=1280,
        height=720,
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        sys.exit(f"[ERROR] Folder not found: {folder}")

    # Get all PLY files sorted by name (chronological order)
    ply_files = sorted(folder.glob("*.ply"))
    if len(ply_files) < 2:
        sys.exit(f"[ERROR] Need at least 2 .ply files, found {len(ply_files)}")

    print("=" * 60)
    print("  Batch ICP Motion Estimation")
    print("=" * 60)
    print(f"  Folder    : {folder}")
    print(f"  Frames    : {len(ply_files)}")
    print(f"  Pairs     : {len(ply_files) - 1}")
    print(f"  Voxel size: {args.voxel} mm")
    print(f"  Threshold : {args.threshold} mm")
    print(f"  Output    : {args.output}")
    print("=" * 60)

    results = []
    total_displacement = 0.0
    failed = 0

    for i in range(len(ply_files) - 1):
        f1 = ply_files[i]
        f2 = ply_files[i + 1]

        # Calculate time delta from filenames
        try:
            t1 = datetime.strptime(f1.stem, "point_cloud_%Y%m%d_%H%M%S_%f")
            t2 = datetime.strptime(f2.stem, "point_cloud_%Y%m%d_%H%M%S_%f")
            dt_ms = (t2 - t1).total_seconds() * 1000
        except ValueError:
            dt_ms = None

        # Load both frames
        source = load_pcd(f1, args.voxel)
        target = load_pcd(f2, args.voxel)

        if source is None or target is None:
            failed += 1
            continue

        # Run ICP
        icp_result = run_icp(source, target, args.threshold)
        translation, displacement, rotation = extract_motion(icp_result.transformation)
        fitness = icp_result.fitness
        rmse = icp_result.inlier_rmse

        total_displacement += displacement

        # Print progress
        dt_str = f"{dt_ms:.1f}ms" if dt_ms else "N/A"
        print(f"  [{i+1:04d}/{len(ply_files)-1}] {f1.name} → {f2.name}")
        print(f"           Δt={dt_str}  disp={displacement:.3f}mm  rot={rotation:.3f}°  fitness={fitness:.3f}  RMSE={rmse:.3f}")

        results.append({
            "pair": i + 1,
            "frame1": f1.name,
            "frame2": f2.name,
            "dt_ms": dt_ms,
            "displacement_mm": displacement,
            "tx_mm": translation[0],
            "ty_mm": translation[1],
            "tz_mm": translation[2],
            "rotation_deg": rotation,
            "fitness": fitness,
            "rmse": rmse,
        })

        # Visualize first pair if requested
        if args.visualize and i == 0:
            print("\n[INFO] Opening visualization of first frame pair...")
            visualize_alignment(source, target, icp_result.transformation)

    # ── Summary ──────────────────────────────────────────────────────────────
    if results:
        displacements = [r["displacement_mm"] for r in results]
        print("\n" + "═" * 60)
        print("  Motion Summary")
        print("═" * 60)
        print(f"  Total frames processed : {len(results)}")
        print(f"  Failed pairs           : {failed}")
        print(f"  Mean displacement      : {np.mean(displacements):.3f} mm")
        print(f"  Max displacement       : {np.max(displacements):.3f} mm")
        print(f"  Min displacement       : {np.min(displacements):.3f} mm")
        print(f"  Std deviation          : {np.std(displacements):.3f} mm")
        print(f"  Total displacement     : {total_displacement:.3f} mm")
        print("═" * 60)

        # Save to CSV
        out_path = Path(args.output)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to: {out_path.resolve()}")

    else:
        print("[WARN] No results — all pairs failed.")


if __name__ == "__main__":
    main()
