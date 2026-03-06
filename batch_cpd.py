"""
batch_cpd.py
─────────────────────────────────────────────────────────────────────────────
Runs CPD (Coherent Point Drift) on all consecutive .ply frame pairs in a
folder and outputs a motion timeline. Designed for linear/rigid motion tracking
for SPECT patient motion correction.

CPD is a probabilistic algorithm that handles both rigid and non-rigid motion,
making it more suitable than ICP for breathing deformation.

Usage:
    python batch_cpd.py --folder point_clouds/point_clouds/

Optional:
    --voxel     Voxel size for downsampling (default: 5.0 mm)
    --mode      rigid | affine | deformable (default: rigid)
    --output    Save results to a CSV file (default: cpd_results.csv)
    --visualize Show visualization of first frame pair
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
        description="Batch CPD motion estimation across consecutive point cloud frames."
    )
    parser.add_argument("--folder", required=True, help="Folder containing .ply files")
    parser.add_argument("--voxel", type=float, default=5.0,
                        help="Voxel size for downsampling in mm (default: 5.0)")
    parser.add_argument("--mode", choices=["rigid", "affine", "deformable"],
                        default="rigid",
                        help="CPD mode: rigid, affine, or deformable (default: rigid)")
    parser.add_argument("--output", default="cpd_results.csv",
                        help="Output CSV file (default: cpd_results.csv)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the first frame pair alignment")
    return parser.parse_args()


# ─── Load and preprocess ─────────────────────────────────────────────────────

def load_pcd(filepath: Path, voxel_size: float) -> np.ndarray:
    """Load a PLY file, remove outliers, downsample, return as numpy array."""
    pcd = o3d.io.read_point_cloud(str(filepath))

    if len(pcd.points) == 0:
        print(f"  [WARN] Empty point cloud: {filepath.name}")
        return None

    # Downsample
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return np.asarray(pcd.points, dtype=np.float64)


# ─── CPD ─────────────────────────────────────────────────────────────────────

def run_cpd(source: np.ndarray, target: np.ndarray, mode: str):
    """Run CPD registration between source and target point clouds."""
    try:
        from pycpd import RigidRegistration, AffineRegistration, DeformableRegistration
    except ImportError:
        sys.exit("[ERROR] pycpd not found. Run: pip install pycpd")

    if mode == "rigid":
        reg = RigidRegistration(X=target, Y=source)
    elif mode == "affine":
        reg = AffineRegistration(X=target, Y=source)
    else:
        reg = DeformableRegistration(X=target, Y=source)

    transformed, params = reg.register()
    return transformed, params


def extract_motion_cpd(source: np.ndarray, transformed: np.ndarray, params, mode: str):
    """Extract displacement and rotation from CPD result."""
    # Per-point displacements
    displacements = np.linalg.norm(transformed - source, axis=1)
    mean_displacement = displacements.mean()
    max_displacement = displacements.max()

    rotation_deg = 0.0
    translation = np.zeros(3)

    if mode == "rigid" and params is not None:
        # params: (s, R, t) — scale, rotation matrix, translation
        try:
            s, R, t = params
            translation = np.array(t).flatten()
            # Extract rotation angle from rotation matrix
            trace = np.trace(R)
            cos_angle = np.clip((trace - 1) / 2, -1, 1)
            rotation_deg = np.degrees(np.arccos(cos_angle))
        except Exception:
            pass

    return mean_displacement, max_displacement, translation, rotation_deg


# ─── Visualization ───────────────────────────────────────────────────────────

def visualize_cpd(source: np.ndarray, target: np.ndarray, transformed: np.ndarray):
    """Visualize CPD alignment."""
    def to_pcd(points, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        return pcd

    source_pcd = to_pcd(source, [0.2, 0.4, 1.0])       # blue = source
    target_pcd = to_pcd(target, [1.0, 0.2, 0.2])       # red = target
    aligned_pcd = to_pcd(transformed, [0.2, 0.8, 0.2]) # green = aligned

    o3d.visualization.draw_geometries(
        [source_pcd, target_pcd, aligned_pcd],
        window_name="CPD Alignment — Red: target, Blue: source, Green: aligned",
        width=1280,
        height=720,
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        sys.exit(f"[ERROR] Folder not found: {folder}")

    ply_files = sorted(folder.glob("*.ply"))
    if len(ply_files) < 2:
        sys.exit(f"[ERROR] Need at least 2 .ply files, found {len(ply_files)}")

    print("=" * 60)
    print("  Batch CPD Motion Estimation")
    print("=" * 60)
    print(f"  Folder    : {folder}")
    print(f"  Frames    : {len(ply_files)}")
    print(f"  Pairs     : {len(ply_files) - 1}")
    print(f"  Voxel size: {args.voxel} mm")
    print(f"  CPD mode  : {args.mode}")
    print(f"  Output    : {args.output}")
    print("=" * 60)
    print("  [NOTE] CPD is slower than ICP — be patient on large datasets.\n")

    results = []
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

        # Run CPD
        try:
            transformed, params = run_cpd(source, target, args.mode)
            mean_disp, max_disp, translation, rotation = extract_motion_cpd(
                source, transformed, params, args.mode
            )
        except Exception as e:
            print(f"  [WARN] CPD failed on pair {i+1}: {e}")
            failed += 1
            continue

        dt_str = f"{dt_ms:.1f}ms" if dt_ms else "N/A"
        print(f"  [{i+1:04d}/{len(ply_files)-1}] {f1.name} → {f2.name}")
        print(f"           Δt={dt_str}  mean_disp={mean_disp:.3f}mm  max_disp={max_disp:.3f}mm  rot={rotation:.3f}°")

        results.append({
            "pair": i + 1,
            "frame1": f1.name,
            "frame2": f2.name,
            "dt_ms": dt_ms,
            "mean_displacement_mm": mean_disp,
            "max_displacement_mm": max_disp,
            "tx_mm": translation[0] if len(translation) > 0 else 0,
            "ty_mm": translation[1] if len(translation) > 1 else 0,
            "tz_mm": translation[2] if len(translation) > 2 else 0,
            "rotation_deg": rotation,
        })

        # Visualize first pair if requested
        if args.visualize and i == 0:
            print("\n[INFO] Opening visualization of first frame pair...")
            visualize_cpd(source, target, transformed)

    # ── Summary ──────────────────────────────────────────────────────────────
    if results:
        displacements = [r["mean_displacement_mm"] for r in results]
        print("\n" + "═" * 60)
        print("  Motion Summary")
        print("═" * 60)
        print(f"  Total frames processed : {len(results)}")
        print(f"  Failed pairs           : {failed}")
        print(f"  Mean displacement      : {np.mean(displacements):.3f} mm")
        print(f"  Max displacement       : {np.max(displacements):.3f} mm")
        print(f"  Min displacement       : {np.min(displacements):.3f} mm")
        print(f"  Std deviation          : {np.std(displacements):.3f} mm")
        print("═" * 60)

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