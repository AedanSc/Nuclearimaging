"""
estimate_motion.py
─────────────────────────────────────────────────────────────────────────────
Takes two consecutive .ply point cloud frames and runs FLOT to estimate
the surface motion between them. Used for SPECT patient motion correction.

Usage:
    python estimate_motion.py --frame1 zivid_frames/frame_0000.ply --frame2 zivid_frames/frame_0001.ply

Optional:
    --npoints   Number of points to sample (default: 2048)
    --output    Save the flow result as a .npy file
    --visualize Show a 3D visualization of the motion vectors
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


# ─── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate motion between two point cloud frames using FLOT."
    )
    parser.add_argument("--frame1", required=True, help="Path to first .ply frame (t=0)")
    parser.add_argument("--frame2", required=True, help="Path to second .ply frame (t=1)")
    parser.add_argument("--npoints", type=int, default=2048,
                        help="Number of points to sample (default: 2048)")
    parser.add_argument("--output", default=None,
                        help="Optional: save flow vectors to a .npy file")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the motion vectors in 3D")
    return parser.parse_args()


# ─── Point cloud loading & preprocessing ────────────────────────────────────

def load_and_preprocess(filepath: str, npoints: int) -> np.ndarray:
    """Load a .ply file, clean it, downsample, and return as numpy array."""
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {filepath}")

    pcd = o3d.io.read_point_cloud(str(path))

    if len(pcd.points) == 0:
        sys.exit(f"[ERROR] Point cloud is empty: {filepath}")

    print(f"  Loaded {len(pcd.points):,} points from {path.name}")

    # Remove statistical outliers (noise reduction)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  After noise removal: {len(pcd.points):,} points")

    points = np.asarray(pcd.points, dtype=np.float32)

    # Randomly sample exactly npoints
    if len(points) >= npoints:
        idx = np.random.choice(len(points), npoints, replace=False)
    else:
        # If fewer points than npoints, sample with replacement
        idx = np.random.choice(len(points), npoints, replace=True)
        print(f"  [WARN] Only {len(points)} points available, sampling with replacement to reach {npoints}")

    return points[idx]


def normalize(pc1: np.ndarray, pc2: np.ndarray):
    """Center and scale both point clouds together for stable inference."""
    combined = np.concatenate([pc1, pc2], axis=0)
    centroid = combined.mean(axis=0)
    scale = np.abs(combined - centroid).max()

    pc1 = (pc1 - centroid) / scale
    pc2 = (pc2 - centroid) / scale

    return pc1, pc2, centroid, scale


# ─── FLOT inference ──────────────────────────────────────────────────────────

def run_flot(pc1: np.ndarray, pc2: np.ndarray, device: torch.device) -> np.ndarray:
    """Run FLOT scene flow estimation between two point clouds."""
    try:
        import sys as _sys
        import os as _os
        _sys.path.insert(0, str(Path("FLOT").resolve()))
        from flot.models import FLOT as FLOTModel
    except ImportError:
        _sys.exit("[ERROR] FLOT not found. Make sure you ran: pip install -e . inside the FLOT folder")

    print("\n[INFO] Loading FLOT model...")
    # nb_iter controls Sinkhorn iterations — 1 is fast, 3 is more accurate
    model = FLOTModel(nb_iter=1)

    # Load pretrained weights
    weights_path = Path("FLOT/flot/pretrained_models/model_2048.tar")
    if weights_path.exists():
        print(f"[INFO] Loading pretrained weights from {weights_path}")
        checkpoint = torch.load(str(weights_path), map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("[INFO] Pretrained weights loaded successfully")
    else:
        print("[WARN] No pretrained weights found, using random weights (results will be inaccurate)")

    model = model.to(device)
    model.eval()

    # Convert to tensors — FLOT expects shape (1, N, 3)
    t1 = torch.tensor(pc1, dtype=torch.float32).unsqueeze(0).to(device)
    t2 = torch.tensor(pc2, dtype=torch.float32).unsqueeze(0).to(device)

    print("[INFO] Running FLOT inference...")
    with torch.no_grad():
        # forward() expects a tuple (pc1, pc2)
        flow = model((t1, t2))  # shape: (1, N, 3)

    return flow.squeeze(0).cpu().numpy()  # shape: (N, 3)


# ─── Visualization ───────────────────────────────────────────────────────────

def visualize_flow(pc1: np.ndarray, flow: np.ndarray, centroid, scale):
    """Visualize motion vectors as lines between original and displaced points."""
    print("\n[INFO] Opening visualization...")

    # Denormalize back to original coordinate space
    pc1_real = pc1 * scale + centroid
    pc2_estimated = (pc1 + flow) * scale + centroid

    # Original point cloud (blue)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1_real)
    pcd1.paint_uniform_color([0.2, 0.4, 1.0])  # blue

    # Estimated displaced point cloud (red)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2_estimated)
    pcd2.paint_uniform_color([1.0, 0.2, 0.2])  # red

    # Draw flow vectors as lines
    points = np.concatenate([pc1_real, pc2_estimated], axis=0)
    n = len(pc1_real)
    lines = [[i, i + n] for i in range(n)]

    # Color lines by flow magnitude
    magnitudes = np.linalg.norm(flow, axis=1)
    mag_norm = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min() + 1e-8)
    line_colors = [[m, 0.3, 1.0 - m] for m in mag_norm]  # purple→orange by magnitude

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    o3d.visualization.draw_geometries(
        [pcd1, pcd2, line_set],
        window_name="FLOT Motion Estimation — Blue: t=0, Red: t=1, Lines: flow vectors",
        width=1280,
        height=720,
    )


# ─── Summary stats ───────────────────────────────────────────────────────────

def print_motion_summary(flow: np.ndarray, scale: float):
    """Print useful motion statistics in real-world units (mm)."""
    # Denormalize flow magnitudes back to mm
    flow_mm = flow * scale
    magnitudes = np.linalg.norm(flow_mm, axis=1)

    print("\n" + "═" * 50)
    print("  Motion Summary (millimetres)")
    print("═" * 50)
    print(f"  Mean displacement  : {magnitudes.mean():.3f} mm")
    print(f"  Max displacement   : {magnitudes.max():.3f} mm")
    print(f"  Min displacement   : {magnitudes.min():.3f} mm")
    print(f"  Std deviation      : {magnitudes.std():.3f} mm")
    print(f"\n  Mean flow vector   : X={flow_mm[:,0].mean():.3f}  Y={flow_mm[:,1].mean():.3f}  Z={flow_mm[:,2].mean():.3f} mm")
    print("═" * 50)

    return magnitudes


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n[INFO] Loading frame 1: {args.frame1}")
    pc1 = load_and_preprocess(args.frame1, args.npoints)

    print(f"\n[INFO] Loading frame 2: {args.frame2}")
    pc2 = load_and_preprocess(args.frame2, args.npoints)

    # Normalize for stable inference
    pc1_norm, pc2_norm, centroid, scale = normalize(pc1, pc2)

    # Run FLOT
    flow = run_flot(pc1_norm, pc2_norm, device)

    # Print motion stats in real-world mm
    print_motion_summary(flow, scale)

    # Optionally save flow to file
    if args.output:
        out_path = Path(args.output)
        np.save(str(out_path), flow)
        print(f"\n[INFO] Flow vectors saved to {out_path}")

    # Optionally visualize
    if args.visualize:
        visualize_flow(pc1_norm, flow, centroid, scale)


if __name__ == "__main__":
    main()