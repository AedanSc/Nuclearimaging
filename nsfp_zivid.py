"""
nsfp_zivid.py
─────────────────────────────────────────────────────────────────────────────
Clean-room reimplementation of Neural Scene Flow Prior (NSFP) for modern
PyTorch 2.x and Python 3.12. Designed for high-density Zivid structured
light point clouds for SPECT patient surface motion tracking.

No C++ compilation, no Conda, no legacy dependencies.
Just PyTorch + NumPy + Open3D.

How it works:
    A small MLP is optimized at runtime to find the smooth flow field
    that maps point cloud 1 to point cloud 2. The MLP acts as a neural
    "prior" that encourages smooth, physically plausible motion.

Usage:
    python nsfp_zivid.py --frame1 frame_0000.ply --frame2 frame_0001.ply

Optional:
    --npoints     Number of points to use (default: 50000)
    --iters       Optimization iterations (default: 500)
    --lr          Learning rate (default: 0.001)
    --visualize   Show 3D visualization of flow
    --output      Save flow vectors to .npy file
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import open3d as o3d


# ─── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural Scene Flow Prior for Zivid point clouds."
    )
    parser.add_argument("--frame1", required=True, help="Path to first .ply frame (t=0)")
    parser.add_argument("--frame2", required=True, help="Path to second .ply frame (t=1)")
    parser.add_argument("--npoints", type=int, default=5000,
                        help="Number of points to use (default: 50000)")
    parser.add_argument("--iters", type=int, default=500,
                        help="Optimization iterations (default: 500)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize flow vectors in 3D")
    parser.add_argument("--output", default=None,
                        help="Save flow vectors to .npy file")
    return parser.parse_args()


# ─── Neural Scene Flow Prior Model ───────────────────────────────────────────

class NSFPrior(nn.Module):
    """
    A simple MLP that acts as a neural prior for scene flow.
    Takes a 3D point (x, y, z) and outputs a flow vector (dx, dy, dz).
    The network's implicit smoothness bias encourages physically plausible motion.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # outputs (dx, dy, dz)
        )

    def forward(self, pc):
        return self.net(pc)


# ─── Point cloud loading ──────────────────────────────────────────────────────

def load_pcd(filepath: str, npoints: int, device: torch.device) -> torch.Tensor:
    """Load PLY, denoise, downsample, return as normalized tensor."""
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {filepath}")

    pcd = o3d.io.read_point_cloud(str(path))

    if len(pcd.points) == 0:
        sys.exit(f"[ERROR] Empty point cloud: {filepath}")

    print(f"  Loaded {len(pcd.points):,} points from {path.name}")

    # Skip denoising for large point clouds — just downsample directly
    # Use a small fixed voxel size to preserve enough points
    points = np.asarray(pcd.points)
    pcd = pcd.voxel_down_sample(voxel_size=2.0)  # 2mm voxel — preserves density

    points = np.asarray(pcd.points, dtype=np.float32)

    # Sample exactly npoints
    if len(points) >= npoints:
        idx = np.random.choice(len(points), npoints, replace=False)
    else:
        idx = np.random.choice(len(points), npoints, replace=True)
        print(f"  [WARN] Only {len(points)} points, sampling with replacement")

    points = points[idx]
    print(f"  After processing: {len(points):,} points")

    return torch.tensor(points, dtype=torch.float32).to(device)


# ─── Normalization ────────────────────────────────────────────────────────────

def normalize_pcds(pc1: torch.Tensor, pc2: torch.Tensor):
    """Normalize both point clouds to [-1, 1] range for stable optimization."""
    combined = torch.cat([pc1, pc2], dim=0)
    centroid = combined.mean(dim=0)
    scale = (combined - centroid).abs().max()

    pc1_norm = (pc1 - centroid) / scale
    pc2_norm = (pc2 - centroid) / scale

    return pc1_norm, pc2_norm, centroid, scale


# ─── Chamfer Distance Loss ────────────────────────────────────────────────────

def chamfer_loss(pc1_warped: torch.Tensor, pc2: torch.Tensor,
                 batch_size: int = 2000) -> torch.Tensor:
    """
    Compute one-directional Chamfer distance between warped pc1 and pc2.
    Uses batching to avoid OOM on large point clouds.
    For each point in pc1_warped, find nearest neighbor in pc2.
    """
    total_loss = torch.tensor(0.0, device=pc1_warped.device)
    n = len(pc1_warped)

    for i in range(0, n, batch_size):
        batch = pc1_warped[i:i+batch_size]  # (B, 3)

        # Compute pairwise distances using torch.cdist
        # batch: (B, 3), pc2: (N, 3) → dist: (B, N)
        dist = torch.cdist(batch.unsqueeze(0), pc2.unsqueeze(0)).squeeze(0)

        # Nearest neighbor distance for each point in batch
        min_dist, _ = dist.min(dim=1)
        total_loss += min_dist.mean()

    return total_loss / (n / batch_size)


def smoothness_loss(flow: torch.Tensor, pc1: torch.Tensor,
                    k: int = 8, batch_size: int = 5000) -> torch.Tensor:
    """
    Encourage spatially smooth flow: nearby points should have similar flow vectors.
    Uses k-nearest neighbors for local smoothness regularization.
    """
    total_loss = torch.tensor(0.0, device=flow.device)
    n = len(pc1)

    for i in range(0, min(n, 10000), batch_size):  # limit for speed
        batch_pts = pc1[i:i+batch_size]
        batch_flow = flow[i:i+batch_size]

        # Find k nearest neighbors
        dist = torch.cdist(batch_pts.unsqueeze(0), pc1.unsqueeze(0)).squeeze(0)
        _, nn_idx = dist.topk(k+1, dim=1, largest=False)
        nn_idx = nn_idx[:, 1:]  # exclude self

        # Flow at neighbors
        nn_flow = flow[nn_idx]  # (B, k, 3)

        # Smoothness: flow should be similar to neighbors
        smooth = (batch_flow.unsqueeze(1) - nn_flow).norm(dim=2).mean()
        total_loss += smooth

    return total_loss / (min(n, 10000) / batch_size)


# ─── Optimization loop ────────────────────────────────────────────────────────

def optimize_flow(pc1: torch.Tensor, pc2: torch.Tensor,
                  iters: int, lr: float, device: torch.device) -> torch.Tensor:
    """
    Optimize the MLP at runtime to find the flow field from pc1 to pc2.
    No pretrained weights — the network is initialized fresh each time.
    """
    model = NSFPrior(hidden_dim=128).to(device)

    # torch.compile requires Triton which is unavailable on Windows — skip
    print("[INFO] Running without torch.compile (Windows not supported)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters)

    print(f"\n[INFO] Optimizing for {iters} iterations...")
    print(f"       Points: {len(pc1):,}  LR: {lr}  Device: {device}")

    best_loss = float('inf')
    best_flow = None

    for i in range(iters):
        optimizer.zero_grad()

        # Predict flow for each point in pc1
        flow = model(pc1)  # (N, 3)

        # Warp pc1 by predicted flow
        pc1_warped = pc1 + flow

        # Chamfer loss: warped pc1 should match pc2
        loss_chamfer = chamfer_loss(pc1_warped, pc2)

        # Smoothness regularization
        loss_smooth = smoothness_loss(flow, pc1)

        # Combined loss
        loss = loss_chamfer + 0.1 * loss_smooth

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_flow = flow.detach().clone()

        if (i + 1) % 50 == 0:
            print(f"  Iter [{i+1:4d}/{iters}]  "
                  f"loss={loss.item():.6f}  "
                  f"chamfer={loss_chamfer.item():.6f}  "
                  f"smooth={loss_smooth.item():.6f}")

    print(f"\n[INFO] Best loss: {best_loss:.6f}")
    return best_flow


# ─── Motion summary ───────────────────────────────────────────────────────────

def print_motion_summary(flow: torch.Tensor, scale: float):
    """Print motion statistics in real-world mm."""
    flow_np = flow.cpu().numpy()
    flow_mm = flow_np * scale.cpu().numpy()
    magnitudes = np.linalg.norm(flow_mm, axis=1)

    print("\n" + "═" * 55)
    print("  Motion Summary (millimetres)")
    print("═" * 55)
    print(f"  Mean displacement  : {magnitudes.mean():.3f} mm")
    print(f"  Max displacement   : {magnitudes.max():.3f} mm")
    print(f"  Min displacement   : {magnitudes.min():.3f} mm")
    print(f"  Std deviation      : {magnitudes.std():.3f} mm")
    print(f"\n  Mean flow vector   : "
          f"X={flow_mm[:,0].mean():.3f}  "
          f"Y={flow_mm[:,1].mean():.3f}  "
          f"Z={flow_mm[:,2].mean():.3f} mm")
    print("═" * 55)

    return magnitudes


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_flow(pc1: torch.Tensor, flow: torch.Tensor, centroid, scale):
    """Visualize flow vectors as lines between original and displaced points."""
    print("\n[INFO] Opening visualization...")

    pc1_np = pc1.cpu().numpy()
    flow_np = flow.cpu().numpy()
    centroid_np = centroid.cpu().numpy()
    scale_val = scale.cpu().item()

    # Denormalize
    pc1_real = pc1_np * scale_val + centroid_np
    pc2_estimated = (pc1_np + flow_np) * scale_val + centroid_np

    # Subsample for visualization (max 5000 vectors)
    n_vis = min(5000, len(pc1_real))
    idx = np.random.choice(len(pc1_real), n_vis, replace=False)
    pc1_vis = pc1_real[idx]
    pc2_vis = pc2_estimated[idx]

    # Original point cloud (blue)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1_real)
    pcd1.paint_uniform_color([0.2, 0.4, 1.0])

    # Estimated displaced points (red)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2_estimated)
    pcd2.paint_uniform_color([1.0, 0.2, 0.2])

    # Flow vectors as lines
    points = np.concatenate([pc1_vis, pc2_vis], axis=0)
    n = len(pc1_vis)
    lines = [[i, i + n] for i in range(n)]

    flow_vis = flow_np[idx]
    magnitudes = np.linalg.norm(flow_vis, axis=1)
    mag_norm = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min() + 1e-8)
    line_colors = [[m, 0.3, 1.0 - m] for m in mag_norm]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    o3d.visualization.draw_geometries(
        [pcd1, pcd2, line_set],
        window_name="NSFP Flow — Blue: t=0, Red: t=1, Lines: flow vectors",
        width=1280, height=720,
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n[INFO] Loading frame 1: {args.frame1}")
    pc1 = load_pcd(args.frame1, args.npoints, device)

    print(f"\n[INFO] Loading frame 2: {args.frame2}")
    pc2 = load_pcd(args.frame2, args.npoints, device)

    # Normalize
    pc1_norm, pc2_norm, centroid, scale = normalize_pcds(pc1, pc2)

    # Optimize flow
    flow = optimize_flow(pc1_norm, pc2_norm, args.iters, args.lr, device)

    # Print motion summary in real-world mm
    print_motion_summary(flow, scale)

    # Save flow
    if args.output:
        np.save(args.output, flow.cpu().numpy())
        print(f"\n[INFO] Flow saved to {args.output}")

    # Visualize
    if args.visualize:
        visualize_flow(pc1_norm, flow, centroid, scale)


if __name__ == "__main__":
    main()