"""
batch_nsfp.py
─────────────────────────────────────────────────────────────────────────────
Runs Neural Scene Flow Prior (NSFP) on all consecutive .ply frame pairs in a
folder and outputs a motion timeline. Designed for SPECT patient surface
motion tracking using Zivid structured light point clouds.

No C++ compilation, no Conda, no legacy dependencies.
Just PyTorch + NumPy + Open3D.

Usage:
    python batch_nsfp.py --folder point_clouds/point_clouds/

Optional:
    --npoints   Number of points per frame (default: 5000)
    --iters     Optimization iterations per pair (default: 300)
    --lr        Learning rate (default: 0.001)
    --output    Output CSV file (default: nsfp_results.csv)
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import open3d as o3d


# ─── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch NSFP motion estimation across consecutive point cloud frames."
    )
    parser.add_argument("--folder", required=True, help="Folder containing .ply files")
    parser.add_argument("--npoints", type=int, default=5000,
                        help="Number of points per frame (default: 5000)")
    parser.add_argument("--iters", type=int, default=300,
                        help="Optimization iterations per pair (default: 300)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--output", default="nsfp_results.csv",
                        help="Output CSV file (default: nsfp_results.csv)")
    return parser.parse_args()


# ─── Neural Scene Flow Prior Model ───────────────────────────────────────────

class NSFPrior(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, pc):
        return self.net(pc)


# ─── Point cloud loading ──────────────────────────────────────────────────────

def load_pcd(filepath: Path, npoints: int, device: torch.device):
    pcd = o3d.io.read_point_cloud(str(filepath))
    if len(pcd.points) == 0:
        return None

    pcd = pcd.voxel_down_sample(voxel_size=2.0)
    points = np.asarray(pcd.points, dtype=np.float32)

    if len(points) == 0:
        return None

    if len(points) >= npoints:
        idx = np.random.choice(len(points), npoints, replace=False)
    else:
        idx = np.random.choice(len(points), npoints, replace=True)

    return torch.tensor(points[idx], dtype=torch.float32).to(device)


# ─── Normalization ────────────────────────────────────────────────────────────

def normalize_pcds(pc1, pc2):
    combined = torch.cat([pc1, pc2], dim=0)
    centroid = combined.mean(dim=0)
    scale = (combined - centroid).abs().max()
    return (pc1 - centroid) / scale, (pc2 - centroid) / scale, centroid, scale


# ─── Chamfer loss ─────────────────────────────────────────────────────────────

def chamfer_loss(pc1_warped, pc2, batch_size=2000):
    total = torch.tensor(0.0, device=pc1_warped.device)
    n = len(pc1_warped)
    count = 0
    for i in range(0, n, batch_size):
        batch = pc1_warped[i:i+batch_size]
        dist = torch.cdist(batch.unsqueeze(0), pc2.unsqueeze(0)).squeeze(0)
        total += dist.min(dim=1).values.mean()
        count += 1
    return total / count


# ─── Smoothness loss ──────────────────────────────────────────────────────────

def smoothness_loss(flow, pc1, k=8, batch_size=2000):
    total = torch.tensor(0.0, device=flow.device)
    n = min(len(pc1), 5000)
    count = 0
    for i in range(0, n, batch_size):
        batch_pts = pc1[i:i+batch_size]
        batch_flow = flow[i:i+batch_size]
        dist = torch.cdist(batch_pts.unsqueeze(0), pc1.unsqueeze(0)).squeeze(0)
        _, nn_idx = dist.topk(k+1, dim=1, largest=False)
        nn_flow = flow[nn_idx[:, 1:]]
        total += (batch_flow.unsqueeze(1) - nn_flow).norm(dim=2).mean()
        count += 1
    return total / count


# ─── Optimize flow ────────────────────────────────────────────────────────────

def optimize_flow(pc1, pc2, iters, lr, device):
    model = NSFPrior(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters)

    best_loss = float('inf')
    best_flow = None

    for i in range(iters):
        optimizer.zero_grad()
        flow = model(pc1)
        pc1_warped = pc1 + flow
        loss = chamfer_loss(pc1_warped, pc2) + 0.1 * smoothness_loss(flow, pc1)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_flow = flow.detach().clone()

    return best_flow, best_loss


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        sys.exit(f"[ERROR] Folder not found: {folder}")

    ply_files = sorted(folder.glob("*.ply"))
    if len(ply_files) < 2:
        sys.exit(f"[ERROR] Need at least 2 .ply files, found {len(ply_files)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Batch NSFP Motion Estimation")
    print("=" * 60)
    print(f"  Folder  : {folder}")
    print(f"  Frames  : {len(ply_files)}")
    print(f"  Pairs   : {len(ply_files) - 1}")
    print(f"  Points  : {args.npoints}")
    print(f"  Iters   : {args.iters}")
    print(f"  Device  : {device}")
    print("=" * 60)

    results = []
    failed = 0

    for i in range(len(ply_files) - 1):
        f1 = ply_files[i]
        f2 = ply_files[i + 1]

        # Time delta
        try:
            t1 = datetime.strptime(f1.stem, "point_cloud_%Y%m%d_%H%M%S_%f")
            t2 = datetime.strptime(f2.stem, "point_cloud_%Y%m%d_%H%M%S_%f")
            dt_ms = (t2 - t1).total_seconds() * 1000
        except ValueError:
            dt_ms = None

        pc1 = load_pcd(f1, args.npoints, device)
        pc2 = load_pcd(f2, args.npoints, device)

        if pc1 is None or pc2 is None:
            failed += 1
            continue

        try:
            pc1_norm, pc2_norm, centroid, scale = normalize_pcds(pc1, pc2)
            flow, best_loss = optimize_flow(pc1_norm, pc2_norm, args.iters, args.lr, device)

            # Denormalize flow to mm
            flow_mm = flow.cpu().numpy() * scale.cpu().item()
            magnitudes = np.linalg.norm(flow_mm, axis=1)

            mean_disp = magnitudes.mean()
            max_disp = magnitudes.max()
            tx = flow_mm[:, 0].mean()
            ty = flow_mm[:, 1].mean()
            tz = flow_mm[:, 2].mean()

            dt_str = f"{dt_ms:.1f}ms" if dt_ms else "N/A"
            print(f"  [{i+1:04d}/{len(ply_files)-1}] {f1.name}")
            print(f"           Δt={dt_str}  mean_disp={mean_disp:.3f}mm  max_disp={max_disp:.3f}mm  loss={best_loss:.6f}")

            results.append({
                "pair": i + 1,
                "frame1": f1.name,
                "frame2": f2.name,
                "dt_ms": dt_ms,
                "mean_displacement_mm": mean_disp,
                "max_displacement_mm": max_disp,
                "tx_mm": tx,
                "ty_mm": ty,
                "tz_mm": tz,
                "best_loss": best_loss,
            })

            # Free GPU memory between pairs
            del pc1, pc2, pc1_norm, pc2_norm, flow
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [WARN] Failed on pair {i+1}: {e}")
            failed += 1
            torch.cuda.empty_cache()
            continue

    # Summary
    if results:
        displacements = [r["mean_displacement_mm"] for r in results]
        print("\n" + "═" * 60)
        print("  Motion Summary")
        print("═" * 60)
        print(f"  Frames processed : {len(results)}")
        print(f"  Failed pairs     : {failed}")
        print(f"  Mean displacement: {np.mean(displacements):.3f} mm")
        print(f"  Max displacement : {np.max(displacements):.3f} mm")
        print(f"  Min displacement : {np.min(displacements):.3f} mm")
        print(f"  Std deviation    : {np.std(displacements):.3f} mm")
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
