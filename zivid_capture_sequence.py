"""
zivid_capture_sequence.py
─────────────────────────────────────────────────────────────────────────────
Capture a series of 3D point clouds with a Zivid camera — like a video.

Three stopping modes (choose at runtime):
  1. duration   – capture every N seconds for a total of T seconds
  2. keypress   – capture continuously until you press Enter
  3. frames     – capture exactly N frames

Supported output formats: .zdf  .ply  .pcd  .xyz

Usage examples
──────────────
  # Mode 1 – 30-second session, one frame every 1.5 s, saved as .ply
  python zivid_capture_sequence.py --mode duration --duration 30 --interval 1.5 --format ply

  # Mode 2 – capture until Enter is pressed, .zdf output
  python zivid_capture_sequence.py --mode keypress --format zdf

  # Mode 3 – exactly 20 frames, .pcd output, 0.5 s between captures
  python zivid_capture_sequence.py --mode frames --frames 20 --interval 0.5 --format pcd

Options
───────
  --mode        duration | keypress | frames   (default: keypress)
  --duration    total seconds to capture       (mode: duration)
  --frames      total number of frames         (mode: frames)
  --interval    seconds between captures       (default: 1.0)
  --format      zdf | ply | pcd | xyz          (default: zdf)
  --output-dir  folder to save frames in       (default: ./zivid_frames)
  --settings    path to a Zivid .yml settings file (optional)
  --no-color    disable color capture (faster, depth-only)
"""

import argparse
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path


# ─── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture a sequence of Zivid 3D point clouds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["duration", "keypress", "frames"],
                        default="keypress",
                        help="Stopping condition (default: keypress)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Total capture duration in seconds (mode: duration)")
    parser.add_argument("--frames", type=int, default=10,
                        help="Number of frames to capture (mode: frames)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between consecutive captures (default: 1.0)")
    parser.add_argument("--format", choices=["zdf", "ply", "pcd", "xyz"],
                        default="zdf", dest="fmt",
                        help="Output file format (default: zdf)")
    parser.add_argument("--output-dir", default="zivid_frames",
                        help="Directory to save frames (default: ./zivid_frames)")
    parser.add_argument("--settings", default=None,
                        help="Path to a Zivid .yml settings file (optional)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colour capture for faster acquisitions")
    return parser.parse_args()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def build_settings(args, zivid):
    """Return a zivid.Settings object — from file or sensible defaults."""
    if args.settings:
        path = Path(args.settings)
        if not path.exists():
            sys.exit(f"[ERROR] Settings file not found: {path}")
        print(f"[INFO] Loading settings from {path}")
        return zivid.Settings.load(str(path))

    # Default: single acquisition with balanced exposure
    acq = zivid.Settings.Acquisition(
        aperture=5.66,
        exposure_time=zivid.Duration(microseconds=10000),
        gain=1.0,
        brightness=1.0,
    )
    settings = zivid.Settings(acquisitions=[acq])

    # Disable colour for speed if requested
    if args.no_color:
        settings.color = zivid.Settings.Color(
            acquisitions=[],
        )

    return settings


def make_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def frame_filename(out_dir: Path, index: int, fmt: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return out_dir / f"frame_{index:04d}_{timestamp}.{fmt}"


def save_frame(frame, filepath: Path, fmt: str):
    """Save a Zivid frame to the requested format."""
    if fmt == "zdf":
        frame.save(str(filepath))
    else:
        # Export via point cloud
        pc = frame.point_cloud()
        if fmt == "ply":
            pc.save(str(filepath))
        elif fmt == "pcd":
            pc.save(str(filepath))
        elif fmt == "xyz":
            pc.save(str(filepath))


def print_summary(out_dir: Path, count: int, elapsed: float, fmt: str):
    print("\n" + "═" * 50)
    print(f"  Capture complete!")
    print(f"  Frames saved : {count}")
    print(f"  Elapsed time : {elapsed:.1f} s")
    print(f"  Format       : .{fmt}")
    print(f"  Output folder: {out_dir.resolve()}")
    print("═" * 50)


# ─── Capture loops ───────────────────────────────────────────────────────────

def capture_loop(camera, settings, args, out_dir: Path):
    """Dispatch to the correct capture loop based on --mode."""
    mode = args.mode

    if mode == "duration":
        return loop_duration(camera, settings, args, out_dir)
    elif mode == "keypress":
        return loop_keypress(camera, settings, args, out_dir)
    elif mode == "frames":
        return loop_frames(camera, settings, args, out_dir)


def _capture_and_save(camera, settings, out_dir, index, fmt) -> float:
    """Capture one frame, save it, return the time taken (seconds)."""
    t0 = time.perf_counter()
    with camera.capture(settings) as frame:
        filepath = frame_filename(out_dir, index, fmt)
        save_frame(frame, filepath, fmt)
    elapsed = time.perf_counter() - t0
    print(f"  [{index:04d}] Saved {filepath.name}  ({elapsed*1000:.0f} ms)")
    return elapsed


def loop_duration(camera, settings, args, out_dir: Path):
    total_duration = args.duration
    interval = args.interval
    fmt = args.fmt

    print(f"\n[START] Duration mode: {total_duration}s, every {interval}s → .{fmt}")
    print("  Press Ctrl-C to stop early.\n")

    count = 0
    session_start = time.perf_counter()

    try:
        while True:
            elapsed_total = time.perf_counter() - session_start
            if elapsed_total >= total_duration:
                break

            capture_start = time.perf_counter()
            _capture_and_save(camera, settings, out_dir, count, fmt)
            count += 1

            # Wait for the remainder of the interval
            capture_time = time.perf_counter() - capture_start
            sleep_time = max(0.0, interval - capture_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    total_elapsed = time.perf_counter() - session_start
    return count, total_elapsed


def loop_keypress(camera, settings, args, out_dir: Path):
    interval = args.interval
    fmt = args.fmt

    print(f"\n[START] Keypress mode: capturing every {interval}s → .{fmt}")
    print("  Press Enter to stop.\n")

    stop_event = threading.Event()

    def wait_for_enter():
        input()  # blocks until Enter
        stop_event.set()

    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    count = 0
    session_start = time.perf_counter()

    try:
        while not stop_event.is_set():
            capture_start = time.perf_counter()
            _capture_and_save(camera, settings, out_dir, count, fmt)
            count += 1

            capture_time = time.perf_counter() - capture_start
            sleep_time = max(0.0, interval - capture_time)

            # Sleep in small chunks so we can react to stop_event quickly
            slept = 0.0
            while slept < sleep_time and not stop_event.is_set():
                chunk = min(0.05, sleep_time - slept)
                time.sleep(chunk)
                slept += chunk

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    total_elapsed = time.perf_counter() - session_start
    return count, total_elapsed


def loop_frames(camera, settings, args, out_dir: Path):
    total_frames = args.frames
    interval = args.interval
    fmt = args.fmt

    print(f"\n[START] Frames mode: {total_frames} frames, every {interval}s → .{fmt}")
    print("  Press Ctrl-C to stop early.\n")

    count = 0
    session_start = time.perf_counter()

    try:
        while count < total_frames:
            capture_start = time.perf_counter()
            _capture_and_save(camera, settings, out_dir, count, fmt)
            count += 1

            if count < total_frames:
                capture_time = time.perf_counter() - capture_start
                sleep_time = max(0.0, interval - capture_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    total_elapsed = time.perf_counter() - session_start
    return count, total_elapsed


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Import Zivid here so missing SDK gives a clean error
    try:
        import zivid
    except ImportError:
        sys.exit(
            "[ERROR] The 'zivid' Python package is not installed.\n"
            "        Install it with:  pip install zivid\n"
            "        or follow the Zivid SDK setup guide for your platform."
        )

    out_dir = make_output_dir(args.output_dir)

    print("=" * 50)
    print("  Zivid Sequential Capture")
    print("=" * 50)

    app = zivid.Application()

    print("[INFO] Connecting to camera...")
    try:
        camera = app.connect_camera()
    except Exception as e:
        sys.exit(f"[ERROR] Could not connect to camera: {e}")

    print(f"[INFO] Camera: {camera.info.model_name}  S/N: {camera.info.serial_number}")

    settings = build_settings(args, zivid)

    # Warm-up capture (optional but reduces first-frame latency)
    print("[INFO] Warm-up capture...")
    with camera.capture(settings):
        pass

    count, elapsed = capture_loop(camera, settings, args, out_dir)
    print_summary(out_dir, count, elapsed, args.fmt)


if __name__ == "__main__":
    main()
