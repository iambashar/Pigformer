#!/usr/bin/env python3
"""
Preprocess ROS2 bag files (.db3) → HDF5.

Extracts color (RGB) and depth images with nanosecond timestamps.
Handles raw Image, CompressedImage, and CameraInfo topics.
Streams frames to keep memory usage low even on 16 GB bags.

HDF5 layout per bag:
    color/images          (N, H, W, 3)  uint8  RGB, LZF-compressed
    color/timestamps_ns   (N,)          int64  nanoseconds since epoch
    depth/images          (M, H, W)     uint16 or float32, LZF-compressed
    depth/timestamps_ns   (M,)          int64
    color/camera_info/    K, D, R, P    (if available)
    depth/camera_info/    K, D, R, P    (if available)

Usage:
    python preprocess_rosbags.py --input_dir /path/to/rosbags_* --output_dir /path/to/hdf5
    python preprocess_rosbags.py --input_dir . --output_dir ./hdf5_out --workers 8
    python preprocess_rosbags.py --input_dir . --output_dir ./hdf5_out --filter rosbags_20251105
"""

import argparse
import csv
import io
import logging
import os
import sqlite3
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, desc=None, unit=None):
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, _n=1):
            return None

        def close(self):
            return None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CDR (Common Data Representation) deserializer
# ---------------------------------------------------------------------------

class CDRReader:
    """Minimal CDR reader for ROS2 message payloads."""

    __slots__ = ("buf", "le", "p", "origin")

    def __init__(self, data: bytes):
        self.buf = data
        # byte 1 of encapsulation header: 0 = big-endian, 1 = little-endian
        self.le = data[1] == 1
        self.p = 4  # skip 4-byte encapsulation header
        self.origin = 4  # CDR alignment is relative to data start

    def _align(self, n: int):
        rem = (self.p - self.origin) % n
        if rem:
            self.p += n - rem

    def u8(self) -> int:
        v = self.buf[self.p]
        self.p += 1
        return v

    def u32(self) -> int:
        self._align(4)
        v = int.from_bytes(self.buf[self.p : self.p + 4], "little" if self.le else "big", signed=False)
        self.p += 4
        return v

    def f64(self) -> float:
        self._align(8)
        fmt = "<d" if self.le else ">d"
        (v,) = struct.unpack_from(fmt, self.buf, self.p)
        self.p += 8
        return v

    def string(self) -> str:
        n = self.u32()  # length including null terminator
        s = self.buf[self.p : self.p + n - 1].decode("utf-8", errors="replace")
        self.p += n
        return s

    def raw_bytes(self) -> bytes:
        n = self.u32()
        b = self.buf[self.p : self.p + n]
        self.p += n
        return b

    def header(self):
        """Parse std_msgs/Header → (timestamp_ns, frame_id)."""
        sec = self.u32()
        nsec = self.u32()
        frame_id = self.string()
        return sec * 1_000_000_000 + nsec, frame_id


# ---------------------------------------------------------------------------
# Message parsers
# ---------------------------------------------------------------------------

def parse_image(data: bytes):
    """sensor_msgs/msg/Image → (ts_ns, h, w, encoding, pixel_bytes)."""
    r = CDRReader(data)
    ts, _ = r.header()
    h = r.u32()
    w = r.u32()
    enc = r.string()
    _bigendian = r.u8()
    _step = r.u32()
    pixels = r.raw_bytes()
    return ts, h, w, enc, pixels


def parse_compressed_image(data: bytes):
    """sensor_msgs/msg/CompressedImage → (ts_ns, format, compressed_bytes)."""
    r = CDRReader(data)
    ts, _ = r.header()
    fmt = r.string()
    payload = r.raw_bytes()
    return ts, fmt, payload


def parse_camera_info(data: bytes) -> dict:
    """sensor_msgs/msg/CameraInfo → dict with K, D, R, P matrices."""
    r = CDRReader(data)
    ts, frame_id = r.header()
    h = r.u32()
    w = r.u32()
    dist_model = r.string()
    d_len = r.u32()
    D = [r.f64() for _ in range(d_len)]
    K = [r.f64() for _ in range(9)]
    R = [r.f64() for _ in range(9)]
    P = [r.f64() for _ in range(12)]
    return dict(
        height=h, width=w, distortion_model=dist_model, frame_id=frame_id,
        D=np.array(D, dtype=np.float64),
        K=np.array(K, dtype=np.float64).reshape(3, 3),
        R=np.array(R, dtype=np.float64).reshape(3, 3),
        P=np.array(P, dtype=np.float64).reshape(3, 4),
    )


# ---------------------------------------------------------------------------
# Pixel decoders
# ---------------------------------------------------------------------------

def decode_color_raw(h, w, enc, pixels) -> np.ndarray:
    """Raw Image → (H, W, 3) uint8 RGB array."""
    if enc == "rgb8":
        return np.frombuffer(pixels, np.uint8).reshape(h, w, 3)
    if enc == "bgr8":
        return cv2.cvtColor(
            np.frombuffer(pixels, np.uint8).reshape(h, w, 3), cv2.COLOR_BGR2RGB
        )
    if enc == "bgra8":
        return cv2.cvtColor(
            np.frombuffer(pixels, np.uint8).reshape(h, w, 4), cv2.COLOR_BGRA2RGB
        )
    if enc == "rgba8":
        return np.frombuffer(pixels, np.uint8).reshape(h, w, 4)[:, :, :3]
    if enc == "mono8":
        g = np.frombuffer(pixels, np.uint8).reshape(h, w)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    # fallback: try 3-ch
    return np.frombuffer(pixels, np.uint8).reshape(h, w, -1)[:, :, :3]


def decode_color_compressed(fmt, payload) -> np.ndarray:
    """CompressedImage → (H, W, 3) uint8 RGB array."""
    img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imdecode failed for format={fmt}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_depth(h, w, enc, pixels) -> np.ndarray:
    """Depth Image → (H, W) uint16 or float32."""
    if enc == "16UC1":
        return np.frombuffer(pixels, np.uint16).reshape(h, w)
    if enc == "32FC1":
        return np.frombuffer(pixels, np.float32).reshape(h, w)
    raise ValueError(f"Unknown depth encoding: {enc}")


# ---------------------------------------------------------------------------
# Bag I/O
# ---------------------------------------------------------------------------

def open_bag_db(db3_path: str):
    """Open a .db3 bag with optimized SQLite settings."""
    conn = sqlite3.connect(db3_path)
    conn.execute("PRAGMA mmap_size=2147483648")   # 2 GB mmap
    conn.execute("PRAGMA cache_size=-131072")      # 128 MB page cache
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    return conn


def identify_topics(conn):
    """Return a dict mapping role → topic_id for the topics we care about."""
    cur = conn.execute("SELECT id, name, type FROM topics")
    rows = cur.fetchall()
    result = {}
    for tid, name, ttype in rows:
        if "color" in name and "camera_info" in name:
            result["color_info"] = tid
        elif "depth" in name and "camera_info" in name:
            result["depth_info"] = tid
        elif "color" in name and ttype == "sensor_msgs/msg/CompressedImage":
            result["color"] = tid
            result["color_type"] = "compressed"
        elif "color" in name and ttype == "sensor_msgs/msg/Image":
            # only set if compressed wasn't found already
            result.setdefault("color", tid)
            result.setdefault("color_type", "raw")
        elif "depth" in name and ttype == "sensor_msgs/msg/Image":
            result["depth"] = tid
    return result


def msg_count(conn, topic_id: int) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (topic_id,))
    return cur.fetchone()[0]


def first_msg(conn, topic_id: int) -> bytes:
    cur = conn.execute(
        "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1",
        (topic_id,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def stream_msgs(conn, topic_id: int):
    """Yield (data,) rows ordered by timestamp — no fetchall, low memory."""
    cur = conn.execute(
        "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (topic_id,),
    )
    while True:
        row = cur.fetchone()
        if row is None:
            break
        yield row[0]


# ---------------------------------------------------------------------------
# Per-bag processing
# ---------------------------------------------------------------------------

def process_bag(bag_dir_str: str, output_dir_str: str) -> tuple:
    """Convert one ROS2 bag directory → one HDF5 file.

    Returns (bag_name, status_string).
    """
    bag_dir = Path(bag_dir_str)
    output_dir = Path(output_dir_str)
    bag_name = bag_dir.name

    output_path = output_dir / f"{bag_name}.h5"
    if output_path.exists():
        return bag_name, "skipped"

    # Find .db3
    db3_files = list(bag_dir.glob("*.db3"))
    if not db3_files:
        return bag_name, "no_db3"
    db3_path = str(db3_files[0])

    try:
        conn = open_bag_db(db3_path)
        tmap = identify_topics(conn)

        if "color" not in tmap or "depth" not in tmap:
            conn.close()
            return bag_name, "missing_topics"

        color_tid = tmap["color"]
        depth_tid = tmap["depth"]
        color_type = tmap["color_type"]

        n_color = msg_count(conn, color_tid)
        n_depth = msg_count(conn, depth_tid)
        if n_color == 0 or n_depth == 0:
            conn.close()
            return bag_name, "empty"

        # Probe first frame for dimensions
        raw0 = first_msg(conn, color_tid)
        if color_type == "compressed":
            _, fmt0, pay0 = parse_compressed_image(raw0)
            c0 = decode_color_compressed(fmt0, pay0)
        else:
            _, h0, w0, enc0, pix0 = parse_image(raw0)
            c0 = decode_color_raw(h0, w0, enc0, pix0)
        ch, cw = c0.shape[:2]

        raw_d0 = first_msg(conn, depth_tid)
        _, dh, dw, d_enc, dpix0 = parse_image(raw_d0)
        depth_dtype = np.uint16 if d_enc == "16UC1" else np.float32

        # Camera info (single message each, optional)
        color_info = None
        depth_info = None
        if "color_info" in tmap:
            ci_data = first_msg(conn, tmap["color_info"])
            if ci_data:
                color_info = parse_camera_info(ci_data)
        if "depth_info" in tmap:
            di_data = first_msg(conn, tmap["depth_info"])
            if di_data:
                depth_info = parse_camera_info(di_data)

        # --- Write HDF5 (atomic via tmp file) ---
        tmp_path = output_path.with_suffix(".h5.tmp")

        with h5py.File(str(tmp_path), "w") as hf:
            # Pre-allocate datasets
            color_ds = hf.create_dataset(
                "color/images",
                shape=(n_color, ch, cw, 3),
                dtype=np.uint8,
                chunks=(1, ch, cw, 3),
                compression="lzf",
            )
            color_ts = hf.create_dataset("color/timestamps_ns", shape=(n_color,), dtype=np.int64)

            depth_ds = hf.create_dataset(
                "depth/images",
                shape=(n_depth, dh, dw),
                dtype=depth_dtype,
                chunks=(1, dh, dw),
                compression="lzf",
            )
            depth_ts = hf.create_dataset("depth/timestamps_ns", shape=(n_depth,), dtype=np.int64)

            # Stream color frames
            for i, data in enumerate(stream_msgs(conn, color_tid)):
                if color_type == "compressed":
                    ts, fmt, pay = parse_compressed_image(data)
                    color_ds[i] = decode_color_compressed(fmt, pay)
                else:
                    ts, h, w, enc, pix = parse_image(data)
                    color_ds[i] = decode_color_raw(h, w, enc, pix)
                color_ts[i] = ts

            # Stream depth frames
            for i, data in enumerate(stream_msgs(conn, depth_tid)):
                ts, h, w, enc, pix = parse_image(data)
                depth_ds[i] = decode_depth(h, w, enc, pix)
                depth_ts[i] = ts

            # Camera intrinsics
            for key, info in [("color/camera_info", color_info), ("depth/camera_info", depth_info)]:
                if info is None:
                    continue
                grp = hf.create_group(key)
                grp.attrs["height"] = info["height"]
                grp.attrs["width"] = info["width"]
                grp.attrs["distortion_model"] = info["distortion_model"]
                grp.attrs["frame_id"] = info["frame_id"]
                grp.create_dataset("D", data=info["D"])
                grp.create_dataset("K", data=info["K"])
                grp.create_dataset("R", data=info["R"])
                grp.create_dataset("P", data=info["P"])

            # Root metadata
            hf.attrs["bag_name"] = bag_name
            hf.attrs["source_path"] = str(bag_dir)
            hf.attrs["n_color"] = n_color
            hf.attrs["n_depth"] = n_depth
            hf.attrs["color_height"] = ch
            hf.attrs["color_width"] = cw
            hf.attrs["depth_height"] = dh
            hf.attrs["depth_width"] = dw
            hf.attrs["depth_encoding"] = d_enc
            hf.attrs["color_stored_as"] = "rgb8"
            hf.attrs["color_source_type"] = color_type

        conn.close()
        tmp_path.rename(output_path)

        sz_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"[OK]   {bag_name}  color:{n_color}  depth:{n_depth}  {sz_mb:.1f} MB")
        return bag_name, "ok"

    except Exception as e:
        tmp_path = output_path.with_suffix(".h5.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        logger.error(f"[FAIL] {bag_name} — {e}", exc_info=False)
        return bag_name, f"error: {e}"


# ---------------------------------------------------------------------------
# Discovery & CLI
# ---------------------------------------------------------------------------

def find_all_bags(root: Path) -> list:
    """Find every directory that directly contains a .db3 file."""
    bags = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(f.endswith(".db3") for f in filenames):
            bags.append(dirpath)
    bags.sort()
    return bags


def resolve_csv_path(input_dir: Path, csv_arg: str | None) -> Path | None:
    if csv_arg:
        csv_path = Path(csv_arg)
        if not csv_path.is_absolute():
            csv_path = input_dir / csv_path if (input_dir / csv_arg).exists() else Path.cwd() / csv_arg
        return csv_path

    default_name = "Body_Condition_Score_Dataset - UNL_ALL_Final.csv"
    for candidate in (input_dir / default_name, Path.cwd() / default_name):
        if candidate.exists():
            return candidate
    return None


def load_bags_from_csv(input_dir: Path, csv_path: Path) -> list:
    """Load unique RosBagPath entries from CSV, preserving row order."""
    bags = []
    seen = set()

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "RosBagPath" not in (reader.fieldnames or []):
            raise ValueError(f"'RosBagPath' column not found in {csv_path}")

        for row in reader:
            bag_value = (row.get("RosBagPath") or "").strip()
            if not bag_value:
                continue
            bag_path = Path(bag_value)
            if not bag_path.is_absolute():
                bag_path = input_dir / bag_path
            bag_path = bag_path.resolve()
            bag_key = str(bag_path)
            if bag_key in seen:
                continue
            seen.add(bag_key)
            bags.append(bag_key)

    return bags


def output_path_for_bag(output_dir: Path, bag_dir_str: str) -> Path:
    bag_dir = Path(bag_dir_str)
    return output_dir / f"{bag_dir.name}.h5"


def main():
    ap = argparse.ArgumentParser(description="ROS2 bags → HDF5")
    ap.add_argument(
        "--input_dir",
        default=".",
        help="Root dir with rosbags_* folders (default: current directory)",
    )
    ap.add_argument(
        "--output_dir",
        default="./hdf5_out",
        help="Output dir for .h5 files (default: ./hdf5_out)",
    )
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers (default 8)")
    ap.add_argument("--filter", type=str, default=None, help="Only process bags whose path contains this string")
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV file with a RosBagPath column. Defaults to Body_Condition_Score_Dataset - UNL_ALL_Final.csv if present.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="List the bags that would be processed and exit without converting.",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_csv_path(input_dir, args.csv)
    if csv_path is not None:
        bags = load_bags_from_csv(input_dir, csv_path)
        logger.info(f"Loaded {len(bags)} bag paths from CSV → {csv_path}")
    else:
        bags = find_all_bags(input_dir)
        logger.info(f"Loaded {len(bags)} bag paths by walking input_dir → {input_dir}")

    if args.filter:
        bags = [b for b in bags if args.filter in b]

    logger.info(f"Processing {len(bags)} bags, using {args.workers} workers, output → {output_dir}")

    if args.dry_run:
        convert_count = 0
        skip_count = 0
        for bag in bags:
            output_path = output_path_for_bag(output_dir, bag)
            status = "skip_exists" if output_path.exists() else "convert"
            if status == "convert":
                convert_count += 1
            else:
                skip_count += 1
            print(f"{status}\t{bag}\t{output_path}")
        logger.info(f"Dry run complete — convert:{convert_count}  skip_exists:{skip_count}")
        return

    t0 = time.time()
    stats = {"ok": 0, "skipped": 0, "error": 0}

    if args.workers <= 1:
        for b in tqdm(bags, total=len(bags), desc="Bags", unit="bag"):
            _, status = process_bag(b, str(output_dir))
            bucket = "ok" if status == "ok" else ("skipped" if status == "skipped" else "error")
            stats[bucket] += 1
    else:
        progress = tqdm(total=len(bags), desc="Bags", unit="bag")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(process_bag, b, str(output_dir)): b for b in bags}
            for fut in as_completed(futs):
                try:
                    _, status = fut.result()
                except Exception as e:
                    status = f"error: {e}"
                bucket = "ok" if status == "ok" else ("skipped" if status == "skipped" else "error")
                stats[bucket] += 1
                progress.update(1)
        progress.close()

    elapsed = time.time() - t0
    logger.info(
        f"Finished in {elapsed:.1f}s  —  ok:{stats['ok']}  skipped:{stats['skipped']}  errors:{stats['error']}"
    )


if __name__ == "__main__":
    main()
