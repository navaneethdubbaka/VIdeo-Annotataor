"""
VLA Batch Dataset Pipeline
===========================
Walks a dataset directory (containing subdirectory parts with .mp4 + .json pairs),
processes each clip through vla_annotator, and merges all JSONL records into a
single dataset manifest.

Usage:
    python vla_pipeline.py \\
        --dataset_root factory001_worker001_part01/ \\
        --macro_task "Factory assembly task" \\
        --steps "Pick component;Align part;Fasten bolt;Inspect result" \\
        --environment "Factory" --scene "Assembly line" \\
        --op_height 170 --robot_height 120 \\
        --skip 3 --no_video

Output:
    vla_dataset/
        <clip_id>.jsonl          per-clip JSONL
        <clip_id>.csv            per-clip 6-DoF CSV
        dataset_manifest.json    full index with metadata
        all_frames.jsonl         merged JSONL for training
"""

import argparse, json, os, glob
from pathlib import Path

from vla_annotator import process_video

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET WALKER
# ═══════════════════════════════════════════════════════════════════════════════

def find_clips(dataset_root: str):
    """
    Recursively find all (.mp4, .json) pairs under dataset_root.
    Returns list of dicts with keys: video_path, meta_path, clip_id.
    """
    clips = []
    root  = Path(dataset_root)
    for mp4 in sorted(root.rglob("*.mp4")):
        json_p = mp4.with_suffix(".json")
        if json_p.exists():
            clip_id = mp4.stem
            clips.append({
                "video_path": str(mp4),
                "meta_path":  str(json_p),
                "clip_id":    clip_id,
            })
    return clips


def load_clip_meta(meta_path: str) -> dict:
    with open(meta_path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEIGHT PARSER  —  "162cm" / "162" / 162 → float
# ═══════════════════════════════════════════════════════════════════════════════

def parse_height(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return float(str(v).lower().replace("cm", "").strip())


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-CLIP PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_clip(clip: dict, out_dir: Path, args) -> dict:
    clip_id   = clip["clip_id"]
    meta      = load_clip_meta(clip["meta_path"])

    csv_path   = str(out_dir / f"{clip_id}.csv")
    jsonl_path = str(out_dir / f"{clip_id}.jsonl")
    video_path = str(out_dir / f"{clip_id}_vis.mp4")

    # Derive per-clip environment / scene from factory/worker IDs in metadata
    factory_id = meta.get("factory_id", args.environment)
    scene_tag  = meta.get("worker_id",  args.scene)
    duration   = meta.get("duration_sec", 0.0)

    steps_list = [s.strip() for s in args.steps.split(";") if s.strip()] if args.steps else []

    stats = process_video(
        input_path   = clip["video_path"],
        csv_path     = csv_path,
        jsonl_path   = jsonl_path,
        out_path     = video_path,
        out_w        = args.width,
        out_h        = args.height,
        skip         = args.skip,
        max_hands    = args.hands,
        conf         = args.conf,
        macro_task   = args.macro_task,
        steps        = steps_list,
        nl_caption   = args.nl_caption,
        ts_start     = 0.0,
        ts_end       = duration,
        environment  = factory_id,
        scene        = scene_tag,
        op_height    = parse_height(args.op_height),
        robot_height = parse_height(args.robot_height),
        write_video  = not args.no_video,
    )

    return {
        "clip_id":       clip_id,
        "video_path":    clip["video_path"],
        "meta":          meta,
        "out_csv":       csv_path,
        "out_jsonl":     jsonl_path,
        "out_video":     video_path if not args.no_video else None,
        "frame_records": stats["frames"],
        "csv_rows":      stats["rows"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MERGE ALL JSONL → all_frames.jsonl
# ═══════════════════════════════════════════════════════════════════════════════

def merge_jsonl(manifest_entries: list, out_path: str):
    total = 0
    with open(out_path, "w") as out_f:
        for entry in manifest_entries:
            jl = entry.get("out_jsonl")
            if jl and os.path.exists(jl):
                with open(jl) as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:
                            out_f.write(line + "\n")
                            total += 1
    return total


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="VLA Batch Dataset Pipeline")
    ap.add_argument("--dataset_root",  required=True,  help="Root folder containing clip sub-directories")
    ap.add_argument("--out_dir",       default="vla_dataset", help="Output directory")
    ap.add_argument("--macro_task",    default="",     help='Long-horizon goal label')
    ap.add_argument("--steps",         default="",     help='Semicolon-separated micro-step labels')
    ap.add_argument("--nl_caption",    default="",     help='NL caption for training')
    ap.add_argument("--environment",   default="Factory")
    ap.add_argument("--scene",         default="Workstation")
    ap.add_argument("--op_height",     default="170",  help="Operator height in cm")
    ap.add_argument("--robot_height",  default="0",    help="Target robot height in cm (0=no normalisation)")
    ap.add_argument("--width",         type=int, default=1600)
    ap.add_argument("--height",        type=int, default=720)
    ap.add_argument("--skip",          type=int, default=3,    help="Process every Nth frame")
    ap.add_argument("--hands",         type=int, default=2)
    ap.add_argument("--conf",          type=float, default=0.55)
    ap.add_argument("--no_video",      action="store_true",    help="Skip video rendering")
    ap.add_argument("--max_clips",     type=int, default=0,    help="Limit number of clips (0=all)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = find_clips(args.dataset_root)
    if not clips:
        print(f"No .mp4+.json pairs found under: {args.dataset_root}")
        return

    if args.max_clips > 0:
        clips = clips[:args.max_clips]

    print(f"Found {len(clips)} clips under {args.dataset_root}")
    print(f"Output → {out_dir}/\n")

    manifest = []
    for i, clip in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {clip['clip_id']}")
        try:
            entry = process_clip(clip, out_dir, args)
            manifest.append(entry)
            print(f"       {entry['frame_records']} frames, {entry['csv_rows']} CSV rows")
        except Exception as e:
            print(f"       ERROR: {e}")
            manifest.append({"clip_id": clip["clip_id"], "error": str(e)})

    # Manifest JSON
    manifest_path = str(out_dir / "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "dataset_root":  args.dataset_root,
            "total_clips":   len(clips),
            "macro_task":    args.macro_task,
            "steps":         args.steps,
            "op_height_cm":  args.op_height,
            "robot_height_cm": args.robot_height,
            "clips":         manifest,
        }, f, indent=2)

    # Merged JSONL
    merged_path = str(out_dir / "all_frames.jsonl")
    total_frames = merge_jsonl(manifest, merged_path)

    print(f"\n{'='*60}")
    print(f"Processed  : {len([e for e in manifest if 'error' not in e])}/{len(clips)} clips")
    print(f"Manifest   : {manifest_path}")
    print(f"All frames : {merged_path}  ({total_frames:,} records)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
