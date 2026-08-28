#!/usr/bin/env python3
"""Collect one resumable clean trajectory for every official RoboTwin task."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_ENVS = {"__init__", "_base_task", "_GLOBAL_CONFIGS"}
REQUIRED_FLOW_CAMERA_FIELDS = {
    "depth_mm",
    "visual_segmentation_ids",
    "entity_segmentation_ids",
    "intrinsic_cv",
    "world_to_camera_cv",
    "camera_to_world_cv",
    "camera_to_world_gl",
}


def official_tasks() -> list[str]:
    tasks = sorted(
        path.stem
        for path in (ROOT / "envs").glob("*.py")
        if path.stem not in EXCLUDED_ENVS
    )
    if len(tasks) != 50:
        raise RuntimeError(f"Expected 50 official tasks, found {len(tasks)}")
    return tasks


def validate_episode(path: Path) -> None:
    with h5py.File(path, "r") as file:
        if set(file["vision"]) != {"cam_head", "cam_left_wrist", "cam_right_wrist"}:
            raise ValueError("expected head and two wrist vision cameras")
        if set(file["flow/cameras"]) != {"cam_head"}:
            raise ValueError("flow must contain only cam_head")
        if "depths" in file["vision/cam_head"]:
            raise ValueError("head depth is duplicated in /vision")
        for camera in ("cam_left_wrist", "cam_right_wrist"):
            group = file[f"vision/{camera}"]
            if "colors" not in group or "depths" not in group:
                raise ValueError(f"{camera} is missing RGB or depth")
            if group["depths"].compression != "lzf" or not group["depths"].shuffle:
                raise ValueError(f"{camera} depth is not LZF+shuffle compressed")
        head = file["flow/cameras/cam_head"]
        missing = REQUIRED_FLOW_CAMERA_FIELDS - set(head)
        if missing:
            raise ValueError(f"head flow fields missing: {sorted(missing)}")
        frame_count = len(file["flow/frame_indices"])
        if head["depth_mm"].shape[0] != frame_count:
            raise ValueError("head depth and flow frame counts differ")
        if file["flow/scene_entities/poses_world"].shape[0] != frame_count:
            raise ValueError("entity poses and flow frame counts differ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="clean_50_rgbd_flow")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds allowed per task")
    return parser.parse_args()


def write_summary(path: Path, records: list[dict[str, object]]) -> None:
    payload = {
        "task_count": len(records),
        "completed": sum(record["status"] in {"collected", "existing"} for record in records),
        "failed": sum(record["status"] not in {"collected", "existing"} for record in records),
        "records": records,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    tasks = official_tasks()
    batch_root = ROOT / "data" / args.config
    log_root = batch_root / "_batch_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    summary_path = batch_root / "collection_summary.json"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["PYTHONWARNINGS"] = "ignore::UserWarning"
    records: list[dict[str, object]] = []

    for index, task in enumerate(tasks, start=1):
        episode = batch_root / task / "aloha_agilex" / "data" / "episode_0000000.hdf5"
        started = time.monotonic()
        if episode.is_file():
            try:
                validate_episode(episode)
                status, detail = "existing", "validated existing episode"
            except Exception as error:
                status, detail = "invalid_existing", str(error)
            elapsed = time.monotonic() - started
        else:
            print(f"[{index:02d}/50] START {task}", flush=True)
            log_path = log_root / f"{task}.log"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "collect_data.py"),
                task,
                args.config,
            ]
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                        check=False,
                    )
                if result.returncode != 0:
                    status, detail = "process_failed", f"exit code {result.returncode}"
                elif not episode.is_file():
                    status, detail = "missing_output", "collector exited without HDF5"
                else:
                    validate_episode(episode)
                    status, detail = "collected", "collected and validated"
            except subprocess.TimeoutExpired:
                status, detail = "timeout", f"exceeded {args.timeout} seconds"
            except Exception as error:
                status, detail = "validation_failed", str(error)
            elapsed = time.monotonic() - started

        record = {
            "task": task,
            "status": status,
            "detail": detail,
            "elapsed_seconds": round(elapsed, 2),
            "episode": str(episode.relative_to(ROOT)) if episode.exists() else "",
            "bytes": episode.stat().st_size if episode.exists() else 0,
        }
        records.append(record)
        write_summary(summary_path, records)
        print(
            f"[{index:02d}/50] {status.upper():18s} {task} "
            f"({elapsed:.1f}s) {detail}",
            flush=True,
        )

    failed = [record for record in records if record["status"] not in {"collected", "existing"}]
    print(f"Finished: {len(records) - len(failed)}/50 valid, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
