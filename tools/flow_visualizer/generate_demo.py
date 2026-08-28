#!/usr/bin/env python3
"""Generate a small, geometrically consistent flow demo HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import cv2
    import h5py
    import numpy as np
except ImportError as error:
    raise SystemExit(
        f"Missing {getattr(error, 'name', error)}. Install tools/flow_visualizer/requirements.txt first."
    ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "demo" / "rigid_flow_demo.hdf5",
    )
    parser.add_argument("--frames", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    height, width = 240, 320
    fx = fy = 260.0
    cx, cy = width / 2, height / 2
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    entity_ids = np.array([11, 27], dtype=np.uint32)

    depths = np.full((args.frames, height, width), 3000.0, dtype=np.float32)
    segmentations = np.zeros((args.frames, height, width), dtype=np.uint32)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, None], args.frames * 2, axis=0).reshape(args.frames, 2, 4, 4)
    encoded_frames: list[np.ndarray] = []

    yy, xx = np.indices((height, width))
    for frame in range(args.frames):
        phase = frame / max(1, args.frames - 1) * 2 * np.pi
        object_x = 0.34 * np.sin(phase)
        object_y = -0.12 + 0.08 * np.cos(phase * 1.3)
        robot_x = -0.38 + 0.26 * frame / max(1, args.frames - 1)
        robot_y = 0.26 + 0.05 * np.sin(phase * 1.7)
        poses[frame, 0, :3, 3] = (object_x, object_y, 0)
        poses[frame, 1, :3, 3] = (robot_x, robot_y, 0)

        object_z = 1.65
        object_u = cx + fx * object_x / object_z
        object_v = cy + fy * object_y / object_z
        object_mask = ((xx - object_u) / 37) ** 2 + ((yy - object_v) / 30) ** 2 <= 1

        robot_z = 2.1
        robot_u = cx + fx * robot_x / robot_z
        robot_v = cy + fy * robot_y / robot_z
        robot_mask = (np.abs(xx - robot_u) <= 26) & (np.abs(yy - robot_v) <= 45)

        # Rasterize farther geometry first and the near object last.
        segmentations[frame, robot_mask] = 27
        depths[frame, robot_mask] = robot_z * 1000
        segmentations[frame, object_mask] = 11
        depths[frame, object_mask] = object_z * 1000

        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:] = (25, 23, 20)
        image += np.linspace(0, 30, width, dtype=np.uint8)[None, :, None]
        image[robot_mask] = (190, 135, 45)
        image[object_mask] = (45, 95, 225)
        cv2.ellipse(image, (int(object_u), int(object_v)), (37, 30), 0, 0, 360, (55, 115, 245), 2)
        cv2.rectangle(
            image,
            (int(robot_u - 26), int(robot_v - 45)),
            (int(robot_u + 26), int(robot_v + 45)),
            (220, 165, 65),
            2,
        )
        cv2.putText(image, f"t={frame:02d}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (185, 190, 195), 1, cv2.LINE_AA)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise RuntimeError("OpenCV failed to encode demo RGB")
        encoded_frames.append(encoded.reshape(-1))

    with h5py.File(args.output, "w") as file:
        file.attrs["source_format"] = "RoboTwin flow visualizer demo"
        flow = file.create_group("flow")
        flow.attrs["depth_unit"] = "millimeter"
        flow.attrs["background_entity_id"] = np.uint32(0)
        flow.attrs["entity_id_semantics"] = "SAPIEN per_scene_id"
        flow.attrs["pose_convention"] = "4x4 local-to-world"
        flow.create_dataset("frame_indices", data=np.arange(args.frames, dtype=np.int32))

        camera = flow.create_group("cameras/cam_head")
        camera.create_dataset("depth_mm", data=depths, compression="lzf", shuffle=True)
        camera.create_dataset("entity_segmentation_ids", data=segmentations, compression="lzf", shuffle=True)
        camera.create_dataset("intrinsic_cv", data=np.repeat(intrinsic[None], args.frames, axis=0))
        camera.create_dataset(
            "world_to_camera_cv",
            data=np.repeat(np.eye(4, dtype=np.float32)[None], args.frames, axis=0),
        )

        entities = flow.create_group("scene_entities")
        entities.create_dataset("entity_ids", data=entity_ids)
        string_type = h5py.string_dtype("utf-8")
        entities.create_dataset("entity_names", data=["demo_object", "demo_robot_link"], dtype=string_type)
        entities.create_dataset("instance_names", data=["object#11", "robot#27"], dtype=string_type)
        entities.create_dataset("link_names", data=["", "arm_link"], dtype=string_type)
        entities.create_dataset("task_roles", data=["object", "robot"], dtype=string_type)
        entities.create_dataset("body_types", data=["rigid_actor", "articulation_link"], dtype=string_type)
        entities.create_dataset("poses_world", data=poses)

        vision = file.create_group("vision/cam_head")
        variable_bytes = h5py.vlen_dtype(np.dtype("uint8"))
        colors = vision.create_dataset("colors", shape=(args.frames,), dtype=variable_bytes)
        for index, encoded in enumerate(encoded_frames):
            colors[index] = encoded
        vision.create_dataset("shape", data=np.array([height, width, 3], dtype=np.int32))

    print(f"Wrote {args.output} ({args.frames} frames, {width}x{height})")


if __name__ == "__main__":
    main()
