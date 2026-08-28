"""HDF5 inspection, rigid scene-flow calculation, and image rendering."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

import cv2
import h5py
import numpy as np


FLOW_ROOT = "flow"
FLOW_FRAME_CACHE_SIZE = 64
_FLOW_CACHE_LOCK = RLock()


class FlowDataError(RuntimeError):
    """Raised when a file does not contain the flow collection schema."""


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _strings(dataset: h5py.Dataset | None, count: int) -> list[str]:
    if dataset is None:
        return [""] * count
    values = np.asarray(dataset).reshape(-1)
    result = [_decode(value) for value in values]
    return (result + [""] * count)[:count]


def _frame_value(dataset: h5py.Dataset, frame: int, trailing_shape: tuple[int, ...]) -> np.ndarray:
    """Read either a per-frame value or one static value."""
    if tuple(dataset.shape) == trailing_shape:
        return np.asarray(dataset, dtype=np.float64)
    if dataset.ndim == len(trailing_shape) + 1 and tuple(dataset.shape[1:]) == trailing_shape:
        index = min(frame, dataset.shape[0] - 1)
        return np.asarray(dataset[index], dtype=np.float64)
    raise FlowDataError(
        f"Unexpected shape for {dataset.name}: {dataset.shape}; expected {trailing_shape} "
        f"or (T, {', '.join(map(str, trailing_shape))})"
    )


def _matrix4(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        result = np.eye(4, dtype=matrix.dtype)
        result[:3] = matrix
        return result
    raise FlowDataError(f"Expected a 3x4 or 4x4 transform, got {matrix.shape}")


def _require(group: h5py.Group, key: str) -> h5py.Dataset:
    if key not in group:
        raise FlowDataError(f"Missing dataset: {group.name}/{key}")
    item = group[key]
    if not isinstance(item, h5py.Dataset):
        raise FlowDataError(f"Expected dataset: {item.name}")
    return item


def inspect_hdf5(path: str | Path) -> dict[str, Any]:
    path = str(Path(path).resolve())
    with h5py.File(path, "r") as file:
        if FLOW_ROOT not in file or "cameras" not in file[FLOW_ROOT]:
            raise FlowDataError("The file has no /flow/cameras group")
        flow = file[FLOW_ROOT]
        cameras = []
        pose_count = 0
        if "scene_entities" in flow and "poses_world" in flow["scene_entities"]:
            pose_count = int(flow["scene_entities/poses_world"].shape[0])

        for name, camera in flow["cameras"].items():
            if not isinstance(camera, h5py.Group):
                continue
            missing = [key for key in ("depth_mm", "entity_segmentation_ids", "intrinsic_cv") if key not in camera]
            world_to_camera = "world_to_camera_cv" in camera or "camera_to_world_cv" in camera
            if not world_to_camera:
                missing.append("world_to_camera_cv (or camera_to_world_cv)")
            if missing:
                cameras.append({"name": name, "ready": False, "missing": missing})
                continue
            depth = camera["depth_mm"]
            segmentation = camera["entity_segmentation_ids"]
            frame_count = min(int(depth.shape[0]), int(segmentation.shape[0]))
            if pose_count:
                frame_count = min(frame_count, pose_count)
            cameras.append(
                {
                    "name": name,
                    "ready": frame_count >= 2 and pose_count > 0,
                    "frame_count": frame_count,
                    "pair_count": max(0, frame_count - 1),
                    "height": int(depth.shape[-2]),
                    "width": int(depth.shape[-1]),
                    "has_rgb": f"vision/{name}/colors" in file,
                    "missing": [] if pose_count else ["/flow/scene_entities/poses_world"],
                }
            )

        entities: list[dict[str, Any]] = []
        if "scene_entities" in flow:
            group = flow["scene_entities"]
            ids = np.asarray(group.get("entity_ids", []), dtype=np.uint32).reshape(-1)
            count = len(ids)
            names = _strings(group.get("entity_names"), count)
            instances = _strings(group.get("instance_names"), count)
            links = _strings(group.get("link_names"), count)
            roles = _strings(group.get("task_roles"), count)
            body_types = _strings(group.get("body_types"), count)
            entities = [
                {
                    "id": int(ids[index]),
                    "name": names[index],
                    "instance": instances[index],
                    "link": links[index],
                    "role": roles[index],
                    "body_type": body_types[index],
                }
                for index in range(count)
            ]

        return {
            "path": path,
            "name": Path(path).name,
            "cameras": cameras,
            "entities": entities,
            "depth_unit": _decode(flow.attrs.get("depth_unit", "millimeter")),
            "pose_convention": _decode(flow.attrs.get("pose_convention", "4x4 local-to-world")),
        }


@dataclass(frozen=True)
class FlowFrame:
    flow: np.ndarray
    depth_m: np.ndarray
    segmentation: np.ndarray
    tracked: np.ndarray
    projected: np.ndarray
    geometric_valid: np.ndarray
    occluded: np.ndarray
    visible_valid: np.ndarray
    target_depth_m: np.ndarray
    entity_ids: np.ndarray
    entity_names: tuple[str, ...]
    rgb_bgr: np.ndarray | None


def _world_to_camera(camera: h5py.Group, frame: int) -> np.ndarray:
    def read_transform(dataset: h5py.Dataset) -> np.ndarray:
        trailing = tuple(dataset.shape[-2:])
        if trailing not in {(3, 4), (4, 4)}:
            raise FlowDataError(f"Unexpected transform shape for {dataset.name}: {dataset.shape}")
        return _matrix4(_frame_value(dataset, frame, trailing))

    if "world_to_camera_cv" in camera:
        return read_transform(camera["world_to_camera_cv"])
    camera_to_world = read_transform(camera["camera_to_world_cv"])
    return np.linalg.inv(camera_to_world)


def _decode_rgb(file: h5py.File, camera_name: str, frame: int) -> np.ndarray | None:
    path = f"vision/{camera_name}/colors"
    if path not in file:
        return None
    dataset = file[path]
    if dataset.shape[0] == 0:
        return None
    encoded = dataset[min(frame, dataset.shape[0] - 1)]
    if isinstance(encoded, (bytes, np.bytes_)):
        payload = np.frombuffer(bytes(encoded), dtype=np.uint8)
    else:
        payload = np.asarray(encoded, dtype=np.uint8).reshape(-1)
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    return image


@lru_cache(maxsize=FLOW_FRAME_CACHE_SIZE)
def compute_scene_flow(
    path: str,
    modified_ns: int,
    camera_name: str,
    frame: int,
    occlusion_threshold_m: float = 0.02,
) -> FlowFrame:
    """Calculate pixel flow from t to t+1 using depth, entity IDs, and rigid poses.

    Depth and camera matrices use OpenCV camera coordinates. Entity pose matrices are
    local-to-world. Background ID 0 is treated as fixed in world coordinates, so this
    also handles moving wrist cameras.
    """
    del modified_ns  # It is part of the cache key and intentionally otherwise unused.
    with h5py.File(path, "r") as file:
        camera_path = f"flow/cameras/{camera_name}"
        if camera_path not in file:
            raise FlowDataError(f"Unknown camera: {camera_name}")
        camera = file[camera_path]
        depth_dataset = _require(camera, "depth_mm")
        segmentation_dataset = _require(camera, "entity_segmentation_ids")
        entities = file.get("flow/scene_entities")
        if entities is None:
            raise FlowDataError("Missing group: /flow/scene_entities")
        poses_dataset = _require(entities, "poses_world")
        entity_ids = np.asarray(_require(entities, "entity_ids"), dtype=np.uint32).reshape(-1)

        frame_count = min(depth_dataset.shape[0], segmentation_dataset.shape[0], poses_dataset.shape[0])
        if frame < 0 or frame + 1 >= frame_count:
            raise FlowDataError(f"Frame {frame} has no next frame (available: 0..{frame_count - 2})")

        depth_m = np.asarray(depth_dataset[frame], dtype=np.float64) / 1000.0
        next_depth_m = np.asarray(depth_dataset[frame + 1], dtype=np.float64) / 1000.0
        segmentation = np.asarray(segmentation_dataset[frame], dtype=np.uint32)
        if depth_m.shape != segmentation.shape:
            raise FlowDataError(f"Depth shape {depth_m.shape} != segmentation shape {segmentation.shape}")

        height, width = depth_m.shape
        intrinsic0 = _frame_value(_require(camera, "intrinsic_cv"), frame, (3, 3))
        intrinsic1 = _frame_value(_require(camera, "intrinsic_cv"), frame + 1, (3, 3))
        world_to_camera0 = _world_to_camera(camera, frame)
        world_to_camera1 = _world_to_camera(camera, frame + 1)
        camera_to_world0 = np.linalg.inv(world_to_camera0)

        rows, columns = np.indices((height, width), dtype=np.float64)
        z = depth_m
        x = (columns - intrinsic0[0, 2]) * z / intrinsic0[0, 0]
        y = (rows - intrinsic0[1, 2]) * z / intrinsic0[1, 1]
        camera_points = np.stack((x, y, z, np.ones_like(z)), axis=-1)
        world_points = camera_points @ camera_to_world0.T
        future_world_points = world_points.copy()

        poses0 = np.asarray(poses_dataset[frame], dtype=np.float64)
        poses1 = np.asarray(poses_dataset[frame + 1], dtype=np.float64)
        if poses0.shape[0] != len(entity_ids):
            raise FlowDataError("entity_ids and poses_world entity dimensions differ")

        tracked = segmentation == 0
        id_to_index = {int(entity_id): index for index, entity_id in enumerate(entity_ids)}
        for visible_id in np.unique(segmentation):
            entity_id = int(visible_id)
            if entity_id == 0 or entity_id not in id_to_index:
                continue
            mask = segmentation == entity_id
            entity_index = id_to_index[entity_id]
            local_to_future = poses1[entity_index] @ np.linalg.inv(poses0[entity_index])
            future_world_points[mask] = world_points[mask] @ local_to_future.T
            tracked[mask] = True

        future_camera_points = future_world_points @ world_to_camera1.T
        future_z = future_camera_points[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            projected_x = intrinsic1[0, 0] * future_camera_points[..., 0] / future_z + intrinsic1[0, 2]
            projected_y = intrinsic1[1, 1] * future_camera_points[..., 1] / future_z + intrinsic1[1, 2]
        projected = np.stack((projected_x, projected_y), axis=-1)
        flow = projected - np.stack((columns, rows), axis=-1)

        geometric_valid = (
            np.isfinite(flow).all(axis=-1)
            & np.isfinite(future_z)
            & (z > 0)
            & (future_z > 0)
            & tracked
            & (projected_x >= 0)
            & (projected_x <= width - 1)
            & (projected_y >= 0)
            & (projected_y <= height - 1)
        )
        safe_projected_x = np.nan_to_num(projected_x, nan=0.0, posinf=width - 1, neginf=0.0)
        safe_projected_y = np.nan_to_num(projected_y, nan=0.0, posinf=height - 1, neginf=0.0)
        sample_x = np.clip(np.rint(safe_projected_x), 0, width - 1).astype(np.int64)
        sample_y = np.clip(np.rint(safe_projected_y), 0, height - 1).astype(np.int64)
        target_depth = next_depth_m[sample_y, sample_x]
        occluded = geometric_valid & (target_depth > 0) & (
            future_z > target_depth + float(occlusion_threshold_m)
        )
        visible_valid = geometric_valid & ~occluded
        flow = flow.astype(np.float32)
        flow[~geometric_valid] = np.nan

        names = tuple(_strings(entities.get("entity_names"), len(entity_ids)))
        rgb = _decode_rgb(file, camera_name, frame)
        return FlowFrame(
            flow=flow,
            depth_m=depth_m.astype(np.float32),
            segmentation=segmentation,
            tracked=tracked,
            projected=projected.astype(np.float32),
            geometric_valid=geometric_valid,
            occluded=occluded,
            visible_valid=visible_valid,
            target_depth_m=target_depth.astype(np.float32),
            entity_ids=entity_ids,
            entity_names=names,
            rgb_bgr=rgb,
        )


def get_flow_frame(path: str | Path, camera_name: str, frame: int, threshold_m: float) -> FlowFrame:
    resolved = Path(path).resolve()
    cache_key = (
        str(resolved), resolved.stat().st_mtime_ns, camera_name,
        int(frame), round(float(threshold_m), 5),
    )
    # functools.lru_cache is thread-safe for its dictionary, but concurrent misses
    # may still execute the wrapped function more than once. A page requests four
    # views plus statistics at once, so single-flight this expensive HDF5 read.
    with _FLOW_CACHE_LOCK:
        return compute_scene_flow(*cache_key)


def flow_cache_info() -> dict[str, int | None]:
    info = compute_scene_flow.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize,
    }


def _valid_mask(data: FlowFrame, hide_occluded: bool) -> np.ndarray:
    return data.visible_valid if hide_occluded else data.geometric_valid


def _flow_scale(data: FlowFrame, mask: np.ndarray, requested: float) -> float:
    if requested > 0:
        return requested
    magnitude = np.linalg.norm(data.flow, axis=-1)
    values = magnitude[mask & np.isfinite(magnitude)]
    return max(1.0, float(np.percentile(values, 95))) if values.size else 1.0


def flow_color(data: FlowFrame, mask: np.ndarray, maximum: float) -> np.ndarray:
    dx = np.nan_to_num(data.flow[..., 0])
    dy = np.nan_to_num(data.flow[..., 1])
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    hsv = np.zeros((*dx.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle / 2.0, 180).astype(np.uint8)
    hsv[..., 1] = 235
    hsv[..., 2] = np.clip(magnitude / max(maximum, 1e-6) * 255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[~mask] = (18, 21, 26)
    return bgr


def _base_rgb(data: FlowFrame) -> np.ndarray:
    height, width = data.depth_m.shape
    if data.rgb_bgr is None:
        return np.full((height, width, 3), (32, 35, 40), dtype=np.uint8)
    if data.rgb_bgr.shape[:2] != (height, width):
        return cv2.resize(data.rgb_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return data.rgb_bgr.copy()


def _segmentation_color(segmentation: np.ndarray) -> np.ndarray:
    values = segmentation.astype(np.uint64)
    result = np.empty((*segmentation.shape, 3), dtype=np.uint8)
    result[..., 0] = ((values * 29 + 47) % 211 + 35).astype(np.uint8)
    result[..., 1] = ((values * 71 + 19) % 211 + 35).astype(np.uint8)
    result[..., 2] = ((values * 113 + 7) % 211 + 35).astype(np.uint8)
    result[segmentation == 0] = (25, 28, 32)
    return result


def render_view(
    data: FlowFrame,
    view: str,
    *,
    maximum: float = 0.0,
    opacity: float = 0.72,
    arrow_step: int = 20,
    hide_occluded: bool = True,
) -> np.ndarray:
    mask = _valid_mask(data, hide_occluded)

    if view == "rgb":
        image = _base_rgb(data)
    elif view == "flow":
        maximum = _flow_scale(data, mask, maximum)
        color = flow_color(data, mask, maximum)
        image = color
    elif view == "overlay":
        maximum = _flow_scale(data, mask, maximum)
        color = flow_color(data, mask, maximum)
        base = _base_rgb(data)
        blended = cv2.addWeighted(base, 1.0 - opacity, color, opacity, 0)
        image = base
        image[mask] = blended[mask]
    elif view == "depth":
        valid_depth = data.depth_m > 0
        limit = float(np.percentile(data.depth_m[valid_depth], 98)) if valid_depth.any() else 1.0
        normalized = np.clip(data.depth_m / max(limit, 1e-6) * 255, 0, 255).astype(np.uint8)
        image = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        image[~valid_depth] = (18, 21, 26)
    elif view == "segmentation":
        image = _segmentation_color(data.segmentation)
    elif view == "magnitude":
        maximum = _flow_scale(data, mask, maximum)
        magnitude = np.linalg.norm(data.flow, axis=-1)
        normalized = np.clip(np.nan_to_num(magnitude) / maximum * 255, 0, 255).astype(np.uint8)
        image = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
        image[~mask] = (18, 21, 26)
    elif view == "validity":
        image = np.full((*mask.shape, 3), (34, 38, 43), dtype=np.uint8)
        image[data.geometric_valid] = (74, 180, 72)
        image[data.occluded] = (34, 120, 235)
        image[~data.tracked & (data.depth_m > 0)] = (191, 72, 214)
    else:
        raise FlowDataError(f"Unknown view: {view}")

    if view in {"overlay", "rgb"} and arrow_step > 0:
        height, width = mask.shape
        offset = arrow_step // 2
        for row in range(offset, height, arrow_step):
            for column in range(offset, width, arrow_step):
                if not mask[row, column]:
                    continue
                dx, dy = data.flow[row, column]
                if not np.isfinite(dx + dy) or np.hypot(dx, dy) < 0.35:
                    continue
                end = (int(round(column + dx)), int(round(row + dy)))
                cv2.arrowedLine(image, (column, row), end, (245, 247, 250), 1, cv2.LINE_AA, tipLength=0.28)
    return image


def frame_summary(data: FlowFrame, hide_occluded: bool = True) -> dict[str, Any]:
    mask = _valid_mask(data, hide_occluded)
    magnitude = np.linalg.norm(data.flow, axis=-1)
    values = magnitude[mask & np.isfinite(magnitude)]
    counts: dict[int, int] = {}
    ids, pixel_counts = np.unique(data.segmentation, return_counts=True)
    counts.update({int(key): int(value) for key, value in zip(ids, pixel_counts)})
    name_by_id = {int(key): value for key, value in zip(data.entity_ids, data.entity_names)}
    visible_entities = [
        {"id": entity_id, "name": name_by_id.get(entity_id, "untracked"), "pixels": pixels}
        for entity_id, pixels in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        if entity_id != 0
    ]
    total = data.depth_m.size
    return {
        "mean_magnitude": float(values.mean()) if values.size else 0.0,
        "p95_magnitude": float(np.percentile(values, 95)) if values.size else 0.0,
        "max_magnitude": float(values.max()) if values.size else 0.0,
        "valid_pixels": int(mask.sum()),
        "valid_ratio": float(mask.sum() / total),
        "occluded_pixels": int(data.occluded.sum()),
        "untracked_pixels": int((~data.tracked & (data.depth_m > 0)).sum()),
        "visible_entities": visible_entities,
    }
