import h5py, pickle
import json
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video


CAMERA_MAP = {
    "head_camera": "cam_head",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
    "front_camera": "cam_third_view",
}


def images_encoding(imgs):
    encode_data = []
    max_len = 0
    for rgb_image in imgs:
        if rgb_image.ndim != 3 or rgb_image.shape[-1] != 3:
            raise ValueError(
                f"Expected an RGB image with shape (H, W, 3), got {rgb_image.shape}"
            )

        # Keep source channel values unchanged for XPolicyLab's OpenCV decoder.
        # OpenCV intentionally interprets this RGB array as BGR while encoding.
        success, encoded_image = cv2.imencode(".jpg", rgb_image)
        if not success:
            raise ValueError("OpenCV failed to encode an RGB frame as JPEG")
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    return encode_data, max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # 如果是叶子节点，直接追加数据
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # 如果是嵌套字典，递归处理
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            value = np.array(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                hdf5_group.create_dataset(key, data=value)
        else:
            return
            try:
                hdf5_group.create_dataset(key, data=str(value))
                print("Not np array")
            except Exception as e:
                print(f"Error storing value for key '{key}': {e}")


def _ensure_2d(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    return values


def _to_4x4(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2 and values.shape == (3, 4):
        result = np.eye(4, dtype=np.float32)
        result[:3, :4] = values
        return result
    if values.ndim == 3 and values.shape[1:] == (3, 4):
        result = np.repeat(np.eye(4, dtype=np.float32)[None], len(values), axis=0)
        result[:, :3, :4] = values
        return result
    return values


def create_xpolicylab_hdf5(data, hdf5_path, instructions, frequency):
    joints = data["joint_action"]
    frame_num = len(joints["left_arm"])
    if frame_num < 2:
        raise ValueError("At least two frames are required to create state/action pairs")

    instruction_values = [str(value) for value in (instructions or []) if str(value)]
    if not instruction_values:
        instruction_values = [""]

    with h5py.File(hdf5_path, "w") as f:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        f.attrs["source_format"] = "RoboTwin"
        f.attrs["source_path"] = "native_collection"
        f.create_dataset("data_format_version", data="v1.0", dtype=string_dtype)
        f.create_dataset(
            "instructions",
            data=json.dumps(instruction_values, ensure_ascii=False),
            dtype=string_dtype,
        )
        f.create_group("additional_info").create_dataset(
            "frequency", data=np.asarray(frequency, dtype=np.int32)
        )

        state = f.create_group("state")
        action = f.create_group("action")
        joint_fields = [
            ("left_arm", "left_arm_joint_states"),
            ("left_gripper", "left_ee_joint_states"),
            ("right_arm", "right_arm_joint_states"),
            ("right_gripper", "right_ee_joint_states"),
        ]
        for source_name, target_name in joint_fields:
            if source_name not in joints:
                continue
            values = _ensure_2d(joints[source_name])
            state.create_dataset(target_name, data=values[:-1])
            action.create_dataset(target_name, data=values[1:])

        endpose = data.get("endpose", {})
        for source_name, target_name in [
            ("left_endpose", "left_ee_poses"),
            ("right_endpose", "right_ee_poses"),
        ]:
            if source_name not in endpose:
                continue
            values = _ensure_2d(endpose[source_name])
            state.create_dataset(target_name, data=values[:-1])
            action.create_dataset(target_name, data=values[1:])

        vision = f.create_group("vision")
        observations = data["observation"]
        for source_name, target_name in CAMERA_MAP.items():
            if source_name not in observations or "rgb" not in observations[source_name]:
                continue
            source_camera = observations[source_name]
            target_camera = vision.create_group(target_name)
            colors = np.asarray(source_camera["rgb"])[:-1]
            encoded_colors, max_len = images_encoding(colors)
            target_camera.create_dataset(
                "colors", data=encoded_colors, dtype=f"S{max_len}"
            )
            target_camera.create_dataset(
                "shape", data=np.asarray(colors[0].shape, dtype=np.int32)
            )

            # A flow camera stores all T depth frames in /flow; avoid duplicating
            # the first T-1 frames in the standard /vision layout.
            if "depth" in source_camera and "entity_segmentation_id" not in source_camera:
                target_camera.create_dataset(
                    "depths",
                    data=np.asarray(source_camera["depth"])[:-1],
                    compression="lzf",
                    shuffle=True,
                )

            intrinsic = source_camera.get(
                "intrinsic_cv", source_camera.get("intrinsic_matrix")
            )
            if intrinsic is not None:
                target_camera.create_dataset(
                    "intrinsic_matrix", data=np.asarray(intrinsic)[:-1]
                )

            extrinsic = source_camera.get("cam2world_gl")
            if extrinsic is None:
                extrinsic = source_camera.get("extrinsic_cv")
            if extrinsic is None:
                extrinsic = source_camera.get("extrinsics_matrix")
            if extrinsic is not None:
                target_camera.create_dataset(
                    "extrinsics_matrix", data=_to_4x4(extrinsic)[:-1]
                )

        flow_state = data.get("flow_state", {})
        has_flow_segmentation = any(
            "entity_segmentation_id" in camera for camera in observations.values()
        )
        if flow_state or has_flow_segmentation:
            flow = f.create_group("flow")
            flow.attrs["depth_unit"] = "millimeter"
            flow.attrs["background_entity_id"] = np.uint32(0)
            flow.attrs["entity_id_semantics"] = "SAPIEN per_scene_id"
            flow.attrs["pose_convention"] = "4x4 local-to-world"
            flow.attrs["tcp_pose_convention"] = "xyz + quaternion(wxyz), TCP-to-world"
            flow.create_dataset("frame_indices", data=np.arange(frame_num, dtype=np.int32))

            flow_cameras = flow.create_group("cameras")
            for source_name, target_name in CAMERA_MAP.items():
                if source_name not in observations:
                    continue
                source_camera = observations[source_name]
                if "entity_segmentation_id" not in source_camera:
                    continue
                target_camera = flow_cameras.create_group(target_name)
                for source_key, target_key in [
                    ("depth", "depth_mm"),
                    ("visual_segmentation_id", "visual_segmentation_ids"),
                    ("entity_segmentation_id", "entity_segmentation_ids"),
                    ("intrinsic_cv", "intrinsic_cv"),
                    ("extrinsic_cv", "world_to_camera_cv"),
                    ("cam2world_gl", "camera_to_world_gl"),
                ]:
                    if source_key not in source_camera:
                        continue
                    values = np.asarray(source_camera[source_key])
                    kwargs = {"compression": "lzf", "shuffle": True} if values.ndim >= 3 else {}
                    target_camera.create_dataset(target_key, data=values, **kwargs)
                if "extrinsic_cv" in source_camera:
                    world_to_camera = np.asarray(source_camera["extrinsic_cv"], dtype=np.float32)
                    target_camera.create_dataset(
                        "camera_to_world_cv", data=np.linalg.inv(_to_4x4(world_to_camera))
                    )

            def write_static_strings(group, key, values):
                group.create_dataset(
                    key, data=np.asarray(values[0], dtype=object), dtype=string_dtype
                )

            if "scene_entities" in flow_state:
                source = flow_state["scene_entities"]
                target = flow.create_group("scene_entities")
                entity_ids = np.asarray(source["entity_ids"], dtype=np.uint32)
                if not np.all(entity_ids == entity_ids[0]):
                    raise ValueError("Scene entity IDs changed within an episode")
                target.create_dataset("entity_ids", data=entity_ids[0])
                for key in (
                    "entity_names", "instance_names", "articulation_names",
                    "link_names", "body_types", "task_roles",
                ):
                    write_static_strings(target, key, source[key])
                target.create_dataset(
                    "poses_world", data=np.asarray(source["poses_world"], dtype=np.float32)
                )

            if "articulations" in flow_state:
                source = flow_state["articulations"]
                target = flow.create_group("articulations")
                for key in ("instance_names", "joint_names"):
                    write_static_strings(target, key, source[key])
                for key in ("root_entity_ids", "dof_offsets"):
                    values = np.asarray(source[key])
                    if not np.all(values == values[0]):
                        raise ValueError(f"Articulation metadata changed: {key}")
                    target.create_dataset(key, data=values[0])
                target.create_dataset("qpos", data=np.asarray(source["qpos"], dtype=np.float32))
                target.create_dataset("qvel", data=np.asarray(source["qvel"], dtype=np.float32))

            if "grippers" in flow_state:
                source = flow_state["grippers"]
                target = flow.create_group("grippers")
                write_static_strings(target, "names", source["names"])
                target.create_dataset(
                    "poses_world_xyz_wxyz",
                    data=np.asarray(source["poses_world_xyz_wxyz"], dtype=np.float32),
                )

    return frame_num - 1


def pkl_files_to_hdf5_and_video(
    pkl_files,
    hdf5_path,
    video_path,
    *,
    instructions=None,
    frequency=15,
):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    images_to_video(np.array(data_list["observation"]["head_camera"]["rgb"]), out_path=video_path)
    return create_xpolicylab_hdf5(data_list, hdf5_path, instructions, frequency)


def process_folder_to_hdf5_video(
    folder_path,
    hdf5_path,
    video_path,
    *,
    instructions=None,
    frequency=15,
):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    return pkl_files_to_hdf5_and_video(
        pkl_files,
        hdf5_path,
        video_path,
        instructions=instructions,
        frequency=frequency,
    )
