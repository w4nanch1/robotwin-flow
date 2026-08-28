# RoboTwin Flow Lab

本地 HDF5 scene-flow 可视化网站。它不会要求预先保存 flow 数组，而是直接使用采集数据中的：

- `depth_mm[t]`
- `entity_segmentation_ids[t]`
- 相机 `intrinsic_cv` 与 `world_to_camera_cv`
- entity 的 `poses_world[t]`（4×4 local-to-world）

将当前像素反投影到三维，按 entity 从 `t` 变换到 `t+1`，再投影到下一帧得到 `(du, dv)`。背景 ID `0` 被视为世界坐标系中的静态点，因此移动的腕部相机也可以计算 camera-induced flow。

## 启动

在 RoboTwin 根目录运行：

```bash
python3 tools/flow_visualizer/server.py --data data
```

浏览器打开 <http://127.0.0.1:8765>。`--data` 后可以给一个或多个 HDF5 文件或目录：

```bash
python3 tools/flow_visualizer/server.py \
  --data data/task_a/episode0.hdf5 data/task_b \
  --port 8765
```

目录扫描会按照 RoboTwin 的
`<collection>/<task>/<embodiment>/data/episode_*.hdf5` 结构提取 task 名。每个 task
只选择路径排序最前的一条轨迹（通常是 `episode_0000000.hdf5`）；即使后续收集了很多
episode，网页下拉框也只会出现一次该任务。同一个 task 出现在多个 collection 中时同样只保留一条。

服务器只监听 `127.0.0.1`，使用 Python 标准库提供 HTTP 服务；数值与图像处理复用 RoboTwin 已有的 NumPy、h5py 和 OpenCV。若当前环境尚未安装：

```bash
python3 -m pip install -r tools/flow_visualizer/requirements.txt
```

## 先看 demo

```bash
python3 tools/flow_visualizer/generate_demo.py
python3 tools/flow_visualizer/server.py --data tools/flow_visualizer/demo
```

demo 包含两个具有已知刚体运动的 entity，可用来检查 flow 方向、幅值和播放。

## HDF5 最小结构

```text
/flow
  /cameras/<camera>
    depth_mm                    (T, H, W)
    entity_segmentation_ids     (T, H, W)
    intrinsic_cv                (T, 3, 3) 或 (3, 3)
    world_to_camera_cv          (T, 4, 4) 或 (4, 4)
  /scene_entities
    entity_ids                  (N,)
    entity_names                (N,)                 可选
    poses_world                 (T, N, 4, 4)
/vision/<camera>/colors         (T,) JPEG bytes      可选
```

也接受 `camera_to_world_cv`，后端会自动求逆。已有 `envs/utils/pkl2hdf5.py` 生成的 flow group 可直接使用。

## 有效性和遮挡

- depth 为 0、投影到相机后 `z <= 0`、落到画面外、或 seg ID 在 entity 表中找不到：标为 invalid。
- 投影点在下一帧的预测深度比实际 depth 更远且超过“遮挡阈值”：标为 occluded。
- “隐藏遮挡 flow”只改变显示与统计，不改变几何计算结果。
- Flow 单位是 pixel/frame。播放 FPS 仅影响网页播放速度，不改变 flow 数值。

键盘：空格播放/暂停，左右方向键切换相邻帧。

## 缓存与播放性能

- 后端保留最近 64 个 scene-flow 帧，并对同帧并发请求做 single-flight 合并。
- 最多缓存 1024 个已编码 PNG 视图；HDF5 修改时间变化后会自动失效。
- 播放会等待当前帧完成再请求下一帧，避免请求积压。设定的 FPS 是上限；计算较慢时会自动降速。
- 前端会保留旧帧，直到新帧下载并完成图像解码后再切换，避免缓存命中时仍出现黑屏闪烁。
- 可访问 `/api/cache-info` 查看 flow 与 render 缓存的 hit/miss 状态。服务重启后内存缓存会清空。
