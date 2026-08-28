#!/usr/bin/env python3
"""Dependency-light local HTTP server for the RoboTwin flow visualizer."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import traceback
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qs, urlparse


HERE = Path(__file__).resolve().parent

try:
    import cv2

    from flow_core import (
        FlowDataError,
        flow_cache_info,
        frame_summary,
        get_flow_frame,
        inspect_hdf5,
        render_view,
    )
except ImportError as error:
    missing = getattr(error, "name", str(error))
    print(
        f"Missing Python dependency: {missing}\n"
        "Install the visualizer dependencies with:\n"
        "  python3 -m pip install -r tools/flow_visualizer/requirements.txt\n"
        "or run this server inside the RoboTwin simulation environment.",
        file=sys.stderr,
    )
    raise SystemExit(2) from error


RENDER_CACHE_SIZE = 1024
_RENDER_CACHE_LOCK = RLock()


@lru_cache(maxsize=RENDER_CACHE_SIZE)
def _render_png(
    path: str,
    modified_ns: int,
    camera: str,
    frame: int,
    threshold: float,
    view: str,
    maximum: float,
    opacity: float,
    arrow_step: int,
    hide_occluded: bool,
) -> bytes:
    del modified_ns  # File mtime invalidates this cache key.
    data = get_flow_frame(path, camera, frame, threshold)
    image = render_view(
        data,
        view,
        maximum=maximum,
        opacity=opacity,
        arrow_step=arrow_step,
        hide_occluded=hide_occluded,
    )
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise FlowDataError("OpenCV failed to encode the rendered frame")
    return encoded.tobytes()


def render_png_cached(*args) -> bytes:
    # Avoid duplicate renders when a browser retries the same URL concurrently.
    with _RENDER_CACHE_LOCK:
        return _render_png(*args)


def render_cache_info() -> dict[str, int | None]:
    info = _render_png.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize,
    }


class DataCatalog:
    def __init__(self, roots: list[str]):
        self.roots = [Path(root).expanduser().resolve() for root in roots]
        self.files: dict[str, Path] = {}
        self.entries: list[dict[str, str | int]] = []
        self.discovered_count = 0
        self.refresh()

    @staticmethod
    def task_name(path: Path) -> str:
        """Infer task from .../<task>/<embodiment>/data/episode_*.hdf5."""
        if path.parent.name.lower() in {"data", "episodes"} and len(path.parents) >= 3:
            return path.parents[2].name
        if path.name.lower().startswith("episode_"):
            return path.parent.name
        return path.stem

    def refresh(self) -> list[dict[str, str | int]]:
        files: list[Path] = []
        for root in self.roots:
            if root.is_file() and root.suffix.lower() in {".h5", ".hdf5"}:
                files.append(root)
            elif root.is_dir():
                files.extend(root.rglob("*.h5"))
                files.extend(root.rglob("*.hdf5"))
        unique = sorted(set(path.resolve() for path in files))
        self.discovered_count = len(unique)
        self.files = {}
        self.entries = []
        selected_tasks: set[str] = set()
        for path in unique:
            task = self.task_name(path)
            task_key = task.casefold()
            if task_key in selected_tasks:
                continue
            selected_tasks.add(task_key)
            try:
                source_path = str(path.relative_to(Path.cwd()))
            except ValueError:
                source_path = str(path)
            key = source_path
            self.files[key] = path
            self.entries.append(
                {
                    "id": key,
                    "label": task,
                    "task": task,
                    "path": source_path,
                    "size": path.stat().st_size,
                }
            )
        return self.entries

    def resolve(self, key: str) -> Path:
        if key not in self.files:
            self.refresh()
        if key not in self.files:
            raise FlowDataError(f"Unknown file: {key}")
        return self.files[key]


def _first(query: dict[str, list[str]], key: str, default: str = "") -> str:
    values = query.get(key)
    return values[0] if values else default


def _bool(query: dict[str, list[str]], key: str, default: bool) -> bool:
    value = _first(query, key, "1" if default else "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


class FlowRequestHandler(BaseHTTPRequestHandler):
    server_version = "RoboTwinFlow/1.0"

    @property
    def catalog(self) -> DataCatalog:
        return self.server.catalog  # type: ignore[attr-defined]

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _bytes(self, body: bytes, content_type: str, cache: bool = False) -> None:
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=31536000" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _flow_args(self, query: dict[str, list[str]]):
        path = self.catalog.resolve(_first(query, "file"))
        camera = _first(query, "camera")
        frame = int(_first(query, "frame", "0"))
        threshold = float(_first(query, "occlusion_threshold", "0.02"))
        return path, camera, frame, threshold

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/files":
                files = self.catalog.refresh()
                self._json(
                    {
                        "files": files,
                        "task_count": len(files),
                        "discovered_episode_count": self.catalog.discovered_count,
                        "episodes_per_task": 1,
                    }
                )
                return
            if parsed.path == "/api/meta":
                path = self.catalog.resolve(_first(query, "file"))
                metadata = inspect_hdf5(path)
                metadata["path"] = _first(query, "file")
                self._json(metadata)
                return
            if parsed.path == "/api/cache-info":
                self._json({"flow": flow_cache_info(), "render": render_cache_info()})
                return
            if parsed.path == "/api/frame-info":
                path, camera, frame, threshold = self._flow_args(query)
                data = get_flow_frame(path, camera, frame, threshold)
                self._json(frame_summary(data, _bool(query, "hide_occluded", True)))
                return
            if parsed.path == "/api/render.png":
                path, camera, frame, threshold = self._flow_args(query)
                resolved = path.resolve()
                body = render_png_cached(
                    str(resolved),
                    resolved.stat().st_mtime_ns,
                    camera,
                    frame,
                    round(threshold, 5),
                    _first(query, "view", "overlay"),
                    round(float(_first(query, "maximum", "0")), 3),
                    round(float(_first(query, "opacity", "0.72")), 3),
                    int(_first(query, "arrow_step", "20")),
                    _bool(query, "hide_occluded", True),
                )
                self._bytes(body, "image/png")
                return
            self._serve_static(parsed.path)
        except (FlowDataError, ValueError, KeyError, OSError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # Keep local debugging useful without exposing it in the UI.
            traceback.print_exc()
            self._json({"error": f"Internal error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        candidate = (HERE / "static" / relative).resolve()
        static_root = (HERE / "static").resolve()
        if static_root not in candidate.parents and candidate != static_root:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime, _ = mimetypes.guess_type(candidate.name)
        self._bytes(candidate.read_bytes(), mime or "application/octet-stream", cache=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize rigid scene flow stored in RoboTwin HDF5 files")
    parser.add_argument(
        "--data",
        nargs="+",
        default=["data"],
        metavar="PATH",
        help="One or more HDF5 files/directories to scan (default: data)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = DataCatalog(args.data)
    server = ThreadingHTTPServer((args.host, args.port), FlowRequestHandler)
    server.catalog = catalog  # type: ignore[attr-defined]
    print(f"RoboTwin Flow Viewer: http://{args.host}:{args.port}")
    print(f"Scanning: {', '.join(str(root) for root in catalog.roots)}")
    print(
        f"Selected {len(catalog.files)} task(s) from {catalog.discovered_count} HDF5 episode(s). "
        "Press Ctrl+C to stop."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
