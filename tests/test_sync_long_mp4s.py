from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sync_long_mp4s as mod


@dataclass
class FakeFrame:
    value: str


class FakeEncoder(mod.FrameEncoder):
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.frames: List[Any] = []
        self.closed = False

    def write_frame(self, frame: Any) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.output_path.write_text("frames=" + ",".join(str(getattr(f, "value", f)) for f in self.frames), encoding="utf-8")
        self.closed = True


class FakeBackend(mod.VideoBackend):
    backend_name = "cpu"

    def __init__(self, frame_map: Dict[str, List[FakeFrame]], geometry_map: Dict[str, tuple[int, int, str]] | None = None) -> None:
        self.frame_map = frame_map
        self.geometry_map = geometry_map or {}
        self.encoders: Dict[str, FakeEncoder] = {}

    def open_encoder(self, output_path: Path, fps: int, width: int, height: int, pixel_format: str) -> mod.FrameEncoder:
        encoder = FakeEncoder(output_path)
        self.encoders[str(output_path)] = encoder
        return encoder

    def decode_summary(self, chunk_path: Path) -> mod.DecodeSummary:
        frames = self.frame_map[str(chunk_path)]
        width, height, pixel_format = self.geometry_map.get(str(chunk_path), (640, 480, "yuv420p"))
        last_frame = frames[-1] if frames else None
        return mod.DecodeSummary(
            decoded_frames=len(frames),
            last_frame=last_frame,
            width=width,
            height=height,
            pixel_format=pixel_format,
            diagnostics={},
        )

    def iter_frames(self, chunk_path: Path) -> Iterator[Any]:
        yield from self.frame_map[str(chunk_path)]


def touch_chunk(root: Path, filename: str) -> Path:
    path = root / filename
    path.write_bytes(b"dummy")
    return path


def make_config(tmp_path: Path, cameras: list[str]) -> mod.SyncConfig:
    return mod.SyncConfig(
        store_id="store_042",
        date="2026-03-19",
        timezone_name="UTC",
        window_start="06:00:00",
        window_end="06:01:00",
        fps_output=20,
        backend="cpu",
        camera_ids=cameras,
        input_root=tmp_path / "input",
        output_root=tmp_path / "output",
    )


def test_normal_aligned_case(tmp_path: Path) -> None:
    config = make_config(tmp_path, ["camA", "camB"])
    config.input_root.mkdir(parents=True)
    config.output_root.mkdir(parents=True)
    chunk_a = touch_chunk(config.input_root, "2026-03-19T06:00:00Z_camA.h264")
    chunk_b = touch_chunk(config.input_root, "2026-03-19T06:00:00Z_camB.h264")
    backend = FakeBackend(
        {
            str(chunk_a): [FakeFrame("a1"), FakeFrame("a2")],
            str(chunk_b): [FakeFrame("b1"), FakeFrame("b2")],
        }
    )

    manifest = mod.run_sync(config, backend=backend)

    assert manifest["status"] == "SUCCESS"
    assert manifest["total_output_frames"] == 2
    assert (config.output_root / "store_store_042_2026-03-19_camA.mp4").read_text(encoding="utf-8") == "frames=a1,a2"
    rows = list(csv.DictReader((config.output_root / "store_store_042_2026-03-19_sync_batches.tsv").open(), delimiter="\t"))
    assert len(rows) == 2
    assert all(row["padding_frames"] == "0" for row in rows)


def test_intra_batch_padding_case(tmp_path: Path) -> None:
    config = make_config(tmp_path, ["camA", "camB", "camC"])
    config.input_root.mkdir(parents=True)
    config.output_root.mkdir(parents=True)
    chunk_a = touch_chunk(config.input_root, "2026-03-19T06:00:00Z_camA.h264")
    chunk_b = touch_chunk(config.input_root, "2026-03-19T06:00:00Z_camB.h264")
    chunk_c = touch_chunk(config.input_root, "2026-03-19T06:00:00Z_camC.h264")
    backend = FakeBackend(
        {
            str(chunk_a): [FakeFrame("a1"), FakeFrame("a2"), FakeFrame("a3")],
            str(chunk_b): [FakeFrame("b1"), FakeFrame("b2")],
            str(chunk_c): [FakeFrame("c1")],
        }
    )

    mod.run_sync(config, backend=backend)

    cam_b = (config.output_root / "store_store_042_2026-03-19_camB.mp4").read_text(encoding="utf-8")
    cam_c = (config.output_root / "store_store_042_2026-03-19_camC.mp4").read_text(encoding="utf-8")
    assert cam_b == "frames=b1,b2,b2"
    assert cam_c == "frames=c1,c1,c1"
    anomalies = list(csv.DictReader((config.output_root / "store_store_042_2026-03-19_sync_anomalies.tsv").open(), delimiter="\t"))
    assert [row["camera_id"] for row in anomalies] == ["camB", "camC"]


def test_missing_chunk_failure(tmp_path: Path) -> None:
    config = make_config(tmp_path, ["camA", "camB"])
    config.input_root.mkdir(parents=True)
    config.output_root.mkdir(parents=True)
    touch_chunk(config.input_root, "2026-03-19T06:00:00Z_camA.h264")

    backend = FakeBackend({})

    with pytest.raises(mod.SyncError, match="has no chunk at all|missing chunk"):
        mod.run_sync(config, backend=backend)


def test_duplicate_chunk_failure(tmp_path: Path) -> None:
    config = make_config(tmp_path, ["camA"])
    config.input_root.mkdir(parents=True)
    config.output_root.mkdir(parents=True)
    touch_chunk(config.input_root, "2026-03-19T06:00:00Z_camA.h264")
    touch_chunk(config.input_root, "2026-03-19T06:00:00Z_camA_copy.h264")

    backend = FakeBackend({})

    with pytest.raises(mod.SyncError, match="duplicate"):
        mod.run_sync(config, backend=backend)


def test_zero_frame_chunk_failure(tmp_path: Path) -> None:
    config = make_config(tmp_path, ["camA"])
    config.input_root.mkdir(parents=True)
    config.output_root.mkdir(parents=True)
    chunk = touch_chunk(config.input_root, "2026-03-19T06:00:00Z_camA.h264")
    backend = FakeBackend({str(chunk): []})

    with pytest.raises(mod.SyncError, match="zero frames"):
        mod.run_sync(config, backend=backend)


def test_unparseable_filename_failure(tmp_path: Path) -> None:
    config = make_config(tmp_path, ["camA"])
    config.input_root.mkdir(parents=True)
    config.output_root.mkdir(parents=True)
    touch_chunk(config.input_root, "badname.h264")

    backend = FakeBackend({})

    with pytest.raises(mod.SyncError, match="Unparseable filename"):
        mod.run_sync(config, backend=backend)


def test_requested_nvidia_backend_unavailable_explicit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "NvidiaBackend", lambda: (_ for _ in ()).throw(mod.SyncError("Requested --backend nvidia, but PyNvVideoCodec is unavailable.")))

    with pytest.raises(mod.SyncError, match="PyNvVideoCodec is unavailable"):
        mod.initialize_backend("nvidia")


def test_intra_camera_resolution_change_failure(tmp_path: Path) -> None:
    config = make_config(tmp_path, ["camA"])
    config.window_end = "06:01:30"
    config.input_root.mkdir(parents=True)
    config.output_root.mkdir(parents=True)
    chunk1 = touch_chunk(config.input_root, "2026-03-19T06:00:00Z_camA.h264")
    chunk2 = touch_chunk(config.input_root, "2026-03-19T06:00:30Z_camA.h264")
    backend = FakeBackend(
        {
            str(chunk1): [FakeFrame("a1")],
            str(chunk2): [FakeFrame("a2")],
        },
        geometry_map={
            str(chunk1): (640, 480, "yuv420p"),
            str(chunk2): (1280, 720, "yuv420p"),
        },
    )

    with pytest.raises(mod.SyncError, match="resolution change"):
        mod.run_sync(config, backend=backend)
