#!/usr/bin/env python3
"""Synchronize per-camera H.264 chunks into long CFR MP4 outputs.

This module implements a strict two-pass synchronizer that groups chunks by the
absolute timestamp encoded in their filenames, counts real decoded frames for
all cameras in each batch, pads only shorter existing chunks by repeating their
own last valid frame, and emits one long MP4 per camera plus manifest/index
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date as date_cls
from datetime import datetime, time, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore


class SyncError(RuntimeError):
    """Raised for fatal synchronization failures."""


@dataclass(frozen=True)
class ChunkRecord:
    camera_id: str
    chunk_start_utc: datetime
    original_filename: str
    full_path: Path


@dataclass(frozen=True)
class ParsedFilename:
    camera_id: str
    chunk_start_utc: datetime
    original_filename: str
    full_path: Path


@dataclass
class DecodeSummary:
    decoded_frames: int
    last_frame: Any
    width: int
    height: int
    pixel_format: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchRow:
    batch_index: int
    batch_start_utc: str
    global_frame_start: int
    global_frame_end: int
    max_frames_in_batch: int
    camera_id: str
    chunk_filename: str
    decoded_frames: int
    padding_frames: int
    status: str


@dataclass
class AnomalyRow:
    batch_index: int
    batch_start_utc: str
    camera_id: str
    anomaly_type: str
    detail: str


@dataclass
class BatchPlan:
    batch_index: int
    batch_start_utc: datetime
    chunks_by_camera: Dict[str, ChunkRecord]


@dataclass
class BatchExecution:
    batch: BatchPlan
    max_frames_in_batch: int
    global_frame_start: int
    global_frame_end: int
    rows: List[BatchRow]


class FilenameParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> ParsedFilename:
        raise NotImplementedError


class DefaultFilenameParser(FilenameParser):
    """Parse filenames containing an ISO-like timestamp and camera id.

    Supported examples:
      * 2026-03-19T06:00:00_camA.h264
      * chunk_1742364000000_camA_0.h264
      * 2026-03-19T06-00-00Z-camA.h264
    """

    ISO_RE = re.compile(
        r"(?P<ts>\d{4}-\d{2}-\d{2}[T_ ]\d{2}[:\-]\d{2}[:\-]\d{2}(?:Z|[+\-]\d{2}:?\d{2})?)"
    )
    EPOCH_MS_RE = re.compile(r"(?:^|[^\d])(?P<epoch_ms>\d{13})(?:[^\d]|$)")

    def __init__(self, expected_cameras: Sequence[str]) -> None:
        self.expected_cameras = sorted(expected_cameras, key=len, reverse=True)

    def parse(self, path: Path) -> ParsedFilename:
        name = path.name
        camera_id = self._extract_camera_id(name)
        if camera_id is None:
            raise SyncError(f"Unparseable filename {name!r}: camera_id could not be extracted")
        chunk_start_utc = self._extract_timestamp_utc(name)
        if chunk_start_utc is None:
            raise SyncError(f"Unparseable filename {name!r}: chunk_start_utc could not be extracted")
        return ParsedFilename(
            camera_id=camera_id,
            chunk_start_utc=chunk_start_utc,
            original_filename=name,
            full_path=path,
        )

    def _extract_camera_id(self, filename: str) -> Optional[str]:
        stem = Path(filename).stem
        for camera_id in self.expected_cameras:
            if re.search(rf"(^|[^A-Za-z0-9]){re.escape(camera_id)}($|[^A-Za-z0-9])", stem):
                return camera_id
        return None

    def _extract_timestamp_utc(self, filename: str) -> Optional[datetime]:
        iso_match = self.ISO_RE.search(filename)
        if iso_match:
            raw = iso_match.group("ts").replace("_", "T").replace(" ", "T")
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            raw = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", raw)
            normalized = raw.replace("T", "T", 1)
            if len(normalized) >= 19:
                normalized = normalized[:13] + normalized[13:16].replace("-", ":") + normalized[16:19].replace("-", ":") + normalized[19:]
            try:
                dt = datetime.fromisoformat(normalized)
            except ValueError:
                dt = None
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)

        epoch_match = self.EPOCH_MS_RE.search(filename)
        if epoch_match:
            epoch_ms = int(epoch_match.group("epoch_ms"))
            return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
        return None


class FrameEncoder(ABC):
    @abstractmethod
    def write_frame(self, frame: Any) -> None:
        raise NotImplementedError

    def write_repeated_frame(self, frame: Any, repeat_count: int) -> None:
        for _ in range(repeat_count):
            self.write_frame(frame)

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class VideoBackend(ABC):
    backend_name: str

    @abstractmethod
    def open_encoder(self, output_path: Path, fps: int, width: int, height: int, pixel_format: str) -> FrameEncoder:
        raise NotImplementedError

    @abstractmethod
    def decode_summary(self, chunk_path: Path) -> DecodeSummary:
        raise NotImplementedError

    @abstractmethod
    def iter_frames(self, chunk_path: Path) -> Iterator[Any]:
        raise NotImplementedError


class CpuFrameEncoder(FrameEncoder):
    def __init__(self, av_module: Any, container: Any, stream: Any) -> None:
        self.av = av_module
        self.container = container
        self.stream = stream
        self.closed = False

    def _normalize_frame(self, frame: Any) -> Any:
        target_format = self.stream.pix_fmt or "yuv420p"
        if isinstance(frame, self.av.VideoFrame):
            if frame.format.name != target_format:
                return frame.reformat(format=target_format)
            return frame
        try:
            return self.av.VideoFrame.from_ndarray(frame, format=target_format)
        except Exception as exc:
            raise SyncError(
                f"CPU encoder expected a PyAV VideoFrame or ndarray compatible with {target_format}, "
                f"got {type(frame).__name__}"
            ) from exc

    def write_frame(self, frame: Any) -> None:
        normalized_frame = self._normalize_frame(frame)
        for packet in self.stream.encode(normalized_frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self.closed:
            return
        for packet in self.stream.encode(None):
            self.container.mux(packet)
        self.container.close()
        self.closed = True


class CpuBackend(VideoBackend):
    backend_name = "cpu"

    def __init__(self) -> None:
        try:
            import av  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise SyncError("CPU backend requires PyAV (`pip install av`).") from exc
        self.av = av

    def open_encoder(self, output_path: Path, fps: int, width: int, height: int, pixel_format: str) -> FrameEncoder:
        container = self.av.open(str(output_path), mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = pixel_format or "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        stream.time_base = Fraction(1, fps)
        return CpuFrameEncoder(self.av, container, stream)

    def decode_summary(self, chunk_path: Path) -> DecodeSummary:
        container = None
        decoded_frames = 0
        last_frame = None
        width = 0
        height = 0
        pix_fmt = "yuv420p"
        try:
            container = self.av.open(str(chunk_path), mode="r", format="h264")
            video_stream = next((s for s in container.streams if s.type == "video"), None)
            if video_stream is None:
                raise SyncError(f"Chunk open/decode failure for {chunk_path}: no video stream")
            for frame in container.decode(video=0):
                decoded_frames += 1
                normalized_frame = frame.reformat(format="yuv420p")
                width = normalized_frame.width
                height = normalized_frame.height
                pix_fmt = normalized_frame.format.name or pix_fmt
                last_frame = normalized_frame.to_ndarray().copy()
                diagnostics = {
                    "source_width": frame.width,
                    "source_height": frame.height,
                    "source_pix_fmt": frame.format.name,
                }
            return DecodeSummary(
                decoded_frames=decoded_frames,
                last_frame=last_frame,
                width=width,
                height=height,
                pixel_format=pix_fmt,
                diagnostics=diagnostics,
            )
        except self.av.AVError as exc:
            raise SyncError(f"Chunk open/decode failure for {chunk_path}: {exc}") from exc
        finally:
            if container is not None:
                container.close()

    def iter_frames(self, chunk_path: Path) -> Iterator[Any]:
        container = None
        try:
            container = self.av.open(str(chunk_path), mode="r", format="h264")
            for frame in container.decode(video=0):
                yield frame.reformat(format="yuv420p")
        except self.av.AVError as exc:
            raise SyncError(f"Chunk open/decode failure for {chunk_path}: {exc}") from exc
        finally:
            if container is not None:
                container.close()


class NvidiaBackend(VideoBackend):
    backend_name = "nvidia"

    def __init__(self) -> None:
        self.codec_module = self._import_pynvvideocodec()
        # The exact PyNvVideoCodec surface is environment dependent. We validate
        # import availability explicitly and keep the runtime implementation in a
        # dedicated backend object so production systems can adapt this thin layer
        # if API names differ.
        self._unavailable_reason = self._probe_runtime_surface()
        if self._unavailable_reason is not None:
            raise SyncError(self._unavailable_reason)

    @staticmethod
    def _import_pynvvideocodec() -> Any:
        import importlib

        candidates = ["PyNvVideoCodec", "pynvvideocodec", "PyNvVideoCodec.py"]
        for name in candidates:
            try:
                return importlib.import_module(name)
            except ImportError:
                continue
        raise SyncError(
            "Requested --backend nvidia, but PyNvVideoCodec is unavailable. "
            "Install the NVIDIA Python video codec bindings and ensure they are importable."
        )

    def _probe_runtime_surface(self) -> Optional[str]:
        # PyNvVideoCodec package naming is known to vary. We intentionally verify
        # only that the module is importable and appears to expose codec-related
        # symbols; the actual decode/encode objects are resolved lazily.
        attrs = set(dir(self.codec_module))
        likely_decode = any("decode" in attr.lower() for attr in attrs)
        likely_encode = any("encode" in attr.lower() for attr in attrs)
        if not (likely_decode and likely_encode):
            return (
                "Requested --backend nvidia, but the imported PyNvVideoCodec module does not expose "
                "recognizable decode/encode entry points. No silent CPU fallback will be used."
            )
        return (
            "Requested --backend nvidia, but this environment lacks a repository-specific "
            "PyNvVideoCodec adapter implementation. No silent CPU fallback will be used."
        )

    def open_encoder(self, output_path: Path, fps: int, width: int, height: int, pixel_format: str) -> FrameEncoder:
        raise SyncError("NVIDIA backend adapter is unavailable in this environment.")

    def decode_summary(self, chunk_path: Path) -> DecodeSummary:
        raise SyncError("NVIDIA backend adapter is unavailable in this environment.")

    def iter_frames(self, chunk_path: Path) -> Iterator[Any]:
        raise SyncError("NVIDIA backend adapter is unavailable in this environment.")
        yield  # pragma: no cover


@dataclass
class AtomicOutputs:
    temp_video_paths: Dict[str, Path]
    final_video_paths: Dict[str, Path]
    temp_manifest_path: Path
    final_manifest_path: Path
    temp_batches_path: Path
    final_batches_path: Path
    temp_anomalies_path: Path
    final_anomalies_path: Path


@dataclass
class SyncConfig:
    store_id: str
    date: str
    timezone_name: str
    window_start: str
    window_end: str
    fps_output: int
    backend: str
    camera_ids: List[str]
    input_root: Path
    output_root: Path

    @property
    def local_date(self) -> date_cls:
        return date_cls.fromisoformat(self.date)

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    @property
    def window_start_local(self) -> datetime:
        return datetime.combine(self.local_date, parse_clock_time(self.window_start), self.tzinfo)

    @property
    def window_end_local(self) -> datetime:
        return datetime.combine(self.local_date, parse_clock_time(self.window_end), self.tzinfo)

    @property
    def window_start_utc(self) -> datetime:
        return self.window_start_local.astimezone(timezone.utc)

    @property
    def window_end_utc(self) -> datetime:
        return self.window_end_local.astimezone(timezone.utc)


def parse_clock_time(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise SyncError(f"Invalid time value {value!r}; expected HH:MM:SS") from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> SyncConfig:
    parser = argparse.ArgumentParser(description="Synchronize H.264 chunks into long MP4s")
    parser.add_argument("--store-id", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--timezone", required=True)
    parser.add_argument("--window-start", required=True)
    parser.add_argument("--window-end", required=True)
    parser.add_argument("--fps-output", required=True, type=int)
    parser.add_argument("--backend", choices=("nvidia", "cpu"), default="nvidia")
    parser.add_argument("--camera", dest="camera_ids", action="append", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    ns = parser.parse_args(argv)
    return SyncConfig(
        store_id=ns.store_id,
        date=ns.date,
        timezone_name=ns.timezone,
        window_start=ns.window_start,
        window_end=ns.window_end,
        fps_output=ns.fps_output,
        backend=ns.backend,
        camera_ids=list(ns.camera_ids),
        input_root=Path(ns.input_root),
        output_root=Path(ns.output_root),
    )


def validate_config(config: SyncConfig) -> None:
    if not config.camera_ids:
        raise SyncError("At least one --camera must be provided")
    if len(set(config.camera_ids)) != len(config.camera_ids):
        raise SyncError("Camera list contains duplicates")
    try:
        _ = config.local_date
    except ValueError as exc:
        raise SyncError(f"Invalid --date value {config.date!r}; expected YYYY-MM-DD") from exc
    try:
        _ = config.tzinfo
    except Exception as exc:
        raise SyncError(f"Invalid --timezone value {config.timezone_name!r}") from exc
    if config.fps_output <= 0:
        raise SyncError("--fps-output must be a positive integer")
    if config.window_end_local <= config.window_start_local:
        raise SyncError("Window end must be strictly after window start")
    config.output_root.mkdir(parents=True, exist_ok=True)


def initialize_backend(name: str) -> VideoBackend:
    if name == "cpu":
        return CpuBackend()
    if name == "nvidia":
        return NvidiaBackend()
    raise SyncError(f"Unsupported backend {name!r}")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def discover_chunks(config: SyncConfig, parser: FilenameParser) -> Tuple[Dict[str, List[ChunkRecord]], List[BatchPlan]]:
    chunks_by_camera: Dict[str, List[ChunkRecord]] = {camera_id: [] for camera_id in config.camera_ids}
    all_files = sorted(path for path in config.input_root.rglob("*") if path.is_file())
    parse_errors: List[str] = []

    for path in all_files:
        if path.suffix.lower() not in {".h264", ".264", ".mp4"}:
            continue
        try:
            parsed = parser.parse(path)
        except SyncError as exc:
            parse_errors.append(str(exc))
            continue
        if parsed.camera_id not in chunks_by_camera:
            continue
        if config.window_start_utc <= parsed.chunk_start_utc < config.window_end_utc:
            chunks_by_camera[parsed.camera_id].append(
                ChunkRecord(
                    camera_id=parsed.camera_id,
                    chunk_start_utc=parsed.chunk_start_utc,
                    original_filename=parsed.original_filename,
                    full_path=parsed.full_path,
                )
            )

    if parse_errors:
        raise SyncError(parse_errors[0])

    for camera_id, records in chunks_by_camera.items():
        records.sort(key=lambda item: (item.chunk_start_utc, item.original_filename))
        if not records:
            raise SyncError(
                f"Expected camera {camera_id!r} has no chunk at all in the interval "
                f"[{config.window_start_utc.isoformat()}, {config.window_end_utc.isoformat()})"
            )

    batches = build_batches(config.camera_ids, chunks_by_camera)
    return chunks_by_camera, batches


def build_batches(camera_ids: Sequence[str], chunks_by_camera: Mapping[str, Sequence[ChunkRecord]]) -> List[BatchPlan]:
    timestamps = sorted({record.chunk_start_utc for records in chunks_by_camera.values() for record in records})
    batch_map: List[BatchPlan] = []
    indexed: Dict[Tuple[str, datetime], List[ChunkRecord]] = {}
    for camera_id, records in chunks_by_camera.items():
        for record in records:
            indexed.setdefault((camera_id, record.chunk_start_utc), []).append(record)

    for batch_index, ts in enumerate(timestamps):
        chunks: Dict[str, ChunkRecord] = {}
        for camera_id in camera_ids:
            matches = indexed.get((camera_id, ts), [])
            if len(matches) != 1:
                if len(matches) == 0:
                    raise SyncError(
                        f"Batch {ts.isoformat()} missing chunk for expected camera {camera_id!r}"
                    )
                raise SyncError(
                    f"Batch {ts.isoformat()} has duplicate chunk entries for camera {camera_id!r}"
                )
            chunks[camera_id] = matches[0]
        batch_map.append(BatchPlan(batch_index=batch_index, batch_start_utc=ts, chunks_by_camera=chunks))
    return batch_map


def output_name_prefix(config: SyncConfig) -> str:
    store_token = config.store_id if config.store_id.startswith("store_") else f"store_{config.store_id}"
    return f"{store_token}_{config.date}"


def make_atomic_outputs(config: SyncConfig) -> AtomicOutputs:
    prefix = output_name_prefix(config)
    temp_video_paths: Dict[str, Path] = {}
    final_video_paths: Dict[str, Path] = {}
    for camera_id in config.camera_ids:
        final_name = f"{prefix}_{camera_id}.mp4"
        final_path = config.output_root / final_name
        temp_video_paths[camera_id] = config.output_root / f".{final_name}.tmp"
        final_video_paths[camera_id] = final_path
    manifest_name = f"{prefix}_sync_manifest.json"
    batches_name = f"{prefix}_sync_batches.tsv"
    anomalies_name = f"{prefix}_sync_anomalies.tsv"
    return AtomicOutputs(
        temp_video_paths=temp_video_paths,
        final_video_paths=final_video_paths,
        temp_manifest_path=config.output_root / f".{manifest_name}.tmp",
        final_manifest_path=config.output_root / manifest_name,
        temp_batches_path=config.output_root / f".{batches_name}.tmp",
        final_batches_path=config.output_root / batches_name,
        temp_anomalies_path=config.output_root / f".{anomalies_name}.tmp",
        final_anomalies_path=config.output_root / anomalies_name,
    )


def open_encoders_for_first_batch(
    backend: VideoBackend,
    outputs: AtomicOutputs,
    first_summaries: Mapping[str, DecodeSummary],
    fps_output: int,
) -> Dict[str, FrameEncoder]:
    encoders: Dict[str, FrameEncoder] = {}
    for camera_id, summary in first_summaries.items():
        encoders[camera_id] = backend.open_encoder(
            outputs.temp_video_paths[camera_id],
            fps=fps_output,
            width=summary.width,
            height=summary.height,
            pixel_format="yuv420p",
        )
    return encoders


def validate_resolution_consistency(
    expected_geometry: Dict[str, Tuple[int, int, str]],
    camera_id: str,
    summary: DecodeSummary,
) -> None:
    geometry = (summary.width, summary.height, summary.pixel_format)
    if camera_id not in expected_geometry:
        expected_geometry[camera_id] = geometry
        return
    if expected_geometry[camera_id] != geometry:
        prev = expected_geometry[camera_id]
        raise SyncError(
            "Intra-camera resolution change across the execution window in V0 is fatal: "
            f"camera={camera_id} previous={prev} current={geometry}"
        )


def execute_batches(
    config: SyncConfig,
    backend: VideoBackend,
    batches: Sequence[BatchPlan],
    outputs: AtomicOutputs,
) -> Tuple[Dict[str, FrameEncoder], List[BatchExecution], List[AnomalyRow], int]:
    if not batches:
        raise SyncError("No batches discovered inside the requested window")

    executions: List[BatchExecution] = []
    anomalies: List[AnomalyRow] = []
    global_frame_cursor = 0
    encoders: Optional[Dict[str, FrameEncoder]] = None
    expected_geometry: Dict[str, Tuple[int, int, str]] = {}

    for batch in batches:
        log(f"Processing batch {batch.batch_index} @ {batch.batch_start_utc.isoformat()}")
        summaries: Dict[str, DecodeSummary] = {}
        for camera_id in config.camera_ids:
            chunk = batch.chunks_by_camera[camera_id]
            summary = backend.decode_summary(chunk.full_path)
            if summary.decoded_frames == 0:
                raise SyncError(f"Chunk decodes to zero frames: {chunk.full_path}")
            validate_resolution_consistency(expected_geometry, camera_id, summary)
            summaries[camera_id] = summary

        if encoders is None:
            encoders = open_encoders_for_first_batch(backend, outputs, summaries, config.fps_output)

        max_frames = max(summary.decoded_frames for summary in summaries.values())
        batch_rows: List[BatchRow] = []
        batch_start_iso = batch.batch_start_utc.isoformat().replace("+00:00", "Z")
        global_frame_start = global_frame_cursor
        global_frame_end = global_frame_cursor + max_frames

        for camera_id in config.camera_ids:
            chunk = batch.chunks_by_camera[camera_id]
            summary = summaries[camera_id]
            frame_count_second_pass = 0
            for frame in backend.iter_frames(chunk.full_path):
                encoders[camera_id].write_frame(frame)
                frame_count_second_pass += 1
            if frame_count_second_pass != summary.decoded_frames:
                raise SyncError(
                    f"Second-pass decode count mismatch for {chunk.full_path}: "
                    f"pass1={summary.decoded_frames} pass2={frame_count_second_pass}"
                )
            padding_frames = max_frames - summary.decoded_frames
            if padding_frames > 0:
                encoders[camera_id].write_repeated_frame(summary.last_frame, padding_frames)
                anomalies.append(
                    AnomalyRow(
                        batch_index=batch.batch_index,
                        batch_start_utc=batch_start_iso,
                        camera_id=camera_id,
                        anomaly_type="PADDING_APPLIED",
                        detail=(
                            f"decoded_frames={summary.decoded_frames}; "
                            f"max_frames_in_batch={max_frames}; padding_frames={padding_frames}"
                        ),
                    )
                )
            batch_rows.append(
                BatchRow(
                    batch_index=batch.batch_index,
                    batch_start_utc=batch_start_iso,
                    global_frame_start=global_frame_start,
                    global_frame_end=global_frame_end,
                    max_frames_in_batch=max_frames,
                    camera_id=camera_id,
                    chunk_filename=chunk.original_filename,
                    decoded_frames=summary.decoded_frames,
                    padding_frames=padding_frames,
                    status="OK" if padding_frames == 0 else "OK_PADDED",
                )
            )
        executions.append(
            BatchExecution(
                batch=batch,
                max_frames_in_batch=max_frames,
                global_frame_start=global_frame_start,
                global_frame_end=global_frame_end,
                rows=batch_rows,
            )
        )
        global_frame_cursor += max_frames

    if encoders is None:
        raise SyncError("No encoders were initialized")
    return encoders, executions, anomalies, global_frame_cursor


def close_encoders(encoders: Mapping[str, FrameEncoder]) -> None:
    close_errors: List[Exception] = []
    for encoder in encoders.values():
        try:
            encoder.close()
        except Exception as exc:  # pragma: no cover - best effort finalization
            close_errors.append(exc)
    if close_errors:
        raise SyncError(f"Final MP4 write failure: {close_errors[0]}")


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_batches_tsv(path: Path, rows: Sequence[BatchRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "batch_index",
                "batch_start_utc",
                "global_frame_start",
                "global_frame_end",
                "max_frames_in_batch",
                "camera_id",
                "chunk_filename",
                "decoded_frames",
                "padding_frames",
                "status",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_anomalies_tsv(path: Path, rows: Sequence[AnomalyRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["batch_index", "batch_start_utc", "camera_id", "anomaly_type", "detail"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def finalize_success(outputs: AtomicOutputs) -> None:
    for final_path in list(outputs.final_video_paths.values()) + [
        outputs.final_manifest_path,
        outputs.final_batches_path,
        outputs.final_anomalies_path,
    ]:
        if final_path.exists():
            final_path.unlink()
    for camera_id, temp_path in outputs.temp_video_paths.items():
        os.replace(temp_path, outputs.final_video_paths[camera_id])
    os.replace(outputs.temp_manifest_path, outputs.final_manifest_path)
    os.replace(outputs.temp_batches_path, outputs.final_batches_path)
    os.replace(outputs.temp_anomalies_path, outputs.final_anomalies_path)


def build_success_manifest(
    config: SyncConfig,
    executions: Sequence[BatchExecution],
    total_output_frames: int,
) -> Dict[str, Any]:
    output_videos = {
        camera_id: f"{output_name_prefix(config)}_{camera_id}.mp4"
        for camera_id in config.camera_ids
    }
    return {
        "store_id": config.store_id,
        "date": config.date,
        "timezone": config.timezone_name,
        "window_start": config.window_start_local.isoformat(),
        "window_end": config.window_end_local.isoformat(),
        "fps_nominal_output": config.fps_output,
        "backend": config.backend,
        "camera_ids": list(config.camera_ids),
        "output_videos": output_videos,
        "total_batches": len(executions),
        "total_output_frames": total_output_frames,
        "status": "SUCCESS",
    }


def build_failed_manifest(config: SyncConfig, backend_name: str, message: str) -> Dict[str, Any]:
    return {
        "store_id": config.store_id,
        "date": config.date,
        "timezone": config.timezone_name,
        "window_start": config.window_start_local.isoformat(),
        "window_end": config.window_end_local.isoformat(),
        "fps_nominal_output": config.fps_output,
        "backend": backend_name,
        "camera_ids": list(config.camera_ids),
        "output_videos": {
            camera_id: f"{output_name_prefix(config)}_{camera_id}.mp4"
            for camera_id in config.camera_ids
        },
        "total_batches": 0,
        "total_output_frames": 0,
        "status": "FAILED",
        "failure_reason": message,
    }


def run_sync(config: SyncConfig, backend: Optional[VideoBackend] = None) -> Dict[str, Any]:
    validate_config(config)
    parser = DefaultFilenameParser(config.camera_ids)
    outputs = make_atomic_outputs(config)
    anomalies: List[AnomalyRow] = []
    batch_rows: List[BatchRow] = []
    backend_impl = backend or initialize_backend(config.backend)

    try:
        _chunks_by_camera, batches = discover_chunks(config, parser)
        encoders, executions, anomalies, total_output_frames = execute_batches(config, backend_impl, batches, outputs)
        close_encoders(encoders)
        for execution in executions:
            batch_rows.extend(execution.rows)
        manifest = build_success_manifest(config, executions, total_output_frames)
        write_manifest(outputs.temp_manifest_path, manifest)
        write_batches_tsv(outputs.temp_batches_path, batch_rows)
        write_anomalies_tsv(outputs.temp_anomalies_path, anomalies)
        finalize_success(outputs)
        return manifest
    except Exception as exc:
        message = str(exc)
        failed_manifest = build_failed_manifest(config, backend_impl.backend_name, message)
        try:
            failed_manifest_path = outputs.temp_manifest_path.with_suffix(outputs.temp_manifest_path.suffix + ".failed")
            failed_anomalies_path = outputs.temp_anomalies_path.with_suffix(outputs.temp_anomalies_path.suffix + ".failed")
            write_manifest(failed_manifest_path, failed_manifest)
            write_anomalies_tsv(failed_anomalies_path, anomalies)
        except Exception:
            pass
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        config = parse_args(argv)
        run_sync(config)
        return 0
    except SyncError as exc:
        log(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive last resort
        log(f"UNEXPECTED ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
