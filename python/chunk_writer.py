"""Sequential chunk file writing with tmp-to-final rename semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class ChunkWriter:
    """Manages chunk file lifecycle for one camera."""

    def __init__(self, output_dir: Path, camera_name: str) -> None:
        self._output_dir = output_dir
        self._camera_name = camera_name
        self._file = None
        self._timestamp = None

    @property
    def chunk_timestamp(self) -> Optional[int]:
        return self._timestamp

    def open_chunk(self, timestamp_ms: int) -> None:
        """Open a new temporary file for the given chunk timestamp."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._timestamp = timestamp_ms
        tmp_path = self._tmp_path(timestamp_ms)
        self._file = tmp_path.open("wb", buffering=0)

    def write_frame(self, frame: bytes) -> None:
        """Write one frame payload to current chunk file."""
        if self._file is None:
            raise RuntimeError("Chunk file not opened")
        self._file.write(frame)

    def close_tmp(self):
        """Close current tmp file and return (tmp_path, timestamp)."""
        if self._file is None or self._timestamp is None:
            return None

        timestamp = self._timestamp
        self._file.close()
        self._file = None
        self._timestamp = None
        return self._tmp_path(timestamp), timestamp

    def finalize_tmp(self, timestamp_ms: int, skip_count: int):
        """Rename tmp file to final h264 filename."""
        tmp_path = self._tmp_path(timestamp_ms)
        final_path = self._final_path(timestamp_ms, skip_count)
        if tmp_path.exists():
            tmp_path.rename(final_path)
        return final_path

    def _tmp_path(self, timestamp_ms: int) -> Path:
        filename = "chunk_{0}_{1}.tmp".format(timestamp_ms, self._camera_name)
        return self._output_dir / filename

    def _final_path(self, timestamp_ms: int, skip_count: int) -> Path:
        filename = "chunk_{0}_{1}_{2}.h264".format(timestamp_ms, self._camera_name, skip_count)
        return self._output_dir / filename
