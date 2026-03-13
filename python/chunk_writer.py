"""Sequential chunk file writing with tmp-to-final rename semantics."""

from pathlib import Path
from typing import Optional, Tuple


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
        if self._file is not None:
            raise RuntimeError("Previous chunk must be closed before opening a new one")

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._timestamp = timestamp_ms
        tmp_path = self._tmp_path(timestamp_ms)
        self._file = tmp_path.open("wb", buffering=0)

    def write_frame(self, frame: bytes) -> None:
        """Write one frame payload to current chunk file."""
        if self._file is None:
            raise RuntimeError("Chunk file not opened")
        self._file.write(frame)

    def close_chunk(self) -> Optional[Tuple[int, Path]]:
        """Close current tmp chunk and return (timestamp, tmp_path)."""
        if self._file is None or self._timestamp is None:
            return None

        timestamp = self._timestamp
        self._file.close()
        self._file = None
        self._timestamp = None

        return timestamp, self._tmp_path(timestamp)

    def finalize_chunk(self, timestamp_ms: int, skip_count: int) -> Path:
        """Rename closed tmp chunk file to final h264 path."""
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
