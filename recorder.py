"""Entrypoint for synchronized multi-camera H264 chunk recorder."""

from __future__ import annotations

import logging
import signal
import sys
import threading
from pathlib import Path

from gi.repository import GLib, Gst

from camera_stream import CameraStream
from config_loader import load_config
from scheduler import ChunkScheduler


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python recorder.py config.yaml")
        return 1

    configure_logging()
    logger = logging.getLogger("recorder")

    config = load_config(argv[1])

    Gst.init(None)

    output_dir = Path("recordings")
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = [
        CameraStream(name=camera.name, rtsp_url=camera.rtsp, output_dir=output_dir, logger=logging.getLogger(f"camera.{camera.name}"))
        for camera in config.cameras
    ]

    scheduler = ChunkScheduler(chunk_duration_ms=config.recording.chunk_duration_ms, logger=logging.getLogger("scheduler"))
    for camera in cameras:
        scheduler.register(camera.rotate_event)

    main_loop = GLib.MainLoop()
    stop_event = threading.Event()

    def _shutdown_handler(signum, _frame) -> None:
        logger.info("received signal=%s, shutting down", signum)
        stop_event.set()
        main_loop.quit()

    signal.signal(signal.SIGINT, _shutdown_handler)

    for camera in cameras:
        camera.start()

    scheduler.start()

    try:
        main_loop.run()
    finally:
        scheduler.stop()
        for camera in cameras:
            camera.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
