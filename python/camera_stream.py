"""RTSP camera ingestion and per-camera chunk file handling."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from gi.repository import Gst

from chunk_writer import ChunkWriter
from h264_chunk_analyzer import analyze_h264_chunk
from h264_utils import has_idr_nal


class CameraStream:
    """Represents one RTSP H264 stream and chunk writer state."""

    def __init__(self, name: str, rtsp_url: str, output_dir: Path, logger: logging.Logger) -> None:
        self.name = name
        self.rtsp_url = rtsp_url
        self._logger = logger
        self._writer = ChunkWriter(output_dir=output_dir, camera_name=name)

        self.current_chunk_timestamp = None
        self.skip_count = 0
        self.first_idr_seen = False
        self.frame_count = 0

        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._reconnect_event = threading.Event()
        self._reconnect_thread = threading.Thread(target=self._reconnect_loop, name="reconnect-{0}".format(name), daemon=True)

        self._pipeline = self._build_pipeline()
        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message", self._on_bus_message)

        self._appsink = self._pipeline.get_by_name("sink")
        self._appsink.connect("new-sample", self._on_new_sample)

    def start(self) -> None:
        self._running = True
        self._reconnect_thread.start()
        self._logger.info("camera started name=%s", self.name)
        self._set_pipeline_playing()

    def stop(self) -> None:
        self._running = False
        self._reconnect_event.set()
        if self._reconnect_thread.is_alive():
            self._reconnect_thread.join(timeout=3)

        self._pipeline.set_state(Gst.State.NULL)
        with self._lock:
            self._close_and_finalize_active_chunk()

    def rotate_event(self, timestamp_ms: int) -> None:
        """Rotate chunk file in global scheduler order."""
        with self._lock:
            if self.current_chunk_timestamp is not None:
                self._close_and_finalize_active_chunk()

            self._writer.open_chunk(timestamp_ms)
            self.current_chunk_timestamp = timestamp_ms
            self.skip_count = 0
            self.first_idr_seen = False
            self.frame_count = 0
            self._logger.info("chunk started camera=%s ts=%s", self.name, timestamp_ms)

    def _close_and_finalize_active_chunk(self):
        closed = self._writer.close_tmp()
        if closed is None:
            return None

        tmp_path, timestamp_ms = closed
        analysis = analyze_h264_chunk(str(tmp_path))
        if analysis.first_idr_frame_index is None:
            analyzed_skip = analysis.frame_count
        else:
            analyzed_skip = analysis.first_idr_frame_index

        if analyzed_skip != self.skip_count:
            self._logger.info(
                "skip mismatch camera=%s ts=%s online_skip=%s analyzed_skip=%s",
                self.name,
                timestamp_ms,
                self.skip_count,
                analyzed_skip,
            )

        final_path = self._writer.finalize_tmp(timestamp_ms=timestamp_ms, skip_count=analyzed_skip)
        self._logger.info(
            "chunk closed camera=%s ts=%s analyzed_frames=%s analyzed_first_idr=%s skip=%s file=%s",
            self.name,
            timestamp_ms,
            analysis.frame_count,
            analysis.first_idr_frame_index,
            analyzed_skip,
            final_path.name,
        )

        self.current_chunk_timestamp = None
        return final_path

    def _build_pipeline(self) -> Gst.Pipeline:
        launch = (
            'rtspsrc location="{0}" protocols=tcp name=src '
            "! rtph264depay "
            "! h264parse "
            "! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        ).format(self.rtsp_url)
        pipeline = Gst.parse_launch(launch)
        if not isinstance(pipeline, Gst.Pipeline):
            raise RuntimeError("Failed to create pipeline for {0}".format(self.name))
        return pipeline

    def _set_pipeline_playing(self) -> None:
        state_change = self._pipeline.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self._logger.warning("rtsp disconnected camera=%s", self.name)
            self._schedule_reconnect()

    def _on_new_sample(self, sink: Gst.Element) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buffer = sample.get_buffer()
        ok, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK

        try:
            frame = bytes(map_info.data)
        finally:
            buffer.unmap(map_info)

        idr_present = has_idr_nal(frame)

        with self._lock:
            if self.current_chunk_timestamp is None:
                return Gst.FlowReturn.OK

            if not self.first_idr_seen:
                if idr_present:
                    self.first_idr_seen = True
                else:
                    self.skip_count += 1

            self.frame_count += 1
            self._writer.write_frame(frame)

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message) -> None:
        message_type = message.type

        if message_type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            self._logger.warning(
                "rtsp disconnected camera=%s error=%s debug=%s",
                self.name,
                error,
                debug,
            )
            self._connected = False
            self._schedule_reconnect()
        elif message_type == Gst.MessageType.EOS:
            self._logger.warning("rtsp disconnected camera=%s reason=eos", self.name)
            self._connected = False
            self._schedule_reconnect()
        elif message_type == Gst.MessageType.STATE_CHANGED and message.src == self._pipeline:
            old_state, new_state, _ = message.parse_state_changed()
            if new_state == Gst.State.PLAYING and old_state != Gst.State.PLAYING:
                if not self._connected:
                    self._connected = True
                    self._logger.info("rtsp connected camera=%s", self.name)

    def _schedule_reconnect(self) -> None:
        if self._running:
            self._reconnect_event.set()

    def _reconnect_loop(self) -> None:
        while self._running:
            triggered = self._reconnect_event.wait(timeout=1.0)
            if not self._running:
                return
            if not triggered:
                continue

            self._reconnect_event.clear()
            attempt = 1
            while self._running and not self._connected:
                self._logger.info("reconnect attempt camera=%s attempt=%s", self.name, attempt)
                self._pipeline.set_state(Gst.State.NULL)
                time.sleep(0.5)
                self._set_pipeline_playing()
                attempt += 1
                time.sleep(2.0)
