"""Offline H264 Annex-B chunk analyzer.

This module performs a lightweight parse of a closed H264 chunk file to count
visual frames and detect the first frame that contains an IDR NAL.

It does not decode, transcode, or alter payload bytes.
"""

from __future__ import annotations

from collections import namedtuple


ChunkAnalysis = namedtuple("ChunkAnalysis", ["frame_count", "first_idr_frame_index"])


class _BitReader(object):
    """Minimal bit reader for RBSP parsing."""

    def __init__(self, data):
        self._data = data
        self._byte_index = 0
        self._bit_index = 0

    def read_bit(self):
        if self._byte_index >= len(self._data):
            raise ValueError("Unexpected end of RBSP while reading bit")

        value = (self._data[self._byte_index] >> (7 - self._bit_index)) & 0x01
        self._bit_index += 1
        if self._bit_index == 8:
            self._bit_index = 0
            self._byte_index += 1
        return value

    def read_ue(self):
        """Read unsigned Exp-Golomb code."""
        leading_zero_bits = 0
        while True:
            bit = self.read_bit()
            if bit == 0:
                leading_zero_bits += 1
            else:
                break

        if leading_zero_bits == 0:
            return 0

        suffix = 0
        for _ in range(leading_zero_bits):
            suffix = (suffix << 1) | self.read_bit()
        return (1 << leading_zero_bits) - 1 + suffix


def _remove_emulation_prevention(nal_payload):
    """Remove emulation prevention bytes (0x03 after 0x00 0x00) from payload."""
    out = bytearray()
    zeros = 0

    for value in nal_payload:
        if zeros >= 2 and value == 0x03:
            zeros = 0
            continue

        out.append(value)
        if value == 0x00:
            zeros += 1
        else:
            zeros = 0

    return bytes(out)


def _start_code_len(buf, index):
    if index + 4 <= len(buf) and buf[index:index + 4] == b"\x00\x00\x00\x01":
        return 4
    if index + 3 <= len(buf) and buf[index:index + 3] == b"\x00\x00\x01":
        return 3
    return 0


def _iter_annexb_nalus_from_file(path, read_size=65536):
    """Yield (nal_type, nal_bytes_without_start_code) from an Annex-B file.

    Parses incrementally to avoid loading full files into memory.
    """
    with open(path, "rb") as f:
        data = f.read(read_size)
        if not data:
            return

        pending = data
        current_nal = None

        while True:
            data = f.read(read_size)
            eof = not data
            if not eof:
                pending += data

            index = 0
            last_safe = max(0, len(pending) - 4)

            while index <= last_safe:
                start_len = _start_code_len(pending, index)
                if start_len == 0:
                    index += 1
                    continue

                if current_nal is not None:
                    current_nal.extend(pending[:index])
                    if current_nal:
                        yield current_nal[0] & 0x1F, bytes(current_nal)

                pending = pending[index + start_len:]
                current_nal = bytearray()
                index = 0
                last_safe = max(0, len(pending) - 4)

            if eof:
                if current_nal is not None:
                    current_nal.extend(pending)
                    if current_nal:
                        yield current_nal[0] & 0x1F, bytes(current_nal)
                return

            if current_nal is not None:
                keep = 4
                if len(pending) > keep:
                    current_nal.extend(pending[:-keep])
                    pending = pending[-keep:]
            else:
                if len(pending) > 4:
                    pending = pending[-4:]


def _first_mb_in_slice(vcl_nal):
    """Extract first_mb_in_slice from a VCL NAL (types 1/5)."""
    if len(vcl_nal) <= 1:
        raise ValueError("VCL NAL too short")

    rbsp = _remove_emulation_prevention(vcl_nal[1:])
    reader = _BitReader(rbsp)
    return reader.read_ue()


def analyze_h264_chunk(path):
    """Analyze a closed chunk file and return frame_count and first IDR frame index.

    Frame boundaries are identified by VCL NAL slice headers where
    first_mb_in_slice == 0.
    """
    frame_count = 0
    first_idr_frame_index = None

    seen_any_vcl = False

    for nal_type, nal in _iter_annexb_nalus_from_file(path):
        if nal_type not in (1, 5):
            continue

        try:
            first_mb = _first_mb_in_slice(nal)
        except ValueError:
            continue

        if not seen_any_vcl:
            frame_count = 1
            seen_any_vcl = True
            if nal_type == 5 and first_idr_frame_index is None:
                first_idr_frame_index = 0
            continue

        if first_mb == 0:
            frame_count += 1
            if nal_type == 5 and first_idr_frame_index is None:
                first_idr_frame_index = frame_count - 1
        else:
            if nal_type == 5 and first_idr_frame_index is None:
                first_idr_frame_index = frame_count - 1

    return ChunkAnalysis(frame_count=frame_count, first_idr_frame_index=first_idr_frame_index)
