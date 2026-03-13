"""Offline H264 Annex-B chunk analysis utilities.

This module parses the raw H264 byte stream directly (no decode/transcode)
and returns frame metadata used for finalized chunk filenames.
"""

from __future__ import absolute_import



class _BitReader(object):
    """Minimal RBSP bit reader with unsigned Exp-Golomb support."""

    def __init__(self, data):
        self._data = data
        self._bit_pos = 0

    def read_bit(self):
        total_bits = len(self._data) * 8
        if self._bit_pos >= total_bits:
            return None

        byte_index = self._bit_pos >> 3
        bit_index = 7 - (self._bit_pos & 7)
        self._bit_pos += 1
        return (self._data[byte_index] >> bit_index) & 0x01

    def read_bits(self, count):
        value = 0
        for _ in range(count):
            bit = self.read_bit()
            if bit is None:
                return None
            value = (value << 1) | bit
        return value

    def read_ue(self):
        """Read unsigned Exp-Golomb value from current bit position."""
        leading_zero_bits = 0

        while True:
            bit = self.read_bit()
            if bit is None:
                return None
            if bit == 0:
                leading_zero_bits += 1
                continue
            break

        if leading_zero_bits == 0:
            return 0

        suffix = self.read_bits(leading_zero_bits)
        if suffix is None:
            return None

        return ((1 << leading_zero_bits) - 1) + suffix


def _iter_annexb_nal_units(path):
    """Yield NAL unit payloads (without start codes) from an Annex-B file.

    Streaming parser:
    - start code is 00 00 01 or 00 00 00 01
    - does not load the whole chunk into memory
    """
    with open(path, "rb") as fp:
        started = False
        nal = bytearray()
        zero_count = 0

        while True:
            block = fp.read(65536)
            if not block:
                break

            for byte in block:
                if not started:
                    if byte == 0:
                        zero_count += 1
                        continue
                    if byte == 1 and zero_count >= 2:
                        started = True
                        zero_count = 0
                        nal = bytearray()
                        continue

                    zero_count = 0
                    continue

                if byte == 0:
                    zero_count += 1
                    continue

                if byte == 1 and zero_count >= 2:
                    if nal:
                        yield bytes(nal)
                    nal = bytearray()
                    zero_count = 0
                    continue

                if zero_count:
                    nal.extend(b"\x00" * zero_count)
                    zero_count = 0

                nal.append(byte)

        if started:
            if zero_count:
                nal.extend(b"\x00" * zero_count)
            if nal:
                yield bytes(nal)


def _ebsp_to_rbsp(ebsp):
    """Remove emulation prevention bytes (00 00 03 -> 00 00)."""
    rbsp = bytearray()
    zero_count = 0

    for byte in ebsp:
        if zero_count >= 2 and byte == 0x03:
            zero_count = 0
            continue

        rbsp.append(byte)

        if byte == 0:
            zero_count += 1
        else:
            zero_count = 0

    return bytes(rbsp)


def _first_mb_in_slice(nal_payload):
    """Parse first_mb_in_slice from a VCL NAL payload."""
    if len(nal_payload) <= 1:
        return None

    rbsp = _ebsp_to_rbsp(nal_payload[1:])
    reader = _BitReader(rbsp)
    return reader.read_ue()


def analyze_h264_chunk(path):
    """Return (frame_count, first_idr_frame_index) for a closed H264 chunk.

    - frame_count: number of visual frames/access units (VCL based)
    - first_idr_frame_index: first frame index that contains an IDR NAL, or None
    """
    frame_count = 0
    first_idr_frame_index = None

    current_frame_index = -1
    current_frame_has_idr = False

    for nal_payload in _iter_annexb_nal_units(path):
        if not nal_payload:
            continue

        nal_type = nal_payload[0] & 0x1F
        if nal_type not in (1, 5):
            continue

        first_mb = _first_mb_in_slice(nal_payload)
        starts_new_frame = (first_mb == 0)

        if frame_count == 0 and first_mb is None:
            starts_new_frame = True
        elif frame_count == 0 and first_mb != 0:
            starts_new_frame = True

        if starts_new_frame:
            if current_frame_index >= 0 and current_frame_has_idr and first_idr_frame_index is None:
                first_idr_frame_index = current_frame_index

            current_frame_index += 1
            frame_count += 1
            current_frame_has_idr = (nal_type == 5)
        else:
            if current_frame_index < 0:
                current_frame_index = 0
                frame_count = 1
                current_frame_has_idr = (nal_type == 5)
            elif nal_type == 5:
                current_frame_has_idr = True

    if current_frame_index >= 0 and current_frame_has_idr and first_idr_frame_index is None:
        first_idr_frame_index = current_frame_index

    return frame_count, first_idr_frame_index
