"""The one audio decode path for every ASR backend: container in, waveform out.

PyAV ships with faster-whisper, so decoding needs no ffmpeg binary in the image.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

SAMPLE_RATE = 16000


def decode(path: str) -> np.ndarray:
    """Decode an audio file (WebM/Opus included) to a float32 mono 16 kHz waveform."""
    # Check the file before a ~1 GB model load, and let os.stat do it rather than
    # os.path.exists: exists() answers False for a permission error too, which
    # transcriber.tracks would read as "this participant's track is missing" and quietly
    # drop them from the transcript. os.stat raises FileNotFoundError with .filename set —
    # the shape tracks.py needs — for a genuine ENOENT, and the true error otherwise.
    os.stat(path)
    # ponytail: PyAV via faster-whisper is already installed; swap for an
    # `ffmpeg -f f32le -ac 1 -ar 16000` subprocess only if faster-whisper is dropped.
    # Lazy import so that importing this module pulls in neither PyAV nor numpy.
    from faster_whisper.audio import decode_audio

    return decode_audio(path, sampling_rate=SAMPLE_RATE)
