"""The one audio decode path for every ASR backend: container in, waveform out.

PyAV ships with faster-whisper, so decoding needs no ffmpeg binary in the image.
"""

from __future__ import annotations

import errno
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

SAMPLE_RATE = 16000


def decode(path: str) -> np.ndarray:
    """Decode an audio file (WebM/Opus included) to a float32 mono 16 kHz waveform."""
    if not os.path.exists(path):
        # PyAV's own "not found" carries no filename, and transcriber.tracks skips a track
        # only on a FileNotFoundError whose .filename is that track's path, as open() raises.
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    # ponytail: PyAV via faster-whisper is already installed; swap for an
    # `ffmpeg -f f32le -ac 1 -ar 16000` subprocess only if faster-whisper is dropped.
    # Lazy import so that importing this module pulls in neither PyAV nor numpy.
    from faster_whisper.audio import decode_audio

    return decode_audio(path, sampling_rate=SAMPLE_RATE)
