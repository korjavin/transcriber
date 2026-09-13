"""Parakeet-tdt-0.6b-v3 on CPU via onnx-asr (int8) — the default ASR backend.

Language (EN and RU among others) is autodetected; the model takes no language option.
"""

from __future__ import annotations

import logging
import os
import shutil

from transcriber import audio
from transcriber.transcribe import Segment

log = logging.getLogger(__name__)

MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
MODEL_SUBDIR = "parakeet-tdt-0.6b-v3"
VAD_NAME = "silero"
# The VAD gets its own subdirectory of MODEL_DIR so that the whole volume is self-contained:
# left pathless it would land in the default Hugging Face cache and be re-fetched from the
# network on every cold start, which also means no start at all when the hub is unreachable.
VAD_SUBDIR = "silero-vad"
# CPU inference is the owner's decision, so say so instead of taking onnxruntime's default
# provider list: on a macOS dev box that list starts with CoreML, which hands the graph to
# the Neural Engine and gets the process killed. The deployed Linux image has CPU only.
# ponytail: widen this list if the service is ever given a GPU.
PROVIDERS = ["CPUExecutionProvider"]

# One instance per process: loading takes seconds and ~1 GB of RAM. The worker runs one
# transcription at a time, so a plain global needs no lock.
_MODEL = None


def _load(loader, name: str, path: str, **kwargs):
    """Load one model from `path`, downloading it there on first use.

    Never pre-create `path`: onnx-asr reads an existing local directory as a complete
    offline copy and then never downloads into it. That same rule is why a download killed
    part-way — a restarted container — would otherwise leave a half-filled directory that
    fails every subsequent start forever, so an incomplete one is cleared and fetched again.
    """
    try:
        return loader(name, path, **kwargs)
    except FileNotFoundError:
        if not os.path.exists(path):
            raise
        log.warning("model directory %s is incomplete, clearing it and downloading again", path)
        shutil.rmtree(path, ignore_errors=True)
        return loader(name, path, **kwargs)


def _get_model():
    """Load the model once per process, downloading it into MODEL_DIR on first use."""
    global _MODEL
    if _MODEL is None:
        # Lazy import: importing this module must not pull onnxruntime in.
        import onnx_asr

        root = os.getenv("MODEL_DIR") or "/models"
        log.info("loading %s (int8) from MODEL_DIR subdirectory %s", MODEL_NAME, MODEL_SUBDIR)
        asr = _load(
            onnx_asr.load_model,
            MODEL_NAME,
            os.path.join(root, MODEL_SUBDIR),
            quantization="int8",
            providers=PROVIDERS,
        )
        # VAD is mandatory, not a nicety: the model tops out at ~20-30 s of audio per
        # chunk while calls run for minutes.
        vad = _load(
            onnx_asr.load_vad, VAD_NAME, os.path.join(root, VAD_SUBDIR), providers=PROVIDERS
        )
        _MODEL = asr.with_vad(vad)
    return _MODEL


def transcribe_parakeet(audio_path: str) -> list[Segment]:
    """Transcribe one audio file into VAD-delimited segments with timecodes.

    Decoding happens first so that a missing file fails before a ~1 GB model load.
    """
    waveform = audio.decode(audio_path)
    model = _get_model()
    return [
        Segment(r.start, r.end, r.text)
        for r in model.recognize(waveform, sample_rate=audio.SAMPLE_RATE)
    ]
