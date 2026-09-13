"""Offline tests: the real onnx_asr is shadowed by a fake, so no model is ever downloaded."""

import importlib
import os
import sys
import types
from typing import ClassVar

import pytest

from transcriber import asr_parakeet as p
from transcriber.transcribe import Segment


class FakeSegmentResult:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class FakeModel:
    """Stands in for the onnx-asr adapter: load_model(...).with_vad(vad).recognize(...)."""

    loads: ClassVar[list] = []
    vads: ClassVar[list] = []

    def __init__(self, args, kwargs):
        FakeModel.loads.append((args, kwargs))
        self.vad = None
        self.calls = []

    def with_vad(self, vad):
        self.vad = vad
        FakeModel.vads.append(vad)
        return self

    def recognize(self, waveform, **kwargs):
        self.calls.append((waveform, kwargs))
        return iter(
            [
                FakeSegmentResult(5.4, 9.0, " hello"),
                FakeSegmentResult(9.0, 12.0, "world"),
            ]
        )


@pytest.fixture(autouse=True)
def fake_onnx_asr(monkeypatch, tmp_path):
    module = types.ModuleType("onnx_asr")
    module.load_model = lambda *args, **kwargs: FakeModel(args, kwargs)
    module.load_vad = lambda name, **kwargs: f"vad:{name}:{kwargs.get('providers')}"
    monkeypatch.setitem(sys.modules, "onnx_asr", module)
    monkeypatch.setattr(p, "_MODEL", None)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    FakeModel.loads.clear()
    FakeModel.vads.clear()


@pytest.fixture
def decoded(monkeypatch):
    paths = []
    monkeypatch.setattr(p.audio, "decode", lambda path: paths.append(path) or "waveform")
    return paths


def test_maps_segment_results_to_segments(decoded):
    assert p.transcribe_parakeet("call.webm") == [
        Segment(start=5.4, end=9.0, text=" hello"),
        Segment(start=9.0, end=12.0, text="world"),
    ]
    assert decoded == ["call.webm"]


def test_recognizes_the_decoded_waveform_at_16khz(decoded):
    p.transcribe_parakeet("call.webm")
    assert p._MODEL.calls == [("waveform", {"sample_rate": 16000})]


def test_model_is_loaded_once_with_int8_and_vad(decoded, tmp_path):
    p.transcribe_parakeet("a.webm")
    p.transcribe_parakeet("b.webm")

    assert FakeModel.loads == [
        (
            (p.MODEL_NAME, os.path.join(str(tmp_path), "parakeet-tdt-0.6b-v3")),
            {"quantization": "int8", "providers": ["CPUExecutionProvider"]},
        )
    ]
    # Both graphs must stay on the CPU provider, the VAD's included.
    assert FakeModel.vads == ["vad:silero:['CPUExecutionProvider']"]
    assert p._MODEL.vad == "vad:silero:['CPUExecutionProvider']"


def test_model_dir_defaults_to_models(decoded, monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    p.transcribe_parakeet("call.webm")
    assert FakeModel.loads[0][0][1] == "/models/parakeet-tdt-0.6b-v3"


def test_model_dir_is_not_pre_created(decoded, tmp_path):
    # An existing local dir makes onnx-asr treat the model as a complete offline copy
    # and never download it, so the first load must find the path absent.
    p.transcribe_parakeet("call.webm")
    assert not (tmp_path / "parakeet-tdt-0.6b-v3").exists()


def test_missing_audio_fails_before_the_model_is_loaded(tmp_path):
    path = str(tmp_path / "gone.webm")

    with pytest.raises(FileNotFoundError) as excinfo:
        p.transcribe_parakeet(path)

    assert excinfo.value.filename == path
    assert FakeModel.loads == []


def test_import_does_not_need_onnx_asr_or_construct_a_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx_asr", None)
    monkeypatch.delitem(sys.modules, "transcriber.asr_parakeet")

    module = importlib.import_module("transcriber.asr_parakeet")

    assert module._MODEL is None
