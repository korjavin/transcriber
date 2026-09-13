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


class FakeVadModel:
    """What with_vad() really returns: a separate adapter, not the model it came from."""

    def __init__(self, asr, vad):
        self.asr = asr
        self.vad = vad
        self.calls = []

    def recognize(self, waveform, **kwargs):
        self.calls.append((waveform, kwargs))
        return iter(
            [
                FakeSegmentResult(5.4, 9.0, " hello"),
                FakeSegmentResult(9.0, 12.0, "world"),
            ]
        )


class FakeModel:
    """Stands in for the onnx-asr adapter: load_model(...).with_vad(vad).recognize(...)."""

    loads: ClassVar[list] = []
    vads: ClassVar[list] = []

    def __init__(self, args, kwargs):
        FakeModel.loads.append((args, kwargs))

    def with_vad(self, vad):
        FakeModel.vads.append(vad)
        return FakeVadModel(self, vad)

    def recognize(self, waveform, **kwargs):
        # Keeping this model instead of what with_vad() returned means no VAD at all, and
        # the real adapter would hand back a plain string here rather than segments.
        raise AssertionError("recognize() reached the model: with_vad()'s return was dropped")


@pytest.fixture(autouse=True)
def fake_onnx_asr(monkeypatch, tmp_path):
    module = types.ModuleType("onnx_asr")
    module.load_model = lambda *args, **kwargs: FakeModel(args, kwargs)
    module.load_vad = lambda name, path=None, **kwargs: f"vad:{name}:{path}:{kwargs.get('providers')}"
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
    # The VAD lives on the same volume, so a cold start needs no network, and both graphs
    # must stay on the CPU provider.
    vad = f"vad:silero:{os.path.join(str(tmp_path), 'silero-vad')}:['CPUExecutionProvider']"
    assert FakeModel.vads == [vad]
    assert p._MODEL.vad == vad


def test_model_dir_defaults_to_models(decoded, monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    p.transcribe_parakeet("call.webm")
    assert FakeModel.loads[0][0][1] == "/models/parakeet-tdt-0.6b-v3"


def test_model_dir_is_not_pre_created(decoded, tmp_path):
    # An existing local dir makes onnx-asr treat the model as a complete offline copy
    # and never download it, so the first load must find the path absent.
    p.transcribe_parakeet("call.webm")
    assert not (tmp_path / "parakeet-tdt-0.6b-v3").exists()
    assert not (tmp_path / "silero-vad").exists()


def test_an_incomplete_model_dir_is_cleared_and_downloaded_again(monkeypatch, tmp_path):
    # A download killed part-way (a restarted container) leaves the directory present but
    # half-filled. onnx-asr would then read it as a complete offline copy and fail every
    # start from then on, so the load has to clear it and fetch again rather than give up.
    partial = tmp_path / "parakeet-tdt-0.6b-v3"
    partial.mkdir()
    (partial / "encoder-model.int8.onnx.incomplete").write_bytes(b"half a model")
    attempts = []

    def loader(name, path, **kwargs):
        attempts.append(os.path.exists(path))
        if len(attempts) == 1:
            raise FileNotFoundError(f"File 'encoder-model.int8.onnx' not found in path {path!r}")
        return "loaded"

    assert p._load(loader, "nemo", str(partial)) == "loaded"
    # First attempt saw the stale directory, the retry saw it gone.
    assert attempts == [True, False]


def test_a_missing_model_dir_does_not_retry(tmp_path):
    # Nothing to clear means nothing to heal: the real failure must surface, not be retried.
    attempts = []

    def loader(name, path, **kwargs):
        attempts.append(path)
        raise FileNotFoundError("the hub is unreachable")

    with pytest.raises(FileNotFoundError):
        p._load(loader, "nemo", str(tmp_path / "absent"))
    assert len(attempts) == 1


def test_missing_audio_fails_before_the_model_is_loaded(tmp_path):
    path = str(tmp_path / "gone.webm")

    with pytest.raises(FileNotFoundError) as excinfo:
        p.transcribe_parakeet(path)

    assert excinfo.value.filename == path
    assert FakeModel.loads == []


def test_import_does_not_need_onnx_asr_or_construct_a_model(monkeypatch):
    import transcriber

    # Re-importing rebinds the submodule on the package too, and `from a.b import c` reads
    # that attribute in preference to sys.modules. Without this the throwaway module below
    # would leak into every later test that imports this one lazily.
    monkeypatch.setattr(transcriber, "asr_parakeet", p)
    # sys.modules[name] = None makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, "onnx_asr", None)
    monkeypatch.delitem(sys.modules, "transcriber.asr_parakeet")

    module = importlib.import_module("transcriber.asr_parakeet")

    assert module._MODEL is None
