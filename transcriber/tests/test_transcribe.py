"""Offline tests: the real faster_whisper is shadowed by a fake, so no model is ever loaded."""

import importlib
import sys
import types
from typing import ClassVar

import pytest

from transcriber import transcribe as t

WHISPER_ENV = ("WHISPER_MODEL", "WHISPER_LANGUAGE", "WHISPER_DEVICE", "WHISPER_COMPUTE_TYPE")


class FakeModel:
    instances: ClassVar[list] = []

    def __init__(self, model, device=None, compute_type=None):
        self.init = (model, device, compute_type)
        self.calls = []
        FakeModel.instances.append(self)

    def transcribe(self, audio_path, **kwargs):
        self.calls.append((audio_path, kwargs))
        segments = iter(
            [
                types.SimpleNamespace(start=5.4, end=9.0, text=" hello"),
                types.SimpleNamespace(start=9.0, end=12.0, text="world"),
            ]
        )
        return segments, types.SimpleNamespace(language="en")


@pytest.fixture(autouse=True)
def fake_whisper(monkeypatch):
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(t, "_MODELS", {})
    FakeModel.instances.clear()
    for name in WHISPER_ENV:
        monkeypatch.delenv(name, raising=False)
    # This file is about the whisper backend; parakeet is the default engine elsewhere.
    monkeypatch.setenv("ASR_ENGINE", "whisper")


def seg(start, text="x", end=None):
    return t.Segment(start=start, end=start if end is None else end, text=text)


def test_transcribe_maps_segments():
    assert t.transcribe("call.webm") == [
        t.Segment(start=5.4, end=9.0, text=" hello"),
        t.Segment(start=9.0, end=12.0, text="world"),
    ]


def test_defaults_come_from_fallbacks():
    t.transcribe("call.webm")
    model = FakeModel.instances[0]
    assert model.init == ("large-v3", "cpu", "int8")
    assert model.calls == [
        ("call.webm", {"language": None, "vad_filter": True, "beam_size": 5})
    ]


def test_env_overrides_device_and_compute_type(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "medium")
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float16")
    monkeypatch.setenv("WHISPER_LANGUAGE", "ru")

    t.transcribe("call.webm")

    assert FakeModel.instances[0].init == ("medium", "cuda", "float16")
    assert FakeModel.instances[0].calls[0][1]["language"] == "ru"


def test_arguments_beat_env(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "medium")
    monkeypatch.setenv("WHISPER_LANGUAGE", "ru")

    t.transcribe("call.webm", model="small", language="en")

    assert FakeModel.instances[0].init[0] == "small"
    assert FakeModel.instances[0].calls[0][1]["language"] == "en"


def test_model_is_cached_per_configuration():
    t.transcribe("a.webm")
    t.transcribe("b.webm")
    assert len(FakeModel.instances) == 1

    t.transcribe("c.webm", model="small")
    assert len(FakeModel.instances) == 2


@pytest.fixture
def routed(monkeypatch):
    """Record which backend transcribe() dispatched to, without running either."""
    calls = []
    monkeypatch.setattr(
        t, "transcribe_whisper", lambda path, **kw: calls.append(("whisper", path, kw))
    )
    module = types.ModuleType("transcriber.asr_parakeet")
    module.transcribe_parakeet = lambda path, **kw: calls.append(("parakeet", path, kw))
    monkeypatch.setitem(sys.modules, "transcriber.asr_parakeet", module)
    return calls


def test_engine_defaults_to_parakeet(monkeypatch):
    # Patched on the real module rather than a sys.modules fake, so the dispatcher's own
    # import line runs: a circular import between these two modules would fail here.
    calls = []
    monkeypatch.setattr(
        "transcriber.asr_parakeet.transcribe_parakeet", lambda path: calls.append(path)
    )
    monkeypatch.delenv("ASR_ENGINE", raising=False)

    t.transcribe("call.webm")

    assert calls == ["call.webm"]


def test_env_selects_whisper(routed, monkeypatch):
    monkeypatch.setenv("ASR_ENGINE", "whisper")
    t.transcribe("call.webm", model="small")
    assert routed == [("whisper", "call.webm", {"model": "small"})]


def test_engine_argument_beats_env(routed, monkeypatch):
    monkeypatch.setenv("ASR_ENGINE", "whisper")
    t.transcribe("call.webm", engine="parakeet")
    assert routed == [("parakeet", "call.webm", {})]


def test_unknown_engine_raises(routed, monkeypatch):
    monkeypatch.setenv("ASR_ENGINE", "vosk")
    with pytest.raises(ValueError, match="vosk"):
        t.transcribe("call.webm")
    assert routed == []


def test_whisper_options_are_rejected_by_parakeet(monkeypatch):
    monkeypatch.delenv("ASR_ENGINE", raising=False)
    module = types.ModuleType("transcriber.asr_parakeet")
    module.transcribe_parakeet = lambda path: []
    monkeypatch.setitem(sys.modules, "transcriber.asr_parakeet", module)

    with pytest.raises(TypeError):
        t.transcribe("call.webm", model="small")


def test_to_markdown_timecodes():
    assert t.to_markdown([seg(5, "Hi"), seg(62, "Later")]) == "[00:05] Hi\n[01:02] Later"


def test_to_markdown_switches_to_hours():
    assert t.to_markdown([seg(3723, "Deep")]) == "[1:02:03] Deep"


def test_to_markdown_skips_blank_and_strips():
    assert t.to_markdown([seg(5, " Hi "), seg(7, "   "), seg(9, "")]) == "[00:05] Hi"


def test_to_markdown_empty():
    assert t.to_markdown([]) == ""


def test_import_does_not_need_or_construct_a_model(monkeypatch):
    import transcriber

    # Re-importing rebinds the submodule on the package too, and `from a.b import c` reads
    # that attribute in preference to sys.modules, so put the real one back afterwards.
    monkeypatch.setattr(transcriber, "transcribe", t)
    # sys.modules[name] = None makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.delitem(sys.modules, "transcriber.transcribe")

    module = importlib.import_module("transcriber.transcribe")

    assert module._MODELS == {}


def test_cli_prints_markdown(capsys):
    assert t.main(["call.webm"]) == 0
    assert capsys.readouterr().out == "[00:05] hello\n[00:09] world\n"


def test_cli_rejects_bad_usage(capsys):
    assert t.main([]) == 2
    assert "usage:" in capsys.readouterr().err
