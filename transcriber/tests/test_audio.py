"""Offline tests: faster_whisper is shadowed by a fake, so PyAV is never reached."""

import sys
import types

import pytest

from transcriber import audio


def test_missing_file_raises_before_importing_faster_whisper(monkeypatch, tmp_path):
    # sys.modules[name] = None makes `import name` raise ImportError, so a decode attempt
    # would blow up as ImportError rather than the FileNotFoundError we assert on.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", None)
    path = str(tmp_path / "gone.webm")

    with pytest.raises(FileNotFoundError) as excinfo:
        audio.decode(path)

    # transcriber.tracks skips a track only when .filename is that track's own path.
    assert excinfo.value.filename == path


def test_decode_passes_path_and_sample_rate(monkeypatch, tmp_path):
    calls = []
    module = types.ModuleType("faster_whisper.audio")
    module.decode_audio = lambda input_file, sampling_rate=16000: (
        calls.append((input_file, sampling_rate)) or "waveform"
    )
    package = types.ModuleType("faster_whisper")
    package.audio = module
    monkeypatch.setitem(sys.modules, "faster_whisper", package)
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", module)
    path = tmp_path / "call.webm"
    path.write_bytes(b"not really audio")

    assert audio.decode(str(path)) == "waveform"
    assert calls == [(str(path), 16000)]
