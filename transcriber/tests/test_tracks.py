"""Pure-function tests: transcription is a fake callable, no model and no files."""

import logging

import pytest

from transcriber.tracks import coalesce, transcribe_tracks
from transcriber.transcribe import Segment, to_markdown


def fake_transcribe(by_path):
    """Build a transcribe callable from {path: segments-or-exception}."""

    def transcribe(path):
        result = by_path[path]
        if isinstance(result, Exception):
            raise result
        return result

    return transcribe


def test_two_tracks_are_shifted_merged_and_named():
    tracks = [
        {"id": "t1", "name": "Alice", "path": "/data/alice.webm", "offset_s": 0},
        {"id": "t2", "name": "Bob", "path": "/data/bob.webm", "offset_s": 12.5},
    ]
    transcribe = fake_transcribe(
        {
            "/data/alice.webm": [Segment(5.0, 8.0, "Hi everyone")],
            "/data/bob.webm": [Segment(5.5, 9.0, "On the second point.")],
        }
    )

    segments = transcribe_tracks(tracks, transcribe)

    assert segments == [
        Segment(5.0, 8.0, "Hi everyone", speaker="Alice"),
        Segment(18.0, 21.5, "On the second point.", speaker="Bob"),
    ]
    assert to_markdown(segments) == "[00:05] Alice: Hi everyone\n[00:18] Bob: On the second point."


def test_missing_track_file_is_skipped_and_logged(caplog):
    tracks = [
        {"id": "t1", "name": "Alice", "path": "/data/gone.webm", "offset_s": 0},
        {"id": "t2", "name": "Bob", "path": "/data/bob.webm", "offset_s": 0},
    ]
    transcribe = fake_transcribe(
        {
            "/data/gone.webm": FileNotFoundError("/data/gone.webm"),
            "/data/bob.webm": [Segment(1.0, 2.0, "Still here")],
        }
    )

    with caplog.at_level(logging.WARNING):
        segments = transcribe_tracks(tracks, transcribe)

    assert segments == [Segment(1.0, 2.0, "Still here", speaker="Bob")]
    assert "t1" in caplog.text
    assert "/data/gone.webm" not in caplog.text


def test_other_exceptions_propagate():
    tracks = [{"id": "t1", "name": "Alice", "path": "/data/alice.webm", "offset_s": 0}]
    transcribe = fake_transcribe({"/data/alice.webm": RuntimeError("model blew up")})

    with pytest.raises(RuntimeError):
        transcribe_tracks(tracks, transcribe)


def test_missing_name_falls_back_to_participant_id():
    tracks = [{"id": "t7", "name": "", "path": "/data/x.webm"}]
    transcribe = fake_transcribe({"/data/x.webm": [Segment(0.0, 1.0, "Hello")]})

    assert transcribe_tracks(tracks, transcribe)[0].speaker == "Participant t7"


def test_no_tracks_is_empty():
    assert transcribe_tracks([], fake_transcribe({})) == []


def test_same_speaker_within_the_gap_is_joined():
    segments = [
        Segment(0.0, 3.0, "First half", speaker="Alice"),
        Segment(4.0, 6.0, "second half", speaker="Alice"),
    ]
    assert coalesce(segments) == [Segment(0.0, 6.0, "First half second half", speaker="Alice")]


def test_different_speakers_stay_separate():
    segments = [
        Segment(0.0, 3.0, "Mine", speaker="Alice"),
        Segment(4.0, 6.0, "Yours", speaker="Bob"),
    ]
    assert coalesce(segments) == segments


def test_a_long_pause_stays_separate():
    segments = [
        Segment(0.0, 3.0, "Before", speaker="Alice"),
        Segment(10.0, 12.0, "After", speaker="Alice"),
    ]
    assert coalesce(segments) == segments


def test_coalesce_empty():
    assert coalesce([]) == []
