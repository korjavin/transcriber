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
            "/data/gone.webm": FileNotFoundError(2, "No such file", "/data/gone.webm"),
            "/data/bob.webm": [Segment(1.0, 2.0, "Still here")],
        }
    )

    with caplog.at_level(logging.WARNING):
        segments = transcribe_tracks(tracks, transcribe)

    assert segments == [Segment(1.0, 2.0, "Still here", speaker="Bob")]
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "t1" in caplog.text
    assert "/data/gone.webm" not in caplog.text


def test_a_missing_file_from_inside_the_asr_stack_is_not_swallowed():
    # A model file the ASR stack cannot open is a real failure, not a missing track.
    tracks = [{"id": "t1", "name": "Alice", "path": "/data/alice.webm", "offset_s": 0}]
    transcribe = fake_transcribe(
        {"/data/alice.webm": FileNotFoundError(2, "No such file", "/models/model.onnx")}
    )

    with pytest.raises(FileNotFoundError):
        transcribe_tracks(tracks, transcribe)


def test_all_tracks_missing_fails_the_job():
    tracks = [
        {"id": "t1", "name": "Alice", "path": "/data/a.webm"},
        {"id": "t2", "name": "Bob", "path": "/data/b.webm"},
    ]
    transcribe = fake_transcribe(
        {
            "/data/a.webm": FileNotFoundError(2, "No such file", "/data/a.webm"),
            "/data/b.webm": FileNotFoundError(2, "No such file", "/data/b.webm"),
        }
    )

    with pytest.raises(FileNotFoundError):
        transcribe_tracks(tracks, transcribe)


def test_tracks_are_merged_by_time_not_by_track_order():
    tracks = [
        {"id": "t1", "name": "Bob", "path": "/data/bob.webm", "offset_s": 30},
        {"id": "t2", "name": "Alice", "path": "/data/alice.webm", "offset_s": 0},
    ]
    transcribe = fake_transcribe(
        {
            "/data/bob.webm": [Segment(0.0, 2.0, "Late")],
            "/data/alice.webm": [Segment(1.0, 2.0, "Early"), Segment(40.0, 41.0, "Latest")],
        }
    )

    assert [(s.start, s.speaker) for s in transcribe_tracks(tracks, transcribe)] == [
        (1.0, "Alice"),
        (30.0, "Bob"),
        (40.0, "Alice"),
    ]


def test_equal_starts_keep_track_order():
    tracks = [
        {"id": "t1", "name": "Alice", "path": "/data/alice.webm", "offset_s": 0},
        {"id": "t2", "name": "Bob", "path": "/data/bob.webm", "offset_s": 0},
    ]
    transcribe = fake_transcribe(
        {
            "/data/alice.webm": [Segment(4.0, 5.0, "A1"), Segment(4.0, 6.0, "A2")],
            "/data/bob.webm": [Segment(4.0, 5.0, "B1")],
        }
    )

    assert [s.text for s in transcribe_tracks(tracks, transcribe)] == ["A1", "A2", "B1"]


def test_blank_name_falls_back_to_participant_id():
    tracks = [{"id": "t7", "name": "   ", "path": "/data/x.webm"}]
    transcribe = fake_transcribe({"/data/x.webm": [Segment(0.0, 1.0, "Hello")]})

    assert transcribe_tracks(tracks, transcribe)[0].speaker == "Participant t7"


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


def test_a_gap_of_exactly_gap_s_still_joins():
    segments = [
        Segment(0.0, 3.0, "Before", speaker="Alice"),
        Segment(5.0, 6.0, "after", speaker="Alice"),
        Segment(8.01, 9.0, "too late", speaker="Alice"),
    ]
    assert coalesce(segments, gap_s=2.0) == [
        Segment(0.0, 6.0, "Before after", speaker="Alice"),
        Segment(8.01, 9.0, "too late", speaker="Alice"),
    ]


def test_overlapping_same_speaker_segments_keep_the_later_end():
    segments = [
        Segment(0.0, 9.0, "Long", speaker="Alice"),
        Segment(1.0, 4.0, "inner", speaker="Alice"),
    ]
    assert coalesce(segments) == [Segment(0.0, 9.0, "Long inner", speaker="Alice")]


def test_segments_without_a_speaker_are_never_joined():
    segments = [Segment(0.0, 3.0, "One"), Segment(4.0, 6.0, "Two")]
    assert coalesce(segments) == segments


def test_coalesce_empty():
    assert coalesce([]) == []
