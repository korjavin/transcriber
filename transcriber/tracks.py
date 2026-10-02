"""Per-participant track transcription merged into one speaker-attributed feed.

No diarization: jitsi-capture records one file per participant, so the speaker's
name comes straight from the track. Each track is transcribed on its own, shifted
by the participant's `offset_s` and merged back by time.

Google Meet has no per-participant tracks, only one mixed file; there the names come
from Meet's caption timeline (`speaker_hints_path`) laid over the mixed transcript.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from collections.abc import Callable

from transcriber.transcribe import Segment

log = logging.getLogger(__name__)


def transcribe_tracks(
    tracks: list[dict], transcribe: Callable[[str], list[Segment]]
) -> list[Segment]:
    """Transcribe every track and merge the segments into one feed, sorted by start time.

    `tracks` are the webhook's `tracks[]` items with `path` already rebased by the
    caller — this module does no path or environment work. A track whose own audio is
    missing is skipped — the callable must raise `FileNotFoundError` with `filename` set
    to that path, as `open()` does. Any other transcription error propagates and fails
    the job, as does losing every track (broken input beats an empty transcript).
    """
    merged: list[Segment] = []
    missing = 0
    for track in tracks:
        speaker = (track.get("name") or "").strip() or f"Participant {track.get('id')}"
        try:
            segments = transcribe(track["path"])
        except FileNotFoundError as exc:
            # Only this track's own audio counts as a gap. A FileNotFoundError from inside
            # the ASR stack (a missing model file, say) is a real failure, not a silent skip.
            if exc.filename != track["path"]:
                raise
            log.warning("track %s: audio file is missing, skipping", track.get("id"))
            missing += 1
            continue
        offset = track.get("offset_s") or 0.0
        merged += [
            Segment(s.start + offset, s.end + offset, s.text, speaker=speaker) for s in segments
        ]
    if missing and missing == len(tracks):
        # Every track gone means broken input: fail the job instead of publishing nothing.
        raise FileNotFoundError("none of the track audio files exist")
    # Stable sort: a speaker's own segments keep their order when starts tie.
    merged.sort(key=lambda s: s.start)
    return merged


def coalesce(segments: list[Segment], gap_s: float = 2.0) -> list[Segment]:
    """Join consecutive same-speaker segments closer than `gap_s` into one turn.

    One line per speaker turn instead of one per VAD chunk. Pure function. Segments
    without a speaker (mixed audio) are never joined — an unknown speaker is not
    evidence of the same speaker.
    """
    turns: list[Segment] = []
    for s in segments:
        prev = turns[-1] if turns else None
        if (
            prev is not None
            and prev.speaker is not None
            and prev.speaker == s.speaker
            and s.start - prev.end <= gap_s
        ):
            turns[-1] = Segment(
                prev.start,
                max(prev.end, s.end),
                f"{prev.text.strip()} {s.text.strip()}".strip(),
                speaker=prev.speaker,
            )
        else:
            turns.append(s)
    return turns


def load_hints(path: str) -> list[tuple[float, str]]:
    """Read a Meet `speaker_hints_path` JSONL file into (offset_s, speaker) pairs.

    One `{"offset_s", "speaker", "text"}` object per line; only offset and name are used
    (the text still comes from the audio). Unusable lines are skipped. Sorted by offset.
    """
    hints = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                hint = json.loads(line)
                offset, speaker = float(hint["offset_s"]), hint["speaker"]
            except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
                continue
            if isinstance(speaker, str) and math.isfinite(offset):
                hints.append((offset, speaker.strip()))
    hints.sort(key=lambda h: h[0])  # stable: equal offsets keep file order
    return hints


def apply_hints(segments: list[Segment], hints: list[tuple[float, str]]) -> list[Segment]:
    """Name each segment after the hint whose window overlaps it most.

    Hint i covers [offset_i, offset_{i+1}); the last one runs to the end of the audio.
    A segment with no overlap (zero-length, say) takes the hint covering its start.
    Meet's '?' (or an empty name) means no name, and so does a segment before the
    first hint. Pure function.
    """
    offsets = [h[0] for h in hints]
    named = []
    for s in segments:
        best, best_overlap = None, 0.0
        first = max(bisect.bisect_right(offsets, s.start) - 1, 0)
        for i in range(first, bisect.bisect_left(offsets, s.end)):
            win_end = offsets[i + 1] if i + 1 < len(offsets) else math.inf
            overlap = min(s.end, win_end) - max(s.start, offsets[i])
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        if best is None and offsets and offsets[0] <= s.start:
            best = bisect.bisect_right(offsets, s.start) - 1
        speaker = hints[best][1] if best is not None else ""
        named.append(
            Segment(s.start, s.end, s.text, speaker=speaker if speaker not in ("", "?") else None)
        )
    return named
