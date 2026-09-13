"""Per-participant track transcription merged into one speaker-attributed feed.

No diarization: jitsi-capture records one file per participant, so the speaker's
name comes straight from the track. Each track is transcribed on its own, shifted
by the participant's `offset_s` and merged back by time.
"""

from __future__ import annotations

import logging
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
