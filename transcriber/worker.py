"""The pipeline: one job at a time through queued -> transcribing -> publishing -> done.

State lives on disk (`transcriber.jobs`), so every step is resumable: a restart picks
up whatever was unfinished and skips the work already on disk — transcript.md means the
ASR is done, `outline_url` means tr2outline already has the document.

A failed job is never retried automatically; re-sending the webhook re-queues it.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time

from transcriber import jobs, publish, tracks
from transcriber import transcribe as transcribe_module

log = logging.getLogger(__name__)


def run_job(
    job_id: str,
    *,
    transcribe=transcribe_module.transcribe,
    transcribe_tracks=tracks.transcribe_tracks,
    send=publish.send_to_tr2outline,
    callback=publish.post_callback,
) -> None:
    """Run one job to completion (or to `failed`). Dependencies are keyword arguments
    so tests inject fakes without monkeypatching modules."""
    stage = "loading"
    try:
        job = jobs.load_job(job_id)
        webhook = jobs.load_webhook(job_id)
        transcript = jobs.job_dir(job_id) / "transcript.md"

        if job["state"] in ("queued", "transcribing"):
            stage = "transcribing"
            # transcript.md is written only once the ASR has finished, so its presence
            # means the transcription is done — a webhook re-sent for a job that failed
            # while publishing (the receiver re-queues it as `queued`) never re-runs the
            # hour of CPU work.
            if not transcript.exists():
                jobs.set_state(job_id, "transcribing")
                items = webhook.get("tracks") or []
                engine = os.getenv("ASR_ENGINE") or "parakeet"
                log.info(
                    "job %s: transcription started (engine=%s, %s)",
                    job_id,
                    engine,
                    f"tracks={len(items)}" if items else "mixed",
                )
                started = time.monotonic()
                if items:
                    # Rebasing is the caller's job: tracks.py does no environment work.
                    rebased = [{**t, "path": jobs.rebase_path(t["path"])} for t in items]
                    segments = tracks.coalesce(transcribe_tracks(rebased, transcribe))
                else:
                    segments = transcribe(jobs.rebase_path(webhook["audio_path"]))
                # jobs' atomic writer: a half-written transcript.md would be read back as
                # the finished text by the very next resume.
                jobs._write_atomic(transcript, transcribe_module.to_markdown(segments).encode())
                log.info(
                    "job %s: transcription finished (%.1fs, %d segments)",
                    job_id,
                    time.monotonic() - started,
                    len(segments),
                )
            job = jobs.set_state(job_id, "publishing")

        if job["state"] == "publishing":
            stage = "publishing"
            text = transcript.read_text(encoding="utf-8")
            url, title = job.get("outline_url"), job.get("outline_title")
            if not url:
                url, title = send(publish.build_anarlog_payload(webhook, text))
                job = jobs.set_state(job_id, "publishing", outline_url=url, outline_title=title)
            callback(webhook["callback_url"], job_id, publish.ready_message(title, url))
            jobs.set_state(job_id, "done")
            log.info("job %s: done", job_id)
    except Exception as exc:
        log.exception("job %s failed while %s", job_id, stage)
        # Bounded: the text lands in job.json. publish keeps secrets out of its exceptions;
        # paths are fine here.
        jobs.set_state(job_id, "failed", error=f"{stage}: {type(exc).__name__}: {exc}"[:500])


class Worker:
    """A single background thread draining a queue of job ids.

    # ponytail: one thread == the CPU lock; add a second worker only if the box has
    cores to spare and jobs pile up.
    """

    def __init__(self, run=run_job):
        self._run = run
        self.q: queue.Queue[str] = queue.Queue()
        self.t = threading.Thread(target=self._loop, name="worker", daemon=True)

    def start(self) -> None:
        """Queue everything left over from the last run, then start consuming."""
        for job_id in jobs.list_unfinished():
            self.q.put(job_id)
        self.t.start()

    def enqueue(self, job_id: str) -> None:
        self.q.put(job_id)

    def _loop(self) -> None:
        while True:
            job_id = self.q.get()
            try:
                self._run(job_id)
            except Exception:
                # run_job records its own failures; this only catches a failure to
                # record one (an unreadable job.json). A dead thread would silently
                # swallow every later job.
                log.exception("job %s: worker could not complete the job", job_id)
