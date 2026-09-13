"""Pipeline tests: real job store on tmp_path, every boundary injected as a fake."""

import json
import logging
import threading
import time

import pytest

from transcriber import jobs, worker
from transcriber.publish import PublishError
from transcriber.transcribe import Segment

JOB_ID = "123456"
CALLBACK_URL = "http://jitsi-capture:8080/notify"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("HOST_DATA_DIR", raising=False)
    return tmp_path


def make_job(tmp_path, state="queued", **job_fields):
    """Write webhook.json + job.json the way the receiver does."""
    webhook = {
        "event": "recording.finished",
        "id": JOB_ID,
        "audio_path": str(tmp_path / "audio.webm"),
        "callback_url": CALLBACK_URL,
        "topic": "standup",
        "participants": ["Alice", "Bob"],
        "ended_at": "2026-09-13T10:00:00Z",
        **job_fields.pop("webhook", {}),
    }
    jobs.save_webhook(JOB_ID, json.dumps(webhook).encode())
    jobs.save_job(
        {
            "id": JOB_ID,
            "state": state,
            "error": "",
            "received_at": jobs.now_rfc3339(),
            "outline_url": "",
            "outline_title": "",
            **job_fields,
        }
    )
    return webhook


def boom(*_args, **_kwargs):
    raise AssertionError("must not be called")


def test_mixed_audio_job_runs_end_to_end(tmp_path):
    make_job(tmp_path)
    seen = {}

    def transcribe(path):
        seen["path"] = path
        return [Segment(0.0, 2.0, "hello there")]

    def send(payload):
        seen["payload"] = payload
        return "https://outline.example/doc/abc", "2026-09-13 standup"

    def callback(url, job_id, content):
        seen["callback"] = (url, job_id, content)

    worker.run_job(
        JOB_ID, transcribe=transcribe, transcribe_tracks=boom, send=send, callback=callback
    )

    assert seen["path"] == str(tmp_path / "audio.webm")
    assert (jobs.job_dir(JOB_ID) / "transcript.md").read_text() == "[00:00] hello there"
    assert seen["payload"]["data"]["transcript_text"] == "[00:00] hello there"
    assert seen["callback"] == (
        CALLBACK_URL,
        JOB_ID,
        "Transcript ready: [2026-09-13 standup](https://outline.example/doc/abc)",
    )
    job = jobs.load_job(JOB_ID)
    assert job["state"] == "done"
    assert job["outline_url"] == "https://outline.example/doc/abc"
    assert job["outline_title"] == "2026-09-13 standup"


def test_tracks_are_rebased_attributed_and_coalesced(tmp_path, monkeypatch):
    monkeypatch.setenv("HOST_DATA_DIR", "/host/data")
    make_job(
        tmp_path,
        webhook={
            "tracks": [
                {"id": "1", "name": "Alice", "path": "/host/data/rec/alice.webm", "offset_s": 3.0}
            ]
        },
    )
    seen = {}

    def transcribe(path):
        seen["path"] = path
        # Two adjacent turns from the same speaker: coalesce must join them.
        return [Segment(0.0, 1.0, "one"), Segment(1.5, 2.0, "two")]

    worker.run_job(
        JOB_ID,
        transcribe=transcribe,
        send=lambda payload: ("https://outline.example/doc/abc", "title"),
        callback=lambda *args: None,
    )

    assert seen["path"] == str(tmp_path / "rec/alice.webm")
    # offset_s shifts the turn to 3.0s, and the two segments become one line.
    assert (jobs.job_dir(JOB_ID) / "transcript.md").read_text() == "[00:03] Alice: one two"
    assert jobs.load_job(JOB_ID)["state"] == "done"


def test_resume_in_publishing_skips_asr_and_tr2outline(tmp_path):
    make_job(
        tmp_path, state="publishing", outline_url="https://outline.example/d", outline_title="t"
    )
    (jobs.job_dir(JOB_ID) / "transcript.md").write_text("[00:00] already transcribed")
    seen = {}

    worker.run_job(
        JOB_ID,
        transcribe=boom,
        transcribe_tracks=boom,
        send=boom,
        callback=lambda url, job_id, content: seen.update(content=content),
    )

    assert seen["content"] == "Transcript ready: [t](https://outline.example/d)"
    assert jobs.load_job(JOB_ID)["state"] == "done"


def test_requeued_job_with_a_transcript_is_not_transcribed_again(tmp_path):
    """The receiver re-queues a failed job as `queued`; a finished transcript stands."""
    make_job(tmp_path, state="queued")
    (jobs.job_dir(JOB_ID) / "transcript.md").write_text("[00:00] Alice: privet", encoding="utf-8")
    seen = {}

    worker.run_job(
        JOB_ID,
        transcribe=boom,
        transcribe_tracks=boom,
        send=lambda payload: seen.update(payload=payload) or ("https://outline.example/d", "t"),
        callback=lambda *args: None,
    )

    assert seen["payload"]["data"]["transcript_text"] == "[00:00] Alice: privet"
    assert jobs.load_job(JOB_ID)["state"] == "done"


def test_resume_in_transcribing_without_transcript_starts_over(tmp_path):
    make_job(tmp_path, state="transcribing")
    calls = []

    worker.run_job(
        JOB_ID,
        transcribe=lambda path: calls.append(path) or [Segment(0.0, 1.0, "again")],
        send=lambda payload: ("https://outline.example/d", "t"),
        callback=lambda *args: None,
    )

    assert calls == [str(tmp_path / "audio.webm")]
    assert (jobs.job_dir(JOB_ID) / "transcript.md").read_text() == "[00:00] again"
    assert jobs.load_job(JOB_ID)["state"] == "done"


def test_transcription_failure_is_recorded_and_nothing_is_published(tmp_path):
    make_job(tmp_path)

    def transcribe(path):
        raise RuntimeError("model exploded")

    worker.run_job(JOB_ID, transcribe=transcribe, send=boom, callback=boom)

    job = jobs.load_job(JOB_ID)
    assert job["state"] == "failed"
    assert job["error"].startswith("transcribing: RuntimeError: model exploded")
    assert not (jobs.job_dir(JOB_ID) / "transcript.md").exists()


def test_publish_failure_is_recorded(tmp_path):
    make_job(tmp_path)

    def send(payload):
        raise PublishError("tr2outline rejected job 123456: HTTP 400")

    worker.run_job(
        JOB_ID, transcribe=lambda path: [Segment(0.0, 1.0, "hi")], send=send, callback=boom
    )

    job = jobs.load_job(JOB_ID)
    assert job["state"] == "failed"
    assert job["error"].startswith("publishing: PublishError:")


def test_unreadable_job_does_not_kill_the_worker_thread(tmp_path):
    """run_job cannot even record the failure of a job with no job.json."""
    done = threading.Event()
    w = worker.Worker(
        run=lambda job_id: done.set() if job_id == "second" else worker.run_job(job_id)
    )
    w.start()
    w.enqueue("unknown-job")
    w.enqueue("second")
    assert done.wait(5)


def test_worker_resumes_unfinished_jobs_in_order_and_serially(tmp_path):
    for index, job_id in enumerate(("job-a", "job-b", "job-c")):
        jobs.save_job(
            {"id": job_id, "state": "queued", "received_at": f"2026-09-13T00:0{index}:00Z"}
        )
    jobs.save_job({"id": "job-done", "state": "done", "received_at": "2026-09-13T00:00:00Z"})

    finished, running, overlaps = [], [], []
    lock = threading.Lock()
    done = threading.Event()

    def run(job_id):
        with lock:
            running.append(job_id)
            if len(running) > 1:
                overlaps.append(list(running))
        time.sleep(0.01)
        with lock:
            running.remove(job_id)
        finished.append(job_id)
        if len(finished) == 3:
            done.set()

    w = worker.Worker(run=run)
    w.start()

    assert done.wait(5)
    time.sleep(0.05)  # let a fourth job, if one was wrongly queued, show up
    assert finished == ["job-a", "job-b", "job-c"]  # done jobs are not resumed
    assert overlaps == []
    assert w.q.empty()


def test_lifecycle_lines_are_logged_at_info(tmp_path, monkeypatch, caplog):
    """The deploy-time breadcrumb: one INFO line per step, each carrying the job id."""
    monkeypatch.setenv("ASR_ENGINE", "whisper")
    make_job(tmp_path)
    caplog.set_level(logging.INFO, logger="transcriber.worker")

    worker.run_job(
        JOB_ID,
        transcribe=lambda path: [Segment(0.0, 2.0, "hello there")],
        transcribe_tracks=boom,
        send=lambda payload: ("https://outline.example/doc/abc", "title"),
        callback=lambda *args: None,
    )

    lines = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    assert f"job {JOB_ID}: transcription started (engine=whisper, mixed)" in lines
    finished = next(line for line in lines if "transcription finished" in line)
    assert finished.startswith(f"job {JOB_ID}: transcription finished (")
    assert finished.endswith("1 segments)")
    assert f"job {JOB_ID}: done" in lines


def test_tracks_job_logs_its_track_count(tmp_path, caplog):
    make_job(tmp_path, webhook={"tracks": [{"id": "1", "name": "Alice", "path": "/data/a.webm"}]})
    caplog.set_level(logging.INFO, logger="transcriber.worker")

    worker.run_job(
        JOB_ID,
        transcribe=lambda path: [Segment(0.0, 1.0, "one")],
        send=lambda payload: ("https://outline.example/doc/abc", "title"),
        callback=lambda *args: None,
    )

    assert f"job {JOB_ID}: transcription started (engine=parakeet, tracks=1)" in [
        r.getMessage() for r in caplog.records
    ]


def test_main_exits_2_and_names_the_missing_variable(monkeypatch, caplog):
    monkeypatch.setenv("WEBHOOK_SECRET", "change-me")
    monkeypatch.setenv("TR2OUTLINE_URL", "http://tr2outline:8080/webhook")
    monkeypatch.delenv("ANARLOG_WEBHOOK_SECRET", raising=False)
    from transcriber.__main__ import main

    assert main() == 2
    assert "ANARLOG_WEBHOOK_SECRET" in caplog.text
    assert "change-me" not in caplog.text
