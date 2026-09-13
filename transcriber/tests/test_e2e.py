"""The whole service offline: a signed webhook in, a published document and a callback out.

Everything here is the shipping code except the ASR and the two peers — the receiver, the
worker, the job store on disk, the track merge and both signed bodies are real. The fake
tr2outline/notify server is the one the publish tests already run on.
"""

import json
import sys
import threading
import time
import urllib.request
from functools import partial
from pathlib import Path

import pytest

from transcriber import jobs, publish, server, worker
from transcriber.publish import build_anarlog_payload
from transcriber.signing import sign_body, verify_signature
from transcriber.tests.test_publish import Fake
from transcriber.transcribe import Segment

JOB_ID = "123456"
SECRET = "change-me"
ANARLOG_SECRET = "anarlog-change-me"
HOST_DATA_DIR = "/host/data"  # the fixture's paths are jitsi-capture's, not ours
FIXTURE = Path(__file__).parent / "fixtures" / "recording_finished.json"

OUTLINE = {
    "status": "success",
    "action": "created",
    "document_id": "d1",
    "title": "2026-09-13 Weekly sync",
    "url": "https://outline.example.com/doc/weekly-sync-abc",
}

# Fake ASR output per audio file. Alice's two opening turns are close enough to coalesce
# into one line, and her last one lands after Bob's — which only happens once his offset_s
# has been applied and the tracks merged by time.
SEGMENTS = {
    "alice.webm": [
        Segment(1.0, 3.0, "good morning"),
        Segment(4.0, 6.0, "shall we start"),
        Segment(12.0, 14.0, "that is all"),
    ],
    "bob.webm": [Segment(2.0, 4.0, "yes go ahead")],
    "mixed.webm": [Segment(2.0, 5.0, "hello everyone")],
}

EXPECTED_FEED = (
    "[00:01] Alice: good morning shall we start\n"
    "[00:07] Bob: yes go ahead\n"
    "[00:12] Alice: that is all"
)


def fake_asr(audio_path: str) -> list[Segment]:
    """Segments by file name — reading the file first, so an unrebased path fails here."""
    Path(audio_path).read_bytes()
    return SEGMENTS[Path(audio_path).name]


def make_audio(tmp_path, *names):
    """Create the files the fixture names, under our own DATA_DIR. Empty: the ASR is fake."""
    directory = tmp_path / "jobs" / JOB_ID
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


def wait_done(timeout=10.0):
    """Poll job.json — the pipeline's only progress signal — until the job leaves the queue."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = jobs.load_job(JOB_ID)
        if job["state"] in jobs.FINAL_STATES:
            assert job["state"] == "done", job.get("error")
            return job
        time.sleep(0.01)
    raise AssertionError(f"job did not finish in {timeout}s: {jobs.load_job(JOB_ID)}")


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HOST_DATA_DIR", HOST_DATA_DIR)
    monkeypatch.setenv("WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("ANARLOG_WEBHOOK_SECRET", ANARLOG_SECRET)
    # requests routes even loopback through HTTP_PROXY unless no_proxy says otherwise.
    for var in ("http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.upper(), raising=False)
    monkeypatch.setattr(publish, "BACKOFF_S", [])  # one attempt per peer, no sleeping
    # None in sys.modules makes `import onnx_asr` raise ImportError: reaching a real ASR
    # backend (a model download, seconds of CPU) fails this test rather than slowing it.
    for module in ("onnx_asr", "faster_whisper"):
        monkeypatch.setitem(sys.modules, module, None)


@pytest.fixture
def service(monkeypatch):
    """start(payload) -> (post, tr2outline, notify) for a running receiver + worker.

    The payload's callback_url is pointed at the fake /notify before it is signed, so the
    bytes the receiver stores are the bytes `post()` sent.
    """
    closers = []

    def start(payload):
        tr2outline, notify = Fake([(200, OUTLINE)]), Fake([(200, {"status": "ok"})])
        closers.extend([tr2outline.close, notify.close])
        monkeypatch.setenv("TR2OUTLINE_URL", tr2outline.url)
        payload["callback_url"] = notify.url
        body = json.dumps(payload).encode()

        # The real worker with only the ASR swapped: run_job binds its dependencies as
        # keyword defaults, which is the seam it offers tests. Everything downstream of it
        # — the payload, both signatures, the HTTP — is the real publish module.
        runner = worker.Worker(run=partial(worker.run_job, transcribe=fake_asr))
        runner.start()
        httpd = server.serve(0, server.make_handler(SECRET, runner.enqueue))
        # Short poll interval: shutdown() waits out one of these per server at teardown.
        threading.Thread(target=httpd.serve_forever, args=(0.01,), daemon=True).start()
        closers.append(lambda: (httpd.shutdown(), httpd.server_close()))
        url = f"http://127.0.0.1:{httpd.server_address[1]}/webhook"

        def post():
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-jitsi-capture-event": "recording.finished",
                    "x-jitsi-capture-signature": sign_body(body, SECRET),
                },
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())

        return post, tr2outline, notify

    yield start
    for close in closers:
        close()


def test_tracked_call_reaches_tr2outline_and_the_ready_callback(tmp_path, service):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    make_audio(tmp_path, "alice.webm", "bob.webm")
    post, tr2outline, notify = service(payload)

    assert post() == (202, {"status": "queued", "id": JOB_ID})
    job = wait_done()

    assert (jobs.job_dir(JOB_ID) / "transcript.md").read_text(encoding="utf-8") == EXPECTED_FEED

    headers, body = tr2outline.requests[0]
    assert headers["x-anarlog-event"] == "note.enhanced"
    assert verify_signature(body, headers["x-anarlog-signature"], ANARLOG_SECRET)
    assert json.loads(body) == build_anarlog_payload(payload, EXPECTED_FEED)
    # tr2outline prefixes the date itself, so the title we send is the Zulip topic alone.
    assert json.loads(body)["data"]["meeting"]["title"] == payload["topic"]

    headers, body = notify.requests[0]
    assert verify_signature(body, headers["x-jitsi-capture-signature"], SECRET)
    assert json.loads(body) == {
        "id": JOB_ID,
        "content": f"Transcript ready: [{OUTLINE['title']}]({OUTLINE['url']})",
    }
    assert (job["outline_url"], job["outline_title"]) == (OUTLINE["url"], OUTLINE["title"])

    # jitsi-capture retries until it gets a 2xx: the retry must not publish a second time.
    assert post() == (200, {"status": "duplicate", "id": JOB_ID})
    assert len(tr2outline.requests) == 1
    assert len(notify.requests) == 1


def test_call_without_tracks_has_no_speaker_names(tmp_path, service):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del payload["tracks"]  # no per-participant recordings: mixed audio_path, no attribution
    make_audio(tmp_path, "mixed.webm")
    post, tr2outline, _notify = service(payload)

    assert post()[0] == 202
    wait_done()

    feed = "[00:02] hello everyone"
    assert (jobs.job_dir(JOB_ID) / "transcript.md").read_text(encoding="utf-8") == feed
    assert json.loads(tr2outline.requests[0][1])["data"]["transcript_text"] == feed
