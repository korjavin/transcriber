"""Receiver tests over a real loopback socket — no network, no framework, no mocks of HTTP."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from transcriber import jobs, server
from transcriber.signing import sign_body

SECRET = "change-me"
PAYLOAD = {
    "event": "recording.finished",
    "id": "123456",
    "audio_path": "/data/jobs/123456/audio.webm",
    "callback_url": "http://jitsi-capture:8080/notify",
    "participants": ["Alice", "Bob"],
}


@pytest.fixture
def queued():
    return []


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def url(queued):
    handler = server.make_handler(SECRET, queued.append)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def request(url, path, *, body=None, headers=None, method=None):
    """(status, parsed json) — HTTP error statuses come back as values, not exceptions."""
    req = urllib.request.Request(url + path, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def post(url, payload=None, *, raw=None, secret=SECRET, event="recording.finished", sig=None):
    body = raw if raw is not None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if event is not None:
        headers["x-jitsi-capture-event"] = event
    headers["x-jitsi-capture-signature"] = sig if sig is not None else sign_body(body, secret)
    return request(url, "/webhook", body=body, headers=headers)


def test_health(url):
    assert request(url, "/health") == (200, {"status": "ok"})


def test_unknown_route_is_404(url):
    assert request(url, "/nope")[0] == 404
    assert post(url, PAYLOAD)[0] == 202  # sanity: the route that does exist
    assert request(url, "/nope", body=b"{}", headers={})[0] == 404


def test_bad_signature_is_401(url, queued):
    assert post(url, PAYLOAD, sig="sha256=deadbeef")[0] == 401
    assert post(url, PAYLOAD, secret="wrong-secret")[0] == 401
    assert post(url, PAYLOAD, sig="")[0] == 401
    assert queued == []


def test_empty_secret_never_authorizes(queued):
    handler = server.make_handler("", queued.append)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        assert post(base, PAYLOAD, secret="")[0] == 401
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert queued == []


def test_other_event_is_ignored(url, queued):
    assert post(url, PAYLOAD, event="recording.started") == (200, {"status": "ignored"})
    assert queued == []


def test_malformed_json_is_400(url, queued):
    assert post(url, raw=b"not json")[0] == 400
    assert queued == []


@pytest.mark.parametrize(
    "payload",
    [
        {**PAYLOAD, "id": 123456},
        {k: v for k, v in PAYLOAD.items() if k != "id"},
        {k: v for k, v in PAYLOAD.items() if k != "audio_path"},
        {k: v for k, v in PAYLOAD.items() if k != "callback_url"},
        {**PAYLOAD, "id": "../escape"},
        [PAYLOAD],
    ],
)
def test_unusable_payloads_are_400(url, queued, payload):
    assert post(url, payload)[0] == 400
    assert queued == []


def test_oversized_body_is_413(url, queued):
    # The declared length is what the guard rejects: the body is never read.
    headers = {
        "Content-Length": str(server.MAX_BODY + 1),
        "x-jitsi-capture-event": "recording.finished",
        "x-jitsi-capture-signature": sign_body(b"x", SECRET),
    }
    assert request(url, "/webhook", body=b"x", headers=headers)[0] == 413
    assert queued == []


def test_happy_path_queues_the_job(url, queued, data_dir):
    body = json.dumps(PAYLOAD).encode()
    status, reply = post(url, raw=body)

    assert (status, reply) == (202, {"status": "queued", "id": "123456"})
    assert queued == ["123456"]
    job_dir = data_dir / "transcriber" / "123456"
    assert (job_dir / "webhook.json").read_bytes() == body
    job = json.loads((job_dir / "job.json").read_bytes())
    assert job["state"] == "queued"
    assert job["id"] == "123456"
    assert job["received_at"].endswith("Z")


def test_repeat_delivery_is_a_duplicate(url, queued):
    assert post(url, PAYLOAD)[0] == 202
    assert post(url, PAYLOAD) == (200, {"status": "duplicate", "id": "123456"})
    assert queued == ["123456"]


@pytest.mark.parametrize("state", ["transcribing", "publishing", "done"])
def test_live_and_done_jobs_are_not_requeued(url, queued, state):
    assert post(url, PAYLOAD)[0] == 202
    jobs.set_state("123456", state)
    assert post(url, PAYLOAD)[0] == 200
    assert queued == ["123456"]


def test_failed_job_is_requeued(url, queued):
    assert post(url, PAYLOAD)[0] == 202
    jobs.set_state("123456", "failed", error="no audio")

    assert post(url, PAYLOAD)[0] == 202

    assert queued == ["123456", "123456"]
    assert jobs.load_job("123456")["state"] == "queued"
