"""Publishing tests over a real loopback socket: a fake tr2outline and a fake /notify.

Nothing leaves the box — the e2e bead reuses this fake server.
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from transcriber import publish
from transcriber.publish import (
    PublishError,
    build_anarlog_payload,
    post_callback,
    ready_message,
    send_to_tr2outline,
)
from transcriber.signing import verify_signature

ANARLOG_SECRET = "anarlog-change-me"
WEBHOOK_SECRET = "change-me"
DEAD_URL = "http://127.0.0.1:1/notify"  # nothing listens on port 1: instant refusal

WEBHOOK = {
    "event": "recording.finished",
    "id": "123456",
    "topic": "Weekly sync",
    "participants": ["Alice", "Bob"],
    "ended_at": "2026-09-13T12:30:34Z",
    "callback_url": "http://jitsi-capture:8080/notify",
}
SUCCESS = {
    "status": "success",
    "action": "created",
    "document_id": "doc-1",
    "title": "2026-09-13 Weekly sync",
    "url": "https://outline.your-domain.com/doc/weekly-sync-abc123",
}


class Fake:
    """Loopback HTTP server: records every request, replies from a scripted list."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []  # (headers, raw body)
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                fake.requests.append((self.headers, body))
                status, payload = fake.replies.pop(0) if fake.replies else (200, SUCCESS)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                if 300 <= status < 400:
                    self.send_header("Location", "/redirected")
                self.end_headers()
                self.wfile.write(data)

            # A followed redirect arrives as a GET: record it, so the test that asserts
            # "one request" actually proves redirects are not followed.
            do_GET = do_POST

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/hook"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    """start(*replies) -> Fake; every server is shut down at the end of the test."""
    started = []

    def start(*replies):
        fake = Fake(replies)
        started.append(fake)
        return fake

    yield start
    for fake in started:
        fake.close()


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("ANARLOG_WEBHOOK_SECRET", ANARLOG_SECRET)
    monkeypatch.setenv("WEBHOOK_SECRET", WEBHOOK_SECRET)
    # requests routes even loopback through HTTP_PROXY unless no_proxy says otherwise,
    # so on a machine with a proxy configured these tests would leave the box.
    for var in ("http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.upper(), raising=False)
    monkeypatch.setattr(publish, "BACKOFF_S", [0])  # one retry, no sleeping


@pytest.fixture
def tr2outline(serve, monkeypatch):
    def start(*replies):
        fake = serve(*replies)
        monkeypatch.setenv("TR2OUTLINE_URL", fake.url)
        return fake

    return start


# --- payload ---------------------------------------------------------------


def test_payload_is_exact():
    assert build_anarlog_payload(WEBHOOK, "[00:05] Alice: hi") == {
        "id": "123456",
        "event": "note.enhanced",
        "created_at": "2026-09-13T12:30:34Z",
        "data": {
            "meeting": {
                "id": "123456",
                "title": "Weekly sync",  # no date: tr2outline prefixes it from created_at
                "note": "",
                "summaries": [],
                "participants": ["Alice", "Bob"],
                "action_items": [],
            },
            "transcript_text": "[00:05] Alice: hi",
        },
    }


@pytest.mark.parametrize("missing", [{}, {"topic": None, "participants": None, "ended_at": None}])
def test_payload_defaults_when_fields_are_absent_or_null(missing):
    payload = build_anarlog_payload({"id": "7", **missing}, "")

    meeting = payload["data"]["meeting"]
    assert (meeting["title"], meeting["participants"]) == ("Meeting", [])
    assert payload["data"]["transcript_text"] == ""
    assert payload["created_at"].endswith("Z")


# --- send_to_tr2outline ----------------------------------------------------


def test_send_signs_the_exact_bytes_and_returns_url_and_title(tr2outline):
    fake = tr2outline((200, SUCCESS))
    payload = build_anarlog_payload(WEBHOOK, "[00:05] Alice: hi")

    assert send_to_tr2outline(payload) == (SUCCESS["url"], SUCCESS["title"])

    headers, body = fake.requests[0]
    assert headers["x-anarlog-event"] == "note.enhanced"
    assert headers["Content-Type"] == "application/json"
    assert verify_signature(body, headers["x-anarlog-signature"], ANARLOG_SECRET)
    assert not verify_signature(body, headers["x-anarlog-signature"], "other-secret")
    assert json.loads(body) == payload


def test_send_retries_a_500_then_succeeds(tr2outline):
    fake = tr2outline((500, {"status": "error"}), (200, SUCCESS))

    url, _ = send_to_tr2outline(build_anarlog_payload(WEBHOOK, "x"))

    assert url == SUCCESS["url"]
    assert len(fake.requests) == 2


def test_send_gives_up_after_the_backoff_table(tr2outline):
    fake = tr2outline((500, {"status": "error"}), (500, {"status": "error"}))

    with pytest.raises(PublishError, match="tr2outline webhook failed for job 123456"):
        send_to_tr2outline(build_anarlog_payload(WEBHOOK, "x"))

    assert len(fake.requests) == 2  # len(BACKOFF_S) + 1


def test_send_retries_a_connection_error(monkeypatch):
    monkeypatch.setenv("TR2OUTLINE_URL", DEAD_URL)

    with pytest.raises(PublishError, match="ConnectionError"):
        send_to_tr2outline(build_anarlog_payload(WEBHOOK, "x"))


@pytest.mark.parametrize(
    "reply",
    [
        (200, {"status": "ignored", "reason": "event not handled"}),
        (200, {"status": "success", "url": ""}),
        (200, {"status": "success"}),
        (200, {"status": "success", "url": {"not": "a string"}}),
        (302, {"status": "moved"}),  # a redirect would drop the signed body
        (200, ["not an object"]),
        (400, {"status": "error"}),
    ],
)
def test_unusable_reply_raises_without_retry(tr2outline, reply):
    fake = tr2outline(reply)

    with pytest.raises(PublishError):
        send_to_tr2outline(build_anarlog_payload(WEBHOOK, "x"))

    assert len(fake.requests) == 1


@pytest.mark.parametrize("title", ["", None, {"not": "a string"}])
def test_title_falls_back_to_the_meeting_title(tr2outline, title):
    tr2outline((200, {**SUCCESS, "title": title}))

    assert send_to_tr2outline(build_anarlog_payload(WEBHOOK, "x"))[1] == "Weekly sync"


def test_a_peer_error_string_cannot_grow_the_persisted_error(tr2outline):
    # The message is stored in job.json: a chatty (or hostile) peer must not fill it.
    tr2outline((200, {"status": "e" * 5000, "url": ""}))

    with pytest.raises(PublishError) as err:
        send_to_tr2outline(build_anarlog_payload(WEBHOOK, "x"))

    assert len(str(err.value)) < 150


# --- callback --------------------------------------------------------------


def test_ready_message():
    assert ready_message("Weekly sync", "https://x/doc/1") == (
        "Transcript ready: [Weekly sync](https://x/doc/1)"
    )


def test_callback_body_and_signature(serve):
    fake = serve((200, {"status": "ok"}))
    content = ready_message(SUCCESS["title"], SUCCESS["url"])

    post_callback(fake.url, "123456", content)

    headers, body = fake.requests[0]
    assert json.loads(body) == {"id": "123456", "content": content}
    assert verify_signature(body, headers["x-jitsi-capture-signature"], WEBHOOK_SECRET)
    assert not verify_signature(body, headers["x-jitsi-capture-signature"], ANARLOG_SECRET)


@pytest.mark.parametrize("status", [401, 404])
def test_callback_operator_errors_do_not_retry(serve, status):
    fake = serve((status, {"error": "nope"}))

    with pytest.raises(PublishError, match=f"rejected: HTTP {status}"):
        post_callback(fake.url, "123456", "x")

    assert len(fake.requests) == 1


def test_callback_retries_a_502(serve):
    fake = serve((502, {"error": "zulip is down"}), (200, {"status": "ok"}))

    post_callback(fake.url, "123456", "x")

    assert len(fake.requests) == 2


def test_callback_gives_up_after_the_backoff_table(serve):
    fake = serve((502, {"error": "x"}), (502, {"error": "x"}))

    with pytest.raises(PublishError, match="ready callback failed for job 123456"):
        post_callback(fake.url, "123456", "x")

    assert len(fake.requests) == 2


# --- secrets and config ----------------------------------------------------


@pytest.mark.parametrize("missing", ["TR2OUTLINE_URL", "ANARLOG_WEBHOOK_SECRET"])
def test_missing_env_var_raises_naming_it(monkeypatch, missing, tr2outline):
    tr2outline((200, SUCCESS))
    monkeypatch.delenv(missing)

    with pytest.raises(PublishError, match=missing):
        send_to_tr2outline(build_anarlog_payload(WEBHOOK, "x"))


def test_callback_missing_secret_raises(monkeypatch):
    monkeypatch.delenv("WEBHOOK_SECRET")

    with pytest.raises(PublishError, match="WEBHOOK_SECRET"):
        post_callback(DEAD_URL, "123456", "x")


def test_no_secret_reaches_a_log_record_or_an_exception(caplog, tr2outline, serve, monkeypatch):
    caplog.set_level(logging.DEBUG)
    # Every failure shape: a 5xx retry, a refused connection, a rejected callback.
    tr2outline((500, {"status": "error"}), (500, {"secret": ANARLOG_SECRET}))
    errors = []
    for call in (
        lambda: send_to_tr2outline(build_anarlog_payload(WEBHOOK, "x")),
        lambda: post_callback(DEAD_URL, "123456", "x"),
        lambda: post_callback(serve((401, {"error": "bad signature"})).url, "123456", "x"),
    ):
        with pytest.raises(PublishError) as err:
            call()
        errors.append(str(err.value) + repr(err.value))

    haystack = caplog.text + "".join(errors)
    for secret in (ANARLOG_SECRET, WEBHOOK_SECRET):
        assert secret not in haystack
