"""Inbound HTTP receiver: POST /webhook (signed, idempotent) and GET /health.

Stdlib http.server only — the surface is two routes, a framework would be dead weight.
The handler does no I/O beyond the job store: validating the audio files is the
worker's job, so a broken path becomes a failed job with a reason, not a 4xx here.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transcriber import jobs
from transcriber.signing import verify_signature

log = logging.getLogger(__name__)

MAX_BODY = 1 << 20  # 1 MiB: the webhook is small JSON, anything bigger is not ours
EVENT_HEADER = "x-jitsi-capture-event"
SIGNATURE_HEADER = "x-jitsi-capture-signature"
EVENT = "recording.finished"

# ponytail: one global lock makes check-and-queue atomic across the server's threads, so
# two simultaneous deliveries of the same id cannot both queue it. Ceiling: it is in-process,
# which is enough for the single receiver this service runs; a second process would need an
# O_EXCL claim on job.json instead.
_QUEUE_LOCK = threading.Lock()


def make_handler(secret: str, on_job: Callable[[str], None]) -> type[BaseHTTPRequestHandler]:
    """Build a handler that verifies with `secret` and hands accepted job ids to `on_job`."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # jitsi-capture reuses the connection
        timeout = 30  # a stalled peer must not hold a thread forever

        def log_message(self, fmt, *args):
            # No default stderr access log: route it through logging at DEBUG.
            log.debug("%s %s", self.address_string(), fmt % args)

        def _reply(self, status: int, payload: dict) -> None:
            if status >= 400:
                # We may have skipped the request body (413), so keep-alive framing is gone.
                self.close_connection = True
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self._route() != "/health":
                self._reply(404, {"status": "not found"})
                return
            self._reply(200, {"status": "ok"})

        def do_POST(self):
            if self._route() != "/webhook":
                self._reply(404, {"status": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._reply(400, {"status": "bad request"})
                return
            if length < 0:
                # read(-1) would block on the socket until the peer goes away.
                self._reply(400, {"status": "bad request"})
                return
            if length > MAX_BODY:
                self._reply(413, {"status": "too large"})
                return

            raw = self.rfile.read(length)
            if not verify_signature(raw, self.headers.get(SIGNATURE_HEADER), secret):
                log.info("webhook rejected: bad signature")
                self._reply(401, {"status": "unauthorized"})
                return
            if self.headers.get(EVENT_HEADER) != EVENT:
                self._reply(200, {"status": "ignored"})
                return

            job_id = self._job_id(raw)
            if job_id is None:
                self._reply(400, {"status": "bad request"})  # never log the body
                return

            try:
                with _QUEUE_LOCK:
                    existing = self._existing(job_id)
                    if existing is not None and existing.get("state") != "failed":
                        # jitsi-capture retries until it gets a 2xx; never re-queue a live job.
                        log.info(
                            "job %s: duplicate webhook (state=%s)", job_id, existing.get("state")
                        )
                        self._reply(200, {"status": "duplicate", "id": job_id})
                        return

                    jobs.save_webhook(job_id, raw)
                    jobs.save_job(
                        {
                            "id": job_id,
                            "state": "queued",
                            "error": "",
                            "received_at": jobs.now_rfc3339(),
                            "outline_url": "",
                            "outline_title": "",
                        }
                    )
                log.info("job %s: queued", job_id)
                on_job(job_id)
            except Exception:
                # Without this the client gets a closed socket and no status line at all.
                # 500 keeps jitsi-capture retrying; a job already written as "queued" is
                # also picked up by the worker's resume scan (jobs.list_unfinished).
                log.exception("job %s: could not be accepted", job_id)
                self._reply(500, {"status": "error", "id": job_id})
                return
            self._reply(202, {"status": "queued", "id": job_id})

        def _route(self) -> str:
            return self.path.split("?", 1)[0]

        def _job_id(self, raw: bytes) -> str | None:
            """The payload's id, or None when the request is not a usable job."""
            try:
                payload = json.loads(raw)
            except ValueError:
                log.info("webhook rejected: malformed JSON")
                return None
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("id"), str)
                or "audio_path" not in payload
                or "callback_url" not in payload
            ):
                log.info("webhook rejected: missing or invalid fields")
                return None
            try:
                jobs.job_dir(payload["id"])  # trust boundary: id becomes a path segment
            except ValueError:
                # The id is body content and unbounded in length: say so, never echo it.
                log.info("webhook rejected: invalid job id")
                return None
            return payload["id"]

        def _existing(self, job_id: str) -> dict | None:
            try:
                return jobs.load_job(job_id)
            except (OSError, ValueError):
                return None

    return Handler


def serve(port: int, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    """Bound server on every interface; the caller runs serve_forever()."""
    return ThreadingHTTPServer(("0.0.0.0", port), handler)
