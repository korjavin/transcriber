"""Outbound publishing: the Anarlog webhook to tr2outline, then the ready callback.

See README "Output: the Anarlog-format webhook to tr2outline" and "Callback to
jitsi-capture". Both bodies are signed with HMAC-SHA256 over the exact bytes sent
(`transcriber.signing`), so each body is built once and never re-serialized.
"""

from __future__ import annotations

import json
import logging
import os
import time

import requests

from transcriber.jobs import now_rfc3339
from transcriber.signing import sign_body

log = logging.getLogger(__name__)

EVENT = "note.enhanced"

# Delays between retries; attempts = len(BACKOFF_S) + 1. Monkeypatched to [0] in tests.
# ponytail: fixed table; the job stays in `publishing` on disk and is retried on
# restart anyway, so giving up here is never the end of the road.
BACKOFF_S = [5, 15, 45, 120, 300]


class PublishError(Exception):
    """Any failure publishing a finished transcript."""


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise PublishError(f"missing environment variable: {name}")
    return value


def _send(url: str, body: bytes, headers: dict, timeout: float, job_id: str, what: str):
    """POST with retries over BACKOFF_S on 5xx and connection errors.

    Returns the first response below 500 — what its status means is the caller's call.
    """
    detail = ""
    for delay in [*BACKOFF_S, None]:
        try:
            # No redirects: a 3xx would rewrite this POST to a GET and drop the signed
            # body, and a cross-host one would hand our signature to the new target.
            response = requests.post(
                url, data=body, headers=headers, timeout=timeout, allow_redirects=False
            )
        except requests.RequestException as exc:
            # Type name only: an exception can carry the request (and its signed
            # headers) along, and neither the log nor the error may leak that.
            detail = type(exc).__name__
        else:
            if response.status_code < 500:
                return response
            detail = f"HTTP {response.status_code}"
        log.warning("%s failed for job %s: %s", what, job_id, detail)
        if delay is None:
            break
        time.sleep(delay)
    raise PublishError(f"{what} failed for job {job_id}: {detail}")


def _reply_object(response) -> dict:
    try:
        reply = response.json()
    except ValueError:
        return {}
    return reply if isinstance(reply, dict) else {}


def build_anarlog_payload(webhook: dict, transcript_text: str) -> dict:
    """The note.enhanced body for tr2outline.

    An empty transcript_text is fine: tr2outline still creates the document (with
    placeholders) and the callback still tells the user where it is.
    """
    job_id = webhook["id"]
    return {
        "id": job_id,
        "event": EVENT,
        "created_at": webhook.get("ended_at") or now_rfc3339(),
        "data": {
            "meeting": {
                "id": job_id,
                # The topic alone: tr2outline prefixes the date from created_at itself
                # (FormatDocumentTitle), so a date here would be printed twice.
                "title": webhook.get("topic") or "Meeting",
                "note": "",
                "summaries": [],
                "participants": webhook.get("participants") or [],
                "action_items": [],
            },
            "transcript_text": transcript_text,
        },
    }


def send_to_tr2outline(payload: dict) -> tuple[str, str]:
    """POST the webhook; return (document url, title) from the reply."""
    job_id = payload["id"]
    body = json.dumps(payload, ensure_ascii=False).encode()
    headers = {
        "Content-Type": "application/json",
        "x-anarlog-event": EVENT,
        "x-anarlog-signature": sign_body(body, _env("ANARLOG_WEBHOOK_SECRET")),
    }
    response = _send(_env("TR2OUTLINE_URL"), body, headers, 60.0, job_id, "tr2outline webhook")
    if not 200 <= response.status_code < 300:
        raise PublishError(f"tr2outline rejected job {job_id}: HTTP {response.status_code}")

    # A 2xx that is not a created document is a contract problem, not a blip: no retry.
    reply = _reply_object(response)
    url, title = reply.get("url"), reply.get("title")
    if reply.get("status") != "success" or not isinstance(url, str) or not url:
        # The status is the peer's string and ends up in job.json: keep it bounded.
        status = repr(reply.get("status"))[:60]
        raise PublishError(f"tr2outline created no document for job {job_id}: {status}")
    log.info("job %s published to tr2outline", job_id)
    return url, title if isinstance(title, str) and title else payload["data"]["meeting"]["title"]


def ready_message(title: str, url: str) -> str:
    return f"Transcript ready: [{title}]({url})"


def post_callback(callback_url: str, job_id: str, content: str) -> None:
    """POST the ready message to jitsi-capture /notify.

    401 (secret mismatch) and 404 (unknown job upstream) are operator problems and
    are never retried; 5xx is.
    """
    body = json.dumps({"id": job_id, "content": content}, ensure_ascii=False).encode()
    headers = {
        "Content-Type": "application/json",
        "x-jitsi-capture-signature": sign_body(body, _env("WEBHOOK_SECRET")),
    }
    response = _send(callback_url, body, headers, 30.0, job_id, "ready callback")
    if not 200 <= response.status_code < 300:
        raise PublishError(f"ready callback for job {job_id} rejected: HTTP {response.status_code}")
    log.info("job %s callback delivered", job_id)
