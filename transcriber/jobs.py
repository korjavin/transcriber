"""Job state on disk: DATA_DIR/transcriber/<job_id>/{webhook.json,job.json,transcript.md}.

Jobs survive restarts, so every write is atomic (tmp file + os.replace) and the
job id — which arrives from the network — is validated before it becomes a path.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

STATES = ("queued", "transcribing", "publishing", "done", "failed")
FINAL_STATES = frozenset({"done", "failed"})

_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "/data"))


def job_dir(job_id: str) -> Path:
    # Trust boundary: the id comes from the webhook body and becomes a path segment.
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise ValueError(f"invalid job id: {job_id!r}")
    return data_dir() / "transcriber" / job_id


def _write_atomic(path: Path, payload: bytes) -> None:
    # The tmp name must be unique: the receiver and the worker can write the same
    # job.json at once, and a shared tmp name makes one of them fail the rename.
    # ponytail: rename-atomic, not crash-durable — a power cut can leave an empty
    # job.json (list_unfinished skips unparseable ones). fsync here if that ever bites.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_job(job: dict) -> None:
    job["updated_at"] = now_rfc3339()
    _write_atomic(job_dir(job["id"]) / "job.json", json.dumps(job).encode())


def load_job(job_id: str) -> dict:
    """The job record; FileNotFoundError when the job is unknown."""
    return json.loads((job_dir(job_id) / "job.json").read_bytes())


def save_webhook(job_id: str, raw: bytes) -> None:
    """Store the inbound payload verbatim — the signature was computed over these bytes."""
    _write_atomic(job_dir(job_id) / "webhook.json", raw)


def load_webhook(job_id: str) -> dict:
    return json.loads((job_dir(job_id) / "webhook.json").read_bytes())


def set_state(job_id: str, state: str, **fields) -> dict:
    if state not in STATES:
        raise ValueError(f"unknown state: {state!r}")
    job = load_job(job_id)
    job.update(state=state, **fields)
    save_job(job)
    return job


def list_unfinished() -> list[str]:
    """Job ids still in flight, oldest first — what a restart has to pick back up."""
    root = data_dir() / "transcriber"
    pending = []
    for entry in sorted(root.iterdir()) if root.is_dir() else []:
        try:
            job = json.loads((entry / "job.json").read_bytes())
        except (OSError, ValueError):
            continue  # half-written or foreign directory: not a job we can resume
        if job.get("state") not in FINAL_STATES:
            pending.append((job.get("received_at", ""), entry.name))
    return [job_id for _, job_id in sorted(pending)]


def rebase_path(p: str) -> str:
    """Map a HOST_DATA_DIR path from jitsi-capture back onto our own DATA_DIR mount."""
    local_root = data_dir()
    host_root = os.environ.get("HOST_DATA_DIR") or str(local_root)
    local = p.replace(host_root, str(local_root), 1)
    if not Path(local).resolve().is_relative_to(local_root.resolve()):
        raise ValueError(f"path escapes DATA_DIR: {p!r}")
    return local
