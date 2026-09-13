"""Process entrypoint: `python -m transcriber` — the receiver plus the worker thread.

Configuration is environment-only (README "Configuration"); the required variables are
checked before the port is bound so a misconfigured container fails immediately.
"""

from __future__ import annotations

import logging
import os
import signal
import threading

from transcriber import server
from transcriber.worker import Worker

log = logging.getLogger("transcriber")

REQUIRED = ("WEBHOOK_SECRET", "TR2OUTLINE_URL", "ANARLOG_WEBHOOK_SECRET")


def _warm_up() -> None:
    # ponytail: best-effort preload so the first webhook does not wait for the model
    # download. Failures are the first job's problem, not startup's.
    try:
        from transcriber.asr_parakeet import _get_model

        _get_model()
        log.info("ASR model ready")
    except Exception:
        log.exception("ASR model warm-up failed")


def main() -> int:
    logging.basicConfig(
        level=(os.getenv("LOG_LEVEL") or "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        # Names only, never values.
        log.error("missing required environment variables: %s", ", ".join(missing))
        return 2

    if (os.getenv("ASR_ENGINE") or "parakeet") == "parakeet":
        threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()

    worker = Worker()
    worker.start()  # resume unfinished jobs before the first new webhook can be served
    port = int(os.getenv("PORT") or 8080)
    httpd = server.serve(port, server.make_handler(os.environ["WEBHOOK_SECRET"], worker.enqueue))

    def _stop(*_args):
        # shutdown() blocks until serve_forever() returns, so it can never be called
        # from the thread sitting in serve_forever() — which is this one.
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    log.info("listening on port %d", port)
    httpd.serve_forever()
    # A job mid-flight is not finished here: its state is on disk and the next start
    # resumes it.
    log.info("shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
