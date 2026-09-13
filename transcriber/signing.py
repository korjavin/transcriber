"""HMAC-SHA256 signing shared by every HTTP boundary of this service.

Inbound: the jitsi-capture webhook (WEBHOOK_SECRET). Outbound: the tr2outline
webhook (ANARLOG_WEBHOOK_SECRET) and the jitsi-capture callback (WEBHOOK_SECRET).
"""

from __future__ import annotations

import hashlib
import hmac

PREFIX = "sha256="


def sign_body(body: bytes, secret: str) -> str:
    """'sha256=<hex HMAC-SHA256(secret, body)>' over the raw bytes, as both peers expect."""
    return PREFIX + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def verify_signature(body: bytes, header: str | None, secret: str) -> bool:
    """Constant-time check of a signature header; the 'sha256=' prefix is optional.

    An empty header or an empty secret is always a rejection: a misconfigured
    service must not accept everything.
    """
    if not header or not secret:
        return False
    expected = sign_body(body, secret).removeprefix(PREFIX).encode()
    return hmac.compare_digest(expected, header.removeprefix(PREFIX).encode())
