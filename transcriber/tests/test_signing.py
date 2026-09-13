"""Signature helpers, pinned to an external vector so a refactor cannot drift silently."""

from transcriber import signing

BODY = b'{"a":1}'
SECRET = "test"
# printf '%s' '{"a":1}' | openssl dgst -sha256 -hmac test
VECTOR = "sha256=3b76df928ca0fb147722b51bd3e4e9f62e88b99eb9575cb0b2ded29e6e812402"


def test_sign_body_matches_openssl():
    assert signing.sign_body(BODY, SECRET) == VECTOR


def test_verify_accepts_with_and_without_prefix():
    assert signing.verify_signature(BODY, VECTOR, SECRET)
    assert signing.verify_signature(BODY, VECTOR.removeprefix("sha256="), SECRET)


def test_verify_rejects_tampered_body():
    assert not signing.verify_signature(b'{"a":2}', VECTOR, SECRET)


def test_verify_rejects_wrong_secret():
    assert not signing.verify_signature(BODY, VECTOR, "change-me")


def test_verify_rejects_empty_secret_header_or_none():
    assert not signing.verify_signature(BODY, VECTOR, "")
    assert not signing.verify_signature(BODY, "", SECRET)
    assert not signing.verify_signature(BODY, None, SECRET)


def test_verify_rejects_non_ascii_header():
    # The header is attacker-controlled: a non-ASCII one must mismatch, not raise
    # (hmac.compare_digest rejects non-ASCII str, hence the encode in verify_signature).
    assert not signing.verify_signature(BODY, "sha256=café", SECRET)
