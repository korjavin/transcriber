"""Job store tests — DATA_DIR points at tmp_path, nothing touches a real volume."""

import json

import pytest

from transcriber import jobs


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("HOST_DATA_DIR", raising=False)
    return tmp_path


def make_job(job_id="123", state="queued", received_at="2026-09-13T12:00:00Z"):
    jobs.save_job(
        {
            "id": job_id,
            "state": state,
            "error": "",
            "received_at": received_at,
            "outline_url": "",
            "outline_title": "",
        }
    )


def test_job_dir_is_under_data_dir(data_dir):
    assert jobs.job_dir("123") == data_dir / "transcriber" / "123"


@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".", "x" * 65, "a b"])
def test_job_dir_rejects_unsafe_ids(bad):
    with pytest.raises(ValueError):
        jobs.job_dir(bad)


def test_save_and_load_job_round_trip():
    make_job()
    job = jobs.load_job("123")
    assert job["id"] == "123"
    assert job["state"] == "queued"
    assert job["updated_at"].endswith("Z")


def test_load_job_missing_raises():
    with pytest.raises(FileNotFoundError):
        jobs.load_job("nope")


def test_webhook_is_stored_verbatim(data_dir):
    raw = b'{"id": "123",  "audio_path": "/data/a.webm"}'
    jobs.save_webhook("123", raw)
    assert (data_dir / "transcriber" / "123" / "webhook.json").read_bytes() == raw
    assert jobs.load_webhook("123")["id"] == "123"


def test_set_state_updates_state_and_timestamp():
    make_job()
    stale = json.loads((jobs.job_dir("123") / "job.json").read_bytes())
    stale["updated_at"] = "2000-01-01T00:00:00Z"
    (jobs.job_dir("123") / "job.json").write_bytes(json.dumps(stale).encode())

    job = jobs.set_state("123", "failed", error="no audio")

    assert job["state"] == "failed"
    assert job["error"] == "no audio"
    assert job["updated_at"] != "2000-01-01T00:00:00Z"
    assert jobs.load_job("123") == job


def test_set_state_rejects_unknown_state():
    make_job()
    with pytest.raises(ValueError):
        jobs.set_state("123", "exploded")


def test_list_unfinished_skips_final_states_and_sorts_by_received_at(data_dir):
    make_job("late", received_at="2026-09-13T13:00:00Z")
    make_job("early", received_at="2026-09-13T11:00:00Z")
    make_job("done1", state="done", received_at="2026-09-13T10:00:00Z")
    make_job("failed1", state="failed", received_at="2026-09-13T10:00:00Z")
    (data_dir / "transcriber" / "junk").mkdir()

    assert jobs.list_unfinished() == ["early", "late"]


def test_list_unfinished_without_a_data_dir():
    assert jobs.list_unfinished() == []


def test_rebase_path_maps_host_dir(data_dir, monkeypatch):
    monkeypatch.setenv("HOST_DATA_DIR", "/srv/host-data")
    assert jobs.rebase_path("/srv/host-data/jobs/1/a.webm") == f"{data_dir}/jobs/1/a.webm"


def test_rebase_path_is_a_noop_when_mounts_match(data_dir):
    path = f"{data_dir}/jobs/1/a.webm"
    assert jobs.rebase_path(path) == path


def test_rebase_path_rejects_escape(data_dir):
    with pytest.raises(ValueError):
        jobs.rebase_path(f"{data_dir}/../etc/passwd")
    with pytest.raises(ValueError):
        jobs.rebase_path("/etc/passwd")
