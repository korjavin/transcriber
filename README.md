# transcriber

Local, CPU-only transcription of **Jitsi Meet** call recordings, with speaker
attribution. Second of three services in the chain:

```text
  [ Zulip ]  🎙️ reaction on a message
      │
      ▼
  [ jitsi-capture ]  (Go + Node/Chromium)
      │  records the call, writes audio under DATA_DIR
      │  ──► signed `recording.finished` webhook
      ▼
  [ transcriber ]  ◄── this repository
      │  CPU ASR + per-track segment merge
      │  ──► Anarlog-format webhook
      ▼
  [ tr2outline ]  (github.com/korjavin/tr2outline)
      │  renders Markdown and publishes a document to Outline
      │  ──► returns the document URL
      ▼
  [ jitsi-capture /notify ]  ──►  "transcript ready" message in the Zulip topic
```

Privacy is the driving requirement: audio and text never leave the local
network, everything runs on the CPU of your own server.

---

## 📥 Input: the `recording.finished` webhook

jitsi-capture sends `POST /webhook`:

| Header | Value |
|---|---|
| `x-jitsi-capture-event` | `recording.finished` |
| `x-jitsi-capture-signature` | `sha256=<hex HMAC-SHA256 of the raw body>`, secret `WEBHOOK_SECRET` |

The signature is computed over the **raw request bytes** (before JSON parsing)
and compared in constant time.

Body:

```json
{
  "event": "recording.finished",
  "id": "123456",
  "message_id": 123456,
  "stream": "<stream>",
  "topic": "<topic>",
  "jitsi_url": "https://meet.jit.si/<room>",
  "audio_path": "/data/jobs/123456/audio.webm",
  "duration_s": 1834.2,
  "started_at": "2026-09-13T12:00:00Z",
  "ended_at": "2026-09-13T12:30:34Z",
  "participants": ["Alice", "Bob"],
  "callback_url": "http://jitsi-capture:8080/notify",
  "tracks": [
    {"id": "a1", "name": "Alice", "path": "/data/jobs/123456/alice.webm", "offset_s": 0.0, "ended_s": 1834.2},
    {"id": "b2", "name": "Bob",   "path": "/data/jobs/123456/bob.webm",   "offset_s": 12.5, "ended_s": 1790.0}
  ]
}
```

`tracks[]` is optional — it only appears once per-participant recording lands in
jitsi-capture, so the code must work without it.

Files at `audio_path` and `tracks[].path` are read directly: both services mount
the same volume at `DATA_DIR`, so nothing is copied over the network.

---

## 🧠 ASR engine

**Owner's decision: [NVIDIA Parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
via [`onnx-asr`](https://github.com/istupakov/onnx-asr), int8 quantization, CPU.**

* 0.6B parameters, multilingual (EN and RU among others), emits word-level
  timestamps — exactly what track merging needs.
* `onnx-asr` is a thin wrapper over onnxruntime, with no torch and no NeMo:
  light to install, predictable on CPU.
* The backend is selected with `ASR_ENGINE=parakeet|whisper`.

`transcriber/transcribe.py` (faster-whisper `large-v3`, `int8`, CPU) moved here
from jitsi-capture and stays as the **temporary/fallback** backend: it works
today and produces timecoded segments, but it is noticeably slower on CPU. Once
the Parakeet backend is in place, whisper remains available as a fallback
(`ASR_ENGINE=whisper`) rather than being deleted.

---

## 🗣️ Speaker attribution

No diarization is needed: the speaker's name is already known from the
recording.

* **When the webhook carries `tracks[]`** — each participant's track is
  transcribed separately, its segments are shifted by `offset_s` (when that
  participant joined, relative to the start of the recording) and merged by time
  into a single feed:

  ```markdown
  [00:05] Alice: Hi everyone, let's start with the architecture.
  [00:18] Bob: On the second point, here is what I suggest...
  ```

  This is more accurate than any diarization over the mixed audio and costs the
  same compute (total track duration ≈ call duration).

* **When there are no tracks** — the mixed `audio_path` is transcribed and the
  feed carries no names; as a fallback the dominant speaker's name may be taken
  from the participant timeline. This is deliberately weaker — per-track is the
  default path.

---

## 📤 Output: the Anarlog-format webhook to tr2outline

`POST $TR2OUTLINE_URL` (`/api/webhooks/anarlog` by default):

| Header | Value |
|---|---|
| `x-anarlog-signature` | `sha256=<hex HMAC-SHA256 of the raw body>`, secret `ANARLOG_WEBHOOK_SECRET` |
| `x-anarlog-event` | `note.enhanced` |

tr2outline processes only `note.enhanced` events; anything else is acknowledged
with `200 OK` and ignored. Body (see `models.go` in tr2outline):

```json
{
  "id": "<job id>",
  "event": "note.enhanced",
  "created_at": "2026-09-13T12:30:34Z",
  "data": {
    "meeting": {
      "id": "<job id>",
      "title": "<topic> (<date>)",
      "participants": ["Alice", "Bob"],
      "note": "",
      "summaries": [],
      "action_items": []
    },
    "transcript_text": "[00:05] Alice: ...\n[00:18] Bob: ..."
  }
}
```

`note`, `summaries` and `action_items` stay empty for now — there is no
summarization step in the chain, and tr2outline renders the document correctly
without them. tr2outline replies with JSON containing the URL of the created
document; extract it and pass it to the callback.

---

## 🔁 Callback to jitsi-capture

`POST <callback_url>` (the address arrives in the webhook; it is jitsi-capture's
`/notify`), signed with the same `WEBHOOK_SECRET` in the
`x-jitsi-capture-signature` header:

```json
{
  "id": "<job id>",
  "content": "Transcript ready: [Meeting title](https://outline.your-domain.com/doc/...)"
}
```

`content` is a ready-to-post Markdown message: jitsi-capture publishes it into
the Zulip topic verbatim.

---

## 🧩 Direct Outline publishing (fallback)

`transcriber/outline_client.py` is a minimal Outline client
(`POST /api/documents.create`) that moved here with the rest of the code. The
primary publishing path is tr2outline; this client stays as a direct fallback
for installations where tr2outline is unavailable or unwanted.

---

## ⚙️ Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `WEBHOOK_SECRET` | yes | — | Shared secret with jitsi-capture: verifies the incoming webhook and signs the callback |
| `DATA_DIR` | no | `/data` | Shared audio volume (same path as in jitsi-capture) |
| `ASR_ENGINE` | no | `parakeet` | `parakeet` (onnx-asr) or `whisper` (faster-whisper) |
| `WHISPER_MODEL` | no | `large-v3` | faster-whisper model |
| `WHISPER_DEVICE` | no | `cpu` | faster-whisper device |
| `WHISPER_COMPUTE_TYPE` | no | `int8` | faster-whisper compute type |
| `WHISPER_LANGUAGE` | no | — | Empty = autodetect |
| `TR2OUTLINE_URL` | yes | — | tr2outline webhook endpoint |
| `ANARLOG_WEBHOOK_SECRET` | yes | — | Secret signing the tr2outline webhook |
| `OUTLINE_BASE_URL` | no | — | Fallback path: Outline base URL |
| `OUTLINE_API_KEY` | no | — | Fallback path: Outline API token |
| `OUTLINE_COLLECTION_ID` | no | — | Fallback path: Outline collection UUID |

Configuration is env-vars **only**. `.env` is gitignored; `.env.example` holds
placeholders exclusively. Logs print the **name** of a variable, never its
value, and never a full Jitsi URL with tokens.

---

## 🚀 Running and developing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

ruff check .          # lint
pytest -q             # tests — fully offline, no network, no model downloads

# one-off transcription of a file with the fallback backend
python -m transcriber.transcribe /path/to/audio.webm
```

The tests shadow `faster_whisper` and `requests` with fakes, so they pass with
no network and no model downloads. CI (GitHub Actions, Python 3.12) runs exactly
these two commands on every push to `master` and on every pull request.

**Docker:** TODO — the image lands together with the HTTP receiver.

---

## 🗺️ Roadmap

- [ ] HTTP receiver: `POST /webhook` (HMAC verification, job queue), `GET /health`
- [ ] Parakeet-tdt-0.6b-v3 backend via onnx-asr (int8, CPU), `ASR_ENGINE` switch
- [ ] Per-track segment merge by `offset_s` → a `Name: text` feed with timecodes
- [ ] Anarlog-format webhook to tr2outline + parsing the document URL from the reply
- [ ] `POST <callback_url>` callback with the signed Zulip message
- [ ] Dockerfile + image build in CI (ghcr.io)
- [ ] Resilience: webhook retries, recovery across service restarts

---

## 🧭 How to continue

Work on this repository is tracked with [bd (beads)](https://github.com/steveyegge/beads),
whose database lives in `.beads/` in this repo:

```bash
bd ready            # issues that are unblocked and ready to pick up
bd show <id>        # the full spec of one issue
bd list             # everything on the board
```

The plan lives in beads, not in this file: the roadmap above is a summary, the
issues are the source of truth for what to build next. The chain diagram and the
contract sections above (input webhook, ASR engine, speaker attribution, Anarlog
output, callback, environment variables) are the **specification** — code against
them, and if something has to change, change the spec here in the same PR.

Everything published here is public: no real domains, hostnames, emails, API
keys, Zulip stream or user names, or user data — in code, tests, fixtures, docs
or commit messages. All public artifacts are written in English.
