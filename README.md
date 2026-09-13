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
      "title": "<topic>",
      "participants": ["Alice", "Bob"],
      "note": "",
      "summaries": [],
      "action_items": []
    },
    "transcript_text": "[00:05] Alice: ...\n[00:18] Bob: ..."
  }
}
```

`title` is the topic alone — the date comes from `created_at`, which tr2outline
prefixes to the Outline title itself (putting it here too would print it twice).

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
| `DATA_DIR` | no | `/data` | Shared audio volume inside the container (same path as in jitsi-capture) |
| `HOST_DATA_DIR` | no | = `DATA_DIR` | Path jitsi-capture reports its recordings under; incoming paths are rebased `HOST_DATA_DIR` → `DATA_DIR`. Leave **unset** in the stack — both services mount the same named volume at the same path, so the rebase is a no-op |
| `PORT` | no | `8080` | Port the receiver listens on |
| `ASR_ENGINE` | no | `parakeet` | `parakeet` (onnx-asr) or `whisper` (faster-whisper) |
| `MODEL_DIR` | no | `/models` | Model cache: onnx-asr model dir and `HF_HOME` for whisper |
| `WHISPER_MODEL` | no | `large-v3` | faster-whisper model |
| `WHISPER_DEVICE` | no | `cpu` | faster-whisper device |
| `WHISPER_COMPUTE_TYPE` | no | `int8` | faster-whisper compute type |
| `WHISPER_LANGUAGE` | no | — | Empty = autodetect |
| `TR2OUTLINE_URL` | yes | — | tr2outline webhook endpoint |
| `ANARLOG_WEBHOOK_SECRET` | yes | — | Secret signing the tr2outline webhook |
| `OUTLINE_BASE_URL` | no | — | Outline base URL. Prefixed onto a **relative** document url returned by tr2outline (logged as a warning); also the fallback path's base |
| `OUTLINE_API_KEY` | no | — | Fallback path: Outline API token |
| `OUTLINE_COLLECTION_ID` | no | — | Fallback path: Outline collection UUID |
| `LOG_LEVEL` | no | `INFO` | Stdlib logging level |

Three more variables are read by `docker-compose.yml` rather than by the code:

| Variable | Required | Default | Description |
|---|---|---|---|
| `DOMAIN` | yes *(compose)* | — | Public hostname Traefik routes to this service |
| `TRAEFIK_NETWORK_NAME` | no | `traefik` | Existing external Docker network Traefik runs on |
| `TRAEFIK_CERTRESOLVER` | no | `myresolver` | Traefik ACME resolver that issues the certificate |

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
these two commands on every push to `master` and on every pull request; the
`docker` job additionally builds the image on every pull request.

**Docker:** build and run the image locally with the same env file the stack uses:

```bash
cp .env.example .env        # then fill in the secrets
docker build -t transcriber .
docker run --rm -p 8080:8080 --env-file .env \
  -v jitsi-capture-data:/data -v transcriber-models:/models transcriber
```

`jitsi-capture-data` is jitsi-capture's own named volume, mounted here at
`DATA_DIR` (`/data`) — the same path it has on the other side, which is why
`HOST_DATA_DIR` stays unset. A run with any required variable missing exits `2`
after a single line naming the variables — no traceback, no restart loop.

The image carries no ASR model: the first transcription downloads it (~1-2 GB)
into `MODEL_DIR`, which is why `/models` is a named volume — otherwise every
container restart re-downloads it.

---

## 🚢 Deploy

One image, one compose stack. `docker-compose.yml` is written for Portainer but
runs the same under plain `docker compose up -d`.

### Run it locally

```bash
cp .env.example .env        # fill in the secrets and DOMAIN
# compose has no `build:`, so build the image under the tag it references
docker build -t ghcr.io/korjavin/transcriber:latest .
docker compose up -d
docker compose logs -f
```

`:latest` is only a local/placeholder tag: from now on the registry is fed SHA
tags alone, so `docker compose pull` either finds nothing or, until the stale
`:latest` left by the old CI job is deleted, silently pulls something very old.
Build it locally, do not pull it. The compose file also expects two
things that already exist on the server: the external Traefik network
(`TRAEFIK_NETWORK_NAME`, default `traefik`) and jitsi-capture's
`jitsi-capture-data` volume. It publishes no ports of its own; Traefik fronts
the service on `DOMAIN`.

### Automated deployment (GitHub Actions → ghcr.io → Portainer)

`.github/workflows/deploy.yml` runs on every push to `master` (and on
`workflow_dispatch`):

1. builds the image and pushes it to `ghcr.io/korjavin/transcriber:<sha>`;
2. checks out a `deploy` branch, rewrites the `image:` line in
   `docker-compose.yml` with that SHA tag, commits `[skip ci]` and force-pushes
   `deploy`;
3. calls the Portainer redeploy webhook stored in the repository secret
   `PORTAINER_REDEPLOY_HOOK` (skipped when the secret is empty).

`master` keeps `image: ghcr.io/korjavin/transcriber:latest` as a placeholder;
only the `deploy` branch carries an immutable SHA tag. **Point the Portainer
git-ops stack at branch `deploy`**, never at `master`, and paste the webhook URL
Portainer generates into the `PORTAINER_REDEPLOY_HOOK` secret.

`.github/workflows/ci.yml` only *builds* the image on pull requests — deploy.yml
is the single publisher, and it publishes the SHA tag alone.

**The ghcr package has to be public.** Portainer pulls anonymously, and the very
first push creates the package as private. There is no workflow step that can
change that; do it once by hand in the package settings
(*Packages → transcriber → Package settings → Change visibility → Public*) and
check it with:

```bash
gh api user/packages/container/transcriber -q .visibility
```

### Stack variables

Portainer pulls the repository, so no `.env` file exists on the node —
`docker-compose.yml` passes every variable through from the stack environment.
Set these in the stack:

| Variable | Notes |
| --- | --- |
| `WEBHOOK_SECRET` | required — shared with jitsi-capture, verifies the inbound webhook and signs the callback |
| `TR2OUTLINE_URL` | required — tr2outline's Anarlog endpoint |
| `ANARLOG_WEBHOOK_SECRET` | required — signs the tr2outline webhook |
| `DOMAIN` | required — public hostname Traefik routes to this service |
| `DATA_DIR` | optional, default `/data` — container path of the shared volume |
| `HOST_DATA_DIR` | leave **unset** — see the shared volume below |
| `PORT` | optional, default `8080` — must match the Traefik `loadbalancer.server.port` label |
| `LOG_LEVEL` | optional, default `INFO` |
| `ASR_ENGINE`, `MODEL_DIR` | optional, defaults `parakeet` / `/models` |
| `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_LANGUAGE` | optional, fallback backend only — see [§ Environment variables](#️-environment-variables) |
| `TRAEFIK_NETWORK_NAME`, `TRAEFIK_CERTRESOLVER` | optional, defaults `traefik` / `myresolver` |

### The shared audio volume

jitsi-capture's `jitsi-capture-data` named volume is mounted here as
`external: true` at the very same path (`DATA_DIR`, `/data`), so the
`audio_path` in the webhook resolves without translation and `HOST_DATA_DIR`
stays unset. jitsi-capture writes `/data/jobs/<id>/audio.webm` (and `tracks/`);
this service only reads those and writes its own state under
`/data/transcriber/`.

**One-time step per volume.** jitsi-capture runs as root, so `/data` and
`/data/jobs` are `755 root` — uid `10001` (this container's user) can read the
audio but cannot create `/data/transcriber` itself. Create it once, from the
jitsi-capture container, before the first transcriber deploy:

```bash
docker compose exec jitsi-capture sh -c 'mkdir -p /data/transcriber && chown 10001 /data/transcriber'
```

Deliberately not automated: neither the Dockerfile, the compose file nor an
entrypoint touches the volume's ownership — a service that chowns a volume it
shares with another service is how the other service loses its data.

The model cache is a second, ordinary named volume (`transcriber-models` at
`MODEL_DIR`): the image ships no ASR model, and the first transcription
downloads ~1–2 GB into it.

### Networking

The stack joins only the external Traefik network. Public traffic reaches
`POST /webhook` and `GET /health` through Traefik on `DOMAIN`; the sibling
services reach the container directly by name on that same network:

* jitsi-capture's `WEBHOOK_URL` → `http://transcriber:8080/webhook`
* transcriber's `TR2OUTLINE_URL` → tr2outline's Anarlog endpoint
* transcriber's callback target arrives in the webhook (`callback_url`,
  jitsi-capture's `/notify`)

---

## 🗺️ Roadmap

- [x] HTTP receiver: `POST /webhook` (HMAC verification, job queue), `GET /health`
- [ ] Parakeet-tdt-0.6b-v3 backend via onnx-asr (int8, CPU), `ASR_ENGINE` switch
- [x] Per-track segment merge by `offset_s` → a `Name: text` feed with timecodes
- [x] Anarlog-format webhook to tr2outline + parsing the document URL from the reply
- [x] `POST <callback_url>` callback with the signed Zulip message
- [x] Dockerfile + image build in CI (ghcr.io)
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
