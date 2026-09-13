# transcribetor

Сервис локальной транскрипции записей звонков **Jitsi Meet** на **CPU** с атрибуцией
спикеров. Второй сервис цепочки из трёх:

```text
  [ Zulip ]  🎙️ реакция на сообщение
      │
      ▼
  [ jitsi-capture ]  (Go + Node/Chromium)
      │  пишет звонок, кладёт аудио в DATA_DIR
      │  ──► подписанный вебхук recording.finished
      ▼
  [ transcribetor ]  ◄── этот репозиторий
      │  ASR на CPU + слияние per-track сегментов
      │  ──► вебхук в формате Anarlog
      ▼
  [ tr2outline ]  (github.com/korjavin/tr2outline)
      │  форматирует Markdown и публикует документ в Outline
      │  ──► возвращает URL документа
      ▼
  [ jitsi-capture /notify ]  ──►  сообщение «transcript ready» в тему Zulip
```

Приватность — основное требование: аудио и текст не покидают локальный контур,
всё считается на CPU собственного сервера.

---

## 📥 Вход: вебхук `recording.finished`

jitsi-capture шлёт `POST /webhook`:

| Заголовок | Значение |
|---|---|
| `x-jitsi-capture-event` | `recording.finished` |
| `x-jitsi-capture-signature` | `sha256=<hex HMAC-SHA256 от сырого тела>`, секрет `WEBHOOK_SECRET` |

Подпись считается по **сырым байтам** тела (до разбора JSON) и сравнивается
константным по времени сравнением.

Тело:

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

`tracks[]` — необязательное поле: оно появляется только после того, как в
jitsi-capture заработает пер-трековая запись. Код обязан работать и без него.

Файлы по `audio_path` и `tracks[].path` видны напрямую: оба сервиса монтируют
один и тот же том в `DATA_DIR`, копирование по сети не нужно.

---

## 🧠 Движок ASR

**Решение владельца: [NVIDIA Parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
через [`onnx-asr`](https://github.com/istupakov/onnx-asr), квантизация int8, CPU.**

* 0.6B параметров, мультиязычная модель (в т.ч. EN и RU), выдаёт пословные
  таймстемпы — ровно то, что нужно для слияния треков.
* `onnx-asr` — тонкая обёртка над onnxruntime, без torch и без NeMo:
  установка лёгкая, инференс на CPU предсказуемый.
* Выбор бэкенда — переменная `ASR_ENGINE=parakeet|whisper`.

`transcribetor/transcribe.py` (faster-whisper `large-v3`, `int8`, CPU) переезжает
сюда из jitsi-capture и остаётся **временным/резервным** бэкендом: работает
сегодня, даёт сегменты с таймкодами, но заметно медленнее на CPU. Как только
Parakeet-бэкенд будет готов, whisper остаётся запасным вариантом
(`ASR_ENGINE=whisper`), а не удаляется.

---

## 🗣️ Атрибуция спикеров

Без диаризации: имя спикера уже известно из записи.

* **Если в вебхуке есть `tracks[]`** — каждый трек участника транскрибируется
  отдельно, все его сегменты сдвигаются на `offset_s` (момент входа участника
  в звонок относительно начала записи) и сливаются по времени в общую ленту:

  ```markdown
  [00:05] Alice: Всем привет, начнём с архитектуры.
  [00:18] Bob: По второму пункту предлагаю такое решение...
  ```

  Это точнее любой диаризации по общему миксу и стоит ровно столько же
  вычислений (суммарная длительность треков ≈ длительность звонка).

* **Если треков нет** — транскрибируется общий микс `audio_path`, лента идёт
  без имён; как запасной вариант имя доминирующего спикера может быть взято из
  тайм-лайна участников. Это осознанно слабее — путь по умолчанию — треки.

---

## 📤 Выход: вебхук в tr2outline (формат Anarlog)

`POST $TR2OUTLINE_URL` (по умолчанию `/api/webhooks/anarlog`):

| Заголовок | Значение |
|---|---|
| `x-anarlog-signature` | `sha256=<hex HMAC-SHA256 от сырого тела>`, секрет `ANARLOG_WEBHOOK_SECRET` |
| `x-anarlog-event` | `note.enhanced` |

tr2outline обрабатывает только события `note.enhanced`; остальные он
подтверждает `200 OK` и игнорирует. Тело (см. `models.go` в tr2outline):

```json
{
  "id": "<job id>",
  "event": "note.enhanced",
  "created_at": "2026-09-13T12:30:34Z",
  "data": {
    "meeting": {
      "id": "<job id>",
      "title": "<topic> (<дата>)",
      "participants": ["Alice", "Bob"],
      "note": "",
      "summaries": [],
      "action_items": []
    },
    "transcript_text": "[00:05] Alice: ...\n[00:18] Bob: ..."
  }
}
```

`note`, `summaries`, `action_items` пока пустые: суммаризации в цепочке нет,
tr2outline корректно рендерит документ и без них. В ответе tr2outline отдаёт
JSON с URL созданного документа — его нужно извлечь и передать в callback.

---

## 🔁 Callback в jitsi-capture

`POST <callback_url>` (адрес приходит в вебхуке, это `/notify` у jitsi-capture),
подпись тем же `WEBHOOK_SECRET` в заголовке `x-jitsi-capture-signature`:

```json
{
  "id": "<job id>",
  "content": "Transcript ready: [Meeting title](https://outline.your-domain.com/doc/...)"
}
```

`content` — готовое Markdown-сообщение **на английском**: jitsi-capture
публикует его в тему Zulip как есть.

---

## 🧩 Прямая публикация в Outline (резерв)

`transcribetor/outline_client.py` — минимальный клиент Outline
(`POST /api/documents.create`), переехавший вместе с кодом. Основной путь
публикации — tr2outline; клиент остаётся как прямой резервный путь на случай,
если tr2outline недоступен или не нужен в конкретной инсталляции.

---

## ⚙️ Переменные окружения

| Переменная | Обяз. | По умолчанию | Описание |
|---|---|---|---|
| `WEBHOOK_SECRET` | да | — | Общий секрет с jitsi-capture: проверка входящего вебхука и подпись callback |
| `DATA_DIR` | нет | `/data` | Общий том с аудио (тот же путь, что у jitsi-capture) |
| `ASR_ENGINE` | нет | `parakeet` | `parakeet` (onnx-asr) или `whisper` (faster-whisper) |
| `WHISPER_MODEL` | нет | `large-v3` | Модель faster-whisper |
| `WHISPER_DEVICE` | нет | `cpu` | Устройство faster-whisper |
| `WHISPER_COMPUTE_TYPE` | нет | `int8` | Тип вычислений faster-whisper |
| `WHISPER_LANGUAGE` | нет | — | Пусто = автоопределение языка |
| `TR2OUTLINE_URL` | да | — | Эндпоинт вебхука tr2outline |
| `ANARLOG_WEBHOOK_SECRET` | да | — | Секрет подписи вебхука в tr2outline |
| `OUTLINE_BASE_URL` | нет | — | Резервный путь: базовый URL Outline |
| `OUTLINE_API_KEY` | нет | — | Резервный путь: API-токен Outline |
| `OUTLINE_COLLECTION_ID` | нет | — | Резервный путь: UUID коллекции Outline |

Конфигурация **только** через переменные окружения. `.env` в git не попадает,
в `.env.example` лежат исключительно заглушки. Логи печатают **имя**
переменной, но никогда значение; полные URL Jitsi с токенами не логируются.

---

## 🚀 Запуск и разработка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

ruff check .          # линтер
pytest -q             # тесты, полностью офлайн — ни сети, ни загрузки моделей

# разовая транскрипция файла резервным бэкендом
python -m transcribetor.transcribe /path/to/audio.webm
```

Тесты подменяют `faster_whisper` и `requests` фейками, поэтому проходят без
сети и без скачивания моделей. CI (GitHub Actions, Python 3.12) гоняет ровно
эти две команды на каждый push в `master` и на каждый pull request.

**Docker:** TODO — образ появится вместе с HTTP-приёмником.

---

## 🗺️ План работ

- [ ] HTTP-приёмник: `POST /webhook` (проверка HMAC, очередь задач), `GET /health`
- [ ] Бэкенд Parakeet-tdt-0.6b-v3 через onnx-asr (int8, CPU), переключатель `ASR_ENGINE`
- [ ] Слияние per-track сегментов по `offset_s` → лента `Имя: текст` с таймкодами
- [ ] Отправка вебхука в формате Anarlog в tr2outline + разбор URL документа из ответа
- [ ] Callback `POST <callback_url>` с подписанным сообщением для Zulip
- [ ] Dockerfile + сборка образа в CI (ghcr.io)
- [ ] Ретраи и устойчивость: повтор отправки вебхука, обработка перезапуска сервиса
