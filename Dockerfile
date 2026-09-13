FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    MODEL_DIR=/models \
    HF_HOME=/models/hf

# No ffmpeg package: audio decoding goes through PyAV, bundled with faster-whisper.
# No ca-certificates package either: python:3.12-slim already ships
# /etc/ssl/certs/ca-certificates.crt, so outbound HTTPS (tr2outline, the HF hub)
# works without an apt layer.
RUN useradd -r -u 10001 app && mkdir -p /data /models && chown app /data /models

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY transcriber ./transcriber

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["python", "-c", "import os,sys,urllib.request;sys.exit(urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/health',timeout=3).status!=200)"]

CMD ["python", "-m", "transcriber"]
