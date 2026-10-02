#!/usr/bin/env bash
# Runs ON the bench instance (Ubuntu 24.04 DLAMI base). Idempotent.
# Layout: /opt/bench/scripts (synced from S3), /opt/bench/work (BENCH_WORK).
set -euo pipefail
export BENCH_WORK=/opt/bench/work HOME=/root PATH=/root/.local/bin:/usr/local/bin:$PATH
mkdir -p "$BENCH_WORK" /opt/bench/bin
cd /opt/bench

if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export DEBIAN_FRONTEND=noninteractive
dpkg -s libsndfile1 >/dev/null 2>&1 || { apt-get update -q; apt-get install -yq libsndfile1 xz-utils; }

# Static ffmpeg (libvmaf + x264 + x265 + lame + libwebp), recorded by sha256.
if [[ ! -x /opt/bench/bin/ffmpeg ]]; then
  curl -sSfL -o /tmp/ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
  sha256sum /tmp/ffmpeg.tar.xz | tee /opt/bench/ffmpeg.sha256
  tar -xJf /tmp/ffmpeg.tar.xz -C /tmp && cp /tmp/ffmpeg-*-static/ffmpeg /tmp/ffmpeg-*-static/ffprobe /opt/bench/bin/
fi
ln -sf /opt/bench/bin/ffmpeg /usr/local/bin/ffmpeg; ln -sf /opt/bench/bin/ffprobe /usr/local/bin/ffprobe
ffmpeg -hide_banner -version | head -1

cd /opt/bench/scripts
if [[ ! -x /opt/bench/.venv/bin/python ]]; then uv venv -p 3.11 /opt/bench/.venv; fi
PY=/opt/bench/.venv/bin/python
uv pip install --python $PY -r requirements.in
$PY fetch_sources.py
uv pip install --python $PY --no-deps -e "$BENCH_WORK/src/videoseal"
uv pip freeze --python $PY > requirements-gpu-main.lock

# SilentCipher pins torch<=2.0 -> isolated env (reference-only row).
if [[ ! -x /opt/bench/.venv-sc/bin/python ]]; then uv venv -p 3.10 /opt/bench/.venv-sc; fi
uv pip install --python /opt/bench/.venv-sc/bin/python "torch==2.0.1" "numpy<2" soundfile librosa==0.10.2 pydub scipy pesq pystoi pyloudnorm soxr pillow matplotlib pyyaml huggingface_hub
uv pip install --python /opt/bench/.venv-sc/bin/python --no-deps -e "$BENCH_WORK/src/silentcipher"
uv pip freeze --python /opt/bench/.venv-sc/bin/python > requirements-gpu-silentcipher.lock
$PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))'
echo BOOTSTRAP_OK
