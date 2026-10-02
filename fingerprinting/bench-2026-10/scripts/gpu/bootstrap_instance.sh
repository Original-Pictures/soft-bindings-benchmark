#!/usr/bin/env bash
# Runs ON the bench instance (Ubuntu 24.04 DLAMI base). Idempotent.
# Layout: /opt/bench/scripts (synced from S3), /opt/bench/work (BENCH_WORK).
set -euo pipefail
export BENCH_WORK=/opt/bench/work HOME=/root PATH=/root/.local/bin:/usr/local/bin:$PATH
mkdir -p "$BENCH_WORK" /opt/bench/bin
cd /opt/bench

if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export DEBIAN_FRONTEND=noninteractive
# libchromaprint-tools: fpcalc (Chromaprint / ISCC audio); g++ + libgomp: TMK build;
# libav*-dev + cmake + pkg-config: vpdq builds its C++ core against FFmpeg; fonts for drawtext.
PKGS="libsndfile1 xz-utils libchromaprint-tools g++ make cmake pkg-config libgomp1 libavcodec-dev libavformat-dev \
libavutil-dev libswscale-dev libavdevice-dev libavfilter-dev fonts-dejavu-core"
dpkg -s $PKGS >/dev/null 2>&1 || { apt-get update -q; apt-get install -yq $PKGS; }
fpcalc -version

# Static ffmpeg (x264, x265, libopus, lame, freetype), recorded by sha256.
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
# laion_clap pins numpy<2 but runs on numpy 2 (checked in the smoke run); install without deps.
uv pip install --python $PY --no-deps laion_clap==1.1.7 torchlibrosa==0.1.0
uv pip install --python $PY --no-deps "git+https://github.com/lyakaap/ISC21-Descriptor-Track-1st@228aad34ac5d"
$PY fetch_sources.py
uv pip install --python $PY --no-deps -e "$BENCH_WORK/src/videoseal"
uv pip install --python $PY vpdq
uv pip freeze --python $PY > requirements-gpu-main.lock

# TMK+PDQF command-line tools (g++, OpenMP).
if [[ ! -x "$BENCH_WORK/src/ThreatExchange/tmk/cpp/tmk-query-parallel" ]]; then
  make -C "$BENCH_WORK/src/ThreatExchange/tmk/cpp" FFMPEG=/opt/bench/bin/ffmpeg tmk-hash-video tmk-query tmk-query-parallel -j8
fi

# NMFP (GPL-3.0, reference row): the authors' TensorFlow 2.13 stack in its own environment.
# CPU inference: the model is a small convnet and 16 vCPUs keep it off the critical path,
# which avoids pairing TF 2.13 with the DLAMI's CUDA 12 runtime.
if [[ ! -x /opt/bench/.venv-nmfp/bin/python ]]; then uv venv -p 3.11 /opt/bench/.venv-nmfp; fi
uv pip install --python /opt/bench/.venv-nmfp/bin/python "tensorflow-cpu==2.13.1" soundfile pyyaml pandas==2.1.4 \
  essentia==2.1b6.dev1110 faiss-cpu==1.7.4
uv pip freeze --python /opt/bench/.venv-nmfp/bin/python > requirements-gpu-nmfp.lock
$PY fetch_weights.py
# iscc-sdk fetches pinned ffmpeg/ffprobe/fpcalc builds on first use; do it once, not per worker.
$PY -c "import iscc_sdk, iscc_sci; iscc_sdk.install(); iscc_sci.get_model()"
$PY -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(CPU helper)")'
echo BOOTSTRAP_OK
