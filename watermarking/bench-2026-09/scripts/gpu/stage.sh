#!/usr/bin/env bash
# Runs ON the instance. Usage: stage.sh <setup|bench-image|bench-audio|bench-video|all|status> [args]
# Every stage logs to /opt/bench/work/logs/<stage>.log and is safe to re-run
# (bench runners skip model configs whose result file already exists).
set -uo pipefail
export BENCH_WORK=/opt/bench/work BENCH_RESULTS=/opt/bench/results HOME=/root
export PATH=/usr/local/bin:/root/.local/bin:$PATH
PY=/opt/bench/.venv/bin/python
LOGS=$BENCH_WORK/logs
mkdir -p "$LOGS" "$BENCH_RESULTS"
cd /opt/bench/scripts
stage=${1:-status}; shift || true
case $stage in
  setup)
    bash gpu/bootstrap_instance.sh > "$LOGS/bootstrap.log" 2>&1 || { echo BOOTSTRAP_FAILED; exit 1; }
    $PY fetch_weights.py > "$LOGS/weights.log" 2>&1 &
    $PY fetch_corpus.py > "$LOGS/corpus.log" 2>&1
    wait
    echo SETUP_DONE ;;
  bootstrap) bash gpu/bootstrap_instance.sh > "$LOGS/bootstrap2.log" 2>&1; tail -2 "$LOGS/bootstrap2.log" ;;
  smoke-sc) BENCH_RESULTS=/tmp/sc-smoke /opt/bench/.venv-sc/bin/python bench_audio.py --only SilentCipher-44.1k --limit 1 2>&1 | grep -v -i warn | tail -12 ;;
  weights) $PY fetch_weights.py "$@" > "$LOGS/weights.log" 2>&1; tail -3 "$LOGS/weights.log" ;;
  corpus) $PY fetch_corpus.py "$@" > "$LOGS/corpus.log" 2>&1; tail -3 "$LOGS/corpus.log" ;;
  bench-image) $PY bench_image.py "$@" >> "$LOGS/bench-image.log" 2>&1; tail -5 "$LOGS/bench-image.log" ;;
  bench-audio) $PY bench_audio.py --only AudioSeal,WavMark,Perth "$@" >> "$LOGS/bench-audio.log" 2>&1; tail -5 "$LOGS/bench-audio.log" ;;
  bench-video) $PY bench_video.py "$@" >> "$LOGS/bench-video.log" 2>&1; tail -5 "$LOGS/bench-video.log" ;;
  bench-sc) /opt/bench/.venv-sc/bin/python bench_audio.py --only SilentCipher "$@" >> "$LOGS/bench-sc.log" 2>&1; tail -5 "$LOGS/bench-sc.log" ;;
  stability) $PY stability.py make "$@" >> "$LOGS/stability.log" 2>&1; tail -5 "$LOGS/stability.log" ;;
  killall) pkill -f "stage.sh queue"; pkill -f "bench_"; pkill -f "stability.py"; sleep 2; pgrep -af "bench_|stage.sh queue" || echo "all stopped" ;;
  kill) pkill -f "bench_${1:?which}.py" && echo killed || echo none ;;
  clean) rm -rf "$BENCH_RESULTS/${1:?which}" "$LOGS/bench-$1.log"; echo cleaned "$1" ;;
  queue)  # run everything remaining, in order; each runner skips finished configs
    # The image/audio runs are CPU-bound (codecs, SSIM, FLIP) on 4 vCPUs while the
    # GPU idles, so configs are sharded across parallel workers.
    for i in 0 1 2 3; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY bench_image.py --shard $i/4 >> "$LOGS/bench-image.$i.log" 2>&1 & done; wait
    OMP_NUM_THREADS=1 $PY bench_audio.py --only AudioSeal,WavMark,Perth --shard 0/2 >> "$LOGS/bench-audio.0.log" 2>&1 &
    OMP_NUM_THREADS=1 $PY bench_audio.py --only AudioSeal,WavMark,Perth --shard 1/2 >> "$LOGS/bench-audio.1.log" 2>&1 &
    /opt/bench/.venv-sc/bin/python bench_audio.py --only SilentCipher >> "$LOGS/bench-sc.log" 2>&1 &
    wait
    for s in bench-video stability; do bash "$0" "$s" >/dev/null 2>&1; done
    bash "$0" bench-image --cpu-latency >/dev/null 2>&1
    bash "$0" bench-audio --cpu-latency >/dev/null 2>&1
    bash "$0" bench-sc --cpu-latency >/dev/null 2>&1
    echo QUEUE_DONE ;;
  rest)  # second-phase driver after the image/audio pass
    $PY bench_video.py >> "$LOGS/bench-video.log" 2>&1 &
    # RivaGAN (reference only) runs single-image ONNX on CPU at ~8 min/DIV2K image; Kodak/CLIC/HDR only.
    OMP_NUM_THREADS=1 $PY bench_image.py --only RivaGAN --div2k 0 >> "$LOGS/bench-image-riva.log" 2>&1 &
    wait
    $PY stability.py make >> "$LOGS/stability.log" 2>&1
    $PY bench_image.py --cpu-latency >> "$LOGS/cpu-latency.log" 2>&1
    $PY bench_audio.py --cpu-latency --only AudioSeal,WavMark,Perth >> "$LOGS/cpu-latency.log" 2>&1
    /opt/bench/.venv-sc/bin/python bench_audio.py --cpu-latency --only SilentCipher-44.1k >> "$LOGS/cpu-latency.log" 2>&1
    echo REST_DONE >> "$LOGS/rest.log" ;;
  video2)  # video rerun after the 5 s OOM: 2.5 s clips, two GPU workers
    rm -rf "$BENCH_RESULTS/video"
    for i in 0 1; do $PY bench_video.py --shard $i/2 >> "$LOGS/bench-video2.$i.log" 2>&1 & done; wait
    echo VIDEO2_DONE >> "$LOGS/rest.log" ;;
  status)
    for f in "$LOGS"/*.log; do echo "== $f"; tail -n "${1:-4}" "$f"; done
    ls "$BENCH_RESULTS" 2>/dev/null | head -80
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    df -h / | tail -1 ;;
  *) echo "unknown stage $stage"; exit 64 ;;
esac
