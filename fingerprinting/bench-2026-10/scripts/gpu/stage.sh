#!/usr/bin/env bash
# Runs ON the instance. Usage: stage.sh <stage> [args]
# Every stage logs to /opt/bench/work/logs/<stage>.log and is safe to re-run: extractors
# skip descriptor files that exist, searches skip existing npz, renders skip existing media.
set -uo pipefail
export BENCH_WORK=/opt/bench/work BENCH_RESULTS=/opt/bench/results HOME=/root PYTHONUNBUFFERED=1
# Pool workers are single-threaded; BLAS/OpenMP pools must be capped before numpy loads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/usr/local/bin:/root/.local/bin:$PATH
export TMK_BIN=$BENCH_WORK/src/ThreatExchange/tmk/cpp FFMPEG=/opt/bench/bin/ffmpeg
PY=/opt/bench/.venv/bin/python
NPY=/opt/bench/.venv-nmfp/bin/python
LOGS=$BENCH_WORK/logs
W=$(( $(nproc --all) - 2 ))  # --all: nproc honours OMP_NUM_THREADS
mkdir -p "$LOGS" "$BENCH_RESULTS"
cd /opt/bench/scripts
stage=${1:-status}; shift || true
run() { local name=$1; shift; echo "== $(date -u +%FT%TZ) $*" >> "$LOGS/$name.log"; "$@" >> "$LOGS/$name.log" 2>&1; echo "== exit $?" >> "$LOGS/$name.log"; }
case $stage in
  setup)
    bash gpu/bootstrap_instance.sh > "$LOGS/bootstrap.log" 2>&1 || { echo BOOTSTRAP_FAILED; tail -20 "$LOGS/bootstrap.log"; exit 1; }
    echo SETUP_DONE ;;
  corpus) run corpus $PY fetch_corpus.py --scale "${1:-full}" --workers 48; tail -5 "$LOGS/corpus.log" ;;
  image)   # retrieval track, all methods
    run image-extract $PY bench_image.py extract --workers $W --sets "${IMAGE_SETS:-reg,dist,pos,neg,hard,disc}"
    run image-search $PY bench_image.py dedup
    run image-search $PY bench_image.py search
    run image-metrics $PY bench_image.py metrics
    tail -30 "$LOGS/image-metrics.log" ;;
  audio)
    run audio-render $PY bench_audio.py render --workers $W
    run audio-extract $PY bench_audio.py extract --workers $W
    bash "$0" nmfp
    run audio-search $PY bench_audio.py search
    run audio-metrics $PY bench_audio.py metrics
    tail -12 "$LOGS/audio-metrics.log" ;;
  nmfp)
    run nmfp $PY nmfp_collect.py list
    D=$BENCH_WORK/weights/nmfp; [[ -d $D/nmfp-triplet ]] || (cd "$D" && unzip -oq nmfp-triplet.zip)
    CFG=$(find "$D" -name config.yaml -path "*triplet*" | head -1)
    (cd "$BENCH_WORK/src/neural-music-fp" && run nmfp $NPY extraction.py "$BENCH_WORK/nmfp_lists/all.txt" "$CFG" "$BENCH_WORK/nmfp_raw" --workers $W)
    run nmfp $PY nmfp_collect.py collect
    tail -8 "$LOGS/nmfp.log" ;;
  video)
    run video-render $PY bench_video.py render --workers $W
    run video-extract $PY bench_video.py extract --workers $W
    run video-search $PY bench_video.py search
    run video-metrics $PY bench_video.py metrics
    tail -12 "$LOGS/video-metrics.log" ;;
  wm)      # watermark combination: needs the retrieval thresholds (image/audio/video metrics) first
    for m in image audio video; do run wm $PY bench_wmcombo.py render $m; done
    run wm $PY bench_image.py extract --workers $W --sets wm_ref,wm_pos
    run wm $PY bench_audio.py extract --workers $W
    run wm $PY nmfp_collect.py list
    run wm $PY bench_video.py extract --workers $W
    for m in image audio video; do run wm $PY bench_wmcombo.py decode $m; done
    run wm $PY bench_wmcombo.py analyze
    tail -40 "$LOGS/wm.log" ;;
  edits)   # partial-edit + localization track
    run localize $PY edits.py
    run localize $PY bench_image.py extract --workers $W --sets edit
    run localize $PY bench_image.py search
    run localize $PY bench_localize.py all
    tail -20 "$LOGS/localize.log" ;;
  security) run security $PY bench_security.py all; tail -20 "$LOGS/security.log" ;;
  stability) run stability $PY stability.py make; tail -5 "$LOGS/stability.log" ;;
  latency) run latency $PY latency.py; tail -40 "$LOGS/latency.log" ;;  # only on an otherwise idle host
  verify)  # second-stage SIFT verification of image top-1 candidates (needs image search)
    run verify $PY bench_verify.py --only "${1:-DINOv2-S,PDQ}" --workers $W
    tail -5 "$LOGS/verify.log" ;;
  rerun)   # 2026-09-28 re-run for per-query outputs: retrieval tracks, watermark combination, verification.
    # DISC21, edits/localization, security and stability are not re-run; their 2026-09-26 results stand.
    for s in image audio video wm verify; do IMAGE_SETS=reg,dist,pos,neg,hard bash "$0" "$s" > /dev/null 2>&1; echo "$(date -u +%FT%TZ) $s done" >> "$LOGS/queue.log"; done
    echo QUEUE_DONE >> "$LOGS/queue.log" ;;
  queue)   # everything, in dependency order
    for s in image audio video wm edits security stability; do bash "$0" "$s" > /dev/null 2>&1; echo "$(date -u +%FT%TZ) $s done" >> "$LOGS/queue.log"; done
    echo QUEUE_DONE >> "$LOGS/queue.log" ;;
  killall) pkill -f "stage.sh queue"; pkill -f "bench_|extraction.py|edits.py|nmfp_collect"; sleep 2; pgrep -af "bench_|extraction.py" || echo "all stopped" ;;
  status)
    for f in "$LOGS"/*.log; do echo "== $f"; tail -n "${1:-3}" "$f"; done
    ls "$BENCH_RESULTS"/* 2>/dev/null | head -80
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    uptime; df -h / | tail -1 ;;
  *) echo "unknown stage $stage"; exit 64 ;;
esac
