#!/usr/bin/env bash
# Runs ON a bench instance. 2026-09-28 re-run that keeps per-query outputs (see README).
# Usage: rerun.sh cpu|gpu
#
# Work split (G-instance vCPU quota leaves one 8-vCPU GPU host, so CPU-bound work goes to the
# 64-vCPU helper), coordinated through markers in the bench bucket:
#   cpu: image hashes -> audio track -> [wait gpu-image-desc] image dedup/search/metrics -> verification
#   gpu: learned image descriptors -> video track -> [wait cpu-image-results] watermark combination
# Descriptors travel through s3://$BUCKET/xfer. Nothing is deleted.
set -uo pipefail
role=${1:?cpu|gpu}
export BENCH_WORK=/opt/bench/work BENCH_RESULTS=/opt/bench/results HOME=/root PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BUCKET=$(cat /opt/bench/bucket)
PY=/opt/bench/.venv/bin/python
ST="bash /opt/bench/scripts/gpu/stage.sh"
LOG=$BENCH_WORK/logs/rerun-$role.log
W=$(( $(nproc --all) - 2 ))
SETS=reg,dist,pos,neg,hard
GPU_IMG=DINOv2-S,DINOv2-B,DINOv2-S-LSH256,OpenCLIP-B32,SSCD-mixup,SSCD-large,ISC21-1st,DINOHash-96
CPU_IMG=PDQ,PDQ-dihedral,aHash-64,dHash-64,pHash-64,pHash-256,wHash-64,BlockMean,MarrHildreth,ColorMoment,Blockhash-256,ISCC-Image-64,ISCC-Image-256
cd /opt/bench/scripts
mkdir -p "$BENCH_WORK/logs"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
mark() { echo "$(date -u +%FT%TZ)" | aws s3 cp - "s3://$BUCKET/markers/$1" --only-show-errors; say "marker $1"; }
wait_for() { say "waiting for $1"; until aws s3 ls "s3://$BUCKET/markers/$1" >/dev/null 2>&1; do sleep 60; done; say "got $1"; }
xfer_up() { for s in ${2//,/ }; do for m in ${1//,/ }; do aws s3 cp "$BENCH_WORK/desc/$s/$m.npy" "s3://$BUCKET/xfer/desc/$s/$m.npy" --only-show-errors; done; aws s3 cp "$BENCH_WORK/desc/$s/ids.json" "s3://$BUCKET/xfer/desc/$s/ids-$role.json" --only-show-errors; done; }
xfer_down() { aws s3 sync "s3://$BUCKET/xfer/desc" "$BENCH_WORK/desc" --exclude '*/ids-*.json' --only-show-errors; }
run() { local name=$1; shift; say "start $name"; "$@" >> "$BENCH_WORK/logs/$name.log" 2>&1; local rc=$?; say "end $name rc=$rc"; return $rc; }

case $role in
  cpu)
    run image-extract-cpu $PY bench_image.py extract --only "$CPU_IMG" --sets $SETS --workers $W
    xfer_up "$CPU_IMG" "$SETS"; mark cpu-image-desc
    $ST audio >/dev/null 2>&1; say "audio stage done"
    aws s3 sync "$BENCH_RESULTS/audio" "s3://$BUCKET/results/audio" --only-show-errors
    # the GPU host's watermark stage searches the audio registry and decodes the attacked negatives
    for d in "$BENCH_WORK"/desc/audio_*; do aws s3 sync "$d" "s3://$BUCKET/xfer/audio/$(basename "$d")" --only-show-errors; done
    aws s3 sync "$BENCH_WORK/out/audio/neg" "s3://$BUCKET/xfer/audio_q/neg" --only-show-errors
    mark cpu-audio
    wait_for gpu-image-desc
    xfer_down
    run image-search $PY bench_image.py dedup
    run image-search $PY bench_image.py search
    run image-metrics $PY bench_image.py metrics
    aws s3 sync "$BENCH_RESULTS/image" "s3://$BUCKET/results/image" --only-show-errors
    mark cpu-image-results
    run verify $PY bench_verify.py --only DINOv2-S,PDQ --workers $W
    aws s3 sync "$BENCH_RESULTS/image" "s3://$BUCKET/results/image" --only-show-errors
    aws s3 sync "$BENCH_WORK/desc/search_abo" "s3://$BUCKET/desc/search_abo" --only-show-errors
    aws s3 sync "$BENCH_WORK/desc/search_audio" "s3://$BUCKET/desc/search_audio" --only-show-errors
    aws s3 sync "$BENCH_WORK/logs" "s3://$BUCKET/logs-cpu" --only-show-errors
    mark cpu-done ;;
  gpu)
    run image-extract-gpu $PY bench_image.py extract --only "$GPU_IMG" --sets $SETS --workers $W
    xfer_up "$GPU_IMG" "$SETS"; mark gpu-image-desc
    $ST video >/dev/null 2>&1; say "video stage done"
    aws s3 sync "$BENCH_RESULTS/video" "s3://$BUCKET/results/video" --only-show-errors
    wait_for cpu-image-results
    wait_for cpu-audio
    xfer_down
    aws s3 sync "s3://$BUCKET/xfer/audio" "$BENCH_WORK/desc" --only-show-errors
    aws s3 sync "s3://$BUCKET/xfer/audio_q/neg" "$BENCH_WORK/out/audio/neg" --only-show-errors
    aws s3 sync "s3://$BUCKET/results/image" "$BENCH_RESULTS/image" --only-show-errors
    aws s3 sync "s3://$BUCKET/results/audio" "$BENCH_RESULTS/audio" --only-show-errors
    $ST wm >/dev/null 2>&1; say "wm stage done"
    aws s3 sync "$BENCH_RESULTS" "s3://$BUCKET/results" --only-show-errors
    aws s3 sync "$BENCH_WORK/desc/search_video" "s3://$BUCKET/desc/search_video" --only-show-errors
    aws s3 sync "$BENCH_WORK/logs" "s3://$BUCKET/logs-gpu" --only-show-errors
    mark gpu-done ;;
esac
say "rerun $role finished"
