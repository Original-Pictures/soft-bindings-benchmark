#!/usr/bin/env bash
# up:   push scripts/ to the scratch bucket and pull them onto the instance
# down: push instance results + thumbnails to the bucket and pull them locally
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
STATE=${BENCH_WORK:-$HOME/op-wm-bench-work}/gpu-state.env
# shellcheck disable=SC1090
. "$STATE"
AWS=(aws --profile "$PROFILE" --region "$REGION")
case ${1:-up} in
  up)
    "${AWS[@]}" s3 sync "$HERE" "s3://$BUCKET/scripts" --exclude '__pycache__/*' --exclude '*.pyc' --only-show-errors
    "$HERE/gpu/remote.sh" -t 300 "mkdir -p /opt/bench/scripts && aws s3 sync s3://$BUCKET/scripts /opt/bench/scripts --only-show-errors && chmod +x /opt/bench/scripts/gpu/*.sh && ls /opt/bench/scripts | wc -l"
    ;;
  down)
    "$HERE/gpu/remote.sh" -t 600 "cd /opt/bench && aws s3 sync /opt/bench/results s3://$BUCKET/results --delete --only-show-errors; aws s3 cp /opt/bench/scripts/requirements-gpu-main.lock s3://$BUCKET/locks/ --only-show-errors; aws s3 cp /opt/bench/scripts/requirements-gpu-silentcipher.lock s3://$BUCKET/locks/ --only-show-errors; aws s3 cp /opt/bench/scripts/weights.lock.json s3://$BUCKET/locks/ --only-show-errors; aws s3 cp /opt/bench/ffmpeg.sha256 s3://$BUCKET/locks/ --only-show-errors; aws s3 sync /opt/bench/work/logs s3://$BUCKET/logs --only-show-errors; aws s3 sync /opt/bench/thumbs s3://$BUCKET/thumbs --only-show-errors; aws s3 cp /opt/bench/corpus_manifest.json s3://$BUCKET/locks/ --only-show-errors; aws s3 sync /opt/bench/work/stability s3://$BUCKET/stability --only-show-errors; true"
    "${AWS[@]}" s3 sync "s3://$BUCKET/stability" "${BENCH_WORK:-$HOME/op-wm-bench-work}/stability" --only-show-errors
    "${AWS[@]}" s3 sync "s3://$BUCKET/thumbs" "$HERE/../thumbs" --only-show-errors
    mkdir -p "$HERE/../results-gpu"
    "${AWS[@]}" s3 sync "s3://$BUCKET/results" "$HERE/../results-gpu" --delete --exclude "stability/framehash-arm64-mac.json" --exclude "stability/arm64-*" --only-show-errors
    "${AWS[@]}" s3 sync "s3://$BUCKET/locks" "$HERE/../results-gpu/locks" --only-show-errors
    "${AWS[@]}" s3 sync "s3://$BUCKET/logs" "${BENCH_WORK:-$HOME/op-wm-bench-work}/gpu-logs" --only-show-errors
    ;;
esac
