#!/usr/bin/env bash
# up:   push scripts/ to the scratch bucket and pull them onto the instance
# down: push instance results, locks, logs and figure media to the bucket and pull them locally
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
WORK=${BENCH_WORK:-$HOME/op-fp-bench-work}
STATE=${BENCH_STATE:-$WORK/gpu-state.env}
# shellcheck disable=SC1090
. "$STATE"
AWS=(aws --profile "$PROFILE" --region "$REGION")
case ${1:-up} in
  up)
    "${AWS[@]}" s3 sync "$HERE" "s3://$BUCKET/scripts" --exclude '__pycache__/*' --exclude '*.pyc' --exclude '.pytest_cache/*' --only-show-errors
    "$HERE/gpu/remote.sh" -t 300 "mkdir -p /opt/bench/scripts && aws s3 sync s3://$BUCKET/scripts /opt/bench/scripts --only-show-errors && chmod +x /opt/bench/scripts/gpu/*.sh && ls /opt/bench/scripts | wc -l"
    ;;
  down)
    "$HERE/gpu/remote.sh" -t 1200 "cd /opt/bench && aws s3 sync /opt/bench/results s3://$BUCKET/results --delete --only-show-errors; for f in requirements-gpu-main.lock requirements-gpu-nmfp.lock weights.lock.json sources.lock.json; do aws s3 cp /opt/bench/scripts/\$f s3://$BUCKET/locks/ --only-show-errors; done; aws s3 cp /opt/bench/ffmpeg.sha256 s3://$BUCKET/locks/ --only-show-errors; aws s3 cp /opt/bench/work/corpus/corpus_manifest.json s3://$BUCKET/locks/ --only-show-errors; aws s3 sync /opt/bench/work/logs s3://$BUCKET/logs --only-show-errors; aws s3 sync /opt/bench/work/out/security s3://$BUCKET/media/security --only-show-errors; aws s3 sync /opt/bench/work/out/localize s3://$BUCKET/media/localize --only-show-errors; true"
    mkdir -p "$HERE/../results-gpu"
    "${AWS[@]}" s3 sync "s3://$BUCKET/results" "$HERE/../results-gpu" --delete --only-show-errors
    "${AWS[@]}" s3 sync "s3://$BUCKET/locks" "$HERE/../results-gpu/locks" --only-show-errors
    "${AWS[@]}" s3 sync "s3://$BUCKET/logs" "$WORK/gpu-logs" --only-show-errors
    "${AWS[@]}" s3 sync "s3://$BUCKET/media" "$WORK/gpu-media" --only-show-errors
    ;;
  pull)  # non-destructive: copy instance results into results-rerun/ and the full search arrays into
         # $WORK/desc-rerun. Never `down` into a directory holding results produced elsewhere:
         # its `s3 sync --delete` removes them.
    "$HERE/gpu/remote.sh" -t 1800 "aws s3 sync /opt/bench/results s3://$BUCKET/results --only-show-errors; aws s3 sync /opt/bench/work/logs s3://$BUCKET/logs --only-show-errors; for d in search_abo search_audio search_video; do aws s3 sync /opt/bench/work/desc/\$d s3://$BUCKET/desc/\$d --only-show-errors; done; for f in requirements-gpu-main.lock requirements-gpu-nmfp.lock; do aws s3 cp /opt/bench/scripts/\$f s3://$BUCKET/locks/ --only-show-errors; done; aws s3 cp /opt/bench/ffmpeg.sha256 s3://$BUCKET/locks/ --only-show-errors; true"
    mkdir -p "$HERE/../results-rerun" "$WORK/desc-rerun" "$WORK/gpu-logs-rerun"
    "${AWS[@]}" s3 sync "s3://$BUCKET/results" "$HERE/../results-rerun" --only-show-errors
    "${AWS[@]}" s3 sync "s3://$BUCKET/locks" "$HERE/../results-rerun/locks" --only-show-errors
    "${AWS[@]}" s3 sync "s3://$BUCKET/desc" "$WORK/desc-rerun" --only-show-errors
    "${AWS[@]}" s3 sync "s3://$BUCKET/logs" "$WORK/gpu-logs-rerun" --only-show-errors
    ;;
esac
