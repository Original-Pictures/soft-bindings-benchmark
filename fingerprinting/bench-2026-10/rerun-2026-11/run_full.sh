#!/usr/bin/env bash
# v1.1 independent rerun of the ISCC image track at FULL scale on this Mac, for one set of ISCC versions:
# the paper's ABO registry (5,000 registered + 100,000 distractors), 2,000 positive and 2,000 negative sources
# under 36 transformations, 1,000 hard pairs. Same stages as the ISCC Foundation's iscc/run.sh (image,
# sensitivity, wm) with three differences: macOS core count, no DISC21 (CC BY-NC, not on this machine, so the
# DISC21 Recall@1/muAP columns are not reproduced), and the watermark-combination decode comes from
# wm_detect300.py (decoder-only path, converted to the harness's decode.json) instead of bench_wmcombo decode.
#
#   run_full.sh image <0.9.5|0.9.7>        extract, search, metrics (thresholds calibrated here, not copied)
#   run_full.sh sensitivity <version>      BENCH_DEDUP=none
#   run_full.sh wm <version>               marked-photo sets, extract, analyze (needs image and wm_detect300 decode)
#
# Inputs: ~/op-v11-work/full/corpus (fetched by iscc/fetch_missing.py, ABO sha256-checked), the paper's
# near-duplicate lists (results-gpu/image/_dedup.json, as iscc/run.sh does). Outputs: results-full/iscc-sdk-<v>/.
set -euo pipefail
stage=$1 version=$2
case $version in
  0.9.5) PY=${PY095:-$HOME/op-fp-bench-work/.venv/bin/python} ;;
  0.9.7) PY=${PY097:-$HOME/op-fp-bench-work/.venv-iscc097/bin/python} ;;
  *) echo "unknown version $version" >&2; exit 2 ;;
esac
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=$(dirname "$HERE")
V11=${V11_WORK:-$HOME/op-v11-work}
export BENCH_WORK=$V11/full-$version BENCH_RESULTS=$HERE/results-full/iscc-sdk-$version PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=$(( $(sysctl -n hw.ncpu) - 2 ))
mkdir -p "$BENCH_WORK/desc" "$BENCH_WORK/out/wm/image" "$BENCH_RESULTS"
ln -sfn "$V11/full/corpus" "$BENCH_WORK/corpus"
ln -sfn "$HOME/op-fp-bench-work/weights" "$BENCH_WORK/weights"
ln -sfn "$V11/wm300/marked" "$BENCH_WORK/out/wm/image/marked"
cd "$BENCH/scripts"
M=ISCC-Image-64,ISCC-Image-256

image() {
  cp "$BENCH/results-gpu/image/_dedup.json" "$BENCH_WORK/desc/dedup.json"
  "$PY" bench_image.py extract --only $M --sets reg,dist,pos,neg,hard --workers $W
  "$PY" bench_image.py search --only $M
  "$PY" bench_image.py metrics --only $M
  uv pip freeze --python "$PY" > "$BENCH_RESULTS/requirements.lock"
}

sensitivity() {
  BENCH_DEDUP=none "$PY" bench_image.py search --only $M
  BENCH_DEDUP=none "$PY" bench_image.py metrics --only $M
}

wm() {
  # render skips the marked photos that exist (drift300.py made them) and registers wm_ref / wm_pos
  "$PY" bench_wmcombo.py render image
  "$PY" "$HERE/wm_detect300.py" export-decode "$BENCH_WORK/out/wm/image/decode.json"
  "$PY" bench_image.py extract --only $M --sets wm_ref,wm_pos --workers $W
  "$PY" bench_wmcombo.py analyze
}

"$stage"
