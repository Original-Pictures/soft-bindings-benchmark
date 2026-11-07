#!/usr/bin/env bash
# v1.1 reproduction of the ISCC rerun on the LOCAL SMOKE SLICE of the image track (1,720 ABO items:
# reg 120, dist 1,500, pos 60, neg 60, hard 30, bg 10), once per set of ISCC versions.
#
# The slice is too small to calibrate the paper's pair-level 1e-7 threshold, so this run checks
# direction only: Recall@1 per transformation, 0.9.5 versus 0.9.7. Full-scale numbers come from the
# fork's records (results-iscc/) and from the cloud rerun plan; nothing here replaces them.
#
#   run_slice.sh <0.9.5|0.9.7>
#
# Inputs are read only: corpus and weights are symlinks into $FP_WORK (default ~/op-fp-bench-work).
# Everything this script writes goes under $V11_WORK (default ~/op-v11-work) and
# rerun-2026-11/results-slice/iscc-sdk-<version>/; results-gpu/ and results-iscc/ are never touched.
set -euo pipefail
version=$1
case $version in
  0.9.5) PY=${PY095:-$HOME/op-fp-bench-work/.venv/bin/python} ;;
  0.9.7) PY=${PY097:-$HOME/op-fp-bench-work/.venv-iscc097/bin/python} ;;
  *) echo "unknown version $version" >&2; exit 2 ;;
esac
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=$(dirname "$HERE")
FP_WORK=${FP_WORK:-$HOME/op-fp-bench-work}
V11_WORK=${V11_WORK:-$HOME/op-v11-work}
export BENCH_WORK=$V11_WORK/slice-$version BENCH_RESULTS=$HERE/results-slice/iscc-sdk-$version PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$BENCH_WORK/desc" "$BENCH_WORK/out" "$BENCH_RESULTS"
ln -sfn "$FP_WORK/corpus" "$BENCH_WORK/corpus"
ln -sfn "$FP_WORK/weights" "$BENCH_WORK/weights"
cp "$FP_WORK/desc/dedup.json" "$BENCH_WORK/desc/dedup.json"
"$PY" -c "import iscc_sdk, iscc_core; print('iscc-sdk', iscc_sdk.__version__, 'iscc-core', iscc_core.__version__)"
cd "$BENCH/scripts"
W=$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) - 3 ))
t0=$(date +%s)
"$PY" bench_image.py extract --only ISCC-Image-64,ISCC-Image-256 --sets reg,dist,pos,neg,hard --workers $W
t1=$(date +%s)
"$PY" bench_image.py search --only ISCC-Image-64,ISCC-Image-256
"$PY" bench_image.py metrics --only ISCC-Image-64,ISCC-Image-256 || true
t2=$(date +%s)
echo "extract_s=$((t1-t0)) search_metrics_s=$((t2-t1))" | tee "$BENCH_RESULTS/timing.txt"
"$PY" -m pip freeze 2>/dev/null > "$BENCH_RESULTS/requirements.lock" || uv pip freeze --python "$PY" > "$BENCH_RESULTS/requirements.lock"
