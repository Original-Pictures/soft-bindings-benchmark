#!/usr/bin/env bash
# Reruns the ISCC image methods of the paper's image track, the image part of its watermark-combination track and
# the evasion and forgery attacks of its security track on a CPU host with the harness in scripts/, once per set of
# ISCC versions. The harness code differs from the paper's only in the ISCC surrogate's trim box
# (scripts/hash_surrogates.py), which follows the installed iscc-sdk; the ISCC versions come from this script, not
# from scripts/requirements.in.
#
#   iscc/run.sh setup <iscc-sdk version>   venv, corpus, TrustMark weights
#   iscc/run.sh image <iscc-sdk version>   image track: extract, search, metrics
#   iscc/run.sh wm <iscc-sdk version>      watermark-combination track (needs the image track first)
#   iscc/run.sh security <iscc-sdk version>     security track: the image attacks that involve ISCC (iscc/security.py
#                                               and bench_security.py collision; needs the image track first)
#   iscc/run.sh sensitivity <iscc-sdk version>  image track without the near-duplicate cleanup (BENCH_DEDUP=none)
#
# Versions: 0.9.5 runs iscc-sdk 0.9.5 with iscc-core 1.4.0 and iscc-lib 0.6.0, as the paper did; 0.9.7 runs the
# released update, iscc-sdk 0.9.7 with iscc-core 1.5.0, iscc-lib 0.7.0 and iscc-sci 0.4.1.
#
# Methods: the paper's ISCC-Image-64 and ISCC-Image-256 with both versions. The update also runs the paper's
# pilot-only ISCC-SCI-256 and two methods made from stored descriptors, ISCC-SCI-64 and ISCC-Combined-512
# (iscc/bench.py adds these three to the paper's method table). The paper reports no Semantic-Code to compare with.
#
# Environment: ROOT (default ~/sbb-iscc) holds one work directory per version; corpus, weights, the watermarked
# images and the transfer perturbations are shared. DISC21 is not downloaded: put a copy in $ROOT/corpus/disc21 or
# point DISC21_DIR to one; ABO_DIR may point to an existing copy of corpus/abo/img. The first setup links them,
# fetches the missing ABO photos and checks every ABO and DISC21 file against the paper's manifest; if fetching or
# the check fails, delete $ROOT/corpus/corpus_manifest.json before running setup again. Results go to
# results-iscc/iscc-sdk-<version>/. The near-duplicate lists come from the paper (results-gpu/image/_dedup.json):
# its dedup stage uses PDQ, SSCD and DINOv2-B, not ISCC.
set -euo pipefail

stage=$1 version=$2
case $version in
  0.9.5) ISCC="iscc-core==1.4.0 iscc-lib==0.6.0 iscc-sdk==0.9.5" SCI="" ;;
  0.9.7) ISCC="iscc-core==1.5.0 iscc-lib==0.7.0 iscc-sdk==0.9.7 iscc-sci==0.4.1" SCI=",ISCC-SCI-256" ;;
  *) echo "unknown iscc-sdk version $version" >&2; exit 2 ;;
esac

HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=$(dirname "$HERE")
ROOT=${ROOT:-$HOME/sbb-iscc}
export BENCH_WORK=$ROOT/work-$version BENCH_RESULTS=$BENCH/results-iscc/iscc-sdk-$version PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # pool workers run single-threaded
PY=$ROOT/venv-$version/bin/python
BENCHPY=$HERE/bench.py
EXTRACTED=ISCC-Image-64,ISCC-Image-256$SCI
ALL=$EXTRACTED${SCI:+,ISCC-SCI-64,ISCC-Combined-512}
W=$(( $(nproc --all) - 2 ))
UV=${UV:-uv}

setup() {
  "$UV" venv -q --allow-existing -p 3.11 "$ROOT/venv-$version"
  "$UV" pip install -q --python "$PY" --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0
  "$UV" pip install -q --python "$PY" -r "$HERE/requirements-cpu.in" $ISCC
  mkdir -p "$ROOT/corpus" "$ROOT/weights" "$ROOT/wm" "$BENCH_WORK/out" "$BENCH_RESULTS"
  ln -sfn "$ROOT/corpus" "$BENCH_WORK/corpus"
  ln -sfn "$ROOT/weights" "$BENCH_WORK/weights"
  ln -sfn "$ROOT/wm" "$BENCH_WORK/out/wm"  # the watermarked images and their decodes do not depend on ISCC
  if [[ ! -f $ROOT/corpus/corpus_manifest.json ]]; then
    gunzip -c "$BENCH/results-gpu/locks/corpus_manifest.json.gz" > "$ROOT/corpus/corpus_manifest.json"
    if [[ -n ${ABO_DIR:-} ]]; then mkdir -p "$ROOT/corpus/abo"; ln -sfn "$ABO_DIR" "$ROOT/corpus/abo/img"; fi
    if [[ -n ${DISC21_DIR:-} ]]; then ln -sfn "$DISC21_DIR" "$ROOT/corpus/disc21"; fi
    "$PY" "$HERE/fetch_missing.py"
  fi
  (cd "$BENCH/scripts" && "$PY" fetch_weights.py --only trustmark_encoder_Q,trustmark_decoder_Q,trustmark_yaml_Q)
  "$UV" pip freeze -q --python "$PY" > "$BENCH_RESULTS/requirements-cpu.lock"
}

image() {
  mkdir -p "$BENCH_WORK/desc"
  cp "$BENCH/results-gpu/image/_dedup.json" "$BENCH_WORK/desc/dedup.json"
  cd "$BENCH/scripts"
  "$PY" "$BENCHPY" image extract --only $EXTRACTED --sets reg,dist,pos,neg,hard,disc --workers $W
  "$PY" "$BENCHPY" derive
  "$PY" "$BENCHPY" image search --only $ALL
  "$PY" "$BENCHPY" image metrics
}

wm() {
  cd "$BENCH/scripts"
  "$PY" "$BENCHPY" wmcombo render image
  "$PY" "$BENCHPY" image extract --only $EXTRACTED --sets wm_ref,wm_pos --workers $W
  [[ -f $ROOT/wm/image/decode.json ]] || "$PY" "$BENCHPY" wmcombo decode image
  "$PY" "$BENCHPY" derive
  "$PY" "$BENCHPY" wmcombo analyze
}

security() {
  "$UV" pip install -q --python "$PY" -r "$HERE/requirements-cpu.in" -r "$HERE/requirements-security.in" $ISCC
  (cd "$BENCH/scripts" && "$PY" fetch_weights.py --only sscd_disc_mixup)
  mkdir -p "$ROOT/security" "$BENCH_RESULTS/security"
  ln -sfn "$ROOT/security" "$BENCH_WORK/out/security"  # the transfer perturbations do not depend on ISCC
  local shards=$(( $(nproc --all) / 4 )) pids=()
  for k in $(seq 0 $((shards - 1))); do
    OMP_NUM_THREADS=4 "$PY" "$HERE/security.py" craft --shard "$k/$shards" & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  export OMP_NUM_THREADS=8
  "$PY" "$HERE/security.py" transfer
  "$PY" "$HERE/security.py" surrogate
  (cd "$BENCH/scripts" && "$PY" bench_security.py collision --device cpu)
  "$UV" pip freeze -q --python "$PY" > "$BENCH_RESULTS/security/requirements-cpu.lock"
}

sensitivity() {
  cd "$BENCH/scripts"
  BENCH_DEDUP=none "$PY" "$BENCHPY" image search --only ISCC-Image-64,ISCC-Image-256
  BENCH_DEDUP=none "$PY" "$BENCHPY" image metrics
}

"$stage"
