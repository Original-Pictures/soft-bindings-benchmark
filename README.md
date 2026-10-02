# Watermarks and Fingerprints as Soft Bindings for Content Provenance

Code, results and LaTeX source for the preprint *Watermarks and Fingerprints as Soft Bindings for Content Provenance: An Open-Licence Benchmark for Images, Audio and Video*. The authors are Seyedmahdi Kazempourradi, Ramtin Mojtahedi and Behrang Mohseni, all at Original Pictures Technologies, Inc.

C2PA soft bindings recover a stripped provenance manifest from the content itself, in one of two ways: an invisible watermark read from the media, or a fingerprint looked up in a registry. This repository benchmarks openly licensed methods of both kinds on public media for images, audio and video, under one protocol:
- false-match rates are calibrated on held-out negatives;
- uncertainty is estimated at the level of the source item.

The compiled paper is [`soft-bindings-preprint.pdf`](soft-bindings-preprint.pdf).

## Layout

| Path | Contents |
|---|---|
| `paper/` | Manuscript: `main.tex`, `sections/`, `refs.bib`. `scripts/assemble.py` collects both tracks' generated numbers, tables and figures. `build.sh` rebuilds the PDF. |
| `watermarking/bench-2026-09/` | Watermarking track. `scripts/` is the harness (corpus fetch, embedding, attacks, decoding, aggregation). `results-gpu/` and `results-followup/` hold the raw per-configuration records, and `results.json` the aggregate. `paper/scripts/` generates this track's numbers, tables and figures. |
| `fingerprinting/bench-2026-10/` | Fingerprinting track. `scripts/` is the harness (fingerprint adapters, retrieval, localization, security, watermark complementarity). `results-gpu/` holds the raw records, including per-query score arrays (`_perquery/`). `tests/` holds the unit tests. `paper/scripts/` generates this track's numbers, tables and figures. |

`corpus_manifest.json` and `results-gpu/locks/` pin the inputs:
- each corpus item, with its source URL, licence and SHA-256;
- model weights (`weights.lock.json`) and upstream source trees (`sources.lock.json`);
- the Python environments of the GPU hosts (`requirements-gpu-*.lock`).

## Rebuild the paper from the recorded results

Requires [uv](https://docs.astral.sh/uv/), Python 3.11+ and Docker.

```sh
./paper/build.sh
```

The script:
1. runs both tracks' `paper/scripts/build_numbers.py`, which writes every cited number to `generated/numbers.tex` and writes every table;
2. assembles them into `paper/`;
3. compiles with pdfLaTeX in `texlive/texlive:latest-medium`;
4. packs the arXiv source and recompiles it without BibTeX;
5. writes `soft-bindings-preprint.pdf`.

No number in the manuscript is typed by hand. Each is a `\val{...}` macro produced in step 1 from the files in `results*/`.

Figures are committed. To redraw one, run the track's `paper/scripts/figures.py` (requires matplotlib). The galleries and residual figures also need the benchmark media, which are not distributed (see below).

## Re-run the experiments

Media never enter the repository. `fetch_corpus.py` downloads every item from its original source into `$BENCH_WORK` and checks it against the manifest hashes. Each track runs on its own:

```sh
cd watermarking/bench-2026-09            # or fingerprinting/bench-2026-10
export BENCH_WORK=~/wm-bench-work         # media, descriptors and caches
uv venv -p 3.11 $BENCH_WORK/.venv
uv pip install --python $BENCH_WORK/.venv/bin/python -r scripts/requirements.in
cd scripts
python fetch_sources.py && python fetch_weights.py   # pinned trees, SHA-256-checked weights
python fetch_corpus.py                               # fingerprinting: --scale full
```

**Watermarking:**
```sh
python bench_image.py && python bench_audio.py && python bench_video.py && python stability.py make
python aggregate.py <results dir>
```

**Fingerprinting:**
- Run `pytest ../tests` first.
- Then run the stages defined in `scripts/gpu/stage.sh`, in this order: image, audio, nmfp, video, wm, edits, security, stability, latency.

**Hardware used for the recorded results:**
- AWS g5.xlarge and g5.2xlarge (NVIDIA A10G 24 GB, AMD EPYC 7R32);
- a c5a CPU host for hash extraction;
- an Apple M5 Pro for the cross-device checks.

**AWS helper scripts.** `scripts/gpu/` holds helpers that provision a tagged temporary instance, move files through a scratch S3 bucket, run stages over SSM and tear everything down.
- They need `AWS_PROFILE_BENCH` and `BENCH_AWS_ACCOUNT`, and refuse any other account.
- `BENCH_PROTECTED_INSTANCE` can name an instance that teardown must never touch.
- `sync.sh down` uses `aws s3 sync --delete`, which deletes local results the host does not have.

The two decoder environments that cannot share dependencies are pinned separately: SilentCipher in `requirements-gpu-silentcipher.lock`, and NMFP (TensorFlow) in `requirements-gpu-nmfp.lock`.

## Licences

- **Code** (everything outside `paper/` and the PDF): [Apache-2.0](LICENSE).
- **Manuscript, figures and tables:** [CC BY 4.0](LICENSE-paper).
- **Third-party models and data:** not redistributed. `fetch_weights.py` and `fetch_corpus.py` download them under their own licences, which the paper's method tables list. Several were measured as reference points only:
  - DISC21 was used under its non-commercial research licence.
  - The SSCD and ISC21 weights were trained on DISC21.
  - NMFP is GPL-3.0 and runs as an external tool in its own environment.

## Citation

See [`CITATION.cff`](CITATION.cff). Correspondence: Seyedmahdi Kazempourradi, mahdi@originalpictures.com.
