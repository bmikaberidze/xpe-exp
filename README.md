# xpe-exp

Experimental code that extends the **Cross-Prompt Encoder (XPE)** paper to generative (decoder-only) transformers, plus visual and quantitative analyses comparing XPE vs SPT (Soft Prompt Tuning) representations.

Companion repo to [`micm-nlp`](https://github.com/bmikaberidze/micm-nlp) — the toolkit ships the building blocks (`MODEL`, `DATASET`, PEFT dispatch, XPE module, the `run-group` runner); this repo holds the experiment configs, dataset prep, the cross-lingual-transfer runner, eval scripts, and analyses.

## Paper

> Mikaberidze, B., Saghinadze, T., Ostermann, S., & Müller, P.
> **Cross-Prompt Encoder for Low-Performing Languages.**
> *Findings of IJCNLP–AACL 2025* — [ACL Anthology](https://aclanthology.org/2025.findings-ijcnlp.144/) · [arXiv:2508.10352](https://arxiv.org/abs/2508.10352)

## Install

```bash
git clone https://github.com/bmikaberidze/xpe-exp.git
cd xpe-exp
pip install -r requirements.txt   # micm-nlp v0.5.0 (git tag)
cp .env.example .env              # add WANDB_API_KEY, HF_TOKEN
```

For active toolkit development, install `micm-nlp` editable from a sibling checkout:

```bash
pip install -e ../micm-nlp
```

Docker:

```bash
docker build -t xpe-exp .
docker run --gpus all -it --rm -v $(pwd):/app -w /app xpe-exp bash
```

The code before micm-nlp 0.4 (the `run_xlt` / `run_xlt_meta` CLIs, `config/meta/` metaconfigs, the XStoryCloze and SIB-200 encoder configs) is at tag `pre-micm-nlp-0.4`.

## Layout

```
config/units/           unit configs: one run's model, data, PEFT and training setup
config/units/archive/   frozen units of old grids (what those runs actually used)
config/groups/          group configs: one experiment = one source group = one output dir
src/                    helpers: xlt_langs (language groups), sib200_meta, hub_upload, utils
scripts/xlt_runner.py   the cross-lingual-transfer runner (tune on source langs, test the rest)
scripts/datasets/       dataset prep (reframe, folds, tokenize)
scripts/evals/          unify / aggregate results, plots, representation analyses
tests/                  unit tests
```

## Datasets

Reframe MCQA benchmarks to FTP (First-Token Prediction) format and tokenize per model. Artefacts land under `artefacts/datasets/benchmarks/mcqa/belebele_ftp/<lang>/`, with tokenized variants in a `tokenized--<org>--<model>` subfolder per tokenizer.

```bash
# 1. Reframe Belebele to FTP (downloads from HF, saves to disk): 122 languages
python -m scripts.datasets.reframe_bebe_to_ftp

# 2. Item-disjoint self-split folds (train 500 / val 100 / test 300 per fold)
python -m scripts.datasets.split_bebe_folds

# 3. Tokenize every language (root + fold0-2) with the chosen tokenizer
python -m scripts.datasets.preprocess_dir --config ./config/units/proc.ds.bebe.tok.aya.yml
python -m scripts.datasets.preprocess_dir --config ./config/units/proc.ds.bebe.tok.bloom.yml
```

## Usage

A **group config** lists runs; each entry names a unit config and the axes it varies
(`seed`, `fold`, `method`, `source_group`, `overrides`). The runner tunes on the entry's
source languages (a key of `LANG_GROUPS` in `src/xlt_langs.py`) and tests on every
other language:

```bash
# One entry locally (index 0); under SLURM the array task id picks the entry
python -m micm_nlp run-group \
  --group-config config/groups/26a_bebe_grid_aya.aya_high.yml \
  --runner scripts.xlt_runner:run --run-index 0

# A single unit config, no cross-lingual loop (e.g. a zero-shot test)
python -m micm_nlp run --config config/units/test.lm.aya.ds.bebe.yml
```

Entry shapes: `separate_test:` = tune then test; `tune_only: true` = tune only (LR
searches); no tuning with `adapter:` = replay a trained adapter; neither = zero-shot.
Results land in `artefacts/runs/groups/{group}/{time_id}_{name}/`; `scripts/evals/`
turns a grid into one table.

Visual / quantitative analyses (XPE vs SPT representations):

```bash
python -m scripts.evals.plot   --config <plot-config>
python -m scripts.evals.quant  --config <quant-config>
```

## License

MIT
