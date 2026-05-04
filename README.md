# xpe-exp

Experimental code that extends the **Cross-Prompt Encoder (XPE)** paper to generative (decoder-only) transformers, plus visual and quantitative analyses comparing XPE vs SPT (Soft Prompt Tuning) representations.

Companion repo to [`micm-nlp`](https://github.com/bmikaberidze/micm-nlp) — the toolkit ships the building blocks (`MODEL`, `DATASET`, PEFT dispatch, XPE module); this repo holds the experiment configs, dataset prep, eval scripts, and analyses.

## Paper

> Mikaberidze, B., Saghinadze, T., Ostermann, S., & Müller, P.
> **Cross-Prompt Encoder for Low-Performing Languages.**
> *Findings of IJCNLP–AACL 2025* — [ACL Anthology](https://aclanthology.org/2025.findings-ijcnlp.144/) · [arXiv:2508.10352](https://arxiv.org/abs/2508.10352)

## Install

```bash
git clone https://github.com/bmikaberidze/xpe-exp.git
cd xpe-exp
pip install -r requirements.txt   # pins micm-nlp==0.1.0
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

## Layout

```
config/   YAML experiment configs (FTP-reframed XSC + Belebele on BLOOM, AYA)
src/      experiment-local helpers (config_utils, hub_upload)
scripts/  runners (run, run_xlt) + dataset prep, model utilities, eval analyses
tests/    unit tests
```

## Datasets

Reframe MCQA benchmarks to FTP (First-Token Prediction) format and tokenize per model. Artefacts land under `artefacts/datasets/benchmarks/mcqa/{belebele_ftp,xstory_cloze_ftp}/<lang>/`, with tokenized variants written to a `tokenized|<org>|<model>` subfolder per tokenizer.

```bash
# 1. Reframe sources to FTP (downloads from HF, saves to disk)
python -m scripts.datasets.reframe_bebe_to_ftp   # 122 Belebele language configs
python -m scripts.datasets.reframe_xsc_to_ftp    # 11 XStoryCloze language configs

# 2. Tokenize every language subdirectory with the chosen tokenizer
python -m scripts.datasets.preprocess_dir --config ./config/proc.ds.bebe.tok.bloom.yml
python -m scripts.datasets.preprocess_dir --config ./config/proc.ds.bebe.tok.aya.yml
python -m scripts.datasets.preprocess_dir --config ./config/proc.ds.xsc.tok.bloom.yml
python -m scripts.datasets.preprocess_dir --config ./config/proc.ds.xsc.tok.aya.yml
```

For a single language, use `preprocess.py` and pin `ds.dirs` (e.g. `mcqa/belebele_ftp/eng_Latn`) in the config.

## Usage

Train / fine-tune / evaluate / test via the unified runner, configured by YAML. The config's `mode:` field selects what runs (`train` / `finetune` / `evaluate` / `test`).

```bash
# Zero-shot test on Belebele (one language at a time; pin ds.dirs in the config)
python -m scripts.run --config ./config/test.lm.bloom.ds.bebe.yml
python -m scripts.run --config ./config/test.lm.aya.ds.bebe.yml

# Zero-shot eval on XStoryCloze validation
python -m scripts.run --config ./config/eval.lm.bloom.ds.xsc.yml
python -m scripts.run --config ./config/eval.lm.aya.ds.xsc.yml

# Tune XPE on XSC (full)
python -m scripts.run --config ./config/tune.xpe.lm.bloom.ds.xsc.yml
python -m scripts.run --config ./config/tune.xpe.lm.aya.ds.xsc.yml

# Smoke variants — 1% subset, 1 epoch — for fast end-to-end verification
python -m scripts.run --config ./config/tune.smoke.xpe.lm.bloom.ds.xsc.yml
python -m scripts.run --config ./config/tune.smoke.xpe.lm.aya.ds.xsc.yml
```

Each run logs to W&B (set `WANDB_API_KEY` in `.env`, or flip `report_to: none` in the config for offline). For aya-8b on a single 48 GB GPU, the configs use `torch_dtype: bfloat16` + `device_map: auto` so weights stream directly to the GPU instead of buffering in CPU RAM (avoids cgroup OOM under tight SLURM allocations).

The full aya tune is dataset-size-agnostic: `max_steps: 6000` with `eval_steps: 500` / `save_steps: 500` and early stopping (`patience: 5`, `early_stopping_after: 0.5`). XPE adapter trains at `learning_rate: 1e-4`; the per-param-group override block is left commented as a template for re-enabling a separate LR on `xpe_embedding`.

Visual / quantitative analyses (XPE vs SPT representations):

```bash
python -m scripts.evals.plot   --config <plot-config>
python -m scripts.evals.quant  --config <quant-config>
```

## License

MIT
