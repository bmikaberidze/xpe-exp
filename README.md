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

## Usage

Train / fine-tune / evaluate via the unified runner, configured by YAML:

```bash
python -m scripts.run --config ./config/tune.xpe.lm.bloom.ds.xsc.yml
python -m scripts.run --config ./config/eval.lm.bloom.ds.xsc.yml
```

Visual / quantitative analyses (XPE vs SPT representations):

```bash
python -m scripts.evals.plot   --config <plot-config>
python -m scripts.evals.quant  --config <quant-config>
```

## License

MIT
