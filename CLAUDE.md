# xpe-exp

## The XPE paper (the work we are extending)

The base/first paper lives at **`docs/papers/xpe.pdf`** (14 pages; ACL Findings IJCNLP
2025, `2025.findings-ijcnlp.144`). **ALWAYS read the appropriate section of this PDF
before making any claim about what the paper says/shows** — do not rely on memory or
on prior summaries in this conversation; quote the section/table you used.
- Extract text with `pypdf` (`pip install -q pypdf` if missing); poppler/pdftotext and
  PDF page-render are NOT installed. Page images can be pasted by the user instead.
- Known gotcha: the paper reports **two** DUAL variants — **DUAL^XPE-30 (encoder_ratio
  0.3)** and **DUAL^XPE-70 (encoder_ratio 0.7)** — both legitimate; neither is a "bug".
  Which wins is regime-dependent (DUAL-30 wins few-source/EnArZho rows, DUAL-70 wins
  Joshi5 rows). Core pattern (Table 1): XPE wins **low-performing/Unseen** targets, SPT
  wins **high-performing/Seen** targets (even at Joshi5, despite XPE's larger param
  count = the architectural-not-parametric proof), DUAL balances; fewer source langs
  shift the advantage from XPE toward SPT/DUAL.

## Run environment

Run repo scripts and tests with the **default `python` / `python3`** directly — no venv, no `source activate`:

```
python -m scripts.run_xlt_meta ...
python -m pytest tests/ -q
```

- If `import micm_nlp` fails (`ModuleNotFoundError`), install it editable from the sibling checkout:
  `python -m pip install -e /fscratch/bmikaberidze/micm-nlp`
- On the SLURM cluster, jobs are launched via `runtime/clusters/pegasus/shell/run.sh ... "python -m scripts...."` (the wrapper sets up the env), but we are already on SLURM cluster.

## GPU partitions / VRAM (SLURM)

In **SBATCH mode `run.sh` does NOT set `--partition` on the inner `srun`** — the job
inherits the `#SBATCH --partition=...` directive, whose default
(`B200,H200,H200-PCI,H100-PCI,H100-Trails,H100,A100-80GB`) **includes ≤80 GB nodes**.
To restrict, pass `--partition=...` to the `sbatch` CLI (it overrides the directive).
VRAM groups (see the reference table at the bottom of `run.sh`):

| group | partitions |
|-------|-----------|
| `> 80 GB` | `B200,H200,H200-PCI,H100-PCI,H100-Trails` (also RTXB6000, but it's ~5–10× slower — exclude) |
| `== 80 GB` | `H100,A100-80GB` |
| `< 80 GB` | `A100-40GB,A100-PCI,L40S,RTXA6000,RTX3090,batch,V100-32GB,V100-16GB` |

- **Aya (`aya-expanse-8b`) needs >80 GB VRAM** → its `sbatch` MUST add
  `--partition=B200,H200,H200-PCI,H100-PCI,H100-Trails`. Without it, jobs land on
  ≤80 GB nodes and OOM.
- **Bloomz-7b1 fits in 80 GB** → default partition is fine, no override needed.

## Core package: micm_nlp

`micm_nlp` is **our** core toolkit (model/trainer/dataset/PEFT machinery this repo
builds on) — **we maintain it alongside this repo**, it is not a frozen third-party dep.

- Lives in a **sibling checkout**: `/fscratch/bmikaberidze/micm-nlp` (dir uses a hyphen;
  the import name is `micm_nlp` with an underscore — `src/micm_nlp/`).
- It is its **own git repo** with its own branches — commit changes there separately.
- Installed editable (`pip install -e /fscratch/bmikaberidze/micm-nlp`), so edits in the
  sibling take effect here immediately (no reinstall).
- When behavior traces into `MODEL` / `TRAINER` / `DATASET` / `PEFT` / fold or seed plumbing,
  the source is in the sibling — read/edit it there, not only in this repo's `scripts/`.

## Repository layout

```
config/                         # all experiment configs (YAML)
  tune.{xpe|spt|dual}.lm.{model}.ds.{xsc|bebe}.yml   # finetune a PEFT method
  test.lm.{model}.ds.bebe[.fold].yml                 # eval; .fold = self-split test split
  proc.ds.{xsc|bebe}.tok.{aya|bloom}.yml             # tokenize a benchmark per backbone
  meta/{N}[a|b]_*.yml             # metaconfigs (LR sweeps etc); stem = wandb run_group
scripts/
  run_xlt.py                     # core: tune_phase, load_concat_dataset, discover_target_langs
  run_xlt_meta.py                # fan a metaconfig matrix over a SLURM --array; --fold N, --seed
  evals/                         # plot/unify; unify_xlt_test_res.py (raw.csv: pool folds + seed std, else per-run fallback), unify_xlt_valid_res.py (valid_res.csv: per-fold mean, seed std)
                                 # NOTE: quant.py / quant_boot_diff_sci.py are for REPRESENTATION evaluation (hidden-state analysis), NOT accuracy/eval-result significance — do not use them to test method-vs-method accuracy gaps.
  datasets/                      # reframe_bebe_to_ftp -> split_bebe_folds -> preprocess_dir
runtime/clusters/pegasus/shell/
  run.sh                         # SLURM+container wrapper (--site-packages, --no-gpu)
  run_minimal.sh                 # light CPU job
  logs/sbatch/{jobid}_{task}.{out,err}   # per-array-task logs; .out has eval metrics, run_name
artefacts/                       # ALL data + outputs (datasets live here, not data/)
  datasets/benchmarks/mcqa/{xstory_cloze_ftp,belebele_ftp}/
    {lang}/                                  # e.g. eng_Latn; 122 langs for belebele_ftp
      {train,validation,test}/               # unfolded root splits
      tokenized|{org}|{model}/               # e.g. tokenized|CohereLabs|aya-expanse-8b
      fold{0,1,2}/                           # item-disjoint self-split (train500/val100/test300)
        {train,validation,test}/  +  tokenized|{org}|{model}/
  models/{family}/{org}/{model}/mcqa_ftp/{uuid}_.../   # saved PEFT adapters
  evals/runs/{uuid}_...                       # single eval runs
  evals/xlt_runs/{llm}/{src_tag}/{run_group}/{timeid}_{run_name}/   # XLT tune/test runs
  wandb/run-*/                                # local wandb (config.yaml, summary, history)
```

Key conventions:
- Dataset `ds.dirs` (e.g. `mcqa/belebele_ftp/eng_Latn/fold0/tokenized|...`) is a **template**:
  the `{lang}` segment is swapped per `--source-group`/`--source-langs` lang and concatenated
  (see [[project_xlt_seen_unseen_assembly]]); `--fold N` swaps the `fold{N}` segment.
- `src_tag` in xlt_runs = the source-group name (or joined source langs, or `zero`).

