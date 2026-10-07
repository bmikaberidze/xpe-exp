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

## Current experimental path: bebe → bebe (xsc→bebe is DROPPED)

- The earlier decoder pipeline (train on XStoryCloze → test on Belebele) is **abandoned**:
  task-type mismatch (2-choice cloze vs 4-choice RC) → no cross-lingual transfer
  (held-out ≈ zero-shot). The xsc configs are gone (tag `pre-micm-nlp-0.4`).
- Current design mirrors the paper (task fixed, language transfers): **train AND test on
  Belebele**. Source (seen) langs use item-disjoint fold splits (train500/val100/test300);
  the 3 folds' test blocks tile the 900 parallel items, so **every item serves as a test
  item exactly once** (pool folds → full 900 per lang). Unseen targets are evaluated
  cross-lingually on the same fold test splits.

## Second track: SIB-200 encoder backbones

The journal extension adds **encoder** rows, because two decoders + one encoder is not
a coherent story when the decoder gaps are small. Backbones: **`mdeberta`**
(`microsoft/mdeberta-v3-base`) and **`mgte`** (`Alibaba-NLP/gte-multilingual-mlm-base`) —
both base-size, both on **SIB-200 topic classification**, zero-shot XLT only.

- **No folds.** SIB-200 ships official `train`/`validation`/`test` splits of
  **701/99/204** per language. **Never set `fold:` on a SIB entry** — `apply_fold` raises when a
  fold is given and `ds.dirs` has no `fold<N>` segment.
- **`LANG_GROUPS` in `src/xlt_langs.py` is the single source of truth for source
  languages** — read groups from there, never from `src/sib200_meta.py`, so a group and
  the runs it produced cannot drift. `src/sib200_meta.py` holds the *metadata* behind
  those lists (205 rows: Joshi tier, family, region, `xlmr`), not a CSV, because
  `artefacts/` is gitignored and `scripts/` holds code. `tests/test_lang_groups.py`
  pins the literals against it.
- **`mdeberta_seen` == the paper's Seen-92** (mDeBERTa trains on CC100 like XLM-R), so
  mDeBERTa is the *bridge row* to the published Table 1: same partition, new code.
  That equality is an assumption, not a documented identical language list — see the
  docstring in `src/sib200_meta.py` before leaning on it in the write-up.
- **The encoders' low-perf group is BORROWED from XLM-R, not measured (2026-08-11).**
  The paper's Low-Performing set is defined by full fine-tuning of **XLM-R-large** in
  the original SIB-200 benchmark (< 60%, sec. 4.2), so it characterises XLM-R — and the
  equivalent has still not been measured for mDeBERTa or mGTE. It is registered in
  `LOW_PERF_LANG_GROUPS` for both as a stand-in so the paper's headline row can be
  computed from existing runs; **every table built from it must say whose measurement
  defined the group**, and it must be replaced by the per-backbone in-language full-FT
  sweep. `low_perf_langs()` in `src/sib200_meta.py` remains the provenance source.
  - **Low-perf must never intersect the backbone's seen set** — `add_seen` encodes it as
    `seen == -1`, a refinement of `seen == 0`, and *raises* on an overlap. So each
    backbone's list is `low_perf_langs() - {backbone}_seen`: **46 for mDeBERTa** (its
    seen set IS the `xlmr` column, from which the list was derived, so nothing drops)
    and **45 for mGTE** (it pretrained on `yor_Latn`). The two rows therefore cover
    DIFFERENT language sets and are not directly comparable across backbones.
  - Measured result: XPE−SPT on low-perf is ≈0 (mDeBERTa) / ≈−1.1 (mGTE) against the
    published **+2.8** — the paper's headline does not reproduce. DUAL-70 does show the
    paper's *gradient* on mDeBERTa (largest margin on low-perf, smallest on seen).
- **The legacy repo is the behavioral spec, and its YAML is NOT the whole recipe.**
  `/fscratch/bmikaberidze/XPE` (`nlpka`) produced the published numbers, but
  `nlpka/configs/scripts/xpe_utils.py`'s SLURM-task table **overrides the YAML at
  runtime**: `optim: adafactor` (no `optim:` key in the YAML at all), `max_steps: 24000`
  (not the YAML's 10 epochs), prompt-group `lr 5e-3 / wd 0.0` via
  `optimizer_grouped_parameters` (`embedding` for SPT, `xpe_embedding` for XPE/DUAL),
  early-stopping patience 20 for source, and 10 seeds. Read both before porting.
- **`peft.modules_to_save` must include the pooler.** PEFT auto-saves only
  `classifier`/`score` and freezes every other child; on both new backbones the head is
  split across `pooler` (randomly initialised, absent from the checkpoint) and
  `classifier`. XLM-R has no pooler, so the published runs never hit this.

Plan: `docs/superpowers/plans/2026-08-05-sib200-encoder-backbones.md`.

## Run environment

**NEVER run python on the login node. Every step goes through `sbatch` + an array,
and the container that `run.sh` starts carries the correct environment** — it is the
only interpreter whose output counts. This covers dataset downloads, tokenization,
probes, training, evaluation, the result tables and pytest.

```
sbatch --array=0 --mem=30G --wait --gpus=0 \
  --partition=RTXA6000,RTX3090,batch,L40S,V100-32GB \
  runtime/clusters/pegasus/shell/run.sh --site-packages --no-gpu \
  "python -m pytest tests/ -q"

sbatch --array=0-59%10 --mem=30G runtime/clusters/pegasus/shell/run.sh --site-packages \
  "python -m micm_nlp run-group --group-config config/groups/<grid>.<src>.yml --runner scripts.xlt_runner:run --root-path /fscratch/bmikaberidze/xpe-exp"
```

### Eras (since 2026-09-18)

`run.sh [--era <name>] [--site-packages] [--no-gpu] "<cmd>"` picks one stack: image +
packages mount (`/fscratch/bmikaberidze/site-packages.<era>`). **`--era` must come
first** (flags are positional). Default is the current era.

| era | stack | code |
|---|---|---|
| `micm-nlp-0.5` (default) | torch 2.14 / CUDA 13.4 (sm_100: B200 works), transformers 5.17, peft 0.21, datasets 5, py3.12 | micm-nlp **v0.5.0**, xpe-exp `main` |
| `pre-micm-nlp-0.4` (frozen) | torch 2.5 (sm_90 max), transformers 4.49, peft 0.14, py3.10 | xpe-exp tag `pre-micm-nlp-0.4` -- produced everything under `artefacts/evals/xlt_runs/` |

- The mount holds micm-nlp **installed editable** from the sibling checkout; after a
  version bump, refresh its metadata inside a job
  (`python -m pip install -q --no-deps -e /fscratch/bmikaberidze/micm-nlp`), or
  `info.json` records the old version.
- **New-era configs set `model.pretrained.args.dtype` and `training_args.args.optim`
  explicitly**: transformers 5 loads a checkpoint in its own dtype (`"auto"`, bf16 for
  Aya) and defaults to `adamw_torch_fused`; transformers 4 used float32 and `adamw_torch`.
- Real jobs download HF models online into the container's own cache (no `HF_HOME`);
  don't set `HF_HUB_OFFLINE` in a job.
- 0.4.x adapters load unchanged on 0.5.0 (verified bit-identical, 16 Aya/Bloomz
  adapters); a replayed Aya adapter reproduced its stored accuracies to ±0.01.

### The rules

- **CPU-only jobs must pin the CPU partitions themselves:**
  `--partition=RTXA6000,RTX3090,batch,L40S,V100-32GB` (the list `--no-gpu` sets at
  `run.sh:69`). That PARTITION variable is used **only** by interactive mode
  (`run.sh:108`), so without the pin the `#SBATCH` GPU list at `run.sh:4` applies,
  which includes `H100-Trails` (not permitted for this uid → the job sits PENDING).
- **CPU-only jobs must ALSO pass `--gpus=0` on the `sbatch` CLI.** In SBATCH mode
  `--no-gpu` does **not** release the GPU: it lowers `MEMORY`, sets `GPUS=0` and
  prints "Running without GPU", but `--gpus=$GPUS` only ever reaches `srun` through
  `SRUN_ARGS`, which is built **inside the interactive branch** (`run.sh:136`). In
  batch mode `SRUN_ARGS` is empty and the job keeps the `#SBATCH --gpus=1` directive
  (`run.sh:7`). Only the `sbatch` CLI overrides a directive. Verified 2026-09-23:
  Peggy flagged job 3432106 (Belebele tokenization, pure CPU) for holding GPU 4 on
  `gifu` idle for 2 h 36 m. Every CPU-only job before this did the same.
  **Running `run.sh` directly is not an alternative from an agent session**: this
  shell is itself inside a small SLURM allocation (`SLURM_JOB_ID` is set, ~4G), so
  `run.sh` takes the SBATCH branch and its `srun` becomes a step of *that* job —
  the ~25 GB container unpack is then OOM-killed (verified 2026-09-16).
- **ALWAYS an array — even for one job** (`--array=0`). Never a bare `sbatch`.
- **NEVER loop `sbatch` over runs.** A shell loop that submits one `sbatch` per run
  puts one row per run in `squeue`, floods the cluster and has **no throttle**. One
  array submission is one row and is throttleable. If you find yourself writing
  `for X in ...; do sbatch ...; done` over *runs*, the matrix belongs in a
  group config instead. (Submitting a grid's few per-source-group files is fine —
  that is a few *array* submissions, and they must still be sent one at a time,
  each confirmed landed before the next; see the dispatch incidents below.)
- **ALWAYS throttle with `%10`** — `--array=0-59%10` runs at most 10 tasks at once.
  Unthrottled arrays starve other users and make black-hole nodes harder to spot.
- `--array`, `--mem`, `--partition` go on the **`sbatch` CLI**, never after `run.sh`:
  the wrapper consumes only `--site-packages` and `--no-gpu` as `$1`, and **anything
  else becomes the command it runs** (so a stray `--mem 30G` silently drops your
  python command). Add `--no-gpu` for CPU-only steps.
- `--mem` stays **≥30G** always — the ~25 GB container image unpacks into the job cgroup.
- Read output from `runtime/clusters/pegasus/shell/logs/sbatch/{jobid}_{task}.{out,err}`,
  not stdout. `--wait` blocks until the job finishes.
- Multi-line inspection snippets: write a `.py` file first, then run
  `"python <path>.py"` through the wrapper — nesting quotes inside `run.sh "…"` is fragile.
- If `import micm_nlp` fails (`ModuleNotFoundError`), install it editable from the
  sibling checkout: `python -m pip install -e /fscratch/bmikaberidze/micm-nlp`.
- Tokenization hazard: `datasets` ≥4 writes a `List` feature type that `datasets` <4
  cannot read (`Feature type 'List' not found`). Everything runs in one container now,
  so this only bites if a dataset was tokenized outside it — round-trip one language
  before tokenizing a whole benchmark.

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
- **The SIB-200 encoders (`mdeberta`, `mgte`) are base-size** → they need no big-VRAM
  node, BUT see the B200 note below: they must still pin a partition.
- **mDeBERTa CANNOT run on B200 -- in the `pre-micm-nlp-0.4` era.** DeBERTa-v2 compiles `build_relative_position` with
  `@torch.jit.script`, and this container's NVRTC does not know Blackwell (sm_100):
  `RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)`.
  Verified 2026-08-05: serv-3310 (H100) trained fine, serv-3324 (B200) failed. The
  default partition INCLUDES B200, so these jobs MUST pin
  `--partition=H100,H100-PCI,H100-Trails,H200,H200-PCI,A100-80GB`. Aya is unaffected
  because it has no `jit.script` — which is why Aya *pins* B200 and mDeBERTa must
  *avoid* it. Use the same pin for mGTE unless it is shown not to need it.
- **`--mem=30G` is the container FLOOR, not a training request.** Use 64G for GPU
  training jobs (the concatenated `mdeberta_seen` source set is 92 x 701 rows); 30G is
  fine only for CPU-only steps like tokenization, the result tables and pytest.
- **Any `run.sh` job needs ≥30 GB RAM** (the ~25 GB container image unpacks into
  node tmpfs and is charged to the job's cgroup) — never lower `--mem` below 30G,
  even for tiny CPU-only jobs like the result-table scripts.

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
- **XPE/SPT/DUAL are ALL one class — `CrossPromptEncoder`** (`models/xpe/encoder.py`); all
  three tune configs set `peft_type: XPE` and differ only by `encoder_ratio` (the XPE
  fraction of the virtual tokens): **SPT = 0** (plain soft prompt, `reparam=NONE`, uses
  `self.embedding`), **XPE = 1** (`self.xpe_embedding` → `xpe_head` MLP), **DUAL = 0<r<1**
  (e.g. 0.3 or 0.7 — a free hyperparam, NOT tied to the dataset; concat of both).
  So "SPT" is NOT a separate module.
  Consequence: anything gated on `isinstance(pe, CrossPromptEncoder)` fires for SPT too —
  e.g. the `normalize_embeddings()` clip/unit callback DOES clip SPT's `self.embedding`
  rows (name has `'embedding'`, 2D, `requires_grad`), so SPT-clip is a real experiment,
  not a no-op *once the callback actually runs*.
- **Normalisation before micm-nlp 0.2.1 NEVER RAN.** `NormalizePromptEncoderEmbeddings`
  was never registered: `runner.py` read `self._config.task.peft`, but `peft` is a
  **top-level** block, so the lookup always returned `None`. Fixed in micm-nlp `d3ae7e1`
  (2026-08-11); registration is now additionally gated on `encoder_embedding_normalize`
  being set. **Any run before that fix did not normalise, whatever its config said** —
  in this repo that is the six clip arms of `15a_stab_probe_aya.yml`, which were
  therefore identical to their base arms (the "clip changes nothing" reading of 15a
  measured nothing). All bebe/sib tune configs set `encoder_embedding_normalize: null`,
  so grids 14a/14b/19a/19b are unaffected — no normalisation intended, none applied.
- **Never read a normalisation claim out of an adapter saved before micm-nlp `9decad2`
  (2026-08-11).** `CrossPromptEncoderConfig` used to default to `'unit'` / max_norm `1.0`
  while `_filtered_kwargs` (`xpe/factory.py`) strips `None`, so a YAML `null` never
  reached the dataclass and every such `adapter_config.json` recorded
  `"encoder_embedding_normalize": "unit"` regardless of what the run did. **That covers
  every adapter this repo has produced so far** — grids 14a/14b/19a/19b included.
  `9decad2` sets both defaults to `None`, so adapters saved from then on record what
  actually happened; it also makes an unknown mode, and `clip` without a max_norm, raise.
  Behaviour is unchanged either way — normalisation is driven by the callback, whose
  registration reads the top-level `peft` block.

## One group config = one experiment = one table

A **group config** (`config/groups/{grid}.{source_group}.yml`) lists run entries for
`python -m micm_nlp run-group --runner scripts.xlt_runner:run`. Its stem is the run
group: output in `artefacts/runs/groups/{stem}/{time_id}_{name}/`. (Replaced the
pre-0.4 `config/meta/` metaconfigs + `run_xlt_meta.py` on 2026-09-30; those are at tag
`pre-micm-nlp-0.4`.)

- **Numbering: one number+letter per GRID** (`26a`, `26b`). A grid's per-source-group
  files share it (`14b_bebe_grid_full_bloomz.{enarzho,joshi5,bloom_seen}`) because
  `target_source_table.py` globs `{grid}.*` into one table; two different grids never
  share a number. Don't repeat the source group in the grid name.
- One file = one source group (named on every entry as `source_group:`); split grids
  by BACKBONE and by ARM (LR regime / recipe variant) too -- the aggregators have no
  method filter. Keep the 4 methods (spt/xpe/d30/d70) together: they are the table's
  columns, and SPT is the baseline every delta is computed against.
- Entries carry `seed`, `fold`, `method`, `source_group`, `llm` (stamped on every result
  row) and optional `overrides`. Shapes: `separate_test:` tune+test; `tune_only: true`
  tune only (LR searches, select on validation); `adapter:` replay; neither = zero-shot.
- Generate regular grids with `scripts/gen_xlt_group_config.sh <grid> <src> <llm> [units] [seeds]`.

## Group configs are immutable contracts

**Once ANY entry of a group config has been dispatched, the file must never be
edited** (the run dir, the result rows' `group` column and wandb all point back to it
by name). Additive only: appending NEW entries is fine; editing or reordering existing
ones is not (array indices are positional). Changing an already-run entry means a NEW
group config (new number/letter), whose header says what it supersedes and why.
Dispatch notes in the header comment may be appended.

Same rule for the units a group config references. A unit that must change after runs
exist makes those runs stale: freeze what they used under `config/units/archive/{grid}/`
first. The old decoder grids 8-16 were rewritten this way on 2026-09-30: their units
were regenerated from each run's wandb `config.yaml` (every run reproduces its record),
because the live tune configs had drifted after those grids ran (LRs, `max_steps`,
early stopping on `eval_loss` vs accuracy, 9a's scheduler). Generator:
`scripts/gen_archive_units.py`.

## Repository layout

```
config/
  units/                          # unit configs (one run's full setup)
    tune.{xpe|spt|dual}.lm.{model}.ds.bebe.yml       # finetune a PEFT method
    test.lm.{model}.ds.bebe[.fold].yml                # eval; .fold = self-split test split
    proc.ds.bebe.tok.{aya|bloom|g3|g4}.yml            # tokenize Belebele per backbone
    archive/{grid}/                                   # frozen units of already-run grids
  groups/{N}{a|b}_{grid}.{source_group}.yml           # group configs; stem = run group
scripts/
  xlt_runner.py                  # the run-group runner: tune on source langs, test the rest
  gen_xlt_group_config.sh        # generate a regular 4-method x 3-fold x seeds grid
  evals/                         # lang_table.py / lr_search_table.py -> target_source_table.py (one table per grid)
                                 # NOTE: quant.py / quant_boot_diff_sci.py are for REPRESENTATION evaluation (hidden-state analysis), NOT accuracy/eval-result significance — do not use them to test method-vs-method accuracy gaps.
  datasets/                      # bebe: reframe_bebe_to_ftp -> split_bebe_folds -> preprocess_dir
src/                             # importable helpers (NOT scripts): xlt_langs.py (LANG_GROUPS,
                                 # LOW_PERF_LANG_GROUPS), sib200_meta.py, utils.py, hub_upload.py
runtime/clusters/pegasus/shell/
  run.sh                         # SLURM+container wrapper (--era, --site-packages, --no-gpu)
  logs/sbatch/{jobid}_{task}.{out,err}   # per-array-task logs
artefacts/                       # ALL data + outputs (datasets live here, not data/)
  datasets/benchmarks/mcqa/belebele_ftp/
    {lang}/                                  # 122 langs
      {train,validation,test}/               # unfolded root splits
      tokenized--{org}--{model}/               # e.g. tokenized--CohereLabs--aya-expanse-8b
      fold{0,1,2}/                           # item-disjoint self-split (train500/val100/test300)
        {train,validation,test}/  +  tokenized--{org}--{model}/
  runs/groups/{group}/{time_id}_{name}/       # 0.4+ runs (info.json, eval_*.csv, separate_test*.csv)
  evals/xlt_runs/{llm}/{src_tag}/{run_group}/{timeid}_{run_name}/   # pre-0.4 runs (read-only)
  models/{family}/{org}/{model}/mcqa_ftp/{uuid}_.../   # saved PEFT adapters
  wandb/run-*/                                # local wandb (config.yaml = a run's resolved config)
```

Key conventions:
- **Dataset folders use `--`, never `|`** (`tokenized--{org}--{model}`, since 2026-09-19).
  datasets ≥4 turns cache paths into regexes when `map()` looks for old cache files; `|` is
  regex alternation, so a second run over a `|` folder crashes (`int(None)`). The old names
  remain as symlinks for the pre-micm-nlp-0.4 era; migration: `scripts/datasets/migrate_pipe_dirs.py`.
- Dataset `ds.dirs` (e.g. `mcqa/belebele_ftp/eng_Latn/fold0/tokenized--...`) is a **template**:
  the runner swaps the `{lang}` segment per source-group lang and concatenates
  (see [[project_xlt_seen_unseen_assembly]]); an entry's `fold: N` swaps the `fold{N}` segment.
