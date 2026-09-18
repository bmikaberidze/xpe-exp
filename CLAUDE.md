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
  (held-out ≈ zero-shot). `*.ds.xsc.yml` configs are legacy.
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
  **701/99/204** per language. **Never pass `--fold`** — `apply_fold` raises when a
  fold is given and `ds.dirs` has no `fold<N>` segment.
- **`LANG_GROUPS` in `scripts/run_xlt.py` is the single source of truth for source
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
probes, training, evaluation, unify/aggregate and pytest.

```
sbatch --array=0 --mem=30G --wait runtime/clusters/pegasus/shell/run.sh --site-packages --no-gpu \
  "python -m pytest tests/ -q"

sbatch --array=0-59%10 --mem=30G runtime/clusters/pegasus/shell/run.sh --site-packages \
  "python -m scripts.run_xlt_meta --meta-config ... --test-config ... --source-group ..."
```

### The rules

- **CPU-only jobs must pin the CPU partitions themselves:**
  `--partition=RTXA6000,RTX3090,batch,L40S,V100-32GB` (the list `--no-gpu` sets at
  `run.sh:69`). `--no-gpu` only lowers memory and asks for 0 GPUs in SBATCH mode —
  that PARTITION variable is used **only** by interactive mode (`run.sh:108`), so
  without the pin the `#SBATCH` GPU list at `run.sh:4` applies, which includes
  `H100-Trails` (not permitted for this uid → the job sits PENDING).
  **Running `run.sh` directly is not an alternative from an agent session**: this
  shell is itself inside a small SLURM allocation (`SLURM_JOB_ID` is set, ~4G), so
  `run.sh` takes the SBATCH branch and its `srun` becomes a step of *that* job —
  the ~25 GB container unpack is then OOM-killed (verified 2026-09-16).
- **ALWAYS an array — even for one job** (`--array=0`). Never a bare `sbatch`.
- **NEVER loop `sbatch` over runs.** A shell loop that submits one `sbatch` per run
  puts one row per run in `squeue`, floods the cluster and has **no throttle**. One
  array submission is one row and is throttleable. If you find yourself writing
  `for X in ...; do sbatch ...; done` over *runs*, the matrix belongs in a
  metaconfig instead. (Looping over a handful of `--source-group` values is fine —
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
- **mDeBERTa CANNOT run on B200.** DeBERTa-v2 compiles `build_relative_position` with
  `@torch.jit.script`, and this container's NVRTC does not know Blackwell (sm_100):
  `RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)`.
  Verified 2026-08-05: serv-3310 (H100) trained fine, serv-3324 (B200) failed. The
  default partition INCLUDES B200, so these jobs MUST pin
  `--partition=H100,H100-PCI,H100-Trails,H200,H200-PCI,A100-80GB`. Aya is unaffected
  because it has no `jit.script` — which is why Aya *pins* B200 and mDeBERTa must
  *avoid* it. Use the same pin for mGTE unless it is shown not to need it.
- **`--mem=30G` is the container FLOOR, not a training request.** Use 64G for GPU
  training jobs (the concatenated `mdeberta_seen` source set is 92 x 701 rows); 30G is
  fine only for CPU-only steps like tokenization, unify and pytest.
- **Any `run.sh` job needs ≥30 GB RAM** (the ~25 GB container image unpacks into
  node tmpfs and is charged to the job's cgroup) — never lower `--mem` below 30G,
  even for tiny CPU-only jobs like the unify scripts.

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

## One metaconfig = one aggregate table

**A metaconfig must be scoped so that `unify_xlt_test_res.py` + `aggregate_xlt_res.py`
run over its run-group and produce ONE conceptually coherent table.** Do not mix
things into a single metaconfig that you would then have to split apart to read.

- Split by BACKBONE and by ARM (LR regime / recipe variant), not just by grid idea.
  `24a_xlmr_base_armA` + `24b_xlmr_base_armC` beats one `24_xlmr_base` holding both:
  the aggregators have no method filter, so a mixed grid yields a table whose columns
  are `xpe` and `xpe_c` side by side and every downstream comparison has to hand-slice
  it. (That is exactly why grid 21a needed the per-arm symlink/copy views in
  `artefacts/_scratch/21a_arms/` to be readable at all -- avoid creating that need.)
- Source group STAYS a CLI axis (`--source-group`), not a matrix column: unify runs
  per source-group dir and aggregate joins them into the target x source cells. That
  is the one split the tooling already understands.
- Seeds and folds stay matrix rows -- they are the replication axes the aggregators
  average over.
- Keep the 4 methods (spt/xpe/d30/d70) together in one metaconfig: they are the
  columns of the table, and SPT is the baseline every delta is computed against.

Rule of thumb: if reading the result requires filtering the metaconfig's own rows,
it should have been two metaconfigs.

## Metaconfigs are immutable contracts

**A metaconfig is the contract artefact for the runs it produced. Once ANY of its
entries has been dispatched, that file must never be edited.** The run dir, the
`run_group`, the wandb group and the `meta_config` column in every `raw.csv` all point
back to it by name — editing it silently rewrites the description of results that
already exist, and nobody reading the file later can tell.

- **Additive changes only.** Appending NEW matrix entries is fine (that is how the
  grid-14 seed extension was done: seeds 15-19 appended as indices 60-119, with a
  dated comment saying so). Editing or reordering an existing entry is not — array
  indices are positional, so a reorder also silently repoints old results.
- **Changing an already-run entry means a NEW metaconfig**, with a new stem and hence
  a new `run_group`: `17a_lr_search_sib_mdeberta.yml` -> `17c_...`, never an in-place
  edit. Header comment should say what it supersedes and why.
- Header comments describing *dispatch* (the sbatch line, partition pins, which array
  indices were re-dispatched after a fix) are the exception: they document the file's
  own use and may be appended to.
- Same rule for the tune/test configs a metaconfig references. If a config must
  change after runs exist, the runs are stale — say so explicitly rather than
  quietly re-pointing.

## Repository layout

```
config/                         # all experiment configs (YAML)
  tune.{xpe|spt|dual}.lm.{model}.ds.{xsc|bebe|sib}.yml  # finetune a PEFT method
  test.lm.{model}.ds.{bebe[.fold]|sib}.yml           # eval; .fold = self-split test split
  proc.ds.{xsc|bebe|sib}.tok.{aya|bloom|mdeberta|mgte}.yml  # tokenize a benchmark per backbone
  meta/{N}[a|b]_*.yml             # metaconfigs (LR searches, grids); stem = wandb run_group
scripts/
  run_xlt.py                     # core: tune_phase, load_concat_dataset, discover_target_langs
  run_xlt_meta.py                # fan a metaconfig matrix over a SLURM --array; --fold N, --seed
  evals/                         # plot/unify; unify_xlt_test_res.py (raw.csv: pool folds + seed std, else per-run fallback), unify_xlt_valid_res.py (valid_res.csv: per-fold mean, seed std)
                                 # aggregate_xlt_res.py: the per-lang test_unified.csv of all source groups -> ONE xlt_runs/{llm}/aggr_res.csv (target-group x source-group cells)
                                 # NOTE: quant.py / quant_boot_diff_sci.py are for REPRESENTATION evaluation (hidden-state analysis), NOT accuracy/eval-result significance — do not use them to test method-vs-method accuracy gaps.
  datasets/                      # bebe: reframe_bebe_to_ftp -> split_bebe_folds -> preprocess_dir
                                 # sib:  download_sib -> preprocess_dir  (no fold step)
src/                             # importable helpers (NOT scripts): utils.py, hub_upload.py,
                                 # sib200_meta.py (205-lang table: xlmr/Seen-92, low_perf/46)
runtime/clusters/pegasus/shell/
  run.sh                         # SLURM+container wrapper (--site-packages, --no-gpu)
  run_minimal.sh                 # light CPU job
  logs/sbatch/{jobid}_{task}.{out,err}   # per-array-task logs; .out has eval metrics, run_name
artefacts/                       # ALL data + outputs (datasets live here, not data/)
  datasets/benchmarks/topic/sib200/          # SIB-200; official splits, NO folds
    {lang}/                                  # 205 langs
      {train,validation,test}/               # 701 / 99 / 204
      tokenized|{org}|{model}/               # e.g. tokenized|microsoft|mdeberta-v3-base
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

