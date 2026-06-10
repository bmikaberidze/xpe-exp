"""
SLURM array dispatcher for cross-lingual transfer experiments.

Reads a meta-experiment YAML describing:
  tune_configs : name -> tune-config-path mapping (only required if at
                 least one matrix entry uses `tune_method`)
  matrix       : list of entries indexed by SLURM_ARRAY_TASK_ID (or
                 --task-id for interactive testing)

Each matrix entry has at most one of:
  tune_method   : <name from tune_configs>  → tune from scratch, then test
  adapter_uuid4 : <uuid>                    → skip tune, test the named adapter
                                              (path resolved by the toolkit's
                                              env/disk registry)
  (neither)                                 → zero-shot. No tune, no adapter
                                              wiring; test config's existing
                                              model.pretrained.adapter is used
                                              as-is (typically null → base model).

Optional per-entry fields:
  run_name      : suffix for run-dir naming (distinguishes sweep variants)
  overrides     : {dotted.path: value} in-memory tune-config mutations
                  (only meaningful with `tune_method`; warned-and-skipped otherwise)
  adapter_path  : absolute path to an existing adapter on disk; when set
                  alongside `adapter_uuid4`, the meta runner registers the
                  path under the uuid in env vars (so the toolkit's
                  resolver finds it). Requires `adapter_uuid4`.

The meta-config file's stem becomes the `run_group` for every entry — one
file = one experiment.

Usage:
    squeue -u bmikaberidze -l  

    sbatch \
      --array=0-3 \
      --gpus=1 \
      --partition=H100 \
      --mem=80G \
      --time=08:00:00 \
      runtime/clusters/pegasus/shell/run.sh \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/1_aya_as_xlmr.yml \
            --test-config ./config/test.lm.aya.ds.bebe.yml \
            --source-langs en,ru,zh,es,ar,hi,id"

   python -m scripts.run_xlt_meta \
        --meta-config ./config/meta/0_zero_shot_eval.yml \
        --test-config ./config/test.lm.aya.bf16.ds.bebe.yml \
        --target-langs kat_Geor,eng_Latn \
        --sequential-test

        --target-langs shn_Mymr \
        --task-id 0    
                
    =====
    
    sbatch \
        --array=0 \
        runtime/clusters/pegasus/shell/run.sh --site-packages \
            "python -m scripts.run_xlt_meta \
                --meta-config ./config/meta/0_zero_shot_eval.yml \
                --test-config ./config/test.lm.aya.ds.bebe.yml"

    sbatch \
        --array=1 \
        runtime/clusters/pegasus/shell/run.sh --site-packages \
            "python -m scripts.run_xlt_meta \
                --meta-config ./config/meta/0_zero_shot_eval.yml \
                --test-config ./config/test.lm.aya.bf16.ds.bebe.yml"

    =====

    sbatch \
        --array=2 \
        runtime/clusters/pegasus/shell/run.sh --site-packages \
            "python -m scripts.run_xlt_meta \
                --meta-config ./config/meta/0_zero_shot_eval.yml \
                --test-config ./config/test.lm.aya.ds.bebe.yml \
                --sequential-test"
    sbatch \
        --array=3 \
        runtime/clusters/pegasus/shell/run.sh --site-packages \
            "python -m scripts.run_xlt_meta \
                --meta-config ./config/meta/0_zero_shot_eval.yml \
                --test-config ./config/test.lm.aya.bf16.ds.bebe.yml \
                --sequential-test"

    python -m scripts.run_xlt_meta \
        --meta-config ./config/meta/0_zero_shot_eval.yml \
        --test-config ./config/test.lm.aya.ds.bebe.yml \
        --task-id 2 \
        --sequential-test

    python -m scripts.run_xlt_meta \
        --meta-config ./config/meta/1_aya_as_xlmr.yml \
        --test-config ./config/test.lm.aya.ds.bebe.yml \
        --source-langs en,ru,zh,es,ar,hi,id \
        --target-langs kat_Geor,eng_Latn \
        --task-id 0

    sbatch \
        --array=0,1 \
        runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/3_full_aya.yml \
            --test-config ./config/test.lm.aya.ds.bebe.yml \
            --source-langs en,ru,zh,es,ar,hi,id \
            --target-langs kat_Geor,eng_Latn"
    
    sbatch \
      --array=0,1 \
      runtime/clusters/pegasus/shell/run.sh --site-packages \
      "python -m scripts.run_xlt_meta \
          --meta-config ./config/meta/4_replay_aya.yml \
          --test-config ./config/test.lm.aya.ds.bebe.yml \
          --source-langs en,ru,zh,es,ar,hi,id \
          --target-langs kat_Geor,eng_Latn"

    =====
    BLOOM

    sbatch --array=0 runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/6_lr_search_bloom.yml \
            --test-config ./config/test.lm.bloom-7b1.ds.bebe.yml \
            --sequential-test"

    sbatch --array=1-6 runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/6_lr_search_bloom.yml \
            --test-config ./config/test.lm.bloom-7b1.ds.bebe.yml \
            --source-langs en,ru,zh,es,ar,hi,id"

    =====
    Inlingual BLOOM
    sbatch --array=3,5 runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/6_lr_search_bloom.yml \
            --test-config ./config/test.lm.bloom-7b1.ds.bebe.yml \
            --source-langs en \
            --target-langs eng_Latn"

    sbatch --array=3,5 runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/6_lr_search_bloom.yml \
            --test-config ./config/test.lm.bloom-7b1.ds.bebe.yml \
            --source-langs es \
            --target-langs spa_Latn"

    sbatch --array=3,5 runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/6_lr_search_bloom.yml \
            --test-config ./config/test.lm.bloom-7b1.ds.bebe.yml \
            --source-langs zh \
            --target-langs zho_Hans"

    sbatch --array=3,5 runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/6_lr_search_bloom.yml \
            --test-config ./config/test.lm.bloom-7b1.ds.bebe.yml \
            --source-langs ru \
            --target-langs rus_Cyrl"

    sbatch --array=3,5 runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/6_lr_search_bloom.yml \
            --test-config ./config/test.lm.bloom-7b1.ds.bebe.yml \
            --source-langs ar \
            --target-langs arb_Arab"

    sbatch --array=3,5 runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/6_lr_search_bloom.yml \
            --test-config ./config/test.lm.bloom-7b1.ds.bebe.yml \
            --source-langs hi \
            --target-langs hin_Deva"

    sbatch --array=3,5 runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/6_lr_search_bloom.yml \
            --test-config ./config/test.lm.bloom-7b1.ds.bebe.yml \
            --source-langs id \
            --target-langs ind_Latn"

    =====
    BeBe to BeBe

    python -m scripts.run_xlt_meta \
        --meta-config ./config/meta/0_zero_shot_eval.yml \
        --test-config ./config/test.lm.bloomz-7b1.ds.bebe.yml \
        --target-langs eng_Latn,spa_Latn,zho_Hans,arb_Arab,hin_Deva,rus_Cyrl,deu_Latn
        
    python -m scripts.run_xlt_meta \
        --meta-config ./config/meta/0_zero_shot_eval.yml \
        --test-config ./config/test.lm.bloomz-7b1.ds.bebe.yml \
        --task-id 2 --sequential-test
        
    sbatch --array=2 runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/0_zero_shot_eval.yml \
            --test-config ./config/test.lm.bloomz-7b1.ds.bebe.yml \
            --sequential-test"
            
    sbatch --array=2 runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/0_zero_shot_eval.yml \
            --test-config ./config/test.lm.bloomz-7b1.ds.bebe.yml \
            --sequential-test \
            --fold 0"

    sbatch --array=1-2 runtime/clusters/pegasus/shell/run.sh --site-packages \
        "python -m scripts.run_xlt_meta \
            --meta-config ./config/meta/8_bebe_self_split_bloomz.yml \
            --test-config ./config/test.lm.bloomz-7b1.ds.bebe.fold.yml \
            --source-langs eng_Latn,spa_Latn,fra_Latn,zho_Hans,hin_Deva,arb_Arab,ind_Latn \
            --fold 0"

    squeue -u bmikaberidze -l
"""

if __name__ == '__main__':
    from src.utils import micm_nlp_setup
    micm_nlp_setup()

import os
import argparse
import yaml
from pathlib import Path

from micm_nlp.config import CONFIG
from scripts.run_xlt import run_xlt


_MODE_KEYS = ('tune_method', 'adapter_uuid4')


def load_meta_config(path: str) -> dict:
    """Load and lightly validate a meta-experiment YAML.

    Each matrix entry has at most one mode key (in `_MODE_KEYS`) set;
    both unset means zero-shot. `tune_configs` is required only when at
    least one entry uses `tune_method`.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f'meta config {path}: top level must be a mapping')
    if not isinstance(data.get('matrix'), list) or not data['matrix']:
        raise ValueError(f'meta config {path}: missing or empty `matrix:` list')

    tune_configs = data.get('tune_configs') or {}
    if not isinstance(tune_configs, dict):
        raise ValueError(f'meta config {path}: `tune_configs:` must be a mapping (or omitted)')

    for i, entry in enumerate(data['matrix']):
        if not isinstance(entry, dict):
            raise ValueError(f'meta config {path}: matrix[{i}] must be a mapping')
        modes_set = [k for k in _MODE_KEYS if entry.get(k)]
        if len(modes_set) > 1:
            raise ValueError(
                f'meta config {path}: matrix[{i}] has both {modes_set} set; at most one allowed'
            )
        if entry.get('adapter_path') and not entry.get('adapter_uuid4'):
            raise ValueError(
                f'meta config {path}: matrix[{i}].adapter_path requires adapter_uuid4 to also be set'
            )
        if 'tune_method' in modes_set:
            method = entry['tune_method']
            if method not in tune_configs:
                raise ValueError(
                    f'meta config {path}: matrix[{i}].tune_method={method!r} '
                    f'not in tune_configs {sorted(tune_configs)}'
                )

    data['tune_configs'] = tune_configs
    return data


def _apply_override(config, dotted_path: str, value) -> None:
    """Set config.a.b.c = value via the path string 'a.b.c'."""
    parts = dotted_path.split('.')
    obj = config
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta-config', type=str, required=True,
                    help='YAML defining tune_configs registry + matrix sweep. '
                         'The file stem (basename without .yml) is used as the run_group.')
    ap.add_argument('--test-config', type=str, required=True)
    ap.add_argument('--source-langs', type=str, default='',
                    help='Comma-separated xsc lang codes. Required for tune mode; '
                         'optional for replay / zero-shot (run-dir naming only).')
    ap.add_argument('--target-langs', type=str, default=None, help='Comma-separated bebe lang codes; default: auto-discover')
    ap.add_argument('--fold', type=int, default=None,
                    help="Belebele self-split fold index; rewrites the tune+test config ds.dirs "
                         "'fold<N>' segment. Omit for fold-less datasets (xstory_cloze).")
    ap.add_argument('--seed', type=int, default=None,
                    help='Pin the training seed (shared across methods → paired '
                         'comparison; recorded in raw.csv). Omit to randomize per run.')
    ap.add_argument('--sequential-test', action='store_true')
    ap.add_argument('--task-id', type=int, default=None, help='Force matrix index for interactive testing of a single entry. '
                                                              'Ignored when SLURM_ARRAY_TASK_ID is set (real array dispatch wins).')
    return ap.parse_args()


def resolve_task_id(cli_task_id: int | None = None) -> tuple[int, bool]:
    """Return (task_id, interactive) for the run.

    Precedence (SLURM wins to keep array dispatch authoritative):
    1. SLURM_ARRAY_TASK_ID env var set → (int(env), False).  Real dispatch.
    2. Else --task-id CLI override     → (cli_task_id, True). Manual probe.
    3. Else default                    → (0, True).           Safe fallback.
    """
    raw = os.environ.get('SLURM_ARRAY_TASK_ID')
    if raw is not None:
        if cli_task_id is not None:
            print(f'[meta] --task-id={cli_task_id} ignored (SLURM_ARRAY_TASK_ID={raw} wins)')
        return int(raw), False
    if cli_task_id is not None:
        print(f'[meta] --task-id={cli_task_id} (manual override; interactive)')
        return cli_task_id, True
    print('[meta] SLURM_ARRAY_TASK_ID not set, no --task-id; defaulting to 0 (interactive)')
    return 0, True


def main():
    args = parse_args()
    meta = load_meta_config(args.meta_config)
    tune_configs = meta['tune_configs']
    matrix = meta['matrix']

    # One meta config = one experiment = one run_group. Use the file stem.
    run_group = Path(args.meta_config).stem

    task_id, interactive = resolve_task_id(cli_task_id=args.task_id)
    if not 0 <= task_id < len(matrix):
        raise ValueError(f'task_id={task_id} out of range [0, {len(matrix)})')

    entry = matrix[task_id]
    run_name_tag = entry.get('run_name')
    overrides = entry.get('overrides') or {}
    method = entry.get('tune_method')
    adapter_uuid4 = entry.get('adapter_uuid4')
    adapter_path = entry.get('adapter_path')

    # Three modes (validated upstream in load_meta_config):
    #   method set        → load + override the tune config
    #   adapter_uuid4 set → skip tune, reuse a trained adapter
    #   neither           → zero-shot
    # run_xlt() owns the actual skip-tune / zero-shot branching; we just
    # hand it the right inputs.
    tune_config = None
    tune_config_path = ''
    if method:
        tune_config_path = tune_configs[method]
        tune_config = CONFIG.from_yaml(tune_config_path)
        for p, v in overrides.items():
            _apply_override(tune_config, p, v)
            print(f'[meta] override: {p} = {v}')
    elif overrides:
        print(f'[meta] WARNING: overrides ignored outside tune mode: {overrides}')

    mode_desc = (
        f'tune({method})' if method
        else f'adapter({adapter_uuid4})' if adapter_uuid4
        else 'zero_shot'
    )
    print(f'[meta] run_group={run_group} | task_id={task_id} interactive={interactive} | '
          f'run_name={run_name_tag!r} | mode={mode_desc}')

    source_langs = [s.strip() for s in args.source_langs.split(',') if s.strip()]
    target_langs = (
        [s.strip() for s in args.target_langs.split(',') if s.strip()]
        if args.target_langs else None
    )
    test_config = CONFIG.from_yaml(args.test_config)

    run_xlt(
        tune_config=tune_config,
        test_config=test_config,
        source_langs=source_langs,
        target_langs=target_langs,
        fold=args.fold,
        seed=args.seed,
        run_group=run_group,
        slurm_task_id=task_id,
        interactive=interactive,
        run_name_tag=run_name_tag,
        sequential_test=args.sequential_test,
        adapter_uuid4=adapter_uuid4,
        adapter_path=adapter_path,
        tune_config_source=tune_config_path,
        test_config_source=args.test_config,
        meta_config_source=args.meta_config,
    )


if __name__ == '__main__':
    main()
