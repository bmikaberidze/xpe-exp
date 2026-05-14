"""
Cross-Lingual Transfer (XLT) runner.

Tunes LM on a concatenated source-language dataset (xstory_cloze / xsc),
then tests per target language (belebele / bebe).

Three modes (CLI / library):
  1. Tune-and-test  — pass --tune-config (CLI) / tune_config (lib).
  2. Test only      — pass --adapter-uuid4 (the trained adapter; path
                      lookup goes through MODEL's env/disk registry).
                      You may pass --adapter-path together with the uuid to
                      register the path in env; the path alone is rejected
                      unless its basename encodes a uuid recoverable via
                      MODEL.extract_uuid_from_name.
  3. Zero-shot      — pass none of the above. The test phase runs against
                      whatever model.pretrained.adapter the test config
                      already specifies (typically `null` → base model).

All XSC source languages: en,ru,zh,es,ar,hi,id,te,sw,eu,my

Examples:
    # Tune-and-test
    python -m scripts.run_xlt \\
      --tune-config ./config/tune.xpe.lm.aya.ds.xsc.yml \\
      --test-config ./config/test.lm.aya.ds.bebe.yml \\
      --source-langs en,ru,zh,es,ar,hi,id \\
      --target-langs kat_Geor,eng_Latn \\
      --run-group xpe_aya

    # Test only, reusing a trained adapter
    python -m scripts.run_xlt \\
      --test-config ./config/test.lm.aya.ds.bebe.yml \\
      --adapter-uuid4 <uuid> \\
      --source-langs en,ru,zh,es,ar,hi,id \\
      --run-group replay

    # Zero-shot (base model from the test config)
    python -m scripts.run_xlt \\
      --test-config ./config/test.lm.aya.ds.bebe.yml \\
      --source-langs en,ru,zh,es,ar,hi,id \\
      --run-group zs

    # Sequential per-lang test (writes one CSV row per lang as it completes)
    ... --sequential-test
"""

if __name__ == '__main__':
    from src.utils import micm_nlp_setup
    micm_nlp_setup()

import os
import wandb
import argparse
import pandas as pd
from pathlib import Path
from datasets import DatasetDict

from micm_nlp import utils
from micm_nlp.path import evals_dir, datasets_dir
from micm_nlp.config import CONFIG, AdapterConfig, _Flex
from micm_nlp.datasets.dataset import DATASET
from micm_nlp.models.model import MODEL
from micm_nlp.tokenizers.tokenizer import load as load_tokenizer
from micm_nlp.training.runner import TRAINER

SLURM_ARRAY_TASK_ID = 'SLURM_ARRAY_TASK_ID'


# --- CLI --------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tune-config', type=str, default=None,
                    help='Path to tune YAML. If omitted and no --adapter-uuid4 is given, runs zero-shot.')
    ap.add_argument('--test-config', type=str, required=True)
    ap.add_argument('--source-langs', type=str, default='',
                    help='Comma-separated xsc lang codes. Required for tune mode (used '
                         'as the concatenated training source); optional for replay / '
                         'zero-shot (used only as run-dir naming metadata).')
    ap.add_argument('--target-langs', type=str, default=None,
                    help='Comma-separated bebe lang codes; default: auto-discover from tokenized dirs')
    ap.add_argument('--run-group', type=str, required=True)
    ap.add_argument('--adapter-uuid4', type=str, default=None,
                    help='uuid4 of an existing trained adapter; skip tune, resolve path via env/disk registry')
    ap.add_argument('--adapter-path', type=str, default=None,
                    help='Absolute path to an existing adapter; the basename must encode the uuid '
                         '(or pass --adapter-uuid4 alongside). Skips tune.')
    ap.add_argument('--sequential-test', action='store_true',
                    help='Run bebe langs one-by-one instead of concat-predict (debug)')
    ap.add_argument('--s-task-id', type=int, default=1,
                    help='Fake SLURM_ARRAY_TASK_ID for interactive mode')
    return ap.parse_args()


# --- Path / naming helpers --------------------------------------------------


def swap_lang_in_dirs(dirs_template: str, lang: str) -> str:
    # ds.dirs shape: '<category_group>/<benchmark>/<lang>/tokenized|<vendor>|<model>'
    parts = dirs_template.split('/')
    parts[2] = lang
    return '/'.join(parts)


def build_run_paths(tune_config, test_config, source_langs_sorted, run_group, slurm_task_id, interactive, run_name_tag=None):
    llm_config = tune_config if tune_config else test_config
    llm = llm_config.model.architecture
    src_tag = '-'.join(source_langs_sorted) if source_langs_sorted else 'zero'

    # Make the resolved task id visible to downstream code (TRAINER etc.)
    # whether we're running under SLURM or interactively.
    os.environ[SLURM_ARRAY_TASK_ID] = str(slurm_task_id)
    if interactive:
        run_group = f'test_{run_group}'

    time_id = utils.get_time_id()
    run_name = f'{time_id}_{run_name_tag}' if run_name_tag else time_id

    run_dir = Path(evals_dir()) / 'xlt_runs' / llm / src_tag / run_group / run_name
    return run_dir, llm, src_tag, run_group, run_name


def discover_target_langs(test_config):
    parts = test_config.ds.dirs.split('/')
    benchmark_rel = '/'.join(parts[:2])
    tok_suffix = parts[3]
    search_root = Path(datasets_dir()) / test_config.ds.category / benchmark_rel
    langs = []
    for lang_dir in sorted(search_root.iterdir()):
        if (lang_dir / tok_suffix).is_dir():
            langs.append(lang_dir.name)
    return langs

# --- Dataset assembly -------------------------------------------------------


def load_concat_dataset(base_config, langs, assign_task_ids=False):
    hf_datasets = []
    tokenizer = load_tokenizer(base_config)
    for i, lang in enumerate(langs):
        per_lang_config = base_config.model_copy(deep=True)
        per_lang_config.ds.dirs = swap_lang_in_dirs(base_config.ds.dirs, lang)
        per_lang_ds = DATASET(per_lang_config)
        per_lang_ds.preprocess(tokenizer)
        hf = per_lang_ds.hf
        if assign_task_ids:
            hf = DatasetDict({
                split: ds.add_column('task_ids', [i] * len(ds))
                for split, ds in hf.items()
                if ds is not None
            })
        hf_datasets.append(hf)
    return DATASET(base_config, hf_datasets=hf_datasets)


# --- Tune / Test phases -----------------------------------------------------


def tune_phase(tune_config, source_langs_sorted):
    tokenizer = load_tokenizer(tune_config)
    dataset = load_concat_dataset(tune_config, source_langs_sorted)
    model = MODEL(tune_config)
    trainer = TRAINER(model, dataset, tokenizer)
    trainer.run()
    return model.uuid4, model.path


def wire_test_to_adapter(test_config, adapter_uuid4, adapter_path=None):
    """Wire the test config's pretrained model to use a trained adapter.

    `adapter_uuid4` is the canonical identity. If `adapter_path` is also
    given, the path is registered in env vars under the uuid so the
    toolkit's resolver can find it; otherwise the toolkit looks the uuid
    up in its env/disk registry on its own.

    Leaves model.pretrained pointing at the base HF model and attaches the
    trained adapter via AdapterConfig, so PEFT.setup_model dispatches via
    PEFT.from_pretrained → load_xpe_pretrained.
    """
    if not adapter_uuid4:
        raise ValueError('adapter_uuid4 is required to wire an adapter')
    if adapter_path:
        MODEL.store_path_by_uuid4_in_envs(adapter_uuid4, adapter_path)
    test_config.model.pretrained.adapter = AdapterConfig(source='local', uuid4=adapter_uuid4)


def _extract_accuracy(metrics, lang=None):
    # HF Trainer prefixes all metric keys with `metric_key_prefix + '_'`.
    # With per_task: keys end '{lang}/accuracy'; without: just 'accuracy'.
    suffix = f'{lang}/accuracy' if lang else 'accuracy'
    for k, v in metrics.items():
        if k.endswith(suffix):
            return float(v)
    return None


def run_test_on_concat(test_config, target_langs, csv_sink):
    # Route per-lang accuracy through eval.py's metric_groups machinery:
    # tag each sample with task_ids, group by task_ids, one metric_group per lang.
    test_config.task.preproc_rules.per_task = 'task_ids'
    test_config.eval.per_task = True
    test_config.task.metric_groups = [
        _Flex(task=_Flex(id=i, name=lang), metrics=['accuracy'])
        for i, lang in enumerate(target_langs)
    ]

    tokenizer = load_tokenizer(test_config)
    dataset = load_concat_dataset(test_config, target_langs, assign_task_ids=True)
    model = MODEL(test_config)
    trainer = TRAINER(model, dataset, tokenizer)
    output = trainer.run()

    pred_out = output.zero_shot or output.full_shot
    metrics = pred_out.metrics
    task_ids = dataset.test['task_ids']
    for i, lang in enumerate(target_langs):
        n = sum(1 for t in task_ids if t == i)
        csv_sink.add({
            'target_lang': lang,
            'accuracy': _extract_accuracy(metrics, lang=lang),
            'n': n,
        })


def run_test_sequential(test_config, target_langs, csv_sink):
    for lang in target_langs:
        per_lang = test_config.model_copy(deep=True)
        per_lang.ds.dirs = swap_lang_in_dirs(test_config.ds.dirs, lang)
        tokenizer = load_tokenizer(per_lang)
        ds = DATASET(per_lang)
        model = MODEL(per_lang)
        trainer = TRAINER(model, ds, tokenizer)
        output = trainer.run()
        pred_out = output.zero_shot or output.full_shot
        csv_sink.add({
            'target_lang': lang,
            'accuracy': _extract_accuracy(pred_out.metrics),
            'n': len(ds.test),
        })


# --- CSV sink ---------------------------------------------------------------


def _slurm_run_id() -> str:
    """SLURM job id with array suffix when present: '2929287_1' or '2929287'."""
    array_jid = os.environ.get('SLURM_ARRAY_JOB_ID')
    array_tid = os.environ.get('SLURM_ARRAY_TASK_ID')
    if array_jid and array_tid:
        return f'{array_jid}_{array_tid}'
    return os.environ.get('SLURM_JOB_ID', '')


class ResultsCSV:
    HEADER = [
        'run_name', 'run_group', 'llm', 'src_tag', 'source_langs', 'target_lang',
        'accuracy', 'n', 'adapter_uuid4', 'adapter_path',
        'tune_config', 'test_config', 'meta_config', 'slurm_run',
    ]

    def __init__(self, path, meta):
        self.path = Path(path) / 'raw.csv'
        self.meta = meta
        self.rows = []

    def add(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows.append({**self.meta, **row})
        pd.DataFrame(self.rows, columns=self.HEADER).to_csv(self.path, index=False)
        print(f'[xlt] wrote {self.path} ({len(self.rows)} rows)')


# --- Library entry point ----------------------------------------------------


def run_xlt(
    *,
    test_config,
    source_langs,
    run_group,
    slurm_task_id: int,
    interactive: bool = False,
    run_name_tag: str | None = None,
    tune_config=None,
    target_langs=None,
    adapter_uuid4: str | None = None,
    adapter_path: str | None = None,
    sequential_test=False,
    tune_config_source='',
    test_config_source='',
    meta_config_source='',
):
    """
    Run cross-lingual transfer with already-loaded CONFIG objects.

    Three modes (mutually exclusive):
      tune_config given            → tune from scratch, then test the trained adapter.
      adapter_uuid4 given          → skip tune, test the named adapter (path
                                      registered in env via adapter_path if
                                      supplied, else resolved by the toolkit).
      neither given                → zero-shot. No tune, no adapter wiring;
                                      the test config's existing
                                      model.pretrained.adapter is used as-is.

    Caller owns task-id resolution: pass `slurm_task_id` as the resolved
    integer. `interactive=True` prefixes the run_group with 'test_' to keep
    interactive probes out of the prod artefacts namespace.
    """
    if tune_config is not None and adapter_uuid4:
        raise ValueError('tune_config and adapter_uuid4 are mutually exclusive')
    if adapter_path and not adapter_uuid4:
        raise ValueError('adapter_path requires adapter_uuid4 to be set')

    skip_tune = tune_config is None
    zero_shot = skip_tune and not adapter_uuid4

    source_langs_sorted = sorted(s.strip() for s in (source_langs or []) if s.strip())
    if not skip_tune and not source_langs_sorted:
        raise ValueError('source_langs is required for tune mode (the training set is built '
                         'by concatenating per-lang datasets)')

    run_dir, llm, src_tag, run_group, run_name = build_run_paths(
        tune_config, test_config, source_langs_sorted, run_group, slurm_task_id, interactive, run_name_tag
    )
    print(f'[xlt] run_dir={run_dir}')

    if not skip_tune:
        adapter_uuid4, adapter_path = tune_phase(tune_config, source_langs_sorted)
        print(f'[xlt] tuned: uuid4={adapter_uuid4} path={adapter_path}')
        run_dir.mkdir(parents=True, exist_ok=True)
        utils.dict_to_yaml_file({
            'run_name': run_name,
            'run_group': run_group,
            'llm': llm,
            'src_tag': src_tag,
            'source_langs': ','.join(source_langs_sorted),
            'adapter_uuid4': adapter_uuid4,
            'adapter_path': str(adapter_path),
            'tune_config': tune_config_source,
        }, str(run_dir / 'tune.yml'))

    if not zero_shot:
        wire_test_to_adapter(test_config, adapter_uuid4=adapter_uuid4, adapter_path=adapter_path)

    if target_langs is None:
        target_langs = discover_target_langs(test_config)
    print(f'[xlt] target langs ({len(target_langs)}): {target_langs}')

    csv_sink = ResultsCSV(run_dir, meta={
        'run_name': run_name,
        'run_group': run_group,
        'llm': llm,
        'src_tag': src_tag,
        'source_langs': ','.join(source_langs_sorted),
        'adapter_uuid4': adapter_uuid4 or '',
        'adapter_path': adapter_path or '',
        'tune_config': tune_config_source,
        'test_config': test_config_source,
        'meta_config': meta_config_source,
        'slurm_run': _slurm_run_id(),
    })

    if sequential_test:
        run_test_sequential(test_config, target_langs, csv_sink)
    else:
        run_test_on_concat(test_config, target_langs, csv_sink)

    if wandb.run is not None:
        wandb.finish()


# --- CLI Main ---------------------------------------------------------------


def main():
    args = parse_args()

    # If the user passed --adapter-path without --adapter-uuid4, try to
    # recover the uuid from the path basename (the dir is named uuid_..._...).
    # No synthetic uuid fallback — fail loudly so the input gets fixed.
    adapter_uuid4 = args.adapter_uuid4
    if args.adapter_path and not adapter_uuid4:
        adapter_uuid4 = MODEL.extract_uuid_from_name(args.adapter_path.split('/')[-1])
        if not adapter_uuid4:
            raise SystemExit(
                f'--adapter-path basename does not encode a uuid; pass --adapter-uuid4 explicitly'
            )

    source_langs = [s.strip() for s in args.source_langs.split(',') if s.strip()]
    target_langs = (
        [s.strip() for s in args.target_langs.split(',') if s.strip()]
        if args.target_langs else None
    )
    tune_config = CONFIG.from_yaml(args.tune_config) if args.tune_config else None
    test_config = CONFIG.from_yaml(args.test_config)

    # Direct CLI invocation = interactive run; SLURM array dispatch is the
    # meta runner's job (scripts/run_xlt_meta.py owns task-id resolution).
    run_xlt(
        tune_config=tune_config,
        test_config=test_config,
        source_langs=source_langs,
        target_langs=target_langs,
        run_group=args.run_group,
        slurm_task_id=args.s_task_id,
        interactive=True,
        adapter_uuid4=adapter_uuid4,
        adapter_path=args.adapter_path,
        sequential_test=args.sequential_test,
        tune_config_source=args.tune_config or '',
        test_config_source=args.test_config,
    )


if __name__ == '__main__':
    main()
