"""
Cross-Lingual Transfer (XLT) runner.

Tunes LM on a concatenated source-language dataset (xstory_cloze / xsc),
then tests per target language (belebele / bebe).

Usage:
    # Full pipeline (tune + test)
    python -m scripts.run_xlt \
        --tune-config ./config/tune.xpe.lm.bloom.ds.xsc.yml \
        --test-config ./config/test.lm.bloom.ds.bebe.yml \
        --source-langs ar,en \
        --target-langs kat_Geor,eng_Latn \
        --run-group smoke

    # Test-only, from an existing source model path
    python -m scripts.run_xlt \
        --test-config ./config/test.lm.bloom.ds.bebe.yml \
        --source-path "/fscratch/bmikaberidze/micm-nlp/exp/artefacts/models/bloom/bigscience/bloom-560m/mcqa_ftp/a6eb0f42-ba4e-478d-a93c-dd40f15714ab_bigscience|bloom-560m_2827473_1_mcqa|xstory_cloze_ftp|ar|tokenized|bigscience|bloom-560m_1_16" \
        --source-langs ar,en \
        --target-langs kat_Geor,eng_Latn \
        --run-group smoke

    # Test-only, from an existing uuid4 (resolves via env-var registry or disk search)
    python -m scripts.run_xlt \
        ... --source-uuid4 <uuid>

    # Sequential per-lang test (debug; skip concat-predict bucketing)
    ... --sequential-test

SLURM dual-mode: if SLURM_ARRAY_TASK_ID is in env, prod mode; else interactive —
run_group is auto-prefixed with 'test_' and a fake SLURM_ARRAY_TASK_ID is set.
"""

if __name__ == '__main__':
    from src.utils import micm_nlp_setup
    micm_nlp_setup()

import os
import uuid
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
    ap.add_argument('--tune-config', type=str, default=None)
    ap.add_argument('--test-config', type=str, required=True)
    ap.add_argument('--source-langs', type=str, required=True,
                    help='Comma-separated xsc lang codes')
    ap.add_argument('--target-langs', type=str, default=None,
                    help='Comma-separated bebe lang codes; default: auto-discover from tokenized dirs')
    ap.add_argument('--run-group', type=str, required=True)
    ap.add_argument('--source-path', type=str, default=None,
                    help='Absolute path to tuned source_model; skip tune phase')
    ap.add_argument('--source-uuid4', type=str, default=None,
                    help='uuid4 of tuned model; skip tune, resolve via MODEL env/disk registry')
    ap.add_argument('--sequential-test', action='store_true',
                    help='Run bebe langs one-by-one instead of concat-predict (debug)')
    ap.add_argument('--s-task-id', type=int, default=1,
                    help='Fake SLURM_ARRAY_TASK_ID for interactive mode')
    args = ap.parse_args()

    if args.source_path and args.source_uuid4:
        ap.error('--source-path and --source-uuid4 are mutually exclusive')
    skip_tune = bool(args.source_path or args.source_uuid4)
    if not skip_tune and not args.tune_config:
        ap.error('--tune-config is required unless --source-path or --source-uuid4 is given')
    return args, skip_tune


# --- Path / naming helpers --------------------------------------------------


def swap_lang_in_dirs(dirs_template: str, lang: str) -> str:
    # ds.dirs shape: '<category_group>/<benchmark>/<lang>/tokenized|<vendor>|<model>'
    parts = dirs_template.split('/')
    parts[2] = lang
    return '/'.join(parts)


def build_run_paths(tune_config, test_config, source_langs_sorted, run_group, s_task_id):
    llm_config = tune_config if tune_config else test_config
    llm = llm_config.model.architecture
    src_tag = '-'.join(source_langs_sorted)

    if SLURM_ARRAY_TASK_ID not in os.environ:
        os.environ[SLURM_ARRAY_TASK_ID] = str(s_task_id)
        run_group = f'test_{run_group}'

    ratio = None
    if tune_config is not None and tune_config.peft is not None:
        ratio = getattr(tune_config.peft, 'encoder_ratio', None)
    run_name = f'{utils.get_time_id()}_{ratio}' if ratio is not None else utils.get_time_id()

    run_dir = Path(evals_dir()) / 'xlt_runs' / llm / src_tag / run_group / run_name
    # run_dir.mkdir(parents=True, exist_ok=True)
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


def wire_test_to_source_model(test_config, source_uuid4=None, source_path=None):
    # Route #2: leave model.pretrained pointing at the base HF model from YAML
    # and attach the trained source_model as an adapter, so PEFT.setup_model
    # dispatches via PEFT.from_pretrained → load_xpe_pretrained.
    if source_path: 
        if not source_uuid4:
            source_name = source_path.split('/')[-1]
            source_uuid4 = MODEL.extract_uuid_from_name(source_name) or str(uuid.uuid4())
        MODEL.store_path_by_uuid4_in_envs(source_uuid4, source_path)
    test_config.model.pretrained.adapter = AdapterConfig(source='local', uuid4=source_uuid4)


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


class ResultsCSV:
    HEADER = [
        'run_name', 'run_group', 'llm', 'src_tag', 'source_langs', 'target_lang',
        'accuracy', 'n', 'source_uuid4', 'source_path',
        'tune_config', 'test_config',
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


# --- Main -------------------------------------------------------------------


def main():
    args, skip_tune = parse_args()

    source_langs_sorted = sorted(x.strip() for x in args.source_langs.split(',') if x.strip())

    tune_config = CONFIG.from_yaml(args.tune_config) if args.tune_config else None
    test_config = CONFIG.from_yaml(args.test_config)

    run_dir, llm, src_tag, run_group, run_name = build_run_paths(
        tune_config, test_config, source_langs_sorted, args.run_group, args.s_task_id
    )
    print(f'[xlt] run_dir={run_dir}')

    if skip_tune:
        source_uuid4 = args.source_uuid4
        source_path = args.source_path
    else:
        source_uuid4, source_path = tune_phase(tune_config, source_langs_sorted)
        print(f'[xlt] tuned: uuid4={source_uuid4} path={source_path}')
        run_dir.mkdir(parents=True, exist_ok=True)
        utils.dict_to_yaml_file({
            'run_name': run_name,
            'run_group': run_group,
            'llm': llm,
            'src_tag': src_tag,
            'source_langs': ','.join(source_langs_sorted),
            'source_uuid4': source_uuid4,
            'source_path': str(source_path),
            'tune_config': args.tune_config,
        }, str(run_dir / 'tune.yml'))

    wire_test_to_source_model(
        test_config,
        source_uuid4=source_uuid4,
        source_path=source_path if args.source_path else None,
    )

    if args.target_langs:
        target_langs = [x.strip() for x in args.target_langs.split(',') if x.strip()]
    else:
        target_langs = discover_target_langs(test_config)
    print(f'[xlt] target langs ({len(target_langs)}): {target_langs}')

    csv_sink = ResultsCSV(run_dir, meta={
        'run_name': run_name,
        'run_group': run_group,
        'llm': llm,
        'src_tag': src_tag,
        'source_langs': ','.join(source_langs_sorted),
        'source_uuid4': source_uuid4 or '',
        'source_path': source_path or '',
        'tune_config': args.tune_config or '',
        'test_config': args.test_config,
    })

    if args.sequential_test:
        run_test_sequential(test_config, target_langs, csv_sink)
    else:
        run_test_on_concat(test_config, target_langs, csv_sink)

    if wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
