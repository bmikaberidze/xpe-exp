"""Cross-lingual transfer as a micm-nlp runner.

``micm-nlp run-group --runner scripts/xlt_runner.py`` calls :func:`run` once per
group entry: tune on the source group's languages concatenated, then test the
trained adapter on every *other* language of the benchmark in one pass, one metric
group per language.

The entry carries what the config cannot, and every key is also stamped on each
result row (`group.scalar_columns`)::

    - {config: xpe, name: xpe_f2_s19, seed: 19, fold: 2, method: xpe,
       source_group: enarzho, separate_test: {config: test}}

``source_group`` names a key of ``LANG_GROUPS`` (``src/xlt_langs.py``, the single
source of truth for those lists) and is required: a group config is for one source
group, never two, so the runs it produces cannot be mixed up afterwards.

The science is the same as the pre-0.4 ``scripts/run_xlt.py`` CLI (tag
``pre-micm-nlp-0.4``); its dataset/fold/adapter helpers live here since that CLI
was removed.
"""

import gc
import re
from pathlib import Path

import torch
from datasets import DatasetDict
from micm_nlp.config import AdapterConfig, _Flex
from micm_nlp.datasets.dataset import DATASET
from micm_nlp.models.model import MODEL
from micm_nlp.path import datasets_dir
from micm_nlp.tokenizers.tokenizer import load as load_tokenizer
from micm_nlp.training.runner import TRAINER

from src.xlt_langs import LANG_GROUPS


# --- Paths: language and fold segments of ds.dirs ----------------------------

def swap_lang_in_dirs(dirs_template: str, lang: str) -> str:
    # ds.dirs shape: '<category_group>/<benchmark>/<lang>/tokenized--<vendor>--<model>'
    # (Belebele fold dirs add a 'fold<N>' segment after <lang>; <lang> stays at
    # index 2, so this swap is unaffected by the fold segment.)
    parts = dirs_template.split('/')
    parts[2] = lang
    return '/'.join(parts)


_FOLD_SEG = re.compile(r'fold\d+')


def apply_fold(dirs_template: str, fold: int | None) -> str:
    """Aim a Belebele self-split config at a specific fold at runtime.

    Belebele fold dirs carry a literal 'fold<N>' segment, e.g.
    '.../eng_Latn/fold0/tokenized--...'. With `fold` given, that segment is
    rewritten to 'fold<fold>'. `fold=None` is a no-op (leaves the config's
    literal default, and leaves fold-less paths like SIB-200 untouched).
    Passing a fold to a fold-less path is a user error and raises.
    """
    if fold is None:
        return dirs_template
    parts = dirs_template.split('/')
    for i, part in enumerate(parts):
        if _FOLD_SEG.fullmatch(part):
            parts[i] = f'fold{fold}'
            return '/'.join(parts)
    raise ValueError(
        f"fold={fold} given but ds.dirs has no 'fold<N>' segment: '{dirs_template}'"
    )


def target_langs_excluding(all_langs, source_langs):
    """All discovered langs minus the source langs (order preserved).

    We never test cross-lingual transfer on a source (in-language) lang."""
    drop = set(source_langs or [])
    return [l for l in all_langs if l not in drop]


def discover_target_langs(test_config, exclude=()):
    parts = test_config.ds.dirs.split('/')
    benchmark_rel = '/'.join(parts[:2])
    # Everything after the <lang> segment (parts[2]) is the per-lang subpath.
    # 4-part => 'tokenized--...'; 5-part bebe fold => 'fold<N>/tokenized--...'.
    lang_subpath = '/'.join(parts[3:])
    search_root = Path(datasets_dir()) / test_config.ds.category / benchmark_rel
    langs = []
    for lang_dir in sorted(search_root.iterdir()):
        if (lang_dir / lang_subpath).is_dir():
            langs.append(lang_dir.name)
    return target_langs_excluding(langs, exclude)


# --- Dataset assembly and adapter wiring -------------------------------------

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


# --- Runner ------------------------------------------------------------------


def source_langs(entry: dict) -> list[str]:
    """The entry's source languages, sorted; raises when the group is missing or unknown.

    Sorted because the tuned adapter must not depend on the order the languages were
    listed in -- the concatenated dataset is built in this order.
    """
    group = entry.get('source_group')
    if not group:
        raise ValueError("each run entry needs `source_group:` naming a key of LANG_GROUPS")
    if group not in LANG_GROUPS:
        raise ValueError(f'unknown source_group {group!r}; known: {sorted(LANG_GROUPS)}')
    return sorted(LANG_GROUPS[group])


def aim_at_fold(config, fold: int | None) -> None:
    """Rewrite the config's ``fold<N>`` path segment in place (no-op without a fold)."""
    if fold is not None:
        config.ds.dirs = apply_fold(config.ds.dirs, fold)


def tune(config, langs: list[str]) -> tuple[str, str]:
    """Train on the source languages concatenated; returns the adapter's ``(uuid4, path)``.

    The trainer writes this run's own files (validation metrics, ``info.json``,
    the checkpoint link) into the run directory the group runner prepared.

    Only the adapter's identity leaves this function: the tuned model and its
    Trainer are freed before returning. Otherwise the test phase loads a second
    8B model next to them, and the test token budget (calibrated on free memory)
    OOMs -- 27s smoke, 2026-09-30. gc.collect() breaks the Trainer<->model
    reference cycles; empty_cache() hands the blocks back to the allocator.
    """
    tokenizer = load_tokenizer(config)
    dataset = load_concat_dataset(config, langs)
    model = MODEL(config)
    trainer = TRAINER(model, dataset, tokenizer)
    trainer.run()
    uuid4, path = model.uuid4, model.path
    del trainer, model, dataset, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return uuid4, path


def test_across_languages(test_config, targets: list[str]):
    """One test pass over every target language, one metric group each.

    Each sample is tagged with its language's ``task_ids`` and the metric groups are
    grouped by that tag, so a single pass yields one accuracy per language -- the
    shape the aggregate tables read. The alternative (a pass per language) costs 119
    model loads for the same numbers.
    """
    test_config.task.preproc_rules.per_task = 'task_ids'
    test_config.eval.per_task = True
    test_config.task.metric_groups = [
        _Flex(task=_Flex(id=i, name=lang), metrics=['accuracy'])
        for i, lang in enumerate(targets)
    ]
    tokenizer = load_tokenizer(test_config)
    dataset = load_concat_dataset(test_config, targets, assign_task_ids=True)
    model = MODEL(test_config)
    return TRAINER(model, dataset, tokenizer).run()


def run(config, ctx):
    """Tune on the entry's source group, then test the adapter cross-lingually.

    Four shapes, chosen by the entry -- the same four the pre-0.4 CLI had:

    ``separate_test:``
        tune-and-test. ``config`` is the tune config; the test phase runs under
        ``ctx.test_config``, which lands in the same run directory with the
        ``separate_`` prefix.
    ``tune_only: true``
        tune, no test phase (the old ``--skip-test``): LR searches select on
        validation accuracy, so testing would only cost time. Returns ``None``.
    no ``separate_test:``, ``adapter: <uuid4>``
        replay. No tuning; ``config`` is the test config and the named adapter is
        wired into it.
    no ``separate_test:``, no ``adapter:``
        zero-shot. ``config`` is the test config, tested as it stands -- whatever
        ``model.pretrained.adapter`` it declares, typically the bare base model.

    :returns: the test phase's ``RunOutput`` -- its ``results`` hold one row per
        target language.
    """
    # A bare zero-shot baseline has no source languages -- nothing was tuned, so
    # nothing is "seen" and every language is a target. Only that shape may omit
    # `source_group:`; tuning and replay still require it (replay must exclude the
    # langs its adapter was tuned on).
    if ctx.test_config is None and not ctx.entry.get('adapter') \
            and 'source_group' not in ctx.entry:
        langs = []
    else:
        langs = source_langs(ctx.entry)
    fold = ctx.entry.get('fold')

    if ctx.entry.get('tune_only'):
        if ctx.test_config is not None or ctx.entry.get('adapter'):
            raise ValueError(f'{ctx.name}: tune_only excludes separate_test and adapter')
        aim_at_fold(config, fold)
        tune(config, langs)
        print(f'[xlt] {ctx.name}: tuned on {len(langs)} source langs, no test phase (tune_only)')
        return None

    if ctx.test_config is None:            # replay or zero-shot: `config` is the test config
        test_config = config
        aim_at_fold(test_config, fold)
        adapter = ctx.entry.get('adapter')
        if adapter:
            wire_test_to_adapter(test_config, adapter)
        phase = f'replaying adapter {adapter}' if adapter else 'zero-shot'
    else:                                  # tune-and-test
        aim_at_fold(config, fold)
        adapter_uuid4, adapter_path = tune(config, langs)
        test_config = ctx.test_config
        aim_at_fold(test_config, fold)
        wire_test_to_adapter(test_config, adapter_uuid4, adapter_path)
        phase = f'tuned on {len(langs)} source langs'

    targets = discover_target_langs(test_config, exclude=langs)
    print(f'[xlt] {ctx.name}: {phase}, testing {len(targets)} targets')
    return test_across_languages(test_config, targets)
