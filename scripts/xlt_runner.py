"""Cross-lingual transfer as a micm-nlp runner.

``micm-nlp run-group --runner scripts/xlt_runner.py`` calls :func:`run` once per
group entry: tune on the source group's languages concatenated, then test the
trained adapter on every *other* language of the benchmark in one pass, one metric
group per language.

The entry carries what the config cannot, and every key is also stamped on each
result row (`group.scalar_columns`)::

    - {config: xpe, name: xpe_f2_s19, seed: 19, fold: 2, method: xpe,
       source_group: enarzho, separate_test: {config: test}}

``source_group`` names a key of ``LANG_GROUPS`` (``scripts/run_xlt.py``, the single
source of truth for those lists) and is required: a group config is for one source
group, never two, so the runs it produces cannot be mixed up afterwards.

The science is the same as the pre-0.4 ``scripts/run_xlt.py`` path, which is why the
helpers are imported from it rather than copied: language lists, the fold segment,
the per-language dataset concatenation, and the adapter wiring.
"""

from micm_nlp.config import _Flex
from micm_nlp.models.model import MODEL
from micm_nlp.tokenizers.tokenizer import load as load_tokenizer
from micm_nlp.training.runner import TRAINER

from scripts.run_xlt import (
    LANG_GROUPS,
    apply_fold,
    discover_target_langs,
    load_concat_dataset,
    wire_test_to_adapter,
)


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


def tune(config, langs: list[str]) -> MODEL:
    """Train on the source languages concatenated; returns the model, adapter included.

    The trainer writes this run's own files (validation metrics, ``info.json``,
    the checkpoint link) into the run directory the group runner prepared.
    """
    tokenizer = load_tokenizer(config)
    dataset = load_concat_dataset(config, langs)
    model = MODEL(config)
    TRAINER(model, dataset, tokenizer).run()
    return model


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

    Three shapes, chosen by the entry -- the same three the pre-0.4 CLI had:

    ``separate_test:``
        tune-and-test. ``config`` is the tune config; the test phase runs under
        ``ctx.test_config``, which lands in the same run directory with the
        ``separate_`` prefix.
    no ``separate_test:``, ``adapter: <uuid4>``
        replay. No tuning; ``config`` is the test config and the named adapter is
        wired into it.
    no ``separate_test:``, no ``adapter:``
        zero-shot. ``config`` is the test config, tested as it stands -- whatever
        ``model.pretrained.adapter`` it declares, typically the bare base model.

    :returns: the test phase's ``RunOutput`` -- its ``results`` hold one row per
        target language.
    """
    langs = source_langs(ctx.entry)
    fold = ctx.entry.get('fold')

    if ctx.test_config is None:            # replay or zero-shot: `config` is the test config
        test_config = config
        aim_at_fold(test_config, fold)
        adapter = ctx.entry.get('adapter')
        if adapter:
            wire_test_to_adapter(test_config, adapter)
        phase = f'replaying adapter {adapter}' if adapter else 'zero-shot'
    else:                                  # tune-and-test
        aim_at_fold(config, fold)
        model = tune(config, langs)
        test_config = ctx.test_config
        aim_at_fold(test_config, fold)
        wire_test_to_adapter(test_config, model.uuid4, model.path)
        phase = f'tuned on {len(langs)} source langs'

    targets = discover_target_langs(test_config, exclude=langs)
    print(f'[xlt] {ctx.name}: {phase}, testing {len(targets)} targets')
    return test_across_languages(test_config, targets)
