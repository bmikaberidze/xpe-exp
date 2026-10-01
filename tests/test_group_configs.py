"""Every group config under config/groups/ must load and resolve before it is dispatched.

Resolves each entry the way `micm_nlp run-group` does (unit config, seed, overrides,
separate_test), without creating run dirs, and checks the xlt_runner rules: a known
source group, a fold only on fold paths, and exactly one of the four entry shapes.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from micm_nlp.config import CONFIG
from micm_nlp.group import apply_override, apply_seed, load_group

import scripts.xlt_runner as xlt_runner
from src.xlt_langs import LANG_GROUPS

GROUPS = sorted((Path(__file__).resolve().parents[1] / 'config' / 'groups').glob('*.yml'))
_CACHE = {}


def _load(path):
    if path not in _CACHE:
        _CACHE[path] = CONFIG.from_yaml(path)
    return _CACHE[path].model_copy(deep=True)


@pytest.mark.parametrize('path', GROUPS, ids=[p.stem for p in GROUPS])
def test_group_config_resolves(path):
    group = load_group(path)
    for i, entry in enumerate(group['runs']):
        where = f"{path.name}[{i}] {entry['name']}"
        config = _load(group['configs'][entry['config']])
        apply_seed(config, entry.get('seed'))
        for dotted, value in (entry.get('overrides') or {}).items():
            apply_override(config, dotted, value)

        if 'source_group' in entry:
            assert entry['source_group'] in LANG_GROUPS, where
        if entry.get('fold') is not None:
            xlt_runner.apply_fold(config.ds.dirs, entry['fold'])     # raises on a fold-less path

        shapes = [bool(entry.get('separate_test')), bool(entry.get('tune_only')), bool(entry.get('adapter'))]
        assert sum(shapes) <= 1, f'{where}: separate_test / tune_only / adapter are exclusive'
        if entry.get('separate_test'):
            _load(group['configs'][entry['separate_test']['config']])
        if entry.get('tune_only') or entry.get('separate_test'):
            assert config.mode in ('finetune', 'train'), f'{where}: tuning entry on a {config.mode} config'
        else:
            assert config.mode in ('test', 'evaluate'), f'{where}: test entry on a {config.mode} config'


def test_tune_only_tunes_and_skips_the_test_phase(monkeypatch):
    calls = []
    monkeypatch.setattr(xlt_runner, 'tune', lambda config, langs: calls.append(('tune', langs)))
    monkeypatch.setattr(xlt_runner, 'test_across_languages', lambda *a: calls.append(('test',)))
    config = SimpleNamespace(ds=SimpleNamespace(dirs='mcqa/belebele_ftp/eng_Latn/fold0/tokenized--x--y'))
    ctx = SimpleNamespace(test_config=None, name='n',
                          entry={'tune_only': True, 'source_group': 'enarzho', 'fold': 2})
    assert xlt_runner.run(config, ctx) is None
    assert calls == [('tune', sorted(LANG_GROUPS['enarzho']))]
    assert config.ds.dirs == 'mcqa/belebele_ftp/eng_Latn/fold2/tokenized--x--y'


def test_tune_only_rejects_a_test_phase():
    ctx = SimpleNamespace(test_config=object(), name='n',
                          entry={'tune_only': True, 'source_group': 'enarzho'})
    with pytest.raises(ValueError, match='tune_only'):
        xlt_runner.run(SimpleNamespace(), ctx)
