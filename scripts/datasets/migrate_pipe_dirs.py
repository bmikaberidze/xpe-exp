"""Rename dataset folders named ``tokenized|org|model`` to ``tokenized--org--model``.

datasets >= 4 turns a cache file's path into a regex when ``map()`` looks for earlier
cache files, and ``|`` in the path is read as regex alternation -- so the second run
over any such folder crashes (``int(None)`` in ``arrow_dataset.map``). micm-nlp now
writes ``--`` names; this moves the existing folders over.

Each folder is renamed in place (no data copied) and a symlink is left at the old name,
so the pre-micm-nlp-0.4 era (datasets 3.4, no such lookup) keeps loading its configs'
``|`` paths. The ``tokenized|...`` spellings in ``config/`` are rewritten as text, so
YAML comments survive. Re-running is a no-op.

usage: python -m scripts.datasets.migrate_pipe_dirs [--apply]    (dry run without --apply)
"""

import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / 'artefacts' / 'datasets'
CONFIGS = ROOT / 'config'
TOKENIZED = re.compile(r'tokenized\|([^|/\s`]+)\|([^|/\s`]+)')


def pipe_dirs():
    """Real folders (not our own symlinks) whose name contains ``|``, not descended into."""
    for dirpath, dirnames, _ in os.walk(DATASETS):
        for name in list(dirnames):
            if '|' in name:
                dirnames.remove(name)
                path = Path(dirpath) / name
                if not path.is_symlink():
                    yield path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--apply', action='store_true', help='rename and rewrite; without it only report')
    apply = ap.parse_args().apply

    dirs = sorted(pipe_dirs())
    conflicts = [d for d in dirs if d.with_name(d.name.replace('|', '--')).exists()]
    print(f'folders to rename: {len(dirs)}   conflicts (new name exists, skipped): {len(conflicts)}')
    for d in conflicts:
        print(f'  CONFLICT {d.relative_to(DATASETS)}')
    for d in dirs[:3]:
        print(f'  e.g. {d.relative_to(DATASETS)} -> {d.name.replace("|", "--")}')

    configs = {}
    for p in sorted(CONFIGS.rglob('*.yml')):
        n = len(TOKENIZED.findall(p.read_text()))
        if n:
            configs[p] = n
    print(f'configs to rewrite: {len(configs)} files, {sum(configs.values())} occurrences')

    if not apply:
        print('dry run: nothing changed (pass --apply)')
        return

    renamed = 0
    for d in dirs:
        if d in conflicts:
            continue
        new = d.with_name(d.name.replace('|', '--'))
        d.rename(new)
        os.symlink(new.name, d)  # relative link at the old name, for the old era
        renamed += 1
    for p in configs:
        p.write_text(TOKENIZED.sub(r'tokenized--\1--\2', p.read_text()))
    print(f'renamed {renamed} folders (symlinked back), rewrote {len(configs)} configs')


if __name__ == '__main__':
    main()
