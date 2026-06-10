"""
Preprocess (tokenize) all language subdirectories in a dataset directory.

Reads config.ds.dirs, strips the language part to get the benchmark base, and
for each language under it tokenizes every dataset found there:

  - the language's own DatasetDict (e.g. the reframed Belebele full-900 `test`,
    or an xsc split set), and
  - any item-disjoint fold subdirs `fold{f}/` written by
    `scripts.datasets.split_bebe_folds` (the cross-lingual-transfer folds).

Datasets without folds (e.g. xstory_cloze) just get the single DatasetDict
tokenized, so this stays backward compatible. Each target is tokenized into its
own `tokenized|<vendor>|<model>` subdir; what was found is logged, not silent.

Config Files:
proc.ds.xsc.tok.aya.yml
proc.ds.xsc.tok.bloom.yml
proc.ds.bebe.tok.aya.yml
proc.ds.bebe.tok.bloom.yml

Usage:
/home/bmikaberidze/.venv/bin/python -m scripts.datasets.preprocess_dir \
    --config ./config/{config_file}
"""

if __name__ == '__main__':

    import re
    from pathlib import PurePosixPath

    import micm_nlp
    from micm_nlp.config import CONFIG
    from micm_nlp.path import datasets_dir
    from micm_nlp.pipeline import preprocess_dataset
    from src.utils import micm_nlp_setup

    micm_nlp_setup()

    config_path = micm_nlp.utils.parse_script_args()
    config = CONFIG.from_yaml(config_path)

    FOLD_RE = re.compile(r'^fold\d+$')

    def is_dataset_dict(path):
        return (path / 'dataset_dict.json').exists()

    def targets_for_lang(lang_dir, base_ds_dirs, lang):
        """Relative ds.dirs to tokenize for one language: the language's own
        DatasetDict (if present) plus any fold{f}/ subdirs."""
        rels = []
        if is_dataset_dict(lang_dir):
            rels.append(f"{base_ds_dirs}/{lang}")
        for sub in sorted(lang_dir.iterdir()):
            if sub.is_dir() and FOLD_RE.match(sub.name) and is_dataset_dict(sub):
                rels.append(f"{base_ds_dirs}/{lang}/{sub.name}")
        return rels

    # config.ds.dirs is e.g. "mcqa/belebele_ftp/kat_Geor" — strip the language part
    ds_dirs_path = PurePosixPath(config.ds.dirs)
    base_ds_dirs = str(ds_dirs_path.parent)
    base_dir = datasets_dir() / config.ds.category / base_ds_dirs
    lang_dirs = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    print(f"Found {len(lang_dirs)} languages in {base_dir}: {lang_dirs}")

    for lang in lang_dirs:
        targets = targets_for_lang(base_dir / lang, base_ds_dirs, lang)
        if not targets:
            print(f"\n[skip] {lang}: no DatasetDict and no fold*/ subdirs")
            continue

        n_folds = sum(1 for t in targets if FOLD_RE.match(PurePosixPath(t).name))
        has_root = any(PurePosixPath(t).name == lang for t in targets)
        print(f"\n{'=' * 60}")
        print(f"Processing: {lang}  (root={'yes' if has_root else 'no'}, folds={n_folds})")
        print(f"{'=' * 60}")

        for rel in targets:
            print(f"  -> {rel}")
            config = CONFIG.from_yaml(config_path)
            config.ds.dirs = rel
            try:
                preprocess_dataset(config)
            except Exception as e:
                print(f"  Error processing {rel}: {e}")
                continue
