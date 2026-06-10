"""Split the reframed Belebele FTP dataset into item-disjoint folds.

Reads the LOCAL reframed dataset (produced by `reframe_bebe_to_ftp`, which
carries the parallel key `(link, question_number)` as provenance) and writes,
per language, N folds of train/validation/test. Belebele is fully parallel —
the same 900 items appear in every language config — so splitting by the stable
key, never by row position, guarantees a held-out test item never leaks into
training via its translation in a source language. The N test blocks tile the
item set (3-fold CV: each item tested exactly once), so folds can be averaged.

This module owns all splitting logic; `reframe_bebe_to_ftp` stays fold-agnostic.

Output layout (per language, per fold):
    benchmarks/mcqa/belebele_ftp/<lang>/fold{f}/   # DatasetDict{train,validation,test}

Usage:
    /home/bmikaberidze/.venv/bin/python -m scripts.datasets.split_bebe_folds
"""

import random

# Fold geometry (Belebele has 900 items per language).
N_SPLITS = 3
TRAIN, VAL, TEST = 500, 100, 300
DEFAULT_SEED = 20260608

# assign_fold_splits emits 'train'|'val'|'test'; map to the canonical HF split
# names the toolkit loader expects.
SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}


def assign_fold_splits(keys, n_splits=N_SPLITS, train=TRAIN, val=VAL, test=TEST, seed=DEFAULT_SEED):
    """Assign each item key to a train/val/test split, per fold.

    Args:
        keys: iterable of hashable item keys (e.g. ``(link, question_number)``).
        n_splits: number of folds; test blocks tile the item set across folds.
        train, val, test: per-fold split sizes. Must satisfy
            ``test * n_splits == len(keys)`` and ``train + val + test == len(keys)``.
        seed: shuffle seed. Output is deterministic and independent of input
            ordering (keys are canonicalised before shuffling).

    Returns:
        list of length ``n_splits``; element ``f`` maps every key to
        ``"train" | "val" | "test"`` for fold ``f``.
    """
    keys = sorted(set(keys))
    total = len(keys)
    if test * n_splits != total:
        raise ValueError(
            f"test*n_splits ({test}*{n_splits}={test * n_splits}) must equal "
            f"total items ({total}) for test blocks to tile the set"
        )
    if train + val + test != total:
        raise ValueError(
            f"train+val+test ({train}+{val}+{test}={train + val + test}) must "
            f"equal total items ({total})"
        )

    order = keys[:]
    random.Random(seed).shuffle(order)

    folds = []
    for f in range(n_splits):
        test_block = set(order[f * test:(f + 1) * test])
        rest = [k for k in order if k not in test_block]
        fmap = {k: "val" for k in rest[:val]}
        fmap.update({k: "train" for k in rest[val:val + train]})
        fmap.update({k: "test" for k in test_block})
        folds.append(fmap)
    return folds


def item_key(record):
    """Stable parallel key, identical across all language configs."""
    return (record["link"], record["question_number"])


def fold_datasetdict(records, fold_map, key_fn=item_key):
    """Bucket reframed records into a DatasetDict by a single fold's split map.

    Args:
        records: list of reframed FTP record dicts (must carry the key fields).
        fold_map: {key: 'train'|'val'|'test'} for one fold.
        key_fn: record -> key (must match the keys in fold_map).

    Returns:
        DatasetDict with canonical HF split names (train / validation / test).
    """
    from datasets import Dataset, DatasetDict

    buckets = {"train": [], "val": [], "test": []}
    for r in records:
        buckets[fold_map[key_fn(r)]].append(r)
    return DatasetDict({
        SPLIT_NAMES[k]: Dataset.from_list(v) for k, v in buckets.items()
    })


def main():
    import logging
    from datasets import DatasetDict
    from micm_nlp.path import datasets_dir
    from src.utils import micm_nlp_setup

    micm_nlp_setup()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    base = datasets_dir() / "benchmarks" / "mcqa" / "belebele_ftp"
    lang_dirs = sorted(d.name for d in base.iterdir() if d.is_dir())
    logger.info(f"Found {len(lang_dirs)} reframed languages in {base}")

    # One language-independent fold geometry, applied identically to every
    # language; each language is asserted to carry the same parallel key set.
    reference_keys = None
    fold_maps = None

    for lang in lang_dirs:
        lang_path = base / lang
        try:
            dd = DatasetDict.load_from_disk(str(lang_path))
        except Exception as e:
            logger.warning(f"  {lang}: not a reframed DatasetDict ({e}); skipping")
            continue
        if "test" not in dd:
            logger.warning(f"  {lang}: no 'test' split; skipping")
            continue

        records = list(dd["test"])
        keys = {item_key(r) for r in records}

        if reference_keys is None:
            reference_keys = keys
            fold_maps = assign_fold_splits(list(reference_keys))
            logger.info(f"  fold geometry: {N_SPLITS} folds, {TRAIN}/{VAL}/{TEST} per fold")
        elif keys != reference_keys:
            logger.error(f"  {lang}: parallel key set differs from reference; skipping")
            continue

        for f, fold_map in enumerate(fold_maps):
            fold_dd = fold_datasetdict(records, fold_map)
            out = lang_path / f"fold{f}"
            out.mkdir(parents=True, exist_ok=True)
            fold_dd.save_to_disk(str(out))
            loaded = DatasetDict.load_from_disk(str(out))
            for split in loaded:
                assert len(loaded[split]) == len(fold_dd[split])
            logger.info(
                f"  {lang}/fold{f}: "
                + ", ".join(f"{s}={len(fold_dd[s])}" for s in ("train", "validation", "test"))
            )

    logger.info(f"Done! Folds written under {base}/<lang>/fold*")


if __name__ == "__main__":
    main()
