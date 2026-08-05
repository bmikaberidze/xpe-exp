"""Download SIB-200 from the Hub, shape it to the legacy layout, and save it.

SIB-200 is already a topic-detection dataset, so there is no reframing step here
(unlike `reframe_bebe_to_ftp`) -- this script downloads, renames the two columns
the configs read, maps the category strings to label ids, and writes one
DatasetDict per language.

The shaping matches the legacy XPE pipeline so results stay comparable to the
published table:
  - `text` -> `inputs`, `category` -> `labels`
  - label ids follow `src.sib200_meta.LABEL_NAMES`, NOT alphabetical order
  - the official 701/99/204 train/validation/test splits are used as-is; SIB-200
    has no fold machinery (unlike Belebele), so `--fold` is never passed

`index_id` keeps its own name on purpose. `run_xlt.load_concat_dataset` injects a
`task_ids` column (the source/target LANGUAGE index) via `Dataset.add_column`,
and `datasets` raises `ValueError: The table can't have duplicated columns` if
one already exists. The legacy pipeline also derived `task_ids` from the language
id, never from the item index -- so reusing that name here would be wrong twice.

Output layout (per language):
    benchmarks/topic/sib200/<lang>/     # DatasetDict{train,validation,test}

Usage (never on the login node -- see CLAUDE.md "Run environment"):
    sbatch --array=0 --mem=30G --wait \\
      runtime/clusters/pegasus/shell/run.sh --site-packages --no-gpu \\
        "python -m scripts.datasets.download_sib"
"""

import logging

from src.sib200_meta import LABEL_NAMES, sib200_codes

logger = logging.getLogger(__name__)

HF_DATASET = 'Davlan/sib200'

# The configs read `ds.input.key: inputs` and `ds.label.key: labels`.
COLUMN_RENAMES = {'category': 'labels', 'text': 'inputs'}

_LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}

SPLITS = ('train', 'validation', 'test')

# Official SIB-200 split sizes, identical for every language (paper Table 1).
EXPECTED_SPLIT_SIZES = {'train': 701, 'validation': 99, 'test': 204}


def language_configs(all_configs):
    """Filter the hub's config list down to the 205 real SIB-200 codes.

    The hub lists a junk `nqo_Nkoo.zip` config alongside the languages. Anything
    that is neither a known SIB-200 code nor that archive artefact is an error,
    not something to drop silently -- a renamed or added config should surface
    here rather than quietly shrink the benchmark.
    """
    known = set(sib200_codes())
    langs = sorted(c for c in all_configs if c in known)
    unknown = [c for c in all_configs if c not in known and not c.endswith('.zip')]
    if unknown:
        raise ValueError(f'hub configs not in the SIB-200 table: {unknown}')
    return langs


def shape_split(records):
    """Rename columns and map category strings to label ids for one split."""
    shaped = []
    for r in records:
        category = r['category']
        if category not in _LABEL_TO_ID:
            raise ValueError(f'unknown category {category!r}; expected one of {LABEL_NAMES}')
        shaped.append({
            'index_id': r['index_id'],
            COLUMN_RENAMES['category']: _LABEL_TO_ID[category],
            COLUMN_RENAMES['text']: r['text'],
        })
    return shaped


def main():
    from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
    from micm_nlp.path import datasets_dir
    from src.utils import micm_nlp_setup

    micm_nlp_setup()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    base = datasets_dir() / 'benchmarks' / 'topic' / 'sib200'
    langs = language_configs(get_dataset_config_names(HF_DATASET))
    logger.info(f'{len(langs)} SIB-200 languages -> {base}')

    for i, lang in enumerate(langs, 1):
        raw = load_dataset(HF_DATASET, lang)
        shaped = DatasetDict({
            split: Dataset.from_list(shape_split(list(raw[split]))) for split in SPLITS
        })
        sizes = {s: len(shaped[s]) for s in SPLITS}
        if sizes != EXPECTED_SPLIT_SIZES:
            raise ValueError(f'{lang}: split sizes {sizes} != official {EXPECTED_SPLIT_SIZES}')

        out = base / lang
        out.mkdir(parents=True, exist_ok=True)
        shaped.save_to_disk(str(out))

        # Round-trip: a dataset that cannot be read back is a silent failure
        # that would otherwise only surface hundreds of languages later.
        loaded = DatasetDict.load_from_disk(str(out))
        for split in SPLITS:
            assert len(loaded[split]) == sizes[split], f'{lang}/{split} round-trip mismatch'
        logger.info(f'  [{i}/{len(langs)}] {lang}: ' + ', '.join(f'{s}={sizes[s]}' for s in SPLITS))

    logger.info(f'Done! {len(langs)} languages under {base}')


if __name__ == '__main__':
    main()
