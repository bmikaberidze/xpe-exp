"""
Analyze tokenized dataset lengths across all language subdirectories.

Reads config.ds.dirs, globs all subdirs under that path, and for each one
preprocesses (tokenizes) and collects length statistics into a single CSV.

Usage:
    python -m scripts.datasets.stats \
        --config ./config/proc.ds.xsc.tok.aya.yml

    python -m scripts.datasets.stats \
        --config ./config/proc.ds.bebe.tok.aya.yml
"""

import os
import csv
from pathlib import PurePosixPath

import micm_nlp
import micm_nlp.path as path
from micm_nlp.config import CONFIG
from micm_nlp.path import datasets_dir
from micm_nlp.pipeline import preprocess_dataset
from src.utils import micm_nlp_setup


def save_stats(stats_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = list(stats_list[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_list)
    print(f"Stats saved to: {output_path}")


if __name__ == '__main__':

    micm_nlp_setup()

    config_path = micm_nlp.utils.parse_script_args()
    config = CONFIG.from_yaml(config_path)

    # Strip the language part from ds.dirs to get the base directory
    ds_dirs_path = PurePosixPath(config.ds.dirs)
    base_ds_dirs = str(ds_dirs_path.parent)
    base_dir = datasets_dir() / config.ds.category / base_ds_dirs
    lang_dirs = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    print(f"Found {len(lang_dirs)} languages in {base_dir}: {lang_dirs}")

    stats_list = []

    for i, lang in enumerate(lang_dirs):
        print(f"\n{'='*60}")
        print(f"Analyzing: {lang}")
        print(f"{'='*60}")

        config = CONFIG.from_yaml(config_path)
        config.ds.dirs = f"{base_ds_dirs}/{lang}"

        try:
            dataset = preprocess_dataset(config)
            stats = {'id': i, 'lang': lang}
            stats.update(dataset.analyze_lengths())
            stats_list.append(stats)
            micm_nlp.utils.p(stats)
        except Exception as e:
            print(f"Error analyzing {lang}: {e}")
            continue

    if stats_list:
        tok_model_name = config.tokenizer.name.split('/')[-1]
        ds_name = PurePosixPath(base_ds_dirs).name
        output_path = f'{path.evals_dir()}/length_stats/{ds_name}_{tok_model_name}.csv'
        save_stats(stats_list, output_path)
