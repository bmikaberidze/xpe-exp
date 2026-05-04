"""
Preprocess all language subdirectories in a dataset directory.

Reads config.ds.dirs, globs all subdirs under that path, and for each one
overrides config.ds.dirs to point to that language's data, then runs preprocessing.

Config Files:
proc.ds.xsc.tok.aya.yml
proc.ds.xsc.tok.bloom.yml
proc.ds.bebe.tok.aya.yml
proc.ds.bebe.tok.bloom.yml

Usage:
python -m scripts.datasets.preprocess_dir \
    --config ./config/{config_file}

"""

if __name__ == '__main__':

    import micm_nlp
    from micm_nlp.config import CONFIG
    from micm_nlp.path import datasets_dir
    from micm_nlp.pipeline import preprocess_dataset
    from src.utils import micm_nlp_setup

    micm_nlp_setup()

    config_path = micm_nlp.utils.parse_script_args()
    config = CONFIG.from_yaml(config_path)

    # config.ds.dirs is e.g. "mcqa/belebele_ftp/kat_Geor" — strip the language part
    from pathlib import PurePosixPath
    ds_dirs_path = PurePosixPath(config.ds.dirs)
    base_ds_dirs = str(ds_dirs_path.parent)
    base_dir = datasets_dir() / config.ds.category / base_ds_dirs
    lang_dirs = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    print(f"Found {len(lang_dirs)} languages in {base_dir}: {lang_dirs}")

    for lang in lang_dirs:
        print(f"\n{'='*60}")
        print(f"Processing: {lang}")
        print(f"{'='*60}")

        config = CONFIG.from_yaml(config_path)
        config.ds.dirs = f"{base_ds_dirs}/{lang}"

        try:
            preprocess_dataset(config)
        except Exception as e:
            print(f"Error processing {lang}: {e}")
            continue
