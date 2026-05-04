"""Shared helpers for uploading reframed datasets to the HuggingFace Hub.

Used by:
- exps/xpe2/scripts/datasets/upload_bebe_ftp_to_hub.py
- exps/xpe2/scripts/datasets/upload_xsc_ftp_to_hub.py
"""

import argparse
from pathlib import Path

import yaml
from datasets import DatasetDict
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError


def discover_local_configs(root: Path) -> list[str]:
    """Return sorted names of immediate subdirectories under `root`."""
    if not root.exists():
        raise FileNotFoundError(f"Local dataset root does not exist: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def prepare_dataset(
    local_dir: Path,
    config_name: str,
    split_remap: dict[str, str] | None = None,
) -> DatasetDict:
    """Load `local_dir/config_name` from disk, optionally renaming splits.

    If `split_remap` is provided, every key MUST be present in the loaded splits;
    splits not in `split_remap` keys are dropped.
    """
    ds = DatasetDict.load_from_disk(str(local_dir / config_name))
    if split_remap is None:
        return ds
    missing = [k for k in split_remap if k not in ds]
    if missing:
        raise KeyError(
            f"split_remap references splits not in {config_name}: {missing}. "
            f"Available splits: {sorted(ds.keys())}"
        )
    return DatasetDict({new: ds[old] for old, new in split_remap.items()})


def render_yaml_frontmatter(
    license_id: str,
    language_codes: list[str],
    tags: list[str],
    pretty_name: str,
    configs: list[str],
    split_paths: dict[str, list[str]],
) -> str:
    """Render the YAML frontmatter block for a multi-config dataset card.

    `split_paths[config]` lists the splits that exist for that config; each one becomes
    a `{split: <name>, path: <config>/<split>-*}` entry under that config's `data_files`.
    """
    payload = {
        "license": license_id,
        "language": list(language_codes),
        "task_categories": ["multiple-choice", "text-generation"],
        "tags": list(tags),
        "pretty_name": pretty_name,
        "configs": [
            {
                "config_name": cfg,
                "data_files": [
                    {"split": s, "path": f"{cfg}/{s}-*"}
                    for s in split_paths[cfg]
                ],
            }
            for cfg in configs
        ],
    }
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=10_000)
    return f"---\n{body}---\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a reframed dataset to the HuggingFace Hub.")
    parser.add_argument("--repo-id", required=True, help="Hub repo id, e.g. mikaberidze/belebele-ftp")
    parser.add_argument("--config", default=None, help="Push only this config (smoke test); default = all")
    parser.add_argument("--private", action="store_true", help="Create a private repo")
    return parser.parse_args(argv)


def ensure_repo(repo_id: str, private: bool, token: str) -> None:
    """Create the dataset repo if it does not already exist. Idempotent."""
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", private=private, token=token, exist_ok=True)
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            raise RuntimeError(
                f"403 creating {repo_id}. The HF_TOKEN likely lacks WRITE scope. "
                f"Regenerate at https://huggingface.co/settings/tokens with `write` scope."
            ) from e
        raise


def push_one_config(
    local_dir: Path,
    config_name: str,
    repo_id: str,
    split_remap: dict[str, str] | None,
    token: str,
) -> None:
    """Load one config from disk, optionally remap splits, push to the Hub."""
    ds = prepare_dataset(local_dir, config_name, split_remap=split_remap)
    ds.push_to_hub(repo_id, config_name=config_name, token=token)


def upload_card(repo_id: str, card_text: str, token: str) -> None:
    """Overwrite the repo's README.md with `card_text`."""
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=card_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="docs: update dataset card",
    )
