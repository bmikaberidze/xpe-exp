from pathlib import Path

import pytest
import yaml as pyyaml
from datasets import Dataset, DatasetDict

from exps.xpe2.src.hub_upload import (
    discover_local_configs,
    prepare_dataset,
    render_yaml_frontmatter,
    parse_args,
)


def test_discover_local_configs_returns_sorted_subdir_names(tmp_path: Path):
    (tmp_path / "kat_Geor").mkdir()
    (tmp_path / "eng_Latn").mkdir()
    (tmp_path / "rus_Cyrl").mkdir()
    (tmp_path / "not_a_config.txt").write_text("")  # files ignored

    assert discover_local_configs(tmp_path) == ["eng_Latn", "kat_Geor", "rus_Cyrl"]


def test_discover_local_configs_empty_dir(tmp_path: Path):
    assert discover_local_configs(tmp_path) == []


def test_discover_local_configs_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discover_local_configs(tmp_path / "does_not_exist")


def _toy_ds(rows: list[dict]) -> Dataset:
    return Dataset.from_list(rows)


def test_prepare_dataset_no_remap(tmp_path: Path):
    cfg = tmp_path / "en"
    DatasetDict({
        "test": _toy_ds([{"x": 1}, {"x": 2}]),
    }).save_to_disk(str(cfg))

    out = prepare_dataset(tmp_path, "en", split_remap=None)
    assert list(out.keys()) == ["test"]
    assert len(out["test"]) == 2


def test_prepare_dataset_with_remap(tmp_path: Path):
    cfg = tmp_path / "en"
    DatasetDict({
        "train":      _toy_ds([{"x": 1}, {"x": 2}, {"x": 3}]),
        "validation": _toy_ds([{"x": 4}]),
    }).save_to_disk(str(cfg))

    # Same swap as xsc: local validation -> public train, local train -> public eval
    remap = {"validation": "train", "train": "eval"}
    out = prepare_dataset(tmp_path, "en", split_remap=remap)

    assert set(out.keys()) == {"train", "eval"}
    assert len(out["train"]) == 1   # was local validation
    assert len(out["eval"]) == 3    # was local train


def test_prepare_dataset_remap_drops_unmapped_splits(tmp_path: Path):
    cfg = tmp_path / "en"
    DatasetDict({
        "train":      _toy_ds([{"x": 1}]),
        "validation": _toy_ds([{"x": 2}]),
        "extra":      _toy_ds([{"x": 3}]),
    }).save_to_disk(str(cfg))

    remap = {"validation": "train", "train": "eval"}
    out = prepare_dataset(tmp_path, "en", split_remap=remap)
    assert set(out.keys()) == {"train", "eval"}  # `extra` dropped


def test_prepare_dataset_remap_missing_key_raises(tmp_path: Path):
    cfg = tmp_path / "en"
    DatasetDict({"train": _toy_ds([{"x": 1}])}).save_to_disk(str(cfg))
    with pytest.raises(KeyError, match="not_a_split"):
        prepare_dataset(tmp_path, "en", split_remap={"not_a_split": "train"})


def test_render_yaml_frontmatter_belebele_shape():
    block = render_yaml_frontmatter(
        license_id="cc-by-sa-4.0",
        language_codes=["eng_Latn", "kat_Geor"],
        tags=["first-token-prediction", "mcqa", "belebele"],
        pretty_name="Belebele-FTP",
        configs=["eng_Latn", "kat_Geor"],
        split_paths={"eng_Latn": ["test"], "kat_Geor": ["test"]},
    )
    assert block.startswith("---\n") and block.endswith("\n---\n"), f"bad fence: {block!r}"
    payload = pyyaml.safe_load(block[len("---\n"):-len("\n---\n")])

    assert payload["license"] == "cc-by-sa-4.0"
    assert payload["language"] == ["eng_Latn", "kat_Geor"]
    assert payload["pretty_name"] == "Belebele-FTP"
    assert {"multiple-choice", "text-generation"}.issubset(payload["task_categories"])
    assert "first-token-prediction" in payload["tags"]

    cfgs = {c["config_name"]: c for c in payload["configs"]}
    assert cfgs["eng_Latn"]["data_files"] == [{"split": "test", "path": "eng_Latn/test-*"}]
    assert cfgs["kat_Geor"]["data_files"] == [{"split": "test", "path": "kat_Geor/test-*"}]


def test_render_yaml_frontmatter_xsc_two_splits():
    block = render_yaml_frontmatter(
        license_id="cc-by-4.0",
        language_codes=["en", "ar"],
        tags=["first-token-prediction", "mcqa", "xstory-cloze"],
        pretty_name="XStoryCloze-FTP",
        configs=["en", "ar"],
        split_paths={"en": ["train", "eval"], "ar": ["train", "eval"]},
    )
    assert block.startswith("---\n") and block.endswith("\n---\n"), f"bad fence: {block!r}"
    payload = pyyaml.safe_load(block[len("---\n"):-len("\n---\n")])
    en_cfg = next(c for c in payload["configs"] if c["config_name"] == "en")
    assert en_cfg["data_files"] == [
        {"split": "train", "path": "en/train-*"},
        {"split": "eval", "path": "en/eval-*"},
    ]


def test_render_yaml_frontmatter_empty_configs():
    block = render_yaml_frontmatter(
        license_id="cc-by-4.0",
        language_codes=[],
        tags=["mcqa"],
        pretty_name="Empty-FTP",
        configs=[],
        split_paths={},
    )
    assert block.startswith("---\n") and block.endswith("\n---\n"), f"bad fence: {block!r}"
    payload = pyyaml.safe_load(block[len("---\n"):-len("\n---\n")])
    assert payload["configs"] == []


def test_parse_args_minimal():
    ns = parse_args(["--repo-id", "mikaberidze/belebele-ftp"])
    assert ns.repo_id == "mikaberidze/belebele-ftp"
    assert ns.config is None
    assert ns.private is False


def test_parse_args_smoke_test_single_config():
    ns = parse_args([
        "--repo-id", "mikaberidze/belebele-ftp",
        "--config", "eng_Latn",
    ])
    assert ns.config == "eng_Latn"


def test_parse_args_private_flag():
    ns = parse_args([
        "--repo-id", "mikaberidze/x",
        "--private",
    ])
    assert ns.private is True


def test_parse_args_repo_id_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_render_card_belebele_renders_without_raising():
    from exps.xpe2.scripts.datasets.upload_bebe_ftp_to_hub import render_card
    card = render_card(["eng_Latn", "kat_Geor"])
    assert card.startswith("---\n")
    assert "cc-by-sa-4.0" in card
    assert "format_ftp_example" in card
    assert "Answer: \n```" in card  # trailing space inside fenced format example
    assert "<<REPLACE" not in card


def test_render_card_xsc_renders_without_raising():
    from exps.xpe2.scripts.datasets.upload_xsc_ftp_to_hub import render_card
    card = render_card(["en", "ar"])
    assert card.startswith("---\n")
    assert "cc-by-4.0" in card
    assert "random.sample(ALL_LABELS, 2)" in card
    assert "{A, B, C, D}" in card  # label-sampling callout, single braces
    assert "Answer: \n```" in card
    assert "<<REPLACE" not in card
