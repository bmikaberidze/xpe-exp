"""Unit tests for the --fold ds.dirs rewriting in run_xlt.

`apply_fold` swaps the `fold<N>` segment of a Belebele tokenized-dirs path so a
single tune/test config can be aimed at any fold at runtime, while leaving
fold-less paths (e.g. xstory_cloze) untouched when no fold is requested.
"""
from types import SimpleNamespace

import pytest

from scripts.run_xlt import apply_fold, apply_seed

BEBE = "mcqa/belebele_ftp/eng_Latn/fold0/tokenized|bigscience|bloom-7b1"
XSC = "mcqa/xstory_cloze_ftp/en/tokenized|bigscience|bloom-7b1"


def test_fold_none_is_noop_on_bebe():
    assert apply_fold(BEBE, None) == BEBE


def test_fold_none_is_noop_on_xsc():
    assert apply_fold(XSC, None) == XSC


def test_swaps_fold_segment():
    assert apply_fold(BEBE, 2) == (
        "mcqa/belebele_ftp/eng_Latn/fold2/tokenized|bigscience|bloom-7b1"
    )


def test_swap_to_same_fold_is_identity():
    assert apply_fold(BEBE, 0) == BEBE


def test_swap_preserves_lang_segment():
    src = "mcqa/belebele_ftp/zho_Hans/fold1/tokenized|bigscience|bloom-7b1"
    assert apply_fold(src, 2) == (
        "mcqa/belebele_ftp/zho_Hans/fold2/tokenized|bigscience|bloom-7b1"
    )


def test_fold_on_foldless_path_raises():
    # Passing --fold to an xsc (fold-less) config is a user error, surfaced loudly.
    with pytest.raises(ValueError, match="fold"):
        apply_fold(XSC, 1)


# --- apply_seed -------------------------------------------------------------


def _cfg(args):
    return SimpleNamespace(training_args=SimpleNamespace(args=args))


def test_seed_none_is_noop():
    cfg = _cfg(SimpleNamespace(seed=999))
    apply_seed(cfg, None)
    assert cfg.training_args.args.seed == 999  # untouched → toolkit randomizes


def test_seed_none_tolerates_none_config():
    apply_seed(None, 7)  # zero-shot / replay: no tune config, must not raise


def test_seed_pins_value():
    cfg = _cfg(SimpleNamespace())
    apply_seed(cfg, 42)
    assert cfg.training_args.args.seed == 42


def test_seed_creates_args_block_if_missing():
    cfg = _cfg(None)
    apply_seed(cfg, 13)
    assert cfg.training_args.args.seed == 13
