"""Unit tests for the fold ds.dirs rewriting in xlt_runner.

`apply_fold` swaps the `fold<N>` segment of a Belebele tokenized-dirs path so a
single tune/test config can be aimed at any fold at runtime, while leaving
fold-less paths (e.g. xstory_cloze) untouched when no fold is requested.
"""

import pytest

from scripts.xlt_runner import apply_fold

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
    # Passing a fold to an xsc (fold-less) config is a user error, surfaced loudly.
    with pytest.raises(ValueError, match="fold"):
        apply_fold(XSC, 1)

