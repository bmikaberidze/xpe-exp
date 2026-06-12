"""Unit tests for per-entry fold/seed resolution in the meta runner."""
import pytest

from scripts.run_xlt_meta import resolve_fold_seed


def test_entry_overrides_cli():
    assert resolve_fold_seed({'fold': 2, 'seed': 11}, cli_fold=0, cli_seed=10) == (2, 11)


def test_cli_used_when_entry_missing():
    assert resolve_fold_seed({}, cli_fold=0, cli_seed=10) == (0, 10)


def test_partial_entry_mixes_with_cli():
    assert resolve_fold_seed({'seed': 12}, cli_fold=1, cli_seed=None) == (1, 12)


def test_none_when_neither_set():
    assert resolve_fold_seed({}, cli_fold=None, cli_seed=None) == (None, None)
