"""Unit tests for the --skip-test guard in run_xlt."""
import pytest

from scripts.run_xlt import check_skip_test


def test_skip_test_requires_tune():
    with pytest.raises(ValueError, match='skip-test'):
        check_skip_test(skip_test=True, has_tune=False)


def test_skip_test_with_tune_ok():
    check_skip_test(skip_test=True, has_tune=True)  # must not raise


def test_no_skip_is_noop_without_tune():
    check_skip_test(skip_test=False, has_tune=False)


def test_no_skip_is_noop_with_tune():
    check_skip_test(skip_test=False, has_tune=True)
