"""Unit tests for dotted-path overrides in run_xlt_meta."""
from types import SimpleNamespace

import pytest

from scripts.run_xlt_meta import _apply_override


def _cfg():
    return SimpleNamespace(
        peft=SimpleNamespace(encoder_ratio=1),
        training_args=SimpleNamespace(args=SimpleNamespace(learning_rate=5e-5, max_steps=24000)),
        custom_training_args=SimpleNamespace(
            optimizer_grouped_parameters=[
                SimpleNamespace(param_name_parts=['xpe_embedding'], lr=5e-3, weight_decay=0.0),
                SimpleNamespace(param_name_parts=['other'], lr=1e-3, weight_decay=0.1),
            ]
        ),
    )


def test_override_plain_attribute_path():
    c = _cfg()
    _apply_override(c, 'peft.encoder_ratio', 0.7)
    assert c.peft.encoder_ratio == 0.7


def test_override_nested_attribute_path():
    c = _cfg()
    _apply_override(c, 'training_args.args.max_steps', 300)
    assert c.training_args.args.max_steps == 300


def test_override_through_a_list_index():
    """The LR searches sweep custom_training_args.optimizer_grouped_parameters.0.lr."""
    c = _cfg()
    _apply_override(c, 'custom_training_args.optimizer_grouped_parameters.0.lr', 1e-2)
    assert c.custom_training_args.optimizer_grouped_parameters[0].lr == 1e-2
    # the sibling group must be untouched
    assert c.custom_training_args.optimizer_grouped_parameters[1].lr == 1e-3


def test_override_a_list_element_itself():
    c = _cfg()
    _apply_override(c, 'custom_training_args.optimizer_grouped_parameters.1.weight_decay', 0.0)
    assert c.custom_training_args.optimizer_grouped_parameters[1].weight_decay == 0.0


def test_override_rejects_out_of_range_index():
    c = _cfg()
    with pytest.raises(IndexError):
        _apply_override(c, 'custom_training_args.optimizer_grouped_parameters.9.lr', 1.0)


def test_override_rejects_unknown_attribute():
    c = _cfg()
    with pytest.raises(AttributeError):
        _apply_override(c, 'peft.no_such_key', 1)
