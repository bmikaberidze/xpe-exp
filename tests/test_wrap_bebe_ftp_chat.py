"""Unit tests for the chat-template layer over the Belebele FTP dataset."""
from datasets import Dataset, DatasetDict

from scripts.datasets.wrap_bebe_ftp_chat import wrap_text, wrap_tree, saved_datasets

FTP = "P: passage\nQ: question?\nA: a\nB: b\nC: c\nD: d\nAnswer："


def test_wrap_text_is_the_harness_layout():
    assert wrap_text(FTP, "gemma") == (
        "<start_of_turn>user\n" + FTP + "<end_of_turn>\n<start_of_turn>model\n"
    )


def test_wrap_tree_mirrors_root_and_fold_splits_and_skips_tokenized(tmp_path):
    src, dst = tmp_path / "belebele_ftp", tmp_path / "belebele_ftp_chat_gemma"
    ds = Dataset.from_list([{"question_id": 0, "text": FTP, "answer_label": "A"}])
    DatasetDict({"test": ds}).save_to_disk(str(src / "eng_Latn"))
    ds.save_to_disk(str(src / "eng_Latn" / "fold0" / "train"))
    ds.save_to_disk(str(src / "eng_Latn" / "tokenized--org--model" / "test"))

    assert wrap_tree(src, dst, "gemma") == 2
    assert [str(p) for p in saved_datasets(dst)] == ["eng_Latn/fold0/train", "eng_Latn/test"]

    root = DatasetDict.load_from_disk(str(dst / "eng_Latn"))      # dataset_dict.json copied
    assert root["test"][0]["text"] == wrap_text(FTP, "gemma")
    assert root["test"][0]["answer_label"] == "A"                  # other columns untouched
    assert Dataset.load_from_disk(str(src / "eng_Latn" / "test"))[0]["text"] == FTP  # input untouched
    assert wrap_tree(src, dst, "gemma") == 0                        # idempotent


def test_gemma4_layout_uses_its_own_markers_and_an_empty_thought_channel():
    out = wrap_text(FTP, "gemma4")
    assert out == "<|turn>user\n" + FTP + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    assert "<start_of_turn>" not in out   # Gemma 3 markers are not Gemma 4 tokens


def test_gemma4_prefill_moves_the_response_template_into_the_model_turn():
    out = wrap_text(FTP, "gemma4_prefill")
    body = FTP[: -len("Answer：")].rstrip("\n")
    assert out == "<|turn>user\n" + body + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>Answer："
    assert out.count("Answer：") == 1
    import pytest
    with pytest.raises(ValueError):
        wrap_text("no response template here", "gemma4_prefill")


def test_gemma_prefill_is_the_gemma3_layout_with_the_response_in_the_model_turn():
    out = wrap_text(FTP, "gemma_prefill")
    body = FTP[: -len("Answer：")].rstrip("\n")
    assert out == "<start_of_turn>user\n" + body + "<end_of_turn>\n<start_of_turn>model\nAnswer："


def test_aya_prefill_uses_the_cohere_turn_tokens_with_the_response_in_the_model_turn():
    out = wrap_text(FTP, "aya_prefill")
    body = FTP[: -len("Answer：")].rstrip("\n")
    assert out == ("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>" + body
                   + "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Answer：")
