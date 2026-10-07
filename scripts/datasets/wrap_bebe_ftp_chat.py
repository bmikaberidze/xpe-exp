"""
Wrap the reframed Belebele FTP dataset in a chat template -- a layer on top of
`reframe_bebe_to_ftp`, the way `lm-eval --apply_chat_template` wraps a task's
`doc_to_text`: the whole FTP text (which already ends in `Answer：`) becomes the
user turn, and the model turn is opened and left empty:

    <start_of_turn>user
    P: <passage>
    Q: <question>
    A: ... D: <answer 4>
    Answer：<end_of_turn>
    <start_of_turn>model
    <A, B, C, or D>

Scoring is unchanged -- first token after the generation prompt, restricted to
the four letters. `<bos>` is NOT part of the text: the tokenizer prepends it
(`add_special_tokens: true`), exactly as for the plain dataset.

Input  = artefacts/datasets/benchmarks/mcqa/belebele_ftp/, every saved split in
         it (the root `{lang}/{split}` AND the `{lang}/fold{N}/{split}` dirs),
         so item ids and folds carry over unchanged. Tokenized dirs are skipped.
Output = the same tree under belebele_ftp_chat_{template}/; the input is untouched.

Usage:
    python -m scripts.datasets.wrap_bebe_ftp_chat --template gemma
"""

import argparse
import logging
import shutil
from pathlib import Path

from datasets import Dataset

from micm_nlp.path import datasets_dir
from src.utils import micm_nlp_setup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# {name: (opens the user turn, closes it and opens the model turn)}, copied from
# the family's tokenizer_config.json `chat_template` with the generation prompt
# included. `<end_of_turn>` follows the user text directly (no newline), as the
# template renders it.
CHAT_TEMPLATES = {
    "gemma": ("<start_of_turn>user\n", "<end_of_turn>\n<start_of_turn>model\n"),
}

TOKENIZED_MARK = "tokenized--"


def wrap_text(text: str, template: str) -> str:
    user_open, model_open = CHAT_TEMPLATES[template]
    return f"{user_open}{text}{model_open}"


def wrap_dataset(ds: Dataset, template: str) -> Dataset:
    return ds.map(lambda ex: {"text": wrap_text(ex["text"], template)})


def saved_datasets(root: Path):
    """Every `Dataset.save_to_disk` dir under root (has state.json), skipping
    tokenized ones, as paths relative to root."""
    return sorted(
        p.parent.relative_to(root)
        for p in root.rglob("state.json")
        if TOKENIZED_MARK not in str(p)
    )


def wrap_tree(src: Path, dst: Path, template: str) -> int:
    """Mirror every saved split of `src` into `dst`, wrapped. A `dataset_dict.json`
    next to the splits is copied so a root `{lang}/` still loads as a DatasetDict.
    Returns the number of splits written."""
    n = 0
    for rel in saved_datasets(src):
        out = dst / rel
        if out.exists():
            logger.info(f"  exists, skipping: {rel}")
            continue
        wrap_dataset(Dataset.load_from_disk(str(src / rel)), template).save_to_disk(str(out))
        n += 1
        dict_json = src / rel.parent / "dataset_dict.json"
        if dict_json.exists():
            shutil.copy(dict_json, out.parent / "dataset_dict.json")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", choices=sorted(CHAT_TEMPLATES), required=True)
    args = ap.parse_args()

    micm_nlp_setup()
    base = datasets_dir() / "benchmarks" / "mcqa"
    src, dst = base / "belebele_ftp", base / f"belebele_ftp_chat_{args.template}"
    n = wrap_tree(src, dst, args.template)
    sample = Dataset.load_from_disk(str(dst / saved_datasets(dst)[0]))[0]["text"]
    logger.info(f"wrote {n} splits to {dst}\nsample:\n{sample}")


if __name__ == "__main__":
    main()
