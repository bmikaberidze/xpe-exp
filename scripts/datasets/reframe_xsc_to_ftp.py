"""
Reframe XStoryCloze dataset to FTP (First-Token Prediction) format.

Each example becomes a single text sequence:
    <4 story sentences>

    A: <ending 1>
    B: <ending 2>
    Answer：<A or B>

Format mirrors lm-evaluation-harness's MCQA template byte-for-byte
(colon-separated choices, fullwidth U+FF1A in `Answer：`, no trailing space),
so XPE/FTP scoring lands on the same letter-token the harness scores.

Note: we randomly choose 2 letters from A, B, C, and D per sample to represent <endings>,
to align with Belebele's 4-option format.

Usage:
    python -m scripts.datasets.reframe_xsc_to_ftp
"""

from datasets import load_dataset, get_dataset_config_names, Dataset, DatasetDict
from tqdm import tqdm
import logging
import random
from micm_nlp.enums import DsSplitSE
from micm_nlp.path import datasets_dir
from src.utils import micm_nlp_setup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALL_LABELS = ["A", "B", "C", "D"]
RESPONSE_TEMPLATE = "Answer："

# XStoryCloze split mapping: original 'train' -> our 'validation', original 'eval' -> our 'train'
SPLIT_MAPPING = {
    DsSplitSE.TRAIN: DsSplitSE.VALIDATION,
    "eval": DsSplitSE.TRAIN,
}


def format_ftp_example(example, idx):
    """Convert a single XStoryCloze example to FTP format."""
    context = "\n".join([
        example["input_sentence_1"],
        example["input_sentence_2"],
        example["input_sentence_3"],
        example["input_sentence_4"],
    ])

    # Randomly pick 2 labels from {A, B, C, D} to avoid bias toward specific letters
    labels = random.sample(ALL_LABELS, 2)

    choices_str = "\n".join(
        f"{labels[i]}: {choice}"
        for i, choice in enumerate([example["sentence_quiz1"], example["sentence_quiz2"]])
    )

    answer_label = labels[int(example["answer_right_ending"]) - 1]

    return {
        "question_id": idx,
        "text": f"{context}\n\n{choices_str}\n{RESPONSE_TEMPLATE}",
        "answer_label": answer_label,
    }


def transform_split(split_data):
    """Transform an entire split to FTP format."""
    return Dataset.from_list([
        format_ftp_example(ex, idx) for idx, ex in enumerate(tqdm(split_data, desc="Transforming"))
    ])


def main():
    micm_nlp_setup()

    dataset_name = "juletxara/xstory_cloze"
    output_base_path = datasets_dir() / "benchmarks" / "mcqa" / "xstory_cloze_ftp"
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    configs = get_dataset_config_names(dataset_name)
    logger.info(f"Found {len(configs)} configs: {configs}")
    
    for config_name in tqdm(configs, desc="Processing configs"):
        try:
            logger.info(f"\nProcessing: {config_name}")
            dataset = load_dataset(dataset_name, config_name)
            
            transformed_splits = {}
            for src_split, tgt_split in SPLIT_MAPPING.items():
                if src_split not in dataset:
                    continue
                transformed = transform_split(dataset[src_split])
                transformed_splits[tgt_split] = transformed
                logger.info(f"  {src_split} -> {tgt_split}: {len(transformed)} examples")
                logger.info(f"  Sample:\n{transformed[0]['text']}")
            
            output_path = output_base_path / config_name
            output_path.mkdir(parents=True, exist_ok=True)
            DatasetDict(transformed_splits).save_to_disk(str(output_path))
            
            # Verify
            loaded = DatasetDict.load_from_disk(str(output_path))
            for split in loaded:
                assert len(loaded[split]) == len(transformed_splits[split])
                logger.info(f"  ✓ {split}: {len(loaded[split])}")
            
        except Exception as e:
            logger.error(f"Error processing {config_name}: {e}")
            continue
    
    logger.info(f"\nDone! Saved to: {output_base_path}")


if __name__ == "__main__":
    main()