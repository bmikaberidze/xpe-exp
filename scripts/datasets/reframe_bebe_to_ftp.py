"""
Transform Belebele dataset to FTP (First-Token Prediction) format.

Each example becomes a single text sequence:
    P: <passage>
    Q: <question>
    A: <answer 1>
    B: <answer 2>
    C: <answer 3>
    D: <answer 4>
    Answer：<A, B, C, or D>

Format mirrors lm-evaluation-harness's belebele template byte-for-byte
(colon-separated choices, fullwidth U+FF1A in `Answer：`, no trailing space),
so XPE/FTP scoring lands on the same letter-token the harness scores.

Usage:
    python -m scripts.datasets.reframe_bebe_to_ftp
"""

from datasets import load_dataset, get_dataset_config_names, Dataset, DatasetDict
from tqdm import tqdm
import logging
from micm_nlp.enums import DsSplitSE
from micm_nlp.path import datasets_dir
from src.utils import micm_nlp_setup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANSWER_LABELS = ["A", "B", "C", "D"]
RESPONSE_TEMPLATE = "Answer："

# Belebele only has 'test' split — we keep it as test
SPLIT_MAPPING = {
    DsSplitSE.TEST: DsSplitSE.TEST,
}


def format_ftp_example(example, idx):
    """Convert a single Belebele example to FTP format."""
    passage = f"P: {example['flores_passage']}"
    question = f"Q: {example['question'].strip()}"

    choices_str = "\n".join(
        f"{ANSWER_LABELS[i]}: {example[f'mc_answer{i + 1}']}"
        for i in range(4)
    )

    answer_label = ANSWER_LABELS[int(example["correct_answer_num"]) - 1]

    return {
        "question_id": idx,
        "text": f"{passage}\n{question}\n{choices_str}\n{RESPONSE_TEMPLATE}",
        "answer_label": answer_label,
    }


def transform_split(split_data):
    """Transform an entire split to FTP format."""
    return Dataset.from_list([
        format_ftp_example(ex, idx) for idx, ex in enumerate(tqdm(split_data, desc="Transforming"))
    ])


def main():
    micm_nlp_setup()

    dataset_name = "facebook/belebele"
    output_base_path = datasets_dir() / "benchmarks" / "mcqa" / "belebele_ftp"
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
                logger.info(f"  Verified {split}: {len(loaded[split])}")

        except Exception as e:
            logger.error(f"Error processing {config_name}: {e}")
            continue

    logger.info(f"\nDone! Saved to: {output_base_path}")


if __name__ == "__main__":
    main()
