"""Upload the locally reframed Belebele-FTP dataset to the HuggingFace Hub.

Smoke test (one config):
    python -m scripts.datasets.upload_bebe_ftp_to_hub \
        --repo-id mikaberidze/belebele-ftp --config eng_Latn

Bulk push (all configs):
    python -m scripts.datasets.upload_bebe_ftp_to_hub \
        --repo-id mikaberidze/belebele-ftp
"""

import logging
import os

from src.hub_upload import (
    discover_local_configs,
    ensure_repo,
    parse_args,
    push_one_config,
    render_yaml_frontmatter,
    upload_card,
)
from src.utils import micm_nlp_setup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
# Belebele has only `test`; used to declare `data_files` in the card YAML.
# On-disk splits load verbatim — `prepare_dataset` is called with split_remap=None.
SPLITS = ["test"]
PRETTY_NAME = "Belebele-FTP"
LICENSE = "cc-by-sa-4.0"
TAGS = ["first-token-prediction", "mcqa", "multilingual", "reframed", "belebele"]

# YAML `language:` mirrors upstream facebook/belebele's README verbatim — bare ISO 639-1
# codes (HF's metadata validator rejects FLORES `lang_Script` here). Config names remain
# FLORES verbatim. List sourced from facebook/belebele/raw/main/README.md on 2026-05-01.
BELEBELE_LANGUAGES = [
    "af", "am", "ar", "az", "as", "bm", "bn", "bo", "bg", "ca", "cs", "ku", "da", "de",
    "el", "en", "es", "et", "eu", "fi", "fr", "ff", "om", "gu", "gn", "ht", "ha", "he",
    "hi", "hr", "hu", "hy", "ig", "id", "it", "is", "jv", "ja", "ka", "kn", "kk", "mn",
    "km", "rw", "ky", "ko", "lo", "ln", "lt", "lg", "lv", "ml", "mr", "mk", "mt", "mi",
    "my", "nl", "no", "ne", "ny", "or", "pa", "ps", "fa", "mg", "pl", "pt", "ro", "ru",
    "sn", "si", "sl", "sv", "sk", "sd", "sw", "ta", "te", "tg", "tl", "th", "ti", "tn",
    "ts", "tr", "uk", "ur", "uz", "vi", "wo", "xh", "yo", "zh", "ms", "zu",
]

# Sourced from facebook/belebele's README on 2026-05-01.
UPSTREAM_BIBTEX = r"""
@inproceedings{bandarkar-etal-2024-belebele,
    title = "The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants",
    author = "Bandarkar, Lucas  and
      Liang, Davis  and
      Muller, Benjamin  and
      Artetxe, Mikel  and
      Shukla, Satya Narayan  and
      Husa, Donald  and
      Goyal, Naman  and
      Krishnan, Abhinandan  and
      Zettlemoyer, Luke  and
      Khabsa, Madian",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.44",
    pages = "749--775",
}
""".strip()


def render_card(configs: list[str]) -> str:
    """Build the full README.md (frontmatter + body) for the given list of pushed configs."""
    frontmatter = render_yaml_frontmatter(
        license_id=LICENSE,
        language_codes=BELEBELE_LANGUAGES,  # ISO 639-1 (HF requires); see BELEBELE_LANGUAGES above
        tags=TAGS,
        pretty_name=PRETTY_NAME,
        configs=configs,  # config_name uses FLORES lang_Script verbatim
        split_paths={c: SPLITS for c in configs},
    )
    body = f"""# {PRETTY_NAME}

A first-token-prediction (FTP) reframing of [facebook/belebele](https://huggingface.co/datasets/facebook/belebele).
Each example is a single text sequence ending in `Answer：` (fullwidth U+FF1A, no trailing space) so a model can predict the answer as one token (A/B/C/D). Format matches lm-evaluation-harness's belebele template byte-for-byte.

## Format

Example (`eng_Latn`):

```
P: <passage>
Q: <question>
A: <choice 1>
B: <choice 2>
C: <choice 3>
D: <choice 4>
Answer：
```

**Schema:** `question_id: int`, `text: str`, `answer_label: str` (one of `A`/`B`/`C`/`D`).

## Reframing

```python
ANSWER_LABELS = ["A", "B", "C", "D"]
RESPONSE_TEMPLATE = "Answer："  # fullwidth U+FF1A, no trailing space

def format_ftp_example(example, idx):
    passage = f"P: {{example['flores_passage']}}"
    question = f"Q: {{example['question'].strip()}}"
    choices_str = "\\n".join(
        f"{{ANSWER_LABELS[i]}}: {{example[f'mc_answer{{i + 1}}']}}"
        for i in range(4)
    )
    answer_label = ANSWER_LABELS[int(example["correct_answer_num"]) - 1]
    return {{
        "question_id": idx,
        "text": f"{{passage}}\\n{{question}}\\n{{choices_str}}\\n{{RESPONSE_TEMPLATE}}",
        "answer_label": answer_label,
    }}
```

## Splits

`test` only (matches upstream — Belebele has no train/validation).

## Citation

```
{UPSTREAM_BIBTEX}
```

## License & credits

All credit to the original authors of Belebele (Bandarkar et al., Meta AI).
This release inherits the upstream license (**CC-BY-SA-4.0**) and adds **no new annotations** — only a reformatting of the existing data.
"""
    return frontmatter + body


def main() -> None:
    micm_nlp_setup()
    from micm_nlp.path import datasets_dir
    local_dir = datasets_dir() / "benchmarks" / "mcqa" / "belebele_ftp"
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set; export it (with `write` scope) and retry.")

    available = discover_local_configs(local_dir)
    if args.config:
        if args.config not in available:
            raise SystemExit(f"--config {args.config!r} not found under {local_dir}")
        configs_to_push = [args.config]
    else:
        configs_to_push = available

    logger.info("Will push %d config(s) to %s", len(configs_to_push), args.repo_id)
    ensure_repo(args.repo_id, private=args.private, token=token)

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    for cfg in configs_to_push:
        try:
            logger.info("[%s] pushing...", cfg)
            push_one_config(local_dir, cfg, args.repo_id, split_remap=None, token=token)
            succeeded.append(cfg)
            logger.info("[%s] ok", cfg)
        except Exception as e:
            failed.append((cfg, str(e)))
            logger.exception("[%s] FAILED", cfg)

    if succeeded:
        logger.info("Rendering and uploading card for %d configs", len(succeeded))
        upload_card(args.repo_id, render_card(succeeded), token=token)

    logger.info("Summary: %d succeeded, %d failed", len(succeeded), len(failed))
    for cfg, err in failed:
        logger.info("  FAILED %s: %s", cfg, err)
    logger.info("Repo: https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
