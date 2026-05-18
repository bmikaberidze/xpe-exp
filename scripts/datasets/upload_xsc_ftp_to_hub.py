"""Upload the locally reframed XStoryCloze-FTP dataset to the HuggingFace Hub.

Smoke test (one config):
    python -m exps.xpe2.scripts.datasets.upload_xsc_ftp_to_hub \
        --repo-id mikaberidze/xstory-cloze-ftp --config en

Bulk push (all configs):
    python -m exps.xpe2.scripts.datasets.upload_xsc_ftp_to_hub \
        --repo-id mikaberidze/xstory-cloze-ftp
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
# Public-facing names mirror upstream (juletxara/xstory_cloze) verbatim.
# Used in the card YAML's `data_files`; on-disk splits are remapped via SPLIT_REMAP.
SPLITS = ["train", "eval"]
SPLIT_REMAP = {
    "validation": "train",  # local validation == upstream train (~360, tuning)
    "train":      "eval",   # local train      == upstream eval  (~1510, test set)
}
PRETTY_NAME = "XStoryCloze-FTP"
LICENSE = "cc-by-4.0"
TAGS = ["first-token-prediction", "mcqa", "multilingual", "reframed", "xstory-cloze"]

# Sourced verbatim from juletxara/xstory_cloze README on 2026-05-01.
UPSTREAM_BIBTEX_XGLM = r"""
@article{DBLP:journals/corr/abs-2112-10668,
  author    = {Xi Victoria Lin and
               Todor Mihaylov and
               Mikel Artetxe and
               Tianlu Wang and
               Shuohui Chen and
               Daniel Simig and
               Myle Ott and
               Naman Goyal and
               Shruti Bhosale and
               Jingfei Du and
               Ramakanth Pasunuru and
               Sam Shleifer and
               Punit Singh Koura and
               Vishrav Chaudhary and
               Brian O'Horo and
               Jeff Wang and
               Luke Zettlemoyer and
               Zornitsa Kozareva and
               Mona T. Diab and
               Veselin Stoyanov and
               Xian Li},
  title     = {Few-shot Learning with Multilingual Language Models},
  journal   = {CoRR},
  volume    = {abs/2112.10668},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.10668},
  eprinttype = {arXiv},
  eprint    = {2112.10668},
  timestamp = {Tue, 04 Jan 2022 15:59:27 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-10668.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
""".strip()

UPSTREAM_BIBTEX_STORY_CLOZE = r"""
@inproceedings{mostafazadeh-etal-2016-corpus,
    title = "A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories",
    author = "Mostafazadeh, Nasrin and Chambers, Nathanael and He, Xiaodong and Parikh, Devi and Batra, Dhruv and Vanderwende, Lucy and Kohli, Pushmeet and Allen, James",
    booktitle = "NAACL-HLT",
    year = "2016",
}
""".strip()


def render_card(configs: list[str]) -> str:
    """Build the full README.md (frontmatter + body) for the given list of pushed configs."""
    frontmatter = render_yaml_frontmatter(
        license_id=LICENSE,
        language_codes=configs,  # ar, en, es, ... verbatim
        tags=TAGS,
        pretty_name=PRETTY_NAME,
        configs=configs,
        split_paths={c: SPLITS for c in configs},
    )
    body = f"""# {PRETTY_NAME}

A first-token-prediction (FTP) reframing of [juletxara/xstory_cloze](https://huggingface.co/datasets/juletxara/xstory_cloze).
Each example is a single text sequence ending in `Answer：` (fullwidth U+FF1A, no trailing space) so a model can predict the answer as one token. Format matches lm-evaluation-harness's MCQA template byte-for-byte.

## Format

Example (`en`):

```
<sentence 1>
<sentence 2>
<sentence 3>
<sentence 4>

A: <ending 1>
B: <ending 2>
Answer：
```

**Schema:** `question_id: int`, `text: str`, `answer_label: str` (one of `A`/`B`/`C`/`D`).

> **Label sampling note.** Despite XStoryCloze having only **2 choices** per example, we randomly sample **2 letters from {{A, B, C, D}}** per example as the choice labels. This aligns the public label space with Belebele's 4-option format and mitigates letter-position bias.

## Reframing

```python
import random
ALL_LABELS = ["A", "B", "C", "D"]
RESPONSE_TEMPLATE = "Answer："  # fullwidth U+FF1A, no trailing space

def format_ftp_example(example, idx):
    context = "\\n".join([
        example["input_sentence_1"],
        example["input_sentence_2"],
        example["input_sentence_3"],
        example["input_sentence_4"],
    ])
    labels = random.sample(ALL_LABELS, 2)  # 2-of-4, see note above
    choices_str = "\\n".join(
        f"{{labels[i]}}: {{choice}}"
        for i, choice in enumerate([example["sentence_quiz1"], example["sentence_quiz2"]])
    )
    answer_label = labels[int(example["answer_right_ending"]) - 1]
    return {{
        "question_id": idx,
        "text": f"{{context}}\\n\\n{{choices_str}}\\n{{RESPONSE_TEMPLATE}}",
        "answer_label": answer_label,
    }}
```

## Splits

Public split names mirror upstream `juletxara/xstory_cloze` verbatim:

- `train` — ~360 examples per language (the upstream tuning split).
- `eval` — ~1510 examples per language (the upstream test set).

## Citation

```
{UPSTREAM_BIBTEX_XGLM}
```

```
{UPSTREAM_BIBTEX_STORY_CLOZE}
```

## License & credits

All credit to the original authors of XStoryCloze (Lin et al., Meta AI / XGLM) and the original English Story Cloze (Mostafazadeh et al., NAACL 2016).
This release inherits the upstream license (**CC-BY-4.0**, as declared by `juletxara/xstory_cloze`) and adds **no new annotations** — only a reformatting of the existing data.
"""
    return frontmatter + body


def main() -> None:
    micm_nlp_setup()
    from micm_nlp.path import datasets_dir
    local_dir = datasets_dir() / "benchmarks" / "mcqa" / "xstory_cloze_ftp"
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
            push_one_config(local_dir, cfg, args.repo_id, split_remap=SPLIT_REMAP, token=token)
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
