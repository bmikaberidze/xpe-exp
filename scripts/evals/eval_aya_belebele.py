#!/usr/bin/env python3
"""
Evaluate CohereForAI/aya-expanse-8b on BeleBele (all language splits)
using lm-evaluation-harness, saving per-language results to CSV.

Usage:
    pip install lm-eval[vllm] "numpy<2" "pyarrow>=14.0" --break-system-packages & \
    python -m scripts.evals.eval_aya_belebele --langs kat_Geor,eng_Latn,deu_Latn

"""
# Evaluating 284 BeleBele splits...

import argparse
import json
import pandas as pd
from pathlib import Path
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM


def get_belebele_tasks(langs: list[str] | None = None) -> list[str]:
    """Get BeleBele task names. If langs is None, use all available."""
    task_manager = tasks.TaskManager()
    all_tasks = task_manager.all_tasks
    belebele_tasks = sorted([t for t in all_tasks if t.startswith("belebele_")])

    if langs:
        selected = [f"belebele_{lang}" for lang in langs]
        missing = [t for t in selected if t not in belebele_tasks]
        if missing:
            raise ValueError(f"Tasks not found: {missing}")
        return selected

    return belebele_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CohereForAI/aya-expanse-8b")
    parser.add_argument("--langs", type=str, default=None, help="Comma-separated language codes, e.g. kat_Geor,eng_Latn")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default="artefacts/evals/aya-8b/belebele_zero_shot_eval_res.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    langs = args.langs.split(",") if args.langs else None
    task_list = get_belebele_tasks(langs)
    print(f"Evaluating {len(task_list)} BeleBele splits...")

    # Initialize model
    lm = HFLM(
        pretrained=args.model,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
    )

    # Run evaluation
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=args.batch_size,
        log_samples=False,
    )

    # Parse results into rows
    rows = []
    for task_name, metrics in results["results"].items():
        lang = task_name.replace("belebele_", "")
        row = {"language": lang, "task": task_name}
        for metric_key, value in metrics.items():
            # Strip ',none' suffix lm-eval appends
            clean_key = metric_key.replace(",none", "")
            row[clean_key] = value
        rows.append(row)

    # Save to CSV
    df = pd.DataFrame(rows).sort_values("language").reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print(df.to_string(index=False))

    # Also save raw JSON for reference
    json_path = Path(args.output).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results["results"], f, indent=2, default=str)
    print(f"Raw JSON saved to {json_path}")


if __name__ == "__main__":
    main()