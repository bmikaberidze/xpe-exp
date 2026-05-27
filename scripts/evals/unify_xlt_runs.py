"""
Unify per-run XLT eval results into one wide accuracy table.

Walks PATH, collects every `raw.csv` found in sub-folders, and concatenates the
`accuracy` column of each into a single CSV. Each column is named after the
run folder's last part with the leading timestamp stripped, e.g.
`20260522_170730_xpe_lr5e5` -> `xpe_lr5e5`. Rows are aligned on `target_lang`.

Usage:
    python -m scripts.evals.unify_xlt_runs artefacts/evals/xlt_runs/aya/ar-en-es-hi-id-ru-zh/5_lr_search_aya
    python -m scripts.evals.unify_xlt_runs <path> --out <path>/unified.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

# Leading run-folder timestamp, e.g. "20260522_170730_".
TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}_")


def column_name(run_dir: Path) -> str:
    """Folder's last part with the leading timestamp stripped."""
    return TIMESTAMP_RE.sub("", run_dir.name)


def unify(path: Path, key: str = "target_lang", value: str = "accuracy") -> pd.DataFrame:
    series = []
    for raw in sorted(path.rglob("raw.csv")):
        name = column_name(raw.parent)
        df = pd.read_csv(raw)
        for col in (key, value):
            if col not in df.columns:
                raise ValueError(f"{raw}: missing '{col}' column (has {list(df.columns)})")
        col = df.set_index(key)[value].rename(name)
        if col.index.has_duplicates:
            raise ValueError(f"{raw}: duplicate '{key}' values, cannot align")
        series.append(col)

    if not series:
        raise SystemExit(f"No raw.csv found under {path}")

    # Disambiguate any colliding column names by appending #2, #3, ...
    seen: dict[str, int] = {}
    for s in series:
        seen[s.name] = seen.get(s.name, 0) + 1
        if seen[s.name] > 1:
            new = f"{s.name}#{seen[s.name]}"
            print(f"warning: duplicate column '{s.name}' -> '{new}'")
            s.rename(new, inplace=True)

    return pd.concat(series, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="root path to walk for raw.csv files")
    ap.add_argument("--out", type=Path, default=None, help="output CSV (default: PATH/unified.csv)")
    ap.add_argument("--key", default="target_lang", help="row-alignment column (default: target_lang)")
    ap.add_argument("--value", default="accuracy", help="column to collect (default: accuracy)")
    args = ap.parse_args()

    out = args.out or (args.path / "unified.csv")
    table = unify(args.path, key=args.key, value=args.value)
    table.to_csv(out)
    print(f"wrote {out}  ({table.shape[0]} rows x {table.shape[1]} cols)")


if __name__ == "__main__":
    main()
