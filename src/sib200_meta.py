"""SIB-200 reference metadata.

Loads `data/sib200_meta.csv` (columns: code, name, class, family, region, xlmr).
`class` is the Joshi resource tier; `xlmr=1` flags languages XLM-R was pretrained on.
"""
from pathlib import Path

import pandas as pd

_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "sib200_meta.csv"

sib200_meta = pd.read_csv(_CSV_PATH)
xlmr_seen_sib200_ds_names = sib200_meta[sib200_meta["xlmr"] == 1]["code"].tolist()
