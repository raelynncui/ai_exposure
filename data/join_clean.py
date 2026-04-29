"""
Pipeline steps 1 and 2:
  1. Join data/ai.csv and data/acs.csv on FIPS code.
  2. Clean invalid values (Census sentinels, zero-employment tracts).

Output: data/joined_cleaned.csv with human-readable column names.
"""

import csv
import logging
import sys
from pathlib import Path

import polars as pl


AI_PATH  = Path("data/ai.csv")
ACS_PATH = Path("data/acs.csv")
OUT_PATH = Path("data/joined_cleaned.csv")

# Set to True  → drop rows where AI Exposure Score = 0 (unpopulated / special-use
#                tracts such as water areas and institutional group quarters).
#                Recommended for modelling.
# Set to False → keep those rows.
DROP_ZERO_AI_SCORE = False

FIPS_COL      = "Geo__geoid_"
AI_SCORE_COL  = "ORG_AIEI_004"   # AI Exposure Score (0-0.29) — the target
AI_DROP_COLS  = [                 # the four other AI cols we don't need
    "ORG_AIEI_001",   # AI Exposure Percentile
    "ORG_AIEI_007",   # AI Exposure Score Residence-based Normalized
    "ORG_AIEI_030",   # SE AI Exposure Index
    "ORG_AIEI_031",   # SE AI Exposure Index (Residence-based)
]
OCC_TOTAL_COL = "ORG_C24010001"  # C24010 total civilian employed — occ-share denominator

# Census uses these integers when an estimate is suppressed or not applicable.
CENSUS_SENTINELS = [-666666666, -888888888, -999999999, 666666666, 888888888]


def build_rename_map(ai_path: Path, acs_path: Path) -> dict[str, str]:
    """
    Read the two header rows from both files and return a
    {variable_code: human_label} mapping.

    Both files share the same two-row header convention:
      row 0  →  human-readable labels
      row 1  →  machine-readable variable codes

    Duplicate human labels (e.g. "Total" appears as the grand-total row in
    every ACS table) get ' ({code})' appended so every column name stays unique.
    """
    rename: dict[str, str] = {}
    seen_labels: set[str]  = set()

    for path in [ai_path, acs_path]:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            labels = next(reader)
            codes  = next(reader)

        for code, label in zip(codes, labels):
            if code in rename:          # shared geo cols already mapped from ai file
                continue
            label = label.strip()
            if label in seen_labels:    # duplicate label → append code to distinguish
                label = f"{label} ({code})"
            seen_labels.add(label)
            rename[code] = label

    return rename


def load_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        skip_rows=1,                  # skip row 0 (human labels); row 1 becomes header
        schema_overrides={FIPS_COL: pl.Utf8},
        infer_schema_length=2000,
        null_values=["", "(X)", "(B)", "(D)", "(N)", "-", "**", "N/A"],
        truncate_ragged_lines=True,
    )


def replace_sentinels(df: pl.DataFrame) -> pl.DataFrame:
    numeric_cols = [
        c for c, t in zip(df.columns, df.dtypes)
        if t in (pl.Int32, pl.Int64, pl.Float32, pl.Float64)
    ]
    return df.with_columns([
        pl.when(pl.col(c).is_in(CENSUS_SENTINELS))
          .then(pl.lit(None, dtype=df.schema[c]))
          .otherwise(pl.col(c))
          .alias(c)
        for c in numeric_cols
    ])



def main() -> None:

    # Step 1: Join 
    rename_map = build_rename_map(AI_PATH, ACS_PATH)

    ai  = load_csv(AI_PATH)
    acs = load_csv(ACS_PATH)

    # Inner join — only keep tracts present in both files.
    # Shared geo cols (GeoLevel, NAME, Qualified Area Name) get a '_right' suffix
    # on the ACS copy; drop those duplicates and keep the AI file's versions.
    df = ai.join(acs, on=FIPS_COL, how="inner", suffix="_right")
    df = df.drop([c for c in df.columns if c.endswith("_right")])

    # Drop the four AI target cols we don't need 
    df = df.drop([c for c in AI_DROP_COLS if c in df.columns])

    # Step 2: Clean 

    # Cast AI score to float
    df = df.with_columns(pl.col(AI_SCORE_COL).cast(pl.Float64, strict=False))

    # Optionally drop rows where AI Exposure Score = 0
    if DROP_ZERO_AI_SCORE:
        before = df.shape[0]
        df = df.filter(pl.col(AI_SCORE_COL).is_not_null() & (pl.col(AI_SCORE_COL) > 0))

    # Replace Census sentinel values (suppressed / N/A estimates) with null
    df = replace_sentinels(df)

    # Drop tracts where the total civilian employed population is zero or null
    df = df.with_columns(pl.col(OCC_TOTAL_COL).cast(pl.Float64, strict=False))
    before = df.shape[0]
    df = df.filter(pl.col(OCC_TOTAL_COL).is_not_null() & (pl.col(OCC_TOTAL_COL) > 0))

    df = df.rename({c: rename_map[c] for c in df.columns if c in rename_map})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT_PATH)

if __name__ == "__main__":
    main()