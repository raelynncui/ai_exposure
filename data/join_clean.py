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
import duckdb


AI_PATH  = Path("data/ai.csv")
ACS_PATH = Path("data/acs.csv")
OUT_PATH = Path("data/joined_cleaned.csv")

DROP_ZERO_AI_SCORE = True

FIPS_COL      = "Geo__geoid_"
AI_SCORE_COL  = "ORG_AIEI_004"   # AI Exposure Score (0-0.29) — the target
AI_DROP_COLS  = [                 # the four other AI cols we don't need
    "ORG_AIEI_001",   # AI Exposure Percentile
    "ORG_AIEI_007",   # AI Exposure Score Residence-based Normalized
    "ORG_AIEI_030",   # SE AI Exposure Index
    "ORG_AIEI_031",   # SE AI Exposure Index (Residence-based)
]
OCC_TOTAL_COL = "ORG_C24010001"  # C24010 total civilian employed — occ-share denominator

# Census uses these integers when an estimate is suppressed or not applicable
CENSUS_SENTINELS = [-666666666, -888888888, -999999999, 666666666, 888888888]


def build_rename_map(ai_path: Path, acs_path: Path) -> dict[str, str]:
    # Read the two header rows from both files and return a {variable_code: human_label} mapping.

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
            if label in seen_labels:    # duplicate label, append code to distinguish
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

    # Inner join
    con = duckdb.connect()
    con.register("ai", ai)
    con.register("acs", acs)

    ai_keep = [c for c in ai.columns if c not in AI_DROP_COLS]
    AI_GEO_COLS = {"Geo__geo_level_", "Geo_NAME", "Geo_qname"}  # adjust to actual ACS col names
    acs_exclude = {FIPS_COL} | AI_GEO_COLS
    select_cols = ", ".join(f'ai."{c}"' for c in ai_keep) + \
            ", " + ", ".join(f'acs."{c}"' for c in acs.columns if c not in acs_exclude)
    df = con.execute(f"""
        SELECT {select_cols}
        FROM ai
        INNER JOIN acs USING ("{FIPS_COL}")
    """).pl()
    

    # Step 2: Clean 
    df = replace_sentinels(df)

    con.register("df", df)
    zero_ai_filter = f'AND CAST("{AI_SCORE_COL}" AS DOUBLE) > 0' if DROP_ZERO_AI_SCORE else ""
    df = con.execute(f"""
        SELECT *
        FROM df
        WHERE CAST("{AI_SCORE_COL}" AS DOUBLE) IS NOT NULL
        {zero_ai_filter}
        AND CAST("{OCC_TOTAL_COL}" AS DOUBLE) IS NOT NULL
        AND CAST("{OCC_TOTAL_COL}" AS DOUBLE) > 0
    """).pl()

    df = df.rename({c: rename_map[c] for c in df.columns if c in rename_map})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT_PATH)

if __name__ == "__main__":
    main()