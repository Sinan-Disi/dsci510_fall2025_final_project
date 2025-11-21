"""Preprocess RAWG, Steam, and VGSales data for the
Video-Game Analytics & Prediction project."""

import re
import numpy as np
import pandas as pd

from config import (
    RAWG_RAW,
    STEAM_RAW,
    VGSALES_RAW,
    RAWG_CLEAN,
    STEAM_CLEAN,
    VGSALES_CLEAN,
    YEAR_MIN,
    YEAR_MAX,
    CANON_GENRES,
    GENRE_MAP,
)

_splitter = re.compile(r"[;,/|]")

def std_genre_cell(s: str) -> str | float:
    """Map a raw genre string to a semicolon-separated list of canonical genres.

    Uses GENRE_MAP and CANON_GENRES from config to standardize labels.
    Returns NaN if no canonical genre can be mapped.
    """
    if pd.isna(s):
        return np.nan
    parts = [p.strip().lower() for p in _splitter.split(str(s)) if p.strip()]
    mapped: set[str] = set()
    for p in parts:
        norm = p.replace("-", " ")
        # First try mapping table
        if norm in GENRE_MAP:
            mapped.add(GENRE_MAP[norm])
            continue
        # Then allow direct canonical genres
        if p in CANON_GENRES:
            mapped.add(p)
            continue
    return "; ".join(sorted(mapped)) if mapped else np.nan


def add_year(df: pd.DataFrame, released_col: str | None) -> pd.DataFrame:
    """Ensure a numeric 'year' column exists using either a date column or Year."""
    if released_col and released_col in df.columns:
        df["year"] = pd.to_datetime(df[released_col], errors="coerce").dt.year
    elif "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        df["year"] = pd.NA
    return df


def in_window(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows to the configured YEAR_MIN..YEAR_MAX window."""
    return df[df["year"].between(YEAR_MIN, YEAR_MAX, inclusive="both")]


def main() -> None:
    """Load raw CSVs, clean genres and years, filter, and save cleaned datasets."""

    # I/O: load raw data using config paths
    rawg = pd.read_csv(RAWG_RAW)
    steam = pd.read_csv(STEAM_RAW)
    sales = pd.read_csv(VGSALES_RAW)

    print("Loaded:", {"rawg": rawg.shape, "steam": steam.shape, "sales": sales.shape})

    # Rename columns to align schemas across the three sources
    rawg.rename(columns={"genres": "genre"}, inplace=True)

    steam.rename(columns={"genres": "genre", "release_date": "released"}, inplace=True)

    sales.rename(
        columns={
            "Genre": "genre",
            "Name": "name",
            "Platform": "platform",
            "Year": "year",
            "Global_Sales": "global_sales",
        },
        inplace=True,
    )

    # Basic genre string cleanup before canonical mapping
    for df in (rawg, steam, sales):
        if "genre" in df.columns:
            df["genre"] = df["genre"].astype(str).str.lower().str.strip()

    # Canonical genre mapping using CANON_GENRES / GENRE_MAP
    for df in (rawg, steam, sales):
        if "genre" in df.columns:
            df["genre_std"] = df["genre"].apply(std_genre_cell)

    # VGSales numeric cleaning
    sales_num_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "global_sales"]
    for c in sales_num_cols:
        if c in sales.columns:
            sales[c] = pd.to_numeric(sales[c], errors="coerce")

    # Recompute global_sales from regions
    sales["global_sales"] = sales[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum(
        axis=1,
        min_count=1,
    )

    # Drop rows where all regional sales are missing
    sales = sales.dropna(
        subset=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"],
        how="all",
    )

    # Keep only games with at least 0.1M units sold
    sales = sales[sales["global_sales"] >= 0.1]

    # RAWG quality filter
    if "rating" in rawg.columns:
        rawg = rawg[rawg["rating"].notna() & (rawg["rating"] > 0)]

    # Keep only rows with a mapped canonical genre
    rawg = rawg[rawg["genre_std"].notna()]
    steam = steam[steam["genre_std"].notna()]
    sales = sales[sales["genre_std"].notna()]

    print(
        "Post-canonicalization:",
        {"rawg": rawg.shape, "steam": steam.shape, "sales": sales.shape},
    )

    # Year handling and filtering by YEAR_MIN..YEAR_MAX
    rawg = add_year(rawg, "released")   
    steam = add_year(steam, "released")  
    sales = add_year(sales, None)        

    rawg = in_window(rawg)
    steam = in_window(steam)
    sales = in_window(sales)

    print(
        "Post-year-filter:",
        {"rawg": rawg.shape, "steam": steam.shape, "sales": sales.shape},
    )

    # Steam specific filters to keep only active games
    steam = steam[
        (steam["positive_ratings"] > 0)
        & (steam["negative_ratings"] > 0)
        & (steam["average_playtime"] > 0)
        & (steam["median_playtime"] > 0)
    ]

    print("After Steam filters:", steam.shape)

    # Save cleaned data using config paths
    rawg.to_csv(RAWG_CLEAN, index=False)
    steam.to_csv(STEAM_CLEAN, index=False)
    sales.to_csv(VGSALES_CLEAN, index=False)

    print("Saved:", RAWG_CLEAN.name, STEAM_CLEAN.name, VGSALES_CLEAN.name)

if __name__ == "__main__":
    main()




