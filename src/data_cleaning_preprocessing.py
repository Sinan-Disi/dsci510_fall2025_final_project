#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import re

from config import (
    RAWG_RAW, STEAM_RAW, VGSALES_RAW, RAWG_CLEAN, STEAM_CLEAN, VGSALES_CLEAN, YEAR_MIN, YEAR_MAX, CANON_GENRES, GENRE_MAP,
)

# I/O: load raw data using config paths
rawg  = pd.read_csv(RAWG_RAW)
steam = pd.read_csv(STEAM_RAW)
sales = pd.read_csv(VGSALES_RAW)

print("Loaded:", {"rawg": rawg.shape, "steam": steam.shape, "sales": sales.shape})

# Renames / normalization to align schemas
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

# basic genre string cleanup
for df in (rawg, steam, sales):
    if "genre" in df.columns:
        df["genre"] = df["genre"].astype(str).str.lower().str.strip()

# Canonical genre mapping using config CANON_GENRES / GENRE_MAP
_splitter = re.compile(r"[;,/|]")

def std_genre_cell(s: str) -> str | float:
    if pd.isna(s):
        return np.nan
    parts = [p.strip().lower() for p in _splitter.split(str(s)) if p.strip()]
    mapped = set()
    for p in parts:
        norm = p.replace("-", " ")
        # first try mapping table
        if norm in GENRE_MAP:
            mapped.add(GENRE_MAP[norm])
            continue
        # then allow direct canonical genres
        if p in CANON_GENRES:
            mapped.add(p)
            continue
    return "; ".join(sorted(mapped)) if mapped else np.nan

for df in (rawg, steam, sales):
    if "genre" in df.columns:
        df["genre_std"] = df["genre"].apply(std_genre_cell)

# VGSales numeric cleaning
sales_num_cols = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "global_sales"]
for c in sales_num_cols:
    if c in sales.columns:
        sales[c] = pd.to_numeric(sales[c], errors="coerce")

# recompute global_sales from regions
sales["global_sales"] = sales[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum(
    axis=1, min_count=1
)

# drop rows where all regional sales are missing
sales = sales.dropna(
    subset=["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"], how="all"
)

# keep only games with at least 0.1M units sold
sales = sales[sales["global_sales"] >= 0.1]

# RAWG quality filter
if "rating" in rawg.columns:
    rawg = rawg[rawg["rating"].notna() & (rawg["rating"] > 0)]

# Keep only mapped/canonical rows
rawg  = rawg[rawg["genre_std"].notna()]
steam = steam[steam["genre_std"].notna()]
sales = sales[sales["genre_std"].notna()]

print(
    "Post-canonicalization:",
    {"rawg": rawg.shape, "steam": steam.shape, "sales": sales.shape},
)

# Year handling 
def add_year(df: pd.DataFrame, released_col: str | None) -> pd.DataFrame:
    if released_col and released_col in df.columns:
        df["year"] = pd.to_datetime(df[released_col], errors="coerce").dt.year
    elif "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        df["year"] = pd.NA
    return df

rawg  = add_year(rawg, "released")   # RAWG has "released"
steam = add_year(steam, "released")  # we renamed release_date -> released
sales = add_year(sales, None)        # already has "year" from rename

def in_window(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["year"].between(YEAR_MIN, YEAR_MAX, inclusive="both")]

rawg  = in_window(rawg)
steam = in_window(steam)
sales = in_window(sales)

print(
    "Post-year-filter:",
    {"rawg": rawg.shape, "steam": steam.shape, "sales": sales.shape},
)

# Steam specific filters 
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


# In[3]:


get_ipython().system('jupyter nbconvert --to script data_cleaning_preprocessing.ipynb')


# In[ ]:




