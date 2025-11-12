import pandas as pd
import numpy as np
import re

# I/O 
rawg  = pd.read_csv("../data/rawg_10000_unfiltered.csv")
steam = pd.read_csv("../data/steam.csv")
sales = pd.read_csv("../data/vgsales.csv")

print("Loaded:", {"rawg": rawg.shape, "steam": steam.shape, "sales": sales.shape})

#  Renames/normalization 
rawg.rename(columns={"genres": "genre"}, inplace=True)
steam.rename(columns={"genres": "genre", "release_date": "released"}, inplace=True)
sales.rename(columns={
    "Genre": "genre", "Name": "name", "Platform": "platform",
    "Year": "year", "Global_Sales": "global_sales"
}, inplace=True)

for df in (rawg, steam, sales):
    if "genre" in df.columns:
        df["genre"] = df["genre"].astype(str).str.lower().str.strip()

#  Canonical genre mapping 
CANON_GENRES = {
    "action","adventure","role-playing","shooter","sports",
    "strategy","simulation","racing","fighting","platform","puzzle","misc"
}
GENRE_MAP = {
    # role-playing
    "rpg":"role-playing","role playing":"role-playing","role-playing":"role-playing",
    # shooter
    "shooter":"shooter","fps":"shooter","tps":"shooter",
    # platform
    "platformer":"platform","platform":"platform",
    # sports
    "sports":"sports","sport":"sports",
    # strategy
    "strategy":"strategy","tactics":"strategy","4x":"strategy",
    # simulation
    "simulation":"simulation","simulator":"simulation","management":"simulation",
    # racing
    "racing":"racing","driving":"racing",
    # fighting
    "fighting":"fighting","brawler":"fighting",
    # puzzle / action / adventure
    "puzzle":"puzzle","action":"action","adventure":"adventure",
    # bucket as misc
    "indie":"misc","casual":"misc","arcade":"misc","party":"misc",
    "board game":"misc","card":"misc","music":"misc","rhythm":"misc",
    "survival":"misc","horror":"misc","visual novel":"misc",
    "sandbox":"misc","roguelike":"misc","roguelite":"misc",
    "metroidvania":"misc","stealth":"misc","hack and slash":"misc",
    "open world":"misc","vr":"misc",
    "action adventure":"action"  
}

_splitter = re.compile(r"[;,/|]")

def std_genre_cell(s: str) -> str | float:
    if pd.isna(s):
        return np.nan
    # normalize separators + spaces/hyphens
    parts = [p.strip().lower() for p in _splitter.split(str(s)) if p.strip()]
    mapped = set()
    for p in parts:
        norm = p.replace("-", " ")  
        if norm in GENRE_MAP:
            mapped.add(GENRE_MAP[norm]); continue
        if p in CANON_GENRES:
            mapped.add(p); continue
    return "; ".join(sorted(mapped)) if mapped else np.nan

for df in (rawg, steam, sales):
    if "genre" in df.columns:
        df["genre_std"] = df["genre"].apply(std_genre_cell)

# RAWG quality filter 
if "rating" in rawg.columns:
    rawg = rawg[rawg["rating"].notna() & (rawg["rating"] > 0)]

# Keep only mapped/canonical rows
rawg  = rawg[rawg["genre_std"].notna()]
steam = steam[steam["genre_std"].notna()]
sales = sales[sales["genre_std"].notna()]

print("Post-canonicalization:", {"rawg": rawg.shape, "steam": steam.shape, "sales": sales.shape})

# Year handling
def add_year(df: pd.DataFrame, released_col: str | None) -> pd.DataFrame:
    if released_col and released_col in df.columns:
        df["year"] = pd.to_datetime(df[released_col], errors="coerce").dt.year
    elif "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        df["year"] = pd.NA
    return df

rawg  = add_year(rawg, "released")
steam = add_year(steam, "released")
sales = add_year(sales, None)

def in_window(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["year"].between(2000, 2020, inclusive="both")]

rawg  = in_window(rawg)
steam = in_window(steam)
sales = in_window(sales)

# Steam-specific filters
steam = steam[
    (steam["positive_ratings"] > 0) &
    (steam["negative_ratings"] > 0) &
    (steam["average_playtime"] > 0) &
    (steam["median_playtime"] > 0)
]

print("After Steam filters:", steam.shape)

# Save
rawg.to_csv("../data/rawg_clean.csv", index=False)
steam.to_csv("../data/steam_clean.csv", index=False)
sales.to_csv("../data/vgsales_clean.csv", index=False)

print("Saved: rawg_clean.csv, steam_clean.csv, vgsales_clean.csv")





