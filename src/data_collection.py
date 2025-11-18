#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Collecting Raw RAWG Data by API  
import os
import time
import random
import requests
import pandas as pd

from config import (
    RAWG_API_KEY,
    DATA_DIR,
    RAWG_RAW,
    STEAM_RAW,
    VGSALES_RAW,
    RAWG_TARGET_TOTAL,
    RAWG_PAGE_LIMIT,
    RAWG_PAGE_SIZE,
    RAWG_SLEEP_SEC,
    ensure_kaggle_creds,
)

API_KEY = RAWG_API_KEY
if not API_KEY:
    raise RuntimeError(
        "RAWG_API_KEY not found. Create a .env file in your project root with:\n"
        "RAWG_API_KEY=your_key_here"
    )

print("API key loaded:", bool(API_KEY))

games = []
target_total = RAWG_TARGET_TOTAL
print("Starting RAWG data collection (unfiltered)…\n")

session = requests.Session()
BASE = "https://api.rawg.io/api/games"

collected = 0
attempts = 0
page_limit = RAWG_PAGE_LIMIT
seen_ids = set()

while collected < target_total and attempts < 5000:
    page = random.randint(1, page_limit)
    params = {
        "key": API_KEY,
        "page_size": RAWG_PAGE_SIZE,
        "page": page,
    }

    try:
        r = session.get(BASE, params=params, timeout=20)
    except requests.RequestException as e:
        print(f"Request failed: {e}. Sleeping and retrying...")
        time.sleep(1)
        continue

    attempts += 1
    if r.status_code != 200:
        print(f"Non-200 status {r.status_code}. Body: {r.text[:200]}")
        time.sleep(1)
        continue

    data = r.json()
    results = data.get("results") or []
    if not results:
        print("No results on this page, skipping...")
        time.sleep(1)
        continue

    for g in results:
        gid = g.get("id")
        if gid in seen_ids:
            continue
        seen_ids.add(gid)

        platforms_list = g.get("platforms") or []
        platform_names = ", ".join(
            p["platform"]["name"]
            for p in platforms_list
            if p and p.get("platform") and p["platform"].get("name")
        )

        genres_list = g.get("genres") or []
        genre_names = ", ".join(
            p.get("name", "")
            for p in genres_list
            if p and p.get("name")
        )

        games.append({
            "name": g.get("name"),
            "rating": g.get("rating"),
            "released": g.get("released"),
            "platforms": platform_names if platform_names else None,
            "genres": genre_names if genre_names else None,
        })
        collected += 1
        if collected >= target_total:
            break

    if collected % 500 == 0:
        print(f"  {collected} games collected so far")
    time.sleep(RAWG_SLEEP_SEC)

df = pd.DataFrame(games)

print("\nCollection complete.")
print("Total collected:", len(df))

# use config paths
out_path = RAWG_RAW
df.to_csv(out_path, index=False)
print("Saved dataset:", out_path)



#Collecting Raw Steam Data by API   & Collecting Raw Data Sales data by API
import shutil
from kaggle import api

# ensure Kaggle credentials are available (env vars or kaggle.json in project root)
ensure_kaggle_creds()

def fetch_kaggle_dataset(dataset_slug: str, dest_map: dict):
    tmp = f"_tmp_{dataset_slug.split('/')[-1]}"
    os.makedirs(tmp, exist_ok=True)
    api.dataset_download_files(dataset_slug, path=tmp, unzip=True)
    for expected_name, out_path in dest_map.items():
        found = None
        for fname in os.listdir(tmp):
            if fname.lower() == expected_name.lower():
                found = os.path.join(tmp, fname)
                break
        if not found:
            raise FileNotFoundError(
                f"Could not find '{expected_name}' in {tmp}. "
                f"Available: {sorted(os.listdir(tmp))[:10]}"
            )
        shutil.copyfile(found, out_path)
    shutil.rmtree(tmp, ignore_errors=True)

fetch_kaggle_dataset(
    "nikdavis/steam-store-games",
    dest_map={"steam.csv": str(STEAM_RAW)}
)

fetch_kaggle_dataset(
    "gregorut/videogamesales",
    dest_map={"vgsales.csv": str(VGSALES_RAW)}
)

steam = pd.read_csv(STEAM_RAW)
sales = pd.read_csv(VGSALES_RAW)

print("steam.csv shape:", steam.shape)
print("vgsales.csv shape:", sales.shape)
print("steam columns (first 15):", list(steam.columns)[:15])
print("vgsales columns:", list(sales.columns))


# In[2]:


get_ipython().system('jupyter nbconvert --to script data_collection.ipynb')


# In[ ]:





# In[ ]:




