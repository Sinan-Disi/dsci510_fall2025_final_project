#Collecting Raw RAWG Data by API  

import os
import time
import random
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RAWG_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "RAWG_API_KEY not found. Create a .env file in your project root with:\n"
        "RAWG_API_KEY=your_key_here"
    )

print("API key loaded:", bool(API_KEY))

games = []
target_total = 10_000  
print("Starting RAWG data collection (unfiltered)…\n")

session = requests.Session()
BASE = "https://api.rawg.io/api/games"

collected = 0
attempts = 0
page_limit = 1200  
seen_ids = set()

while collected < target_total and attempts < 5000:
    page = random.randint(1, page_limit)
    params = {
        "key": API_KEY,
        "page_size": 100,   
        "page": page,       
    }

    try:
        r = session.get(BASE, params=params, timeout=20)
    except requests.RequestException:
        attempts += 1
        time.sleep(1)
        continue

    attempts += 1
    if r.status_code != 200:
        print(f"  Skipped page {page} (error {r.status_code})")
        time.sleep(1)
        continue

    data = r.json().get("results", [])
    if not data:
        time.sleep(0.5)
        continue

    for g in data:
        gid = g.get("id")
        if gid in seen_ids:
            continue
        seen_ids.add(gid)

        platforms = g.get("platforms") or []
        platform_names = ", ".join(
            p["platform"]["name"]
            for p in platforms
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
    time.sleep(1)

df = pd.DataFrame(games)

print("\nCollection complete.")
print("Total collected:", len(df))

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

out_path = os.path.join(data_dir, "rawg_10000_unfiltered.csv")
df.to_csv(out_path, index=False)
print("Saved dataset:", out_path)



#Collecting Raw Steam Data by API   & Collecting Raw Data Sales data by API
import os
import shutil
from kaggle import api
import pandas as pd

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

home = os.path.expanduser("~")
kaggle_dir = os.path.join(home, ".kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

src_cfg = "kaggle.json"
dst_cfg = os.path.join(kaggle_dir, "kaggle.json")
if os.path.exists(src_cfg):
    shutil.copyfile(src_cfg, dst_cfg)
try:
    if os.name == "posix":
        os.chmod(dst_cfg, 0o600)
except Exception:
    pass

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
    dest_map={"steam.csv": os.path.join(data_dir, "steam.csv")}
)

fetch_kaggle_dataset(
    "gregorut/videogamesales",
    dest_map={"vgsales.csv": os.path.join(data_dir, "vgsales.csv")}
)

steam = pd.read_csv(os.path.join(data_dir, "steam.csv"))
sales = pd.read_csv(os.path.join(data_dir, "vgsales.csv"))




print("steam.csv shape:", steam.shape)
print("vgsales.csv shape:", sales.shape)
print("steam columns (first 15):", list(steam.columns)[:15])
print("vgsales columns:", list(sales.columns))





