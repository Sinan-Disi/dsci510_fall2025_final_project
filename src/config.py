"""
Shared configuration for the video game analysis project.

Centralizes paths, API keys, cleaning parameters, and plotting settings
used by data collection, cleaning, and analysis scripts.
"""
from pathlib import Path
import os
import shutil
from dotenv import load_dotenv

# project root and .env
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

# secrets
RAWG_API_KEY = os.getenv("RAWG_API_KEY", "")

def ensure_kaggle_creds():
    """
    Make sure Kaggle API credentials are available.

    Priority:
      1) KAGGLE_USERNAME / KAGGLE_KEY env vars
      2) kaggle.json in ~/.kaggle
      3) kaggle.json in project ROOT (copied into ~/.kaggle)

    Returns:
      "env" or "file" on success, raises RuntimeError if nothing is found.
    """
    # 1) env vars
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return "env"

    home = Path.home()
    kaggle_dir = home / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"

    # 2) existing ~/.kaggle/kaggle.json
    if kaggle_file.exists():
        return "file"

    # 3) kaggle.json in project root
    src = ROOT / "kaggle.json"
    if src.exists():
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, kaggle_file)
        try:
            os.chmod(kaggle_file, 0o600)
        except Exception:
            pass
        return "file"

    raise RuntimeError(
        "Kaggle credentials not found.\n"
        "Either set KAGGLE_USERNAME and KAGGLE_KEY environment variables,\n"
        "or put kaggle.json in ~/.kaggle/ or in the project root."
    )

# folders
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Raw filenames
RAWG_RAW = DATA_DIR / "rawg_10000_unfiltered.csv"
STEAM_RAW = DATA_DIR / "steam.csv"
VGSALES_RAW = DATA_DIR / "vgsales.csv"
# Clean filenames
RAWG_CLEAN = DATA_DIR / "rawg_clean.csv"
STEAM_CLEAN = DATA_DIR / "steam_clean.csv"
VGSALES_CLEAN = DATA_DIR / "vgsales_clean.csv"

# data collection parameters
RAWG_TARGET_TOTAL = int(os.getenv("RAWG_TARGET_TOTAL", "10000"))
RAWG_PAGE_SIZE = 100
RAWG_PAGE_LIMIT = 1200
RAWG_SLEEP_SEC = float(os.getenv("RAWG_SLEEP_SEC", "0.2"))
SKIP_IF_EXISTS = True

# cleaning/analysis parameters
YEAR_MIN, YEAR_MAX = 2000, 2020
MIN_STEAM_REVIEWS = 10
POS_RATIO_HIGH_THRESH = 0.75

# Excluding certain keywords before canonical genre mapping
EXCLUDE_KEYWORDS = [
    "sexual", "nudity", "porn", "adult", "hentai",
    "nsfw", "mature", "gore",
    "audio production", "video production", "utilities",
    "education", "software", "design", "animation",
]

# Canonical genre mapping to create 'genre_std'
CANON_GENRES = {
    "action", "adventure", "role-playing", "shooter", "strategy",
    "simulation", "racing", "fighting", "platform", "puzzle", "sports", "misc"
}
GENRE_MAP = {
    "rpg": "role-playing", "role playing": "role-playing", "role-playing": "role-playing",
    "fps": "shooter", "tps": "shooter", "shooter": "shooter",
    "platformer": "platform", "platform": "platform",
    "strategy": "strategy", "tactics": "strategy", "4x": "strategy",
    "simulation": "simulation", "simulator": "simulation", "management": "simulation",
    "racing": "racing", "driving": "racing",
    "fighting": "fighting", "brawler": "fighting",
    "puzzle": "puzzle", "action": "action", "adventure": "adventure",
    "indie": "misc", "casual": "misc", "arcade": "misc", "party": "misc",
    "board game": "misc", "card": "misc", "music": "misc", "rhythm": "misc",
    "survival": "misc", "horror": "misc", "visual novel": "misc",
    "sandbox": "misc", "roguelike": "misc", "roguelite": "misc",
    "metroidvania": "misc", "stealth": "misc", "hack and slash": "misc",
    "open world": "misc", "vr": "misc",
}

# Platform mapping for RAWG platforms for Q2
def map_platform(p: str):
    """
    Map a RAWG platform string into a broad platform group for Q2 charts.

    Returns one of {"PC", "PlayStation", "Xbox", "Switch"} or None
    if the platform should be ignored.
    """
    p = str(p).lower()
    if "pc" in p:
        return "PC"
    if "playstation" in p:
        return "PlayStation"
    if "xbox" in p:
        return "Xbox"
    if "switch" in p:
        return "Switch"
    return None

# Shared price bins with explicit Free bucket (used in Q3 and Q7)
PRICE_BINS_FREE = [-0.01, 0, 5, 15, 30, 60]
PRICE_LABELS_Q3 = ["Free", "$0-5", "$5-15", "$15-30", "$30-60"]
PRICE_LABELS_Q7 = ["Free", "0_5", "5_15", "15_30", "30_60"]

# Review-volume buckets for Q4
REVIEW_BINS = [0, 100, 1000, float("inf")]
REVIEW_LABELS = ["<100", "100-999", "1,000+"]

# for prediction modeling and figures
TEST_SIZE = 0.20
RANDOM_STATE = 42
FIG_DPI = 150