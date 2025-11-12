from pathlib import Path
import os, shutil
from dotenv import load_dotenv

# project root and .env 
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

# secrets
RAWG_API_KEY = os.getenv("RAWG_API_KEY", "")

def ensure_kaggle_creds():
    """
    Prefer env vars KAGGLE_USERNAME/KAGGLE_KEY.
    Otherwise, if project-root kaggle.json exists, copy it to ~/.kaggle/kaggle.json.
    Returns: "env" | "file" | "missing"
    """
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return "env"
    src = ROOT / "kaggle.json"
    if not src.exists():
        return "missing"
    dest_dir = Path.home() / ".kaggle"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "kaggle.json"
    if not dest.exists():
        shutil.copyfile(src, dest)
        try:
            os.chmod(dest, 0o600)  
        except Exception:
            pass
    return "file"

#folders
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#filenames
RAWG_RAW    = DATA_DIR / "rawg_10000_unfiltered.csv"
STEAM_RAW   = DATA_DIR / "steam.csv"
VGSALES_RAW = DATA_DIR / "vgsales.csv"

RAWG_CLEAN    = DATA_DIR / "rawg_clean.csv"
STEAM_CLEAN   = DATA_DIR / "steam_clean.csv"
VGSALES_CLEAN = DATA_DIR / "vgsales_clean.csv"

#collection parameters
RAWG_TARGET_TOTAL = int(os.getenv("RAWG_TARGET_TOTAL", "10000")) 
RAWG_PAGE_SIZE    = 100
RAWG_PAGE_LIMIT   = 1200
RAWG_SLEEP_SEC    = float(os.getenv("RAWG_SLEEP_SEC", "0.2"))
SKIP_IF_EXISTS    = True

# cleaning/analysis parameters
YEAR_MIN, YEAR_MAX    = 2000, 2020
MIN_STEAM_REVIEWS     = 10
POS_RATIO_HIGH_THRESH = 0.75

EXCLUDE_KEYWORDS = [
    "sexual", "nudity", "porn", "adult", "hentai",
    "nsfw", "mature", "gore",
    "audio production", "video production", "utilities",
    "education", "software", "design", "animation",
]

# canonical genre mapping to create 'genre_std'
CANON_GENRES = {
    "action","adventure","role-playing","shooter","strategy",
    "simulation","racing","fighting","platform","puzzle","sports","misc"
}
GENRE_MAP = {
    "rpg":"role-playing","role playing":"role-playing","role-playing":"role-playing",
    "fps":"shooter","tps":"shooter","shooter":"shooter",
    "platformer":"platform","platform":"platform",
    "strategy":"strategy","tactics":"strategy","4x":"strategy",
    "simulation":"simulation","simulator":"simulation","management":"simulation",
    "racing":"racing","driving":"racing",
    "fighting":"fighting","brawler":"fighting",
    "puzzle":"puzzle","action":"action","adventure":"adventure",
    "indie":"misc","casual":"misc","arcade":"misc","party":"misc",
    "board game":"misc","card":"misc","music":"misc","rhythm":"misc",
    "survival":"misc","horror":"misc","visual novel":"misc",
    "sandbox":"misc","roguelike":"misc","roguelite":"misc",
    "metroidvania":"misc","stealth":"misc","hack and slash":"misc",
    "open world":"misc","vr":"misc",
}

# Q3 price bins 
PRICE_BINS   = [0, 5, 15, 30, 60, float("inf")]
PRICE_LABELS = ["$0–5", "$5–15", "$15–30", "$30–60", "$60+"]

# Q4 review-volume buckets
REVIEW_BINS   = [0, 100, 1000, float("inf")]
REVIEW_LABELS = ["<100", "100–999", "1,000+"]

# for prediction modeling
TEST_SIZE    = 0.20
RANDOM_STATE = 42
FIG_DPI      = 150
