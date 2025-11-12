#Tests
from pathlib import Path
import os
import pandas as pd
import requests
from dotenv import load_dotenv

# Resolve project root whether running from project/ or project/src/
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = HERE if (HERE / "data").exists() else HERE.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

def run_test(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
        return "pass"
    except AssertionError as e:
        print(f"[FAIL] {name} -> {e}")
        return "fail"
    except Exception as e:
        print(f"[ERROR] {name} -> {e}")
        return "fail"

def test_rawg_api_smoke():
    """Fetch 1 small page from RAWG to prove at-least-one API works."""
    load_dotenv()
    key = os.getenv("RAWG_API_KEY")
    if not key:
        print("  SKIP: RAWG_API_KEY not set (set it in .env to run this API test).")
        return
    r = requests.get(
        "https://api.rawg.io/api/games",
        params={"key": key, "page_size": 1, "page": 1},
        timeout=15,
    )
    assert r.status_code == 200, f"RAWG status {r.status_code}"
    js = r.json()
    assert "results" in js and len(js["results"]) >= 1, "RAWG response had no results"

def test_files_exist():
    """Your three collected datasets exist."""
    for fp in [DATA / "rawg_10000_unfiltered.csv",
               DATA / "steam.csv",
               DATA / "vgsales.csv"]:
        assert fp.exists(), f"Missing file: {fp}"
#for rawg
def test_rawg_schema():
    df = pd.read_csv(DATA / "rawg_10000_unfiltered.csv", nrows=5)
    need = {"name", "rating", "released", "platforms", "genres"}
    assert need.issubset(df.columns), f"RAWG missing columns: {need - set(df.columns)}"
#for steam
def test_steam_schema():
    df = pd.read_csv(DATA / "steam.csv", nrows=5)
    need = {
        "positive_ratings", "negative_ratings", "price",
        "average_playtime", "owners", "categories",
        "platforms", "publisher", "release_date"
    }
    assert need.issubset(df.columns), f"Steam missing columns: {need - set(df.columns)}"
#for vgsales
def test_vgsales_schema():
    df = pd.read_csv(DATA / "vgsales.csv", nrows=5)
    need = {"Name", "Platform", "Year", "Genre", "Global_Sales"}
    assert need.issubset(df.columns), f"VGSales missing columns: {need - set(df.columns)}"

def test_cleaned_files_if_present():
    """If cleaned CSVs exist, sanity-check year range."""
    cleaned = [
        (DATA / "rawg_clean.csv",  "year"),
        (DATA / "steam_clean.csv", "year"),
        (DATA / "vgsales_clean.csv", "year"),
    ]
    for fp, ycol in cleaned:
        if not fp.exists():
            continue
        df = pd.read_csv(fp, usecols=lambda c: c==ycol, low_memory=False)
        assert not df.empty, f"{fp.name} is empty"
        yr = pd.to_numeric(df[ycol], errors="coerce").dropna()
        assert ((yr >= 2000) & (yr <= 2020)).all(), f"{fp.name} has year outside 2000–2020"

def test_results_dir_writable():
    RESULTS.mkdir(parents=True, exist_ok=True)
    probe = RESULTS / "_write_test.txt"
    probe.write_text("ok", encoding="utf-8")
    assert probe.exists(), "Could not create file in results/"
    probe.unlink(missing_ok=True)

if __name__ == "__main__":
    print("Running project tests...\n")
    tests = [
        ("RAWG API smoke",                test_rawg_api_smoke),
        ("Files exist (rawg/steam/sales)",test_files_exist),
        ("RAWG schema",                   test_rawg_schema),
        ("Steam schema",                  test_steam_schema),
        ("VGSales schema",                test_vgsales_schema),
        ("Cleaned files (if present)",    test_cleaned_files_if_present),
        ("results/ writable",             test_results_dir_writable),
    ]
    counts = {"pass":0, "fail":0, "skip":0}
    for name, fn in tests:
        res = run_test(name, fn)
        counts[res] += 1
    print(f"\nSummary: {counts['pass']} passed, {counts['fail']} failed")