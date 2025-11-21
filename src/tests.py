"""Lightweight smoke tests for the Video-Game Analytics & Prediction project."""

from pathlib import Path
import os
import pandas as pd
import requests

from config import (
    RAWG_API_KEY,
    DATA_DIR,
    RESULTS_DIR,
    RAWG_RAW,
    STEAM_RAW,
    VGSALES_RAW,
    RAWG_CLEAN,
    STEAM_CLEAN,
    VGSALES_CLEAN,
    YEAR_MIN,
    YEAR_MAX,
)

DATA = DATA_DIR
RESULTS = RESULTS_DIR

def run_test(name, fn):
    """Run a single test function and print a PASS/FAIL/SKIP line."""
    try:
        rv = fn()
        if rv == "skip":
            print(f"[SKIP] {name}")
        else:
            print(f"[PASS] {name}")
        return rv
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
        return "fail"
    except Exception as e:
        print(f"[FAIL] {name}: unexpected error {e}")
        return "fail"

def test_rawg_api_smoke():
    key = RAWG_API_KEY
    if not key:
        print("  SKIP: RAWG_API_KEY not set (put it in .env).")
        return "skip"
    try:
        r = requests.get(
            "https://api.rawg.io/api/games",
            params={"key": key, "page_size": 1},
            timeout=10,
        )
    except requests.RequestException as e:
        raise AssertionError(f"Request failed: {e}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "results" in data, "RAWG response missing 'results'"

def test_source_files_if_present():
    missing = []
    for fp in [RAWG_RAW, STEAM_RAW, VGSALES_RAW]:
        if not fp.exists():
            print(f"  SKIP: {fp} does not exist (run data_collection.py).")
            missing.append(fp)
    if missing:
        return "skip"
    for fp in [RAWG_RAW, STEAM_RAW, VGSALES_RAW]:
        df = pd.read_csv(fp, nrows=5)
        assert not df.empty, f"{fp.name} is empty"
        assert df.shape[1] > 0, f"{fp.name} has no columns"

def test_rawg_schema_if_present():
    fp = RAWG_RAW
    if not fp.exists():
        print("  SKIP: rawg_10000_unfiltered.csv not present.")
        return "skip"
    df = pd.read_csv(fp, nrows=100)
    for col in ["name", "rating", "released", "platforms", "genres"]:
        assert col in df.columns, f"RAWG missing column {col}"

def test_steam_schema_if_present():
    fp = STEAM_RAW
    if not fp.exists():
        print("  SKIP: steam.csv not present.")
        return "skip"
    df = pd.read_csv(fp, nrows=100)
    for col in ["appid", "name", "positive_ratings", "negative_ratings", "price"]:
        assert col in df.columns, f"Steam missing column {col}"

def test_vgsales_schema_if_present():
    fp = VGSALES_RAW
    if not fp.exists():
        print("  SKIP: vgsales.csv not present.")
        return "skip"
    df = pd.read_csv(fp, nrows=100)
    for col in ["Name", "Platform", "Year", "Genre", "Global_Sales"]:
        assert col in df.columns, f"VGSales missing column {col}"

def test_cleaned_files_if_present():
    cleaned = [
        (RAWG_CLEAN,  "year"),
        (STEAM_CLEAN, "year"),
        (VGSALES_CLEAN, "year"),
    ]
    saw_any = False
    for fp, ycol in cleaned:
        if not fp.exists():
            print(f"  SKIP: {fp} not found (run preprocessing).")
            continue
        saw_any = True
        df = pd.read_csv(fp, usecols=lambda c: c == ycol, low_memory=False)
        assert not df.empty, f"{fp.name} is empty"
        yr = pd.to_numeric(df[ycol], errors="coerce").dropna()
        assert ((yr >= YEAR_MIN) & (yr <= YEAR_MAX)).all(), f"{fp.name} has year outside {YEAR_MIN}-{YEAR_MAX}"
    if not saw_any:
        print("  SKIP: cleaned CSVs not found (run preprocessing).")
        return "skip"

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
        ("Source files (if present)",     test_source_files_if_present),
        ("RAWG schema (if present)",      test_rawg_schema_if_present),
        ("Steam schema (if present)",     test_steam_schema_if_present),
        ("VGSales schema (if present)",   test_vgsales_schema_if_present),
        ("Cleaned files (if present)",    test_cleaned_files_if_present),
        ("results/ writable",             test_results_dir_writable),
    ]
    counts = {"pass": 0, "fail": 0, "skip": 0}
    for name, fn in tests:
        res = run_test(name, fn)
        counts[res] += 1
    print(f"\nSummary: {counts['pass']} passed, {counts['fail']} failed, {counts['skip']} skipped")
