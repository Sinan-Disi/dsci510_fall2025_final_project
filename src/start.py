from data_collection import main as collect_data
from data_cleaning_preprocessing import main as preprocess_data
from main import main as run_analysis

def main():
    print("STEP 1 — Collecting data...")
    collect_data()

    print("\nSTEP 2 — Cleaning data...")
    preprocess_data()

    print("\nSTEP 3 — Running analysis...")
    run_analysis()

if __name__ == "__main__":
    main()
    