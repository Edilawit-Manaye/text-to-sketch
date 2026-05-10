"""

CLI entry-point: download the Kaggle anime dataset into data/raw/.
"""


import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from utils.paths import RAW_DATA_DIR, project_path

load_dotenv()

MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Kaggle anime dataset.")
    parser.add_argument(
        "--dataset",
        default=os.getenv("KAGGLE_DATASET", "diraizel/anime-images-dataset"),
        help="Kaggle dataset slug.",
    )
    parser.add_argument(
        "--target-dir",
        default=os.getenv("DATA_RAW_DIR", str(RAW_DATA_DIR)),
        help="Destination directory for downloaded data.",
    )
    return parser.parse_args()


def check_kaggle_auth() -> bool:
    username    = os.getenv("KAGGLE_USERNAME")
    key         = os.getenv("KAGGLE_KEY")
    json_exists = Path("~/.kaggle/kaggle.json").expanduser().exists()

    if json_exists or (username and key):
        return True

    print("ERROR: Kaggle credentials not found.")
    print("  Set either:")
    print("    1) ~/.kaggle/kaggle.json")
    print("    2) env vars KAGGLE_USERNAME and KAGGLE_KEY")
    return False


def download_dataset(dataset_slug: str, target_path: str | Path) -> str:
    """Download *dataset_slug* from Kaggle into *target_path*.

    Uses kagglehub's built-in resume support; retries up to MAX_RETRIES times
    on network timeouts.  Returns the final destination path.
    """
    import kagglehub  # lazy — only needed when actually downloading

    target_path = project_path(target_path).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"[download] Fetching dataset  : {dataset_slug}")
    print(f"[download] Destination       : {target_path}")

    cached_path = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            cached_path = kagglehub.dataset_download(dataset_slug)
            break
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            TimeoutError,
        ) as exc:
            if attempt == MAX_RETRIES:
                print(f"[download] All {MAX_RETRIES} attempts failed. Giving up.")
                raise
            print(f"[download] Timeout on attempt {attempt}/{MAX_RETRIES}: {exc}")
            print(f"[download] Retrying in {RETRY_DELAY}s … (kagglehub will resume)")
            time.sleep(RETRY_DELAY)

    print(f"[download] Cached at         : {cached_path}")

    cached_path = Path(cached_path)
    if cached_path.is_dir():
        for item in cached_path.iterdir():
            dst = target_path / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
    else:
        shutil.copy2(cached_path, target_path)

    print(f"[download] Stored at         : {target_path}")
    return str(target_path)


def main() -> None:
    """Download the Kaggle anime dataset into data/raw/."""
    args = parse_args()

    if not check_kaggle_auth():
        sys.exit(1)

    download_dataset(args.dataset, args.target_dir)


if __name__ == "__main__":
    main()
