"""Download the UCI Adult (Census Income) dataset into data/raw/.

    python scripts/download_data.py          # skips files that already exist
    python scripts/download_data.py --force  # re-download everything

data/raw/ is gitignored, so each environment (including the deployed app) runs this once.
See docs/DATASET.md for what the data is.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deploymate.data.download import URL, download  # noqa: E402
from deploymate.data.load import RAW_DIR  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    try:
        download(parser.parse_args().force)
    except OSError as err:
        sys.exit(f"Download failed: {err}\nManual fallback: unzip {URL} into {RAW_DIR}")
