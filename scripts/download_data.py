"""Download the UCI Adult (Census Income) dataset into data/raw/.

    python scripts/download_data.py          # skips files that already exist
    python scripts/download_data.py --force  # re-download everything

data/raw/ is gitignored, so every teammate (and the live server) runs this once.
See docs/DATASET.md for what the data is.
"""

import argparse
import io
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import certifi

URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
FILES = ["adult.data", "adult.test", "adult.names"]
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not force and all((RAW_DIR / f).exists() for f in FILES):
        print(f"Already downloaded: {RAW_DIR} (use --force to re-download)")
        return

    print(f"Downloading {URL} ...")
    # certifi's CA bundle: python.org installs on macOS don't use the system certificates.
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(URL, timeout=60, context=context) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    for name in FILES:
        (RAW_DIR / name).write_bytes(archive.read(name))
        print(f"  saved data/raw/{name}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    try:
        download(parser.parse_args().force)
    except OSError as err:
        sys.exit(f"Download failed: {err}\nManual fallback: unzip {URL} into {RAW_DIR}")
