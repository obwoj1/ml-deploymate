"""Download the UCI Adult (Census Income) dataset into data/raw/.

Used by scripts/download_data.py and by the service (so the live site can set
itself up). data/raw/ is gitignored. See docs/DATASET.md for what the data is.
"""

import io
import ssl
import urllib.request
import zipfile

import certifi

from deploymate.data.load import RAW_DIR

URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
FILES = ["adult.data", "adult.test", "adult.names"]


def is_downloaded(raw_dir=RAW_DIR) -> bool:
    return all((raw_dir / f).exists() for f in FILES)


def download(force: bool = False, raw_dir=RAW_DIR, verbose: bool = True) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not force and is_downloaded(raw_dir):
        if verbose:
            print(f"Already downloaded: {raw_dir} (use --force to re-download)")
        return

    if verbose:
        print(f"Downloading {URL} ...")
    # certifi's CA bundle: python.org installs on macOS don't use the system certificates.
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(URL, timeout=60, context=context) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    for name in FILES:
        (raw_dir / name).write_bytes(archive.read(name))
        if verbose:
            print(f"  saved data/raw/{name}")
    if verbose:
        print("Done.")
