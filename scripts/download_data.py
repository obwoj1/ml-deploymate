"""Download the UCI Adult (Census Income) dataset into data/raw/."""

from pathlib import Path          # tool for working with file/folder locations
import urllib.request             # built-in Python tool for downloading from the internet

# Web address of the folder on the UCI website that holds the dataset
BASE_URL = "https://archive.ics.uci.edu/dataset/2/adult"

# The three files we want to download
#   adult.data  = training data (~32,000 rows)
#   adult.test  = test data (~16,000 rows)
#   adult.names = text file describing the columns
FILES = ["adult.data", "adult.test", "adult.names"]

# Where to save the files: <project root>/data/raw
# __file__ is this script's location, parents[1] goes up two folders to the project root
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    # Create the data/raw folder if it doesn't already exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Go through each file in the list, one at a time
    for name in FILES:
        dest = RAW_DIR / name                    # full save path for this file

        # If we already downloaded it, skip it
        if dest.exists():
            print(f"Already have {name}, skipping.")
            continue

        print(f"Downloading {name}...")          # progress message
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", dest)   # download and save

    print(f"Done. Files are in {RAW_DIR}")


# Only run main() when you run this file directly (not when another file imports it)
if __name__ == "__main__":
    main()