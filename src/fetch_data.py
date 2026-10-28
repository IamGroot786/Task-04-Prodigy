"""Download the official Prodigy InfoTech Task 4 training CSV.

The official Task 4 folder currently exposes twitter_training.csv. A validation
CSV is optional and can be placed in data/raw/ if separately obtained.
"""
from __future__ import annotations
import csv
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/main/Task%204/twitter_training.csv"
EXPECTED_COLUMNS = 4

def main() -> None:
    destination = ROOT / "data" / "raw" / "twitter_training.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent": "Prodigy-Task4-Student-Project/1.0"})
    with urlopen(request, timeout=90) as response:
        payload = response.read()
    if len(payload) < 100:
        raise RuntimeError("Downloaded file is unexpectedly small.")
    try:
        first_line = payload.decode("utf-8-sig", errors="strict").splitlines()[0]
        first_row = next(csv.reader([first_line]))
    except (UnicodeDecodeError, IndexError, csv.Error) as exc:
        raise RuntimeError("Downloaded content is not a readable CSV.") from exc
    if len(first_row) != EXPECTED_COLUMNS:
        raise RuntimeError(f"Expected a headerless four-column Twitter CSV; found {len(first_row)} columns.")
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        temp.write(payload)
        temp_path = Path(temp.name)
    temp_path.replace(destination)
    print(f"Saved {destination.name}: {len(payload):,} bytes")
    print("Run: python src/analyze_sentiment.py")

if __name__ == "__main__":
    main()
