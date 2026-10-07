"""Fetch the one OULAD table too large for GitHub (studentVle.csv, 454 MB) and check the rest.

    python scripts/get_data.py

The Open University's own download link now returns 404, so this uses the copy on the UCI
Machine Learning Repository (dataset 349, CC BY 4.0). The archive's SHA-256 is pinned; the
six smaller tables already committed in data/ are compared with the archive row for row.
"""
from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
SHA256 = "f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4"
DATA = Path(__file__).resolve().parents[1] / "data"
SMALL = ["studentInfo", "studentRegistration", "studentAssessment", "assessments", "courses", "vle"]


def main() -> None:
    target = DATA / "studentVle.csv"
    if target.exists():
        print("studentVle.csv already present")
        return
    print("downloading", URL)
    blob = urllib.request.urlopen(URL, timeout=600).read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SHA256:
        sys.exit("checksum mismatch: %s" % digest)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in SMALL:
            theirs = pd.read_csv(z.open(name + ".csv"), na_values="?")
            ours = pd.read_csv(DATA / (name + ".csv"))
            if not theirs.equals(ours):
                sys.exit("committed %s.csv differs from the archive" % name)
        target.write_bytes(z.read("studentVle.csv"))
    print("wrote", target, "; the six committed tables match the archive")


if __name__ == "__main__":
    main()
