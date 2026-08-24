"""Télécharge le dataset Telco Customer Churn (IBM Sample) dans data/raw/.

Usage :
    python scripts/download_data.py

Le fichier n'est pas versionné dans Git (voir .gitignore) : ce script
garantit que n'importe qui peut reconstituer data/ à l'identique.
"""

import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
EXPECTED_MD5 = "3b0bfab28a8101b4e4fdd08025a5c235"
DEST = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if DEST.exists():
        print(f"Déjà présent : {DEST}")
    else:
        print(f"Téléchargement depuis {URL}")
        urllib.request.urlretrieve(URL, DEST)

    md5 = hashlib.md5(DEST.read_bytes()).hexdigest()
    if md5 != EXPECTED_MD5:
        raise SystemExit(
            f"Somme de contrôle inattendue : {md5} (attendu {EXPECTED_MD5}). "
            "Le fichier source a peut-être changé."
        )

    print(f"OK — {DEST} ({DEST.stat().st_size:,} octets, md5 vérifié)")


if __name__ == "__main__":
    main()
