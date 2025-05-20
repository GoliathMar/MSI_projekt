# opensea_collections.py
import time
import requests
from datetime import datetime
from typing import List, Dict

API_KEY = "350091df21b748a18368a329b8be8e79"
HEADERS = {"accept": "application/json", "x-api-key": API_KEY}

def log(msg, level="INFO"):
    colors = {"INFO":"\033[94m", "OK":"\033[92m",
              "WARN":"\033[93m", "ERR":"\033[91m", "END":"\033[0m"}
    print(f"{colors.get(level,'')}{datetime.now():%H:%M:%S} [{level}] {msg}{colors['END']}")


def fetch_nfts(slug: str, limit: int = 15) -> List[Dict]:
    """
    Fetch up to `limit` NFTs from a given collection `slug`
    using OpenSea API v2.
    """
    url = f"https://api.opensea.io/api/v2/collection/{slug}/nfts?limit={limit}"
    try:
         r = requests.get(url, headers=HEADERS, timeout=10)
    except Exception as e:
        return []

    if r.status_code != 200:
        log(f"/nfts {slug} → {r.status_code}", "WARN")
        return []
    return r.json().get("nfts", [])
