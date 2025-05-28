# opensea_collections.py
import time
import requests
from datetime import datetime
from typing import List, Dict, Optional

API_KEY  = "350091df21b748a18368a329b8be8e79"
HEADERS  = {"accept": "application/json", "x-api-key": API_KEY}
SLEEP_API = 1.0

def log(msg, level="INFO"):
    colors = {"INFO":"\033[94m", "OK":"\033[92m",
              "WARN":"\033[93m", "ERR":"\033[91m", "END":"\033[0m"}
    print(f"{colors[level]}{datetime.now():%H:%M:%S} [{level}] {msg}{colors['END']}")

# ──────────────────────────────────────────────────────────────
def get_collections_batch(limit: int = 100, next_cursor: Optional[str] = None) -> Dict[str, any]:
    """
    Zwraca jedną partię kolekcji z API (max 100) oraz kursor do następnej strony.
    Zwraca słownik: {"collections": [...], "next": "cursor_string" or None}
    """
    url = f"https://api.opensea.io/api/v2/collections?limit={limit}&order_by=market_cap"
    if next_cursor:
        url += f"&next={next_cursor}"

    try:
        r = requests.get(url, headers=HEADERS, timeout=(3, 5))
        if r.status_code == 200:
            data = r.json()
            collections = data.get("collections", [])
            next_page = data.get("next")
            log(f"Batch: {len(collections)} collections", "OK")
            return {"collections": collections, "next": next_page}
        else:
            log(f"/collections → {r.status_code}", "ERR")
    except Exception as e:
        log(f"Request error → {e}", "ERR")
   

    return {"collections": [], "next": None}
