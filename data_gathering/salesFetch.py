# salesFetch.py

import requests
from datetime import datetime

API_KEY = "350091df21b748a18368a329b8be8e79"
HEADERS = {"accept": "application/json", "x-api-key": API_KEY}

def log(msg, level="INFO"):
    colors = {"INFO":"\033[94m", "OK":"\033[92m",
              "WARN":"\033[93m", "ERR":"\033[91m", "END":"\033[0m"}
    print(f"{colors.get(level,'')}{datetime.now():%H:%M:%S} [{level}] {msg}{colors['END']}")

def fetch_last_sale(contract_address: str, token_id: str, chain: str) -> dict:
    """
    Fetch the last successful sale price for a given NFT
    using OpenSea API v2 with correct endpoint.

    Endpoint: 
    /api/v2/events/chain/{chain}/contract/{contract}/nfts/{identifier}

    Returns: {
        "last_sale_price": float or None,
        "num_sales": int
    }
    """
    url = (
        f"https://api.opensea.io/api/v2/events/chain/{chain}"
        f"/contract/{contract_address}/nfts/{token_id}"
        f"?event_type=sale&limit=1"
    )
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
    except Exception as e:
        return {"last_sale_price": None, "num_sales": 0}
    if response.status_code != 200:
        log(f"/events/chain/{chain}/contract/{contract_address}/nfts/{token_id} → {response.status_code}", "WARN")
        try:
            log(response.json(), "WARN")
        except:
            log(response.text, "WARN")
        return {"last_sale_price": None, "num_sales": 0}

    events = response.json().get("asset_events", [])
    if not events:
        return {"last_sale_price": None, "num_sales": 0}

    try:
        payment = events[0].get("payment", {})
        quantity = float(payment.get("quantity", 0))
        decimals = int(payment.get("decimals", 0))
        price = quantity / (10 ** decimals) if decimals else None
        return {"last_sale_price": price, "num_sales": 1}
    except Exception as e:
        log(f"Error parsing sale data for token {token_id}: {e}", "WARN")
        return {"last_sale_price": None, "num_sales": 0}
