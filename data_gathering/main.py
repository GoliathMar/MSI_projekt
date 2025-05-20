import os
import time
import requests
import pandas as pd
from datetime import datetime

# Assuming the required imports from user modules are already available
from collectionsList import get_collections_batch
from nftsFetch import fetch_nfts
from salesFetch import fetch_last_sale

# Logging function
def log(msg, level="INFO"):
    colors = {"INFO": "\033[94m", "OK": "\033[92m",
              "WARN": "\033[93m", "ERR": "\033[91m", "END": "\033[0m"}
    print(f"{colors.get(level, '')}{datetime.now():%H:%M:%S} [{level}] {msg}{colors['END']}")

# Save image function
def save_img(url: str, path: str):
    try:
        r = requests.get(url, timeout=(3, 5))
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
    except Exception as e:
        log(f"img {url[-25:]} … {e}", "WARN")

# Configuration
TARGET_NFT_COUNT = 35000
MAX_NFTS_PER_COLLECTION = 15
BATCH_SIZE_COLLECTIONS = 10
IMG_DIR = "images"
CSV_PATH = "nft_dataset.csv"
SLEEP_API_COLLECTION = 1.0
SLEEP_NFT_LOOP = 0.3

os.makedirs(IMG_DIR, exist_ok=True)

valid_nft_count = 0
used_slugs = set()

if os.path.exists("last_cursor.txt"):
    with open("last_cursor.txt", "r") as f:
        next_cursor = f.read().strip()
    log(f"▶️ Wznawiam od kursora: {next_cursor}", "INFO")
else:
    next_cursor = None

# Initialize CSV with headers if file does not exist
if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=[
        "collection", "name", "image_url", "token_id", "contract_address",
        "chain", "last_sale_price", "last_sale_symbol", "last_sale_timestamp"
    ]).to_csv(CSV_PATH, index=False)

while valid_nft_count < TARGET_NFT_COUNT:
    result = get_collections_batch(limit=BATCH_SIZE_COLLECTIONS, next_cursor=next_cursor)
    collections = result["collections"]
    next_cursor = result["next"]
    if not collections:
        log("❗ API nie zwróciło więcej kolekcji – kończę pętlę", "WARN")
        break

    for col in collections:
        if valid_nft_count >= TARGET_NFT_COUNT:
            break

        slug = col.get("collection")
        contracts = col.get("contracts", [])
        if not slug or slug in used_slugs or not contracts:
            log(f"Brak nazwy: {slug} #{token_id}", "WARN")
            continue
        used_slugs.add(slug)

        contract_addr = contracts[0].get("address")
        chain_name = contracts[0].get("chain", "ethereum")

        log(f"[{valid_nft_count}/{TARGET_NFT_COUNT}] {slug} ({chain_name})")

        nfts = fetch_nfts(slug, limit=MAX_NFTS_PER_COLLECTION)
        log(f"   → {len(nfts)} NFT-ów z API", "OK" if nfts else "WARN")

        collected_here = 0
        for nft in nfts:
            if valid_nft_count >= TARGET_NFT_COUNT or collected_here >= MAX_NFTS_PER_COLLECTION:
                break

            token_id = str(nft.get("identifier")) if nft.get("identifier") else None
            image_url = nft.get("image_url")
            name = nft.get("name") or f"{slug}_{token_id}"

            if not token_id or not image_url:
                log(f"Brak image url: {slug} #{token_id}", "WARN")
                continue

            sale = fetch_last_sale(contract_addr, token_id, chain_name)
            if sale["last_sale_price"] is None:
                log(f"Brak ceny: {slug} #{token_id}", "WARN")
                continue

            row = {
                "collection": slug,
                "name": name,
                "image_url": image_url,
                "token_id": token_id,
                "contract_address": contract_addr,
                "chain": chain_name,
                **sale
            }

            # Append row directly to CSV
            pd.DataFrame([row]).to_csv(CSV_PATH, mode='a', header=False, index=False)

            valid_nft_count += 1
            collected_here += 1

            safe = name.replace(" ", "_").replace("/", "-").replace("\\", "-").replace("#", "")
            save_img(image_url, os.path.join(IMG_DIR, f"{slug}_{safe}.png"))

            time.sleep(SLEEP_NFT_LOOP)

    time.sleep(SLEEP_API_COLLECTION)
if next_cursor:
    with open("last_cursor.txt", "w") as f:
        f.write(next_cursor)
log(f"✅ Uzbierano {valid_nft_count} NFT (cel: {TARGET_NFT_COUNT})", "OK")
log(f"CSV saved → {CSV_PATH}", "OK")
