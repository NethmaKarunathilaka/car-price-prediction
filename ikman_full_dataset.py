import re
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup

LIST_URL = "https://ikman.lk/en/ads/sri-lanka/cars?page={}"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------- helpers ----------
def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def to_int(text: str):
    if not text:
        return None
    m = re.search(r"([\d,]+)", text)
    return int(m.group(1).replace(",", "")) if m else None

def extract_kv(page_text: str, key: str):
    """
    Extract value from 'Key: Value' lines in the page text.
    """
    m = re.search(rf"{re.escape(key)}\s*:\s*(.+)", page_text, flags=re.I)
    if not m:
        return None
    val = m.group(1).split("\n")[0]
    return clean(val)

def extract_location_from_url(url: str):
    """
    Extract location from URLs like:
    .../honda-crv-2013-for-sale-colombo-157
    Returns: 'Colombo'
    """
    m = re.search(r"for-sale-([a-z\-]+)-\d+$", url)
    if not m:
        return None
    loc = m.group(1).replace("-", " ")
    return loc.title()

def get_ad_urls_from_list_page(page: int):
    url = LIST_URL.format(page)
    r = requests.get(url, headers=HEADERS, timeout=25)
    if r.status_code != 200:
        print(f"List page blocked/failed: {r.status_code}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    urls = []
    for a in soup.select("a[href*='/ad/']"):
        href = a.get("href")
        if not href:
            continue
        full = "https://ikman.lk" + href if href.startswith("/") else href
        urls.append(full)

    # de-dup while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq

def parse_ad_detail(ad_url: str):
    r = requests.get(ad_url, headers=HEADERS, timeout=25)
    if r.status_code != 200:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    page_text = soup.get_text("\n", strip=True)

    # price (top of page)
    m = re.search(r"Rs\.?\s*([\d,]+)", page_text)
    price_lkr = to_int(m.group(1)) if m else None

    # specs block (your screenshot shows these exact labels)
    brand = extract_kv(page_text, "Brand")
    model = extract_kv(page_text, "Model")
    year = to_int(extract_kv(page_text, "Year of Manufacture")) or to_int(extract_kv(page_text, "Year"))
    transmission = extract_kv(page_text, "Transmission")
    fuel_type = extract_kv(page_text, "Fuel type")
    engine_capacity_cc = to_int(extract_kv(page_text, "Engine capacity"))
    mileage_km = to_int(extract_kv(page_text, "Mileage"))

    # ✅ Location extracted from URL (reliable)
    location = extract_location_from_url(ad_url)

    # title: try HTML title tag first, else fallback
    title = None
    if soup.title:
        title = clean(soup.title.get_text())
    if not title:
        title = None

    return {
        "url": ad_url,
        "title_text": title,
        "price_lkr": price_lkr,
        "brand": brand,
        "model": model,
        "year": year,
        "mileage_km": mileage_km,
        "fuel_type": fuel_type,
        "engine_capacity_cc": engine_capacity_cc,
        "transmission": transmission,
        "location": location
    }

# ---------- main ----------
def main():
    PAGES = 35  # start with 10 pages (~200-ish ads). Increase to 20–30 if needed.
    all_urls = []
    for p in range(1, PAGES + 1):
        print(f"Collecting URLs from list page {p}/{PAGES}")
        all_urls.extend(get_ad_urls_from_list_page(p))
        time.sleep(random.uniform(0.8, 1.5))

    # de-dup
    all_urls = list(dict.fromkeys(all_urls))
    print("Total unique ad URLs:", len(all_urls))

    rows = []
    for i, u in enumerate(all_urls, 1):
        print(f"[{i}/{len(all_urls)}] Fetching ad detail")
        row = parse_ad_detail(u)
        if row:
            rows.append(row)

        # polite delay
        time.sleep(random.uniform(1.0, 2.0))

        # save progress
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv("ikman_cars_dataset_partial.csv", index=False)
            print("Saved progress -> ikman_cars_dataset_partial.csv")

    df = pd.DataFrame(rows)
    df.to_csv("ikman_cars_dataset.csv", index=False)
    print("Done. Saved -> ikman_cars_dataset.csv")
    print(df.head())

if __name__ == "__main__":
    main()
