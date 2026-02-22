import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

LIST_URL = "https://ikman.lk/en/ads/sri-lanka/cars?page={}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

PAGES = 50
LIST_PAGE_DELAY_RANGE = (1.0, 2.0)
DETAIL_PAGE_DELAY_RANGE = (1.2, 2.4)
SAVE_EVERY = 10
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 1.5

PARTIAL_OUTPUT = Path("cars_dataset_partial.csv")
FINAL_OUTPUT = Path("cars_dataset_raw.csv")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def to_int(text: str | None):
    if not text:
        return None
    match = re.search(r"([\d,]+)", text)
    return int(match.group(1).replace(",", "")) if match else None


def extract_kv(page_text: str, key: str):
    match = re.search(rf"{re.escape(key)}\s*:\s*(.+)", page_text, flags=re.I)
    if not match:
        return None
    return clean(match.group(1).split("\n")[0])


def extract_location_from_url(url: str):
    match = re.search(r"for-sale-([a-z\-]+)-\d+$", url)
    if not match:
        return None
    location = match.group(1).replace("-", " ")
    return location.title()


def request_with_retry(session: requests.Session, url: str):
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, headers=HEADERS, timeout=30)
            if response.status_code == 200:
                return response

            if response.status_code in {403, 429, 500, 502, 503, 504}:
                sleep_seconds = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0.2, 0.8)
                print(
                    f"Request failed ({response.status_code}) for {url}. "
                    f"Retry {attempt}/{MAX_RETRIES} in {sleep_seconds:.1f}s"
                )
                time.sleep(sleep_seconds)
                continue

            print(f"Request failed ({response.status_code}) for {url}. No retry.")
            return None
        except requests.RequestException as error:
            last_exception = error
            sleep_seconds = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0.2, 0.8)
            print(
                f"Request exception for {url}: {error}. "
                f"Retry {attempt}/{MAX_RETRIES} in {sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)

    print(f"Giving up on {url}. Last error: {last_exception}")
    return None


def get_ad_urls_from_list_page(session: requests.Session, page_number: int):
    url = LIST_URL.format(page_number)
    response = request_with_retry(session, url)
    if response is None:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    urls = []
    for anchor in soup.select("a[href*='/ad/']"):
        href = anchor.get("href")
        if not href:
            continue
        full_url = "https://ikman.lk" + href if href.startswith("/") else href
        urls.append(full_url)

    unique_urls = list(dict.fromkeys(urls))
    return unique_urls


def parse_ad_detail(session: requests.Session, ad_url: str):
    response = request_with_retry(session, ad_url)
    if response is None:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    page_text = soup.get_text("\n", strip=True)

    price_match = re.search(r"Rs\.?\s*([\d,]+)", page_text)
    price_lkr = to_int(price_match.group(1)) if price_match else None

    brand = extract_kv(page_text, "Brand")
    model = extract_kv(page_text, "Model")
    year = to_int(extract_kv(page_text, "Year of Manufacture")) or to_int(extract_kv(page_text, "Year"))
    transmission = extract_kv(page_text, "Transmission")
    fuel_type = extract_kv(page_text, "Fuel type")
    engine_capacity_cc = to_int(extract_kv(page_text, "Engine capacity"))
    mileage_km = to_int(extract_kv(page_text, "Mileage"))
    location = extract_location_from_url(ad_url)

    title_text = clean(soup.title.get_text()) if soup.title else None

    return {
        "url": ad_url,
        "title_text": title_text,
        "price_lkr": price_lkr,
        "brand": brand,
        "model": model,
        "year": year,
        "mileage_km": mileage_km,
        "fuel_type": fuel_type,
        "engine_capacity_cc": engine_capacity_cc,
        "transmission": transmission,
        "location": location,
    }


def load_partial_rows(partial_file: Path):
    if not partial_file.exists():
        return []

    try:
        partial_df = pd.read_csv(partial_file, low_memory=False)
        rows = partial_df.to_dict("records")
        print(f"Loaded {len(rows):,} rows from partial file: {partial_file}")
        return rows
    except Exception as error:
        print(f"Could not read partial file {partial_file}: {error}")
        return []


def save_rows(rows: list[dict], output_file: Path):
    pd.DataFrame(rows).to_csv(output_file, index=False)


def main():
    print("Starting resilient Ikman car scraper...")
    session = requests.Session()

    rows = load_partial_rows(PARTIAL_OUTPUT)
    seen_urls = {row.get("url") for row in rows if row.get("url")}

    all_urls = []
    for page_number in range(1, PAGES + 1):
        print(f"Collecting URLs from list page {page_number}/{PAGES}")
        page_urls = get_ad_urls_from_list_page(session, page_number)
        all_urls.extend(page_urls)
        time.sleep(random.uniform(*LIST_PAGE_DELAY_RANGE))

    all_urls = list(dict.fromkeys(all_urls))
    pending_urls = [url for url in all_urls if url not in seen_urls]

    print(f"Total unique ad URLs discovered: {len(all_urls):,}")
    print(f"Already completed from partial file: {len(seen_urls):,}")
    print(f"Pending ad URLs to fetch: {len(pending_urls):,}")

    for index, ad_url in enumerate(pending_urls, start=1):
        print(f"[{index}/{len(pending_urls)}] Fetching ad detail")
        row = parse_ad_detail(session, ad_url)
        if row is not None:
            rows.append(row)
            seen_urls.add(ad_url)

        if index % SAVE_EVERY == 0:
            save_rows(rows, PARTIAL_OUTPUT)
            print(f"Checkpoint saved -> {PARTIAL_OUTPUT} ({len(rows):,} rows)")

        time.sleep(random.uniform(*DETAIL_PAGE_DELAY_RANGE))

    save_rows(rows, PARTIAL_OUTPUT)
    save_rows(rows, FINAL_OUTPUT)

    print(f"Done. Saved partial -> {PARTIAL_OUTPUT}")
    print(f"Done. Saved final -> {FINAL_OUTPUT}")
    print(f"Final row count: {len(rows):,}")


if __name__ == "__main__":
    main()
