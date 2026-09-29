"""
CodeAlpha - Data Analytics Internship
TASK 1: Web Scraping

Scrapes book data (title, price, rating, availability, category, link) from
https://books.toscrape.com (a website built for practicing scraping) and
saves it as a clean CSV dataset.

Install:  pip install requests beautifulsoup4 pandas
Run:      python task1_web_scraping.py
"""

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
SITE_ROOT = "https://books.toscrape.com/catalogue/"
MAX_PAGES = 50            # the site has 50 pages (1000 books). Lower it for a quick test.
FETCH_CATEGORY = False    # True = also open each book page for its category (slower: 1000 extra requests)
DELAY = 0.5               # seconds between requests (be polite to the server)
HEADERS = {"User-Agent": "Mozilla/5.0 (CodeAlpha Internship Scraper)"}

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_soup(url):
    """Download a page and return a BeautifulSoup object (None on failure)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [!] Failed to fetch {url}: {e}")
        return None


def get_category(book_url):
    """Open a single book page and read its category from the breadcrumb."""
    soup = get_soup(book_url)
    if soup is None:
        return None
    crumbs = soup.select("ul.breadcrumb li a")
    return crumbs[2].text.strip() if len(crumbs) >= 3 else None


def parse_page(soup):
    """Extract all books from one listing page."""
    books = []
    for article in soup.select("article.product_pod"):
        title = article.h3.a["title"]
        link = urljoin(SITE_ROOT, article.h3.a["href"])
        price = float(article.select_one("p.price_color").text.replace("£", "").replace("Â", "").strip())
        rating_word = article.select_one("p.star-rating")["class"][1]
        availability = article.select_one("p.instock.availability").text.strip()

        books.append({
            "title": title,
            "price_gbp": price,
            "rating": RATING_MAP.get(rating_word),
            "availability": availability,
            "url": link,
        })
    return books


def main():
    all_books = []
    for page in range(1, MAX_PAGES + 1):
        url = BASE_URL.format(page)
        print(f"Scraping page {page}/{MAX_PAGES}: {url}")
        soup = get_soup(url)
        if soup is None:
            break
        page_books = parse_page(soup)
        if not page_books:
            break
        all_books.extend(page_books)
        time.sleep(DELAY)

    df = pd.DataFrame(all_books)

    if FETCH_CATEGORY and not df.empty:
        print("Fetching categories (this takes a while)...")
        cats = []
        for i, link in enumerate(df["url"], 1):
            cats.append(get_category(link))
            if i % 50 == 0:
                print(f"  {i}/{len(df)} done")
            time.sleep(DELAY / 2)
        df["category"] = cats

    # Basic cleaning
    df.drop_duplicates(subset="url", inplace=True)
    df.dropna(subset=["title", "price_gbp"], inplace=True)

    df.to_csv("books_dataset.csv", index=False, encoding="utf-8")
    print(f"\nDone! Scraped {len(df)} books -> books_dataset.csv")
    print(df.head())
    print("\nSummary:")
    print(df.describe(include="all").T)


if __name__ == "__main__":
    main()
