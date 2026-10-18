"""
Load real financial news headlines from FNSPID (Dong, Fan & Peng, 2024).

FNSPID ships two news files on Hugging Face (Zihan1004/FNSPID):
  - Stock_news/All_external.csv        (~5.7 GB, multiple outlets, headlines + URL)
  - Stock_news/nasdaq_exteral_data.csv (~23 GB, nasdaq.com articles incl. full text)

Only headline-level columns are read, in chunks, and filtered to the study
tickers and date window, so the full files never need to fit in memory.

Usage:
    python -m src.fnspid_loader              # download (if needed) + extract + coverage
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import (
    FNSPID_DIR,
    FNSPID_REPO,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    STUDY_END,
    STUDY_START,
    STUDY_TICKERS,
)

FNSPID_FILES = {
    "all_external": "Stock_news/All_external.csv",
    "nasdaq": "Stock_news/nasdaq_exteral_data.csv",
}
NEWS_COLUMNS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]
NEWS_PARQUET = PROCESSED_DATA_DIR / "fnspid_news.parquet"
COVERAGE_DIR = RESULTS_DIR / "data_coverage"


def download_fnspid(files: Optional[List[str]] = None) -> Dict[str, Path]:
    """Download FNSPID news files into data/raw/fnspid (skips files already present)."""
    from huggingface_hub import hf_hub_download

    paths = {}
    for key in files or list(FNSPID_FILES):
        local = FNSPID_DIR / FNSPID_FILES[key]
        if not local.exists():
            print(f"Downloading {FNSPID_FILES[key]} ...")
            hf_hub_download(
                FNSPID_REPO, FNSPID_FILES[key], repo_type="dataset", local_dir=FNSPID_DIR
            )
        paths[key] = local
    return paths


def source_from_url(url: str) -> str:
    """News outlet = URL domain without 'www.' (Publisher in FNSPID is often an author)."""
    try:
        host = urlparse(str(url)).netloc.lower()
    except ValueError:
        return "unknown"
    return host[4:] if host.startswith("www.") else (host or "unknown")


def extract_ticker_news(
    tickers: List[str] = STUDY_TICKERS,
    start: str = STUDY_START,
    end: str = STUDY_END,
    files: Optional[List[str]] = None,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """
    Stream the FNSPID CSVs and keep headlines for the given tickers and window.

    Returns one row per unique (ticker, headline, publication time) with columns:
    published_utc, ticker, headline, url, publisher, source, fnspid_file
    """
    paths = download_fnspid(files)
    wanted = {t.upper() for t in tickers}
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)

    frames = []
    for key, path in paths.items():
        kept = scanned = 0
        reader = pd.read_csv(
            path,
            usecols=NEWS_COLUMNS,
            dtype=str,
            chunksize=chunksize,
            on_bad_lines="skip",
        )
        for chunk in reader:
            scanned += len(chunk)
            chunk = chunk[chunk["Stock_symbol"].str.upper().isin(wanted)]
            if chunk.empty:
                continue
            chunk = chunk.assign(
                published_utc=pd.to_datetime(chunk["Date"], utc=True, errors="coerce")
            )
            chunk = chunk[
                (chunk["published_utc"] >= start_ts) & (chunk["published_utc"] < end_ts)
            ]
            chunk = chunk.dropna(subset=["published_utc", "Article_title"])
            kept += len(chunk)
            frames.append(
                pd.DataFrame(
                    {
                        "published_utc": chunk["published_utc"],
                        "ticker": chunk["Stock_symbol"].str.upper(),
                        "headline": chunk["Article_title"].str.strip(),
                        "url": chunk["Url"],
                        "publisher": chunk["Publisher"],
                        "fnspid_file": key,
                    }
                )
            )
        print(f"[OK] {key}: scanned {scanned:,} rows, kept {kept:,}")

    news = pd.concat(frames, ignore_index=True)
    news["source"] = news["url"].map(source_from_url)

    before = len(news)
    # Same story syndicated twice (same URL, or same headline on the same day)
    news["_key"] = news["headline"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    news["_day"] = news["published_utc"].dt.date
    news = (
        news.sort_values("published_utc")
        .drop_duplicates(subset=["ticker", "url"])
        .drop_duplicates(subset=["ticker", "_key", "_day"])
        .drop(columns=["_key", "_day"])
        .reset_index(drop=True)
    )
    print(f"[OK] Removed {before - len(news):,} duplicate headlines; {len(news):,} remain")
    return news


def scan_all_ticker_coverage(
    start: str = STUDY_START, end: str = STUDY_END, chunksize: int = 500_000
) -> pd.DataFrame:
    """
    Headline count and days-with-news per ticker x year x file for EVERY ticker,
    used to choose study tickers with dense, overlapping news coverage.
    """
    paths = download_fnspid()
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)

    parts = []
    for key, path in paths.items():
        for chunk in pd.read_csv(path, usecols=["Date", "Stock_symbol"], dtype=str,
                                 chunksize=chunksize, on_bad_lines="skip"):
            ts = pd.to_datetime(chunk["Date"], utc=True, errors="coerce")
            mask = (ts >= start_ts) & (ts < end_ts)
            if mask.any():
                parts.append(pd.DataFrame({"ticker": chunk.loc[mask, "Stock_symbol"].str.upper(),
                                           "day": ts[mask].dt.date, "file": key}))
        print(f"[OK] scanned {key}")

    rows = pd.concat(parts, ignore_index=True)
    rows["year"] = pd.to_datetime(rows["day"]).dt.year
    coverage = (
        rows.groupby(["ticker", "year"])
        .agg(headlines=("day", "size"), days_with_news=("day", "nunique"))
        .reset_index()
    )
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(COVERAGE_DIR / "fnspid_all_tickers_by_year.csv", index=False)
    return coverage


def coverage_report(news: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Headline counts per ticker x year and per ticker x source."""
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    by_year = (
        news.assign(year=news["published_utc"].dt.year)
        .pivot_table(index="ticker", columns="year", values="headline",
                     aggfunc="count", fill_value=0)
    )
    by_source = (
        news.pivot_table(index="source", columns="ticker", values="headline",
                         aggfunc="count", fill_value=0)
        .assign(total=lambda d: d.sum(axis=1))
        .sort_values("total", ascending=False)
    )
    by_year.to_csv(COVERAGE_DIR / "headlines_by_ticker_year.csv")
    by_source.to_csv(COVERAGE_DIR / "headlines_by_source_ticker.csv")
    return {"by_year": by_year, "by_source": by_source}


def load_news(rebuild: bool = False) -> pd.DataFrame:
    """Load the extracted headline table, building it from FNSPID if needed."""
    if NEWS_PARQUET.exists() and not rebuild:
        return pd.read_parquet(NEWS_PARQUET)
    news = extract_ticker_news()
    NEWS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    news.to_parquet(NEWS_PARQUET, index=False)
    print(f"[OK] Saved {len(news):,} headlines to {NEWS_PARQUET}")
    return news


if __name__ == "__main__":
    news = load_news(rebuild=True)
    report = coverage_report(news)
    print("\nHeadlines per ticker per year:")
    print(report["by_year"].to_string())
    print("\nTop sources:")
    print(report["by_source"].head(15).to_string())
