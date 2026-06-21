"""
TakeMeter — Hacker News comment collection.

Pulls public HN comments via the Algolia HN Search API (no auth), applies a
*conservative* automated pre-filter, dedupes, and writes a CSV for manual
annotation. Designed to run as a single Google Colab cell.

The automated filter intentionally only makes cheap, reliable cuts:
  - empty / deleted
  - ultra-short (< 15 words)
  - link-only
  - duplicate text
Nuanced junk (jokes, pleasantries, bare questions, snark) is left for the
human annotator to mark as label="junk" while reading — rule-based detection
of those is unreliable and would silently drop good data.
"""

import re
import html
import time
import requests
import pandas as pd

ALGOLIA = "https://hn.algolia.com/api/v1/search_by_date"
MIN_WORDS = 15            # ultra-short cutoff

# Topic focus: AI / programming. The query matches comment text and story
# title, so this pulls comments from AI/coding threads (consistent vocabulary,
# topic-coherent dataset) rather than the whole multi-topic HN firehose.
QUERIES = [
    "LLM", "GPT", "Claude", "Copilot", "machine learning",
    "programming", "software engineering", "compiler", "Python", "Rust",
]
PAGES_PER_QUERY = 2       # ~200 raw comments per query


def fetch_comments(queries=QUERIES, pages_per_query=PAGES_PER_QUERY, hits_per_page=100):
    """Pull recent HN comments from AI/programming threads, one query at a time."""
    hits = []
    for q in queries:
        for page in range(pages_per_query):
            r = requests.get(
                ALGOLIA,
                params={"tags": "comment", "query": q,
                        "hitsPerPage": hits_per_page, "page": page},
                timeout=30,
            )
            r.raise_for_status()
            hits.extend(r.json()["hits"])
            time.sleep(0.3)  # be polite to the API
    return hits


def clean_text(raw):
    if not raw:
        return ""
    t = re.sub(r"<[^>]+>", " ", raw)        # strip HTML tags
    t = html.unescape(t)                     # &gt; -> >, etc.
    t = re.sub(r"https?://\S+", "", t)       # strip inline URLs (noise for a text classifier)
    t = re.sub(r"\s+", " ", t).strip()       # collapse whitespace
    return t


def is_link_only(t):
    no_urls = re.sub(r"https?://\S+", "", t).strip()
    return len(no_urls.split()) < 5


def build_dataset(raw):
    seen, records = set(), []
    for h in raw:
        cid = h.get("objectID")
        text = clean_text(h.get("comment_text"))
        if not text or cid in seen:
            continue
        seen.add(cid)
        wc = len(text.split())
        if wc < MIN_WORDS or is_link_only(text):
            continue
        records.append({
            "text": text,
            "label": "",        # annotate: argument | hot_take | explainer | junk
            "notes": "",        # note for difficult/borderline cases
        })
    df = pd.DataFrame(records).drop_duplicates(subset="text").reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw = fetch_comments()
    df = build_dataset(raw)
    print(f"raw fetched: {len(raw)} | after clean + filter: {len(df)}")
    df.to_csv("hn_comments_to_annotate.csv", index=False)
    print("wrote hn_comments_to_annotate.csv")
