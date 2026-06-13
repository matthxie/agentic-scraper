# Agentic Product Scraper

A resilient, state-driven web scraper that crawls e-commerce category pages and extracts structured product data using a combination of deterministic HTML parsing and LLM-based extraction. Built with LangGraph, Playwright, and GPT-4o-mini via the Instructor library.

---

## Architecture Overview

```
[run_scraper loop]
      │
      ▼
[SQLite URL Queue] ──pulls pending URL──▶ [LangGraph Graph]
      │                                          │
      │                              ┌───────────▼───────────┐
      │                              │     classify_page      │
      │                              │  Playwright + GPT-4o  │
      │                              └───────────┬───────────┘
      │                                          │
      │                              ┌───────────▼──────────────┐
      │                         "category"    "product"    other
      │                              │            │           │
      │                    ┌─────────▼──┐  ┌──────▼───────┐  END
      │                    │ navigate_  │  │ extract_     │
      │                    │ category   │  │ product      │
      │                    │ (HTML +    │  │ (BS4 → LLM   │
      │                    │  JSON-LD)  │  │  fallback)   │
      │                    └─────────┬──┘  └──────┬───────┘
      │                              │             │
      │              queues subcats  │      ┌──────▼───────┐
      └◀─────────────& products ─────┘      │ validate_and │
                     into DB                │ _store       │
                                            └──────┬───────┘
                                                   │
                                          products.jsonl + SQLite
```

The graph runs **once per URL**. The outer loop in `run_scraper` owns the queue and drives execution. Category URLs enqueue their discovered products and subcategories into the DB, then end. Product URLs run through extraction and validation. This separation keeps the graph simple and makes every URL independently resumable.

---

## Why This Approach

**Code-first, LLM-as-fallback.** Deterministic HTML parsing (BeautifulSoup + JSON-LD) handles the common case cheaply. The LLM is only invoked when structural extraction fails or for tasks that genuinely require language understanding (page classification, irregular product layouts). This keeps cost and latency low at scale.

**SQLite as the queue.** A persistent queue means the scraper can be killed and resumed at any point without losing progress. Every URL is checkpointed the moment it's discovered. `INSERT OR IGNORE` prevents duplicates across overlapping category pages.

**LangGraph for orchestration.** The graph makes control flow explicit and auditable. LangGraph's built-in `RetryPolicy` and `AsyncSqliteSaver` checkpointer provide node-level retry handling and mid-graph crash recovery without custom infrastructure.

**Separation of transient and permanent failures.** `TransientBrowserError` (timeouts, 429s) triggers retry with exponential backoff. `PermanentBrowserError` (404, 403) fails fast. This avoids wasting retries on dead URLs while tolerating recoverable network conditions.

---

## Agent Responsibilities

### `classify_page`
Navigates to a URL using Playwright and calls GPT-4o-mini to classify it as `category`, `product`, `blocked`, or `unknown` based on page content. Routes the graph accordingly.

### `navigate_category`
Fetches the raw JS-rendered HTML of a category page. Extracts product URLs from the embedded JSON-LD `ItemList` schema (primary) with a regex fallback. Extracts subcategory links filtered to the root URL subtree (in restricted mode). Detects and queues the next pagination page. All discovered URLs are written to the SQLite queue.

### `extract_product`
Attempts structured extraction via BeautifulSoup selectors first (product name, brand, breadcrumbs, SKU, price). Falls back to converting the page to Markdown and passing it to GPT-4o-mini via Instructor to parse a typed `ProductData` schema if required fields are missing.

### `validate_and_store`
Validates that extracted data has a product name and at least one SKU. On success, upserts each SKU to the `scraped_products` SQLite table and appends the product to `products.jsonl`. Only writes a new JSONL line if at least one SKU is genuinely new (prevents duplicates on resume).

---

## Setup & Execution

### Requirements

- Python 3.12+
- An OpenAI API key

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### Configure

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

### Run

```bash
# Scrape a specific category, stop after 50 products
python3 graph.py https://www.safcodental.com/catalog/sutures-surgical-products --max-products 50

# Scrape unrestricted (follow any category link on the same host)
python3 graph.py https://www.safcodental.com/catalog --unrestricted

# Resume a previous run (queue state is preserved in scraper_cache.db)
python3 graph.py https://www.safcodental.com/catalog/sutures-surgical-products
```

### CLI Arguments

| Argument | Description |
|---|---|
| `start_url` | Entry-point URL (default: safcodental.com/catalog) |
| `--max-products N` | Stop after N new products written to JSONL |
| `--unrestricted` | Follow category links outside the start URL's subtree |

### Output Files

| File | Description |
|---|---|
| `products.jsonl` | One JSON object per product, one per line |
| `scraper_cache.db` | SQLite queue tracking URL status |
| `checkpoints.db` | LangGraph node-level execution checkpoints |
| `scraper.log` | Full run log (also written to terminal) |

---

## Sample Output Schema

```json
{
  "product_name": "Ethicon Surgifoam",
  "brand": "Ethicon",
  "category_hierarchy": ["Dental Supplies", "Sutures & surgical products"],
  "product_url": "https://www.safcodental.com/product/ethicon-surgifoam",
  "description": "Absorbable gelatin sponge for surgical hemostasis.",
  "image_urls": ["https://www.safcodental.com/media/catalog/product/s/u/surgifoam.jpg"],
  "variations": [
    {
      "sku": "ETH1974",
      "price": 214.49,
      "package_size": "Box of 24",
      "in_stock": true
    },
    {
      "sku": "ETH1975",
      "price": 189.99,
      "package_size": "Box of 12",
      "in_stock": true
    }
  ]
}
```

---

## Limitations

- **JavaScript-rendered content.** The site renders product listings via Algolia/Alpine.js. Product card links are not in static HTML — they are discovered through JSON-LD `ItemList` schema embedded on category pages. Sites without JSON-LD require a different discovery strategy (e.g. clicking each card).
- **Single-threaded.** The scraper processes one URL at a time. A 56-product category with multiple subcategories takes several minutes due to browser navigation overhead and polite delays.
- **LLM cost on fallback.** Pages with non-standard layouts that the The BS4 extractor cannot cover fall back to the LLM which adds latency and cost per product.
- **Login-gated content.** Products requiring account login are not accessible. The scraper will classify these as `blocked` and skip them.
- **Restricted mode heuristic.** Subcategory URL filtering relies on path prefix matching. Sites with non-hierarchical URL structures (e.g. Magento ID-based category URLs) may not be fully crawled.

---

## Failure Handling

| Failure type | Behaviour |
|---|---|
| Browser timeout | Retried up to 3× in `browser.py` with exponential backoff, then 3× more at the LangGraph node level |
| HTTP 429 / 503 | `Retry-After` header respected; falls back to exponential backoff if absent |
| HTTP 404 / 403 | `PermanentBrowserError` raised immediately, URL marked `failed`, no retry |
| OpenAI timeout / rate limit | Caught by `RetryPolicy` on classify and extract nodes, retried up to 3× |
| Validation failure | URL marked `failed`, logged with specific missing fields, visible in `scraper.log` |
| Process crash mid-graph | `_reset_stale_processing` on next startup resets `processing` URLs to `pending`; LangGraph checkpoint resumes from last completed node |
| Duplicate URLs | `INSERT OR IGNORE` in the URL queue; `completed` URLs are never re-queued regardless of how many category pages link to them |

---

## Scaling to Full-Site Crawling in Production

The current architecture already handles resumability and deduplication correctly. The main changes needed for full-site scale:

1. **Concurrency.** Replace the single-threaded loop with a worker pool. Each worker gets its own Playwright browser context. A semaphore limits concurrent requests per domain to respect rate limits. LangGraph supports async parallel execution natively.

2. **Distributed queue.** Swap SQLite for Redis or a proper task queue (Celery, RQ, or a cloud queue like SQS). This allows multiple scraper instances to pull from the same queue without coordination logic.

3. **Proxy rotation.** A rotating residential proxy pool (Bright Data, Oxylabs) prevents IP-level rate limiting at scale. Route all Playwright traffic through it.

4. **Dedicated checkpoint storage.** Replace the SQLite checkpointer with LangGraph's Postgres checkpointer for concurrent access and durability.

5. **Scheduled re-crawls.** Product prices and stock change frequently. A scheduler (Airflow, cron, or LangGraph's built-in scheduling) triggers incremental re-crawls on a cadence, only re-processing URLs whose content has likely changed.

6. **Anti-detection hardening.** Add `playwright-stealth` to patch headless fingerprints, rotate browser contexts periodically, and simulate human-like interaction patterns.

---

## Monitoring Data Quality

1. **Field completeness tracking.** Log the percentage of extracted products missing each optional field (brand, description, image URLs, price) per run. A sudden drop in completeness signals a site layout change.

2. **Price sanity checks.** Flag variations where price is 0, null, or outside a reasonable range for the category. Store these in a separate `flagged_products` table for manual review.

3. **SKU format validation.** Each site has consistent SKU patterns. A regex check on extracted SKUs catches LLM hallucinations (e.g. a SKU that is actually a product name).

4. **Extraction method ratio.** Track what percentage of products fell back to the LLM extractor vs. the BS4 path. A rising LLM fallback rate indicates the deterministic selectors are breaking and need updating.

5. **Run-over-run diff.** Compare the current run's product count per category against the previous run. A large drop suggests the navigator is missing pages; a large increase suggests deduplication is broken.

6. **Structured log aggregation.** `scraper.log` writes structured records that can be ingested into any log aggregation tool (Datadog, Grafana Loki) for dashboarding error rates, throughput, and per-URL latency.
