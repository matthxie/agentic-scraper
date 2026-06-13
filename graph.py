"""
graph.py — LangGraph orchestration for the Safco Dental agentic web scraper.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiosqlite
import instructor
import openai
from dotenv import load_dotenv

load_dotenv()
from bs4 import BeautifulSoup
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import RetryPolicy

from browser import browser_manager, PermanentBrowserError, TransientBrowserError
from schemas import (
    PageClassification,
    ProductData,
    ProductVariation,
    ScraperState,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = "scraper_cache.db"
JSONL_PATH = "products.jsonl"
MODEL = "gpt-4o-mini"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# instructor-wrapped async OpenAI client (module-level singleton)
_openai_client = openai.AsyncOpenAI()
_instructor_client = instructor.from_openai(_openai_client)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """Create the SQLite tables if they do not already exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS url_queue (
                url          TEXT PRIMARY KEY,
                url_type     TEXT,
                status       TEXT,
                last_updated TEXT
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS scraped_products (
                sku              TEXT PRIMARY KEY,
                product_name     TEXT,
                raw_json_payload TEXT,
                timestamp        TEXT
            )
            """
        )
        await db.commit()
    logger.info("Database initialised at %s", DB_PATH)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


async def upsert_url(url: str, url_type: Optional[str], status: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        # INSERT OR IGNORE: if the URL is already queued (any status), leave it alone.
        # This prevents a URL discovered via two category pages from being re-scraped
        # after it's already been marked completed.
        await db.execute(
            """
            INSERT OR IGNORE INTO url_queue (url, url_type, status, last_updated)
            VALUES (?, ?, ?, ?)
            """,
            (url, url_type, status, _now_iso()),
        )
        await db.commit()


async def _reset_stale_processing() -> None:
    """Reset any URLs left in 'processing' from a previous crashed run."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE url_queue SET status = 'pending' WHERE status = 'processing'"
        )
        await db.commit()


async def mark_url_status(url: str, status: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE url_queue SET status = ?, last_updated = ? WHERE url = ?",
            (status, _now_iso(), url),
        )
        await db.commit()


async def pull_next_pending() -> Optional[str]:
    """Return and atomically mark the next pending URL as 'processing'.

    Categories are drained before products so the full URL tree is discovered
    before individual product pages are visited.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """
            SELECT url FROM url_queue
            WHERE status = 'pending'
            ORDER BY
                CASE url_type WHEN 'category' THEN 0 ELSE 1 END,
                rowid
            LIMIT 1
            """
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        url = row[0]
        await db.execute(
            "UPDATE url_queue SET status = 'processing', last_updated = ? WHERE url = ?",
            (_now_iso(), url),
        )
        await db.commit()
    return url


async def upsert_product(sku: str, product_name: str, raw: Dict[str, Any]) -> bool:
    """Insert or update a product SKU. Returns True if this was a new insert."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT 1 FROM scraped_products WHERE sku = ?", (sku,)
        ) as cursor:
            already_exists = await cursor.fetchone() is not None
        await db.execute(
            """
            INSERT INTO scraped_products (sku, product_name, raw_json_payload, timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(sku) DO UPDATE SET
                product_name     = excluded.product_name,
                raw_json_payload = excluded.raw_json_payload,
                timestamp        = excluded.timestamp
            """,
            (sku, product_name, json.dumps(raw), _now_iso()),
        )
        await db.commit()
    return not already_exists


def _append_jsonl(data: Dict[str, Any]) -> None:
    with open(JSONL_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(data, ensure_ascii=False) + "\n")


def _count_jsonl_lines() -> int:
    try:
        with open(JSONL_PATH, "rb") as fh:
            return sum(1 for line in fh if line.strip())
    except FileNotFoundError:
        return 0


# ---------------------------------------------------------------------------
# Node 1 — classify_page
# ---------------------------------------------------------------------------


async def classify_page(state: ScraperState) -> ScraperState:
    """Navigate to the current URL, retrieve content, and classify the page."""
    logger.info("classify_page: %s", state.current_url)

    try:
        body_text, _ = await browser_manager.navigate(state.current_url)
        state = state.model_copy(update={"html_content": body_text})
    except PermanentBrowserError as exc:
        logger.error("classify_page permanent failure: %s", exc)
        return state.model_copy(update={"page_type": "blocked"})
    except TransientBrowserError:
        raise  # let RetryPolicy handle it

    # Truncate content to avoid excessive token usage
    content_for_llm = (state.html_content or "")[:12_000]

    try:
        classification: PageClassification = await _instructor_client.chat.completions.create(
            model=MODEL,
            response_model=PageClassification,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a web page classifier for Safco Dental's website. "
                        "Category pages contain product listing grids, filters, and pagination. "
                        "Product pages contain SKU tables, 'Add to Cart' buttons, price tables, "
                        "and product variation selectors. "
                        "Blocked pages show captchas or access denied. "
                        "Classify the given page content."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Page URL: {state.current_url}\n\nPage content:\n{content_for_llm}",
                },
            ],
        )
        logger.info("classify_page result: %s", classification.page_type)
        return state.model_copy(update={"page_type": classification.page_type})
    except Exception as exc:
        logger.error("classify_page LLM call failed: %s", exc)
        return state.model_copy(update={"page_type": "unknown"})


# ---------------------------------------------------------------------------
# Node 2 — navigate_category
# ---------------------------------------------------------------------------

_PRODUCT_URL_RE = re.compile(r"/product/", re.IGNORECASE)
_root_url: str = ""


def _extract_jsonld_products(html: str) -> List[str]:
    """Pull product URLs from JSON-LD ItemList scripts embedded in the page."""
    soup = BeautifulSoup(html, "html.parser")
    urls: List[str] = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            if not isinstance(data, dict):
                continue
            if data.get("@type") == "ItemList":
                for item in data.get("itemListElement", []):
                    url = item.get("url") or (item.get("item") or {}).get("@id")
                    if url and "/product/" in url:
                        urls.append(url)
        except (json.JSONDecodeError, AttributeError):
            continue
    return urls


def _extract_subcategory_links(html: str, current_url: str) -> List[str]:
    """Return sub-category links found on the page.

    In restricted mode (_root_url is set) only links within the root subtree
    are returned. In unrestricted mode all same-host category links are returned.
    """
    from urllib.parse import urlparse
    soup = BeautifulSoup(html, "html.parser")
    root_path = urlparse(_root_url).path.rstrip("/") if _root_url else None
    current_host = urlparse(current_url).netloc
    links: List[str] = []
    for tag in soup.find_all("a", href=True):
        full = _resolve_url(current_url, tag["href"])
        parsed = urlparse(full)
        if parsed.netloc != current_host:
            continue
        if _PRODUCT_URL_RE.search(parsed.path):
            continue
        if root_path is not None and not parsed.path.rstrip("/").startswith(root_path + "/"):
            continue
        links.append(full)
    return list(set(links))


def _next_page_url(current_url: str, html: str) -> Optional[str]:
    """Derive the next pagination URL from the current URL or HTML links."""
    soup = BeautifulSoup(html, "html.parser")

    # 1. Standard rel=next / class-based selectors
    for selector in ['a[rel="next"]', ".next-page a", "a.pagination-next", "a.action.next"]:
        tag = soup.select_one(selector)
        if tag and tag.get("href"):
            return _resolve_url(current_url, tag["href"])

    # 2. Algolia pagination (ais-Pagination-link) or generic text "Next Page"
    for a in soup.find_all("a"):
        text = a.get_text(strip=True).lower()
        if "next" in text and a.get("href") and "page=" in a["href"]:
            resolved = _resolve_url(current_url, a["href"])
            # Sanity check: must actually advance beyond the current page
            if resolved != current_url:
                return resolved

    return None


async def navigate_category(state: ScraperState) -> ScraperState:
    """Extract product and sub-category links from a category page."""
    logger.info("navigate_category: %s", state.current_url)

    # Fetch raw HTML (html_content holds inner text from classify; we need HTML for JSON-LD)
    try:
        raw_html = await browser_manager.get_raw_html(state.current_url)
    except PermanentBrowserError as exc:
        logger.error("navigate_category permanent failure: %s", exc)
        return state
    except TransientBrowserError:
        raise  # let RetryPolicy handle it

    # 1. Product links via JSON-LD ItemList (primary)
    product_links = _extract_jsonld_products(raw_html)

    # 2. Sub-category links (queue as category pages)
    subcat_links = _extract_subcategory_links(raw_html, state.current_url)

    # 3. Pagination — only follow if this page actually had products; an empty
    #    page means we've gone past the last real page.
    next_url = _next_page_url(state.current_url, raw_html) if product_links else None

    # If JSON-LD gave nothing, fall back to regex over all <a> hrefs
    if not product_links:
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup.find_all("a", href=True):
            href: str = tag["href"]
            if _PRODUCT_URL_RE.search(href):
                full_url = _resolve_url(state.current_url, href)
                if full_url not in product_links:
                    product_links.append(full_url)

    # Persist to queue
    for url in product_links:
        await upsert_url(url, "product", "pending")
    for url in subcat_links:
        await upsert_url(url, "category", "pending")
    if next_url:
        await upsert_url(next_url, "category", "pending")

    all_discovered = set(state.discovered_urls) | set(product_links) | set(subcat_links)
    if next_url:
        all_discovered.add(next_url)

    logger.info(
        "navigate_category: %d products, %d subcats, next_url=%s",
        len(product_links), len(subcat_links), next_url,
    )

    return state.model_copy(update={"discovered_urls": list(all_discovered)})


def _resolve_url(base: str, href: str) -> str:
    from urllib.parse import urljoin
    return urljoin(base, href)


# ---------------------------------------------------------------------------
# Node 3 — extract_product
# ---------------------------------------------------------------------------


async def extract_product(state: ScraperState) -> ScraperState:
    """Extract structured product data from a product page."""
    logger.info("extract_product: %s", state.current_url)

    html = state.html_content or ""
    soup = BeautifulSoup(html, "html.parser")

    extracted = _bs4_extract(soup, state.current_url)

    # Fall back to markdown + LLM if name or variations are missing
    if not extracted.get("product_name") or not extracted.get("variations"):
        logger.info("extract_product: BS4 extraction incomplete, falling back to LLM")
        extracted = await _llm_extract(state.current_url)

    return state.model_copy(update={"extracted_data": extracted})


def _bs4_extract(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {"product_url": url}

    # product_name
    for selector in ["h1.product-title", "h1"]:
        tag = soup.select_one(selector)
        if tag:
            data["product_name"] = tag.get_text(strip=True)
            break

    # brand
    for selector in ["[data-brand]", ".brand-name"]:
        tag = soup.select_one(selector)
        if tag:
            data["brand"] = tag.get("data-brand") or tag.get_text(strip=True)
            break

    # breadcrumbs
    crumbs: List[str] = []
    for selector in [".breadcrumb a", "nav[aria-label='breadcrumb'] a"]:
        tags = soup.select(selector)
        if tags:
            crumbs = [t.get_text(strip=True) for t in tags]
            break
    data["category_hierarchy"] = crumbs

    # SKU — prefer data attribute, then class, then table
    sku: Optional[str] = None
    for selector in ["[data-sku]", ".sku-value"]:
        tag = soup.select_one(selector)
        if tag:
            sku = tag.get("data-sku") or tag.get_text(strip=True)
            break
    if not sku:
        for row in soup.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) >= 2 and "sku" in cells[0].get_text(strip=True).lower():
                sku = cells[1].get_text(strip=True)
                break

    # price
    price: Optional[float] = None
    for selector in [".price", "[data-price]"]:
        tag = soup.select_one(selector)
        if tag:
            raw = tag.get("data-price") or tag.get_text(strip=True)
            price = _parse_price(raw)
            break

    if sku:
        data["variations"] = [{"sku": sku, "price": price, "in_stock": True}]
    else:
        data["variations"] = []

    return data


def _parse_price(raw: str) -> Optional[float]:
    cleaned = re.sub(r"[^\d.]", "", raw)
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


async def _llm_extract(url: str) -> Dict[str, Any]:
    try:
        markdown = await browser_manager.get_markdown_content(url)
        product: ProductData = await _instructor_client.chat.completions.create(
            model=MODEL,
            response_model=ProductData,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a product data extraction assistant for Safco Dental's website. "
                        "Extract structured product information from the provided page content. "
                        "Include all SKUs, prices, and package sizes found in variation tables."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Product page URL: {url}\n\nPage content (markdown):\n{markdown[:14_000]}"
                    ),
                },
            ],
        )
        return product.model_dump()
    except PermanentBrowserError as exc:
        logger.error("_llm_extract permanent failure for %s: %s", url, exc)
        return {"product_url": url, "product_name": None, "variations": []}
    except TransientBrowserError as exc:
        logger.warning("_llm_extract transient failure for %s: %s", url, exc)
        return {"product_url": url, "product_name": None, "variations": []}
    except Exception as exc:
        logger.error("_llm_extract failed for %s: %s", url, exc)
        return {"product_url": url, "product_name": None, "variations": []}


# ---------------------------------------------------------------------------
# Node 4 — validate_and_store
# ---------------------------------------------------------------------------


async def validate_and_store(state: ScraperState) -> ScraperState:
    """Validate extracted product data and persist it."""
    logger.info("validate_and_store: %s", state.current_url)

    data = state.extracted_data or {}
    errors: List[str] = list(state.validation_errors)

    product_name = data.get("product_name")
    variations = data.get("variations", [])

    valid_skus = [v for v in variations if isinstance(v, dict) and v.get("sku")]
    if not valid_skus and isinstance(variations, list):
        # Support both dict and ProductVariation objects
        valid_skus = [
            v.model_dump() if hasattr(v, "model_dump") else v
            for v in variations
            if (v.sku if hasattr(v, "sku") else v.get("sku"))
        ]

    if not product_name:
        errors.append(f"Missing product_name for {state.current_url}")

    if not valid_skus:
        errors.append(f"No valid SKUs found for {state.current_url}")

    if product_name and valid_skus:
        # Persist each variation as its own row
        any_new_sku = False
        for variation in valid_skus:
            sku = variation.get("sku") if isinstance(variation, dict) else variation.sku
            try:
                is_new = await upsert_product(sku, product_name, data)
                any_new_sku = any_new_sku or is_new
            except Exception as exc:
                logger.error("upsert_product failed for sku=%s: %s", sku, exc)
                errors.append(f"DB write failed for SKU {sku}: {exc}")

        # Append to JSONL only if this is the first time we store any of these SKUs
        if any_new_sku:
            try:
                _append_jsonl(data)
            except Exception as exc:
                logger.error("JSONL write failed: %s", exc)

        await mark_url_status(state.current_url, "completed")
        logger.info(
            "validate_and_store: stored %d SKUs for '%s'",
            len(valid_skus),
            product_name,
        )
    else:
        await mark_url_status(state.current_url, "failed")
        logger.warning(
            "validate_and_store: validation failed for %s — %s",
            state.current_url,
            errors,
        )

    return state.model_copy(update={"validation_errors": errors})


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


def _route_after_classify(state: ScraperState) -> str:
    page_type = state.page_type
    if page_type == "category":
        return "navigate_category"
    elif page_type == "product":
        return "extract_product"
    else:
        return END


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------
#
# The graph runs ONCE PER URL. The run_scraper loop owns the queue:
#
#   category URL → classify → navigate_category (enqueues product/subcat URLs) → END
#   product  URL → classify → extract_product → validate_and_store → END
#
# navigate_category intentionally ends the graph — the outer loop pulls the
# next queued URL and re-enters the graph, eventually hitting product pages.


def _build_graph(checkpointer: Any = None) -> Any:
    builder = StateGraph(ScraperState)

    # Retry transient browser and OpenAI errors; give up on permanent failures.
    _transient_policy = RetryPolicy(
        max_attempts=3,
        initial_interval=2.0,
        backoff_factor=2.0,
        max_interval=30.0,
        retry_on=lambda exc: isinstance(exc, (TransientBrowserError, openai.APITimeoutError, openai.RateLimitError)),
    )

    builder.add_node("classify_page", classify_page, retry_policy=_transient_policy)
    builder.add_node("navigate_category", navigate_category, retry_policy=_transient_policy)
    builder.add_node("extract_product", extract_product, retry_policy=_transient_policy)
    builder.add_node("validate_and_store", validate_and_store)

    builder.set_entry_point("classify_page")

    builder.add_conditional_edges(
        "classify_page",
        _route_after_classify,
        {
            "navigate_category": "navigate_category",
            "extract_product": "extract_product",
            END: END,
        },
    )

    builder.add_edge("navigate_category", END)
    builder.add_edge("extract_product", "validate_and_store")
    builder.add_edge("validate_and_store", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------



async def _log_run_summary(initial_jsonl_lines: int) -> None:
    """Log a one-shot summary of what the run accomplished."""
    products_written = _count_jsonl_lines() - initial_jsonl_lines
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT status, COUNT(*) FROM url_queue GROUP BY status") as cur:
            counts = {row[0]: row[1] async for row in cur}
    completed = counts.get("completed", 0)
    failed = counts.get("failed", 0)
    pending = counts.get("pending", 0)
    logger.info(
        "Run complete — products written: %d | URLs completed: %d | failed: %d | still pending: %d",
        products_written, completed, failed, pending,
    )


async def run_scraper(
    start_url: str,
    max_products: Optional[int] = None,
    restricted: bool = True,
) -> None:
    """
    Initialise the database, seed the start URL, and process all queued URLs.

    Args:
        start_url:    The entry-point category URL to begin crawling from.
        max_products: Stop after this many products written to JSONL.
                      None means run until the queue is exhausted.
        restricted:   If True, subcategory discovery is limited to URLs that
                      fall under start_url's path. If False, the scraper follows
                      any category link found on the page.
    """
    global _root_url
    _root_url = start_url if restricted else ""

    await init_db()
    await _reset_stale_processing()
    await browser_manager.start()
    await upsert_url(start_url, "category", "pending")

    initial_jsonl_lines = _count_jsonl_lines()
    logger.info("Starting scraper from: %s (max_products=%s)", start_url, max_products)

    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        graph = _build_graph(checkpointer=checkpointer)

        try:
            while True:
                if max_products is not None:
                    written = _count_jsonl_lines() - initial_jsonl_lines
                    if written >= max_products:
                        logger.info(
                            "Reached max_products limit (%d/%d) — scraper finished.",
                            written, max_products,
                        )
                        break

                url = await pull_next_pending()
                if url is None:
                    logger.info("URL queue exhausted — scraper finished.")
                    break

                logger.info("Processing URL: %s", url)
                config = {"configurable": {"thread_id": url}}
                initial_state = ScraperState(current_url=url)

                try:
                    await graph.ainvoke(initial_state, config=config)
                    await mark_url_status(url, "completed")
                    logger.debug("Graph completed for %s", url)
                except Exception as exc:
                    logger.error("Graph execution failed for %s: %s", url, exc)
                    await mark_url_status(url, "failed")
        finally:
            await _log_run_summary(initial_jsonl_lines)
            await browser_manager.stop()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Safco Dental agentic scraper")
    parser.add_argument("start_url", nargs="?", default="https://www.safcodental.com/catalog")
    parser.add_argument("--max-products", type=int, default=None)
    parser.add_argument("--unrestricted", action="store_true", help="Follow category links outside the start URL's subtree")
    args = parser.parse_args()

    asyncio.run(run_scraper(args.start_url, max_products=args.max_products, restricted=not args.unrestricted))
