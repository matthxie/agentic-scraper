from __future__ import annotations

import asyncio
import random
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Response,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
)


# ---------------------------------------------------------------------------
# Public exception types
# ---------------------------------------------------------------------------

class TransientBrowserError(Exception):
    """Raised for failures that are worth retrying (timeout, 429, 503)."""


class PermanentBrowserError(Exception):
    """Raised for failures that should not be retried (404, 403, hard block)."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

_POLITE_DELAY_MIN: float = 1.5
_POLITE_DELAY_MAX: float = 3.0
_NAVIGATION_TIMEOUT_MS: int = 60_000
_MAX_RETRIES: int = 3
_BACKOFF_BASE: float = 2.0   # seconds; doubles each attempt
_BACKOFF_MAX: float = 60.0   # ceiling

_TRANSIENT_STATUSES = {429, 503, 502, 504}
_PERMANENT_STATUSES = {403, 404, 410}


# ---------------------------------------------------------------------------
# BrowserManager
# ---------------------------------------------------------------------------

class BrowserManager:
    """Singleton-style manager for a single Playwright browser instance."""

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._lock = asyncio.Lock()
        # Per-hostname consecutive failure count for exponential backoff
        self._failure_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        async with self._lock:
            if self._playwright is not None:
                return
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._context = await self._browser.new_context(
                user_agent=random.choice(_USER_AGENTS),
                viewport={"width": 1280, "height": 800},
                java_script_enabled=True,
            )

    async def stop(self) -> None:
        async with self._lock:
            if self._context is not None:
                try:
                    await self._context.close()
                except Exception:
                    pass
                self._context = None
            if self._browser is not None:
                try:
                    await self._browser.close()
                except Exception:
                    pass
                self._browser = None
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _new_page(self) -> Page:
        if self._context is None:
            raise RuntimeError("BrowserManager has not been started. Call await start() first.")
        page = await self._context.new_page()
        await page.set_extra_http_headers({"User-Agent": random.choice(_USER_AGENTS)})
        return page

    @staticmethod
    async def _polite_delay() -> None:
        await asyncio.sleep(random.uniform(_POLITE_DELAY_MIN, _POLITE_DELAY_MAX))

    def _backoff_delay(self, hostname: str) -> float:
        n = self._failure_counts.get(hostname, 0)
        return min(_BACKOFF_BASE * (2 ** n) + random.uniform(0, 1), _BACKOFF_MAX)

    def _record_success(self, hostname: str) -> None:
        self._failure_counts.pop(hostname, None)

    def _record_failure(self, hostname: str) -> None:
        self._failure_counts[hostname] = self._failure_counts.get(hostname, 0) + 1

    async def _navigate_to(self, url: str) -> Page:
        """Navigate to *url* with retry/backoff. Returns an open Page on success.

        The caller is responsible for closing the page.

        Raises
        ------
        TransientBrowserError
            Retries exhausted on a recoverable failure (timeout, 429, 503 …).
        PermanentBrowserError
            Non-recoverable failure (404, 403, hard block).
        """
        hostname = urlparse(url).hostname or url
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(1, _MAX_RETRIES + 1):
            page = await self._new_page()
            try:
                response: Optional[Response] = await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=_NAVIGATION_TIMEOUT_MS,
                )

                if response is None:
                    raise TransientBrowserError(f"No response from '{url}'")

                status = response.status

                if status in _PERMANENT_STATUSES:
                    raise PermanentBrowserError(
                        f"HTTP {status} for '{url}' — will not retry"
                    )

                if status in _TRANSIENT_STATUSES:
                    retry_after = response.headers.get("retry-after")
                    wait = float(retry_after) if retry_after else self._backoff_delay(hostname)
                    self._record_failure(hostname)
                    await page.close()
                    raise TransientBrowserError(
                        f"HTTP {status} for '{url}' — backing off {wait:.1f}s"
                    )

                # Settle JS-rendered content; timeout here is non-fatal
                try:
                    await page.wait_for_load_state("networkidle", timeout=8_000)
                except PlaywrightTimeoutError:
                    pass

                await self._polite_delay()
                self._record_success(hostname)
                return page

            except PermanentBrowserError:
                await page.close()
                raise

            except asyncio.CancelledError:
                await page.close()
                raise

            except TransientBrowserError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = self._backoff_delay(hostname)
                    await asyncio.sleep(delay)

            except PlaywrightTimeoutError as exc:
                self._record_failure(hostname)
                last_exc = TransientBrowserError(
                    f"Timeout navigating to '{url}' (attempt {attempt}/{_MAX_RETRIES})"
                )
                await page.close()
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(self._backoff_delay(hostname))

            except Exception as exc:
                self._record_failure(hostname)
                last_exc = TransientBrowserError(
                    f"Unexpected error navigating to '{url}': {exc}"
                )
                await page.close()
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(self._backoff_delay(hostname))

        raise TransientBrowserError(
            f"All {_MAX_RETRIES} attempts failed for '{url}'"
        ) from last_exc

    # ------------------------------------------------------------------
    # Public navigation API
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> tuple[str, str]:
        """Return ``(body_inner_text, final_url)``."""
        page = await self._navigate_to(url)
        try:
            return await page.inner_text("body"), page.url
        finally:
            await page.close()

    async def get_raw_html(self, url: str) -> str:
        """Return the full JS-rendered HTML of *url*."""
        page = await self._navigate_to(url)
        try:
            return await page.content()
        finally:
            await page.close()

    async def get_markdown_content(self, url: str) -> str:
        """Return a clean Markdown representation of *url*."""
        page = await self._navigate_to(url)
        try:
            html = await page.content()
        finally:
            await page.close()

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe", "head"]):
            tag.decompose()

        markdown_text: str = md(
            str(soup),
            heading_style="ATX",
            bullets="-",
            strip=["a"],
        )

        lines = markdown_text.splitlines()
        deduplicated: list[str] = []
        blank_streak = 0
        for line in lines:
            if line.strip() == "":
                blank_streak += 1
                if blank_streak <= 2:
                    deduplicated.append(line)
            else:
                blank_streak = 0
                deduplicated.append(line)

        return "\n".join(deduplicated).strip()


# Module-level singleton
browser_manager = BrowserManager()
