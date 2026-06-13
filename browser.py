from __future__ import annotations

import asyncio
import random
import time
from typing import Optional

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
)

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


class BrowserManager:
    """Singleton-style manager for a single Playwright browser instance."""

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch Playwright, a headless Chromium browser, and a browser context."""
        async with self._lock:
            if self._playwright is not None:
                return  # Already started

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._context = await self._browser.new_context(
                user_agent=random.choice(_USER_AGENTS),
                viewport={"width": 1280, "height": 800},
                java_script_enabled=True,
            )

    async def stop(self) -> None:
        """Tear down context, browser, and Playwright in order."""
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
        """Open a new page with a freshly-rotated user agent."""
        if self._context is None:
            raise RuntimeError("BrowserManager has not been started. Call await start() first.")
        page = await self._context.new_page()
        await page.set_extra_http_headers(
            {"User-Agent": random.choice(_USER_AGENTS)}
        )
        return page

    @staticmethod
    async def _polite_delay() -> None:
        """Sleep for a random polite interval to avoid hammering servers."""
        delay = random.uniform(_POLITE_DELAY_MIN, _POLITE_DELAY_MAX)
        await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Public navigation API
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> tuple[str, str]:
        """Navigate to *url* and return ``(body_inner_text, final_url)``.

        Parameters
        ----------
        url:
            The URL to load.

        Returns
        -------
        tuple[str, str]
            A 2-tuple of ``(page_content_as_text, final_url)`` where
            *page_content_as_text* is the ``innerText`` of the ``<body>``
            element and *final_url* is the URL after any redirects.

        Raises
        ------
        RuntimeError
            If navigation times out or another Playwright error occurs.
        """
        page = await self._new_page()
        try:
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=_NAVIGATION_TIMEOUT_MS,
            )
            if response is None:
                raise RuntimeError(f"Navigation to '{url}' returned no response.")

            # Wait for the page to settle without requiring full networkidle.
            # Catches JS-rendered content while tolerating persistent background XHRs.
            try:
                await page.wait_for_load_state("networkidle", timeout=8_000)
            except PlaywrightTimeoutError:
                pass  # partial settle is fine — body content is already loaded

            await self._polite_delay()

            final_url: str = page.url
            body_text: str = await page.inner_text("body")
            return body_text, final_url

        except PlaywrightTimeoutError as exc:
            raise RuntimeError(
                f"Navigation to '{url}' timed out after {_NAVIGATION_TIMEOUT_MS / 1000:.0f}s."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"An error occurred while navigating to '{url}': {exc}"
            ) from exc
        finally:
            await page.close()

    async def get_raw_html(self, url: str) -> str:
        """Fetch *url* and return the full page HTML after JS rendering."""
        page = await self._new_page()
        try:
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=_NAVIGATION_TIMEOUT_MS,
            )
            if response is None:
                raise RuntimeError(f"Navigation to '{url}' returned no response.")
            try:
                await page.wait_for_load_state("networkidle", timeout=8_000)
            except PlaywrightTimeoutError:
                pass
            await self._polite_delay()
            return await page.content()
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(
                f"Navigation to '{url}' timed out after {_NAVIGATION_TIMEOUT_MS / 1000:.0f}s."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"An error occurred while fetching HTML from '{url}': {exc}"
            ) from exc
        finally:
            await page.close()

    async def get_markdown_content(self, url: str) -> str:
        """Fetch *url* and return a clean Markdown representation of its content.

        Scripts, styles, and other non-content tags are stripped before the
        HTML is converted to Markdown via ``markdownify``.

        Parameters
        ----------
        url:
            The URL to fetch.

        Returns
        -------
        str
            Markdown text derived from the page's HTML body.

        Raises
        ------
        RuntimeError
            If navigation times out or another Playwright error occurs.
        """
        page = await self._new_page()
        try:
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=_NAVIGATION_TIMEOUT_MS,
            )
            if response is None:
                raise RuntimeError(f"Navigation to '{url}' returned no response.")

            try:
                await page.wait_for_load_state("networkidle", timeout=8_000)
            except PlaywrightTimeoutError:
                pass

            await self._polite_delay()

            html: str = await page.content()

        except PlaywrightTimeoutError as exc:
            raise RuntimeError(
                f"Navigation to '{url}' timed out after {_NAVIGATION_TIMEOUT_MS / 1000:.0f}s."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"An error occurred while fetching markdown from '{url}': {exc}"
            ) from exc
        finally:
            await page.close()

        # Strip noise tags before converting to Markdown
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe", "head"]):
            tag.decompose()

        clean_html = str(soup)
        markdown_text: str = md(
            clean_html,
            heading_style="ATX",
            bullets="-",
            strip=["a"],
        )

        # Collapse excessive blank lines introduced by markdownify
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
