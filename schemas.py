from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ProductVariation(BaseModel):
    sku: str
    price: Optional[float] = None
    package_size: Optional[str] = None
    in_stock: bool = True


class ProductData(BaseModel):
    product_name: str
    brand: Optional[str] = None
    category_hierarchy: List[str] = Field(default_factory=list)
    product_url: str
    description: Optional[str] = None
    image_urls: List[str] = Field(default_factory=list)
    variations: List[ProductVariation] = Field(..., min_length=1)


class ScraperState(BaseModel):
    current_url: str
    page_type: Optional[Literal["category", "product", "blocked", "unknown"]] = None
    html_content: Optional[str] = None
    discovered_urls: List[str] = Field(default_factory=list)
    extracted_data: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = Field(default_factory=list)
    retry_count: int = 0


class PageClassification(BaseModel):
    """
    LLM classifier output for a scraped page.

    page_type values:
      - "category": a listing or navigation page containing links to products or subcategories
      - "product": a detail page for a single product or product family
      - "blocked": the scraper was blocked (CAPTCHA, login wall, rate-limit, etc.)
      - "unknown": the page does not fit any of the above categories
    """

    page_type: Literal["category", "product", "blocked", "unknown"]


class URLQueueEntry(BaseModel):
    url: str
    url_type: Optional[str] = None
    status: Literal["pending", "processing", "completed", "failed"]
    last_updated: datetime
