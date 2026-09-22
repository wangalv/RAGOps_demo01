"""
LangGraph tools for fetching NSW legislation content.

Each function is decorated with @tool so it can be used both:
  - In M1/M2 pipeline nodes via tool.invoke({...})
  - In M3 ReAct agent via create_react_agent(llm, tools)
"""

from __future__ import annotations

from langchain_core.tools import tool
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from bs4 import BeautifulSoup


def _get_page_html(url: str) -> str:
    """Launch a stealth browser, navigate to url, return full page HTML."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            locale="en-AU",
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        Stealth().apply_stealth_sync(page)
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)
        html = page.content()
        browser.close()
    return html


# ── Tool 1 ────────────────────────────────────────────────────────────────────

@tool
def fetch_ministers_list(url: str) -> list[str]:
    """Fetch the NSW Administrative Arrangements Order and return all minister
    names listed in Schedule 1.

    Use this when you have an Administrative Arrangements Order URL and need
    to discover which ministers administer Acts in NSW.

    Args:
        url: Full URL of the Administrative Arrangements Order
             e.g. https://legislation.nsw.gov.au/view/whole/html/inforce/2025-05-16/sl-2023-0139

    Returns:
        List of minister title strings e.g. ["Premier", "Minister for Health", ...]
    """
    html = _get_page_html(url)
    soup = BeautifulSoup(html, "html.parser")

    ministers = []
    for span in soup.select('[id^="sch.1-sec-oc"] span.frag-heading'):
        text = span.get_text(strip=True)
        if text:
            ministers.append(text)
    return ministers


# ── Tool 2 ────────────────────────────────────────────────────────────────────

@tool
def fetch_acts_by_minister(url: str, minister: str) -> list[dict]:
    """Fetch all Acts administered by a specific minister from an Administrative
    Arrangements Order.

    Use this after fetch_ministers_list when the user has selected a minister
    and you need the list of Acts under that minister.

    Args:
        url:      Full URL of the Administrative Arrangements Order.
        minister: Exact minister title e.g. "Minister for Health"

    Returns:
        List of dicts: [{"title": "Health Services Act 1997", "url": "https://..."}]
    """
    html = _get_page_html(url)
    soup = BeautifulSoup(html, "html.parser")

    # Find the section occurrence block whose heading matches the minister
    target_block = None
    for block in soup.select('[id^="sch.1-sec-oc"]'):
        heading = block.select_one("span.frag-heading")
        if heading and heading.get_text(strip=True) == minister:
            target_block = block
            break

    if target_block is None:
        return []

    base = "https://legislation.nsw.gov.au"
    acts = []
    for link in target_block.select('a[href*="/view/html/inforce/current/"]'):
        title = link.get_text(strip=True)
        href = link.get("href", "")
        if title and href:
            full_url = href if href.startswith("http") else base + href
            acts.append({"title": title, "url": full_url})
    return acts


# ── Tool 3 ────────────────────────────────────────────────────────────────────

@tool
def fetch_act_html(url: str) -> dict:
    """Fetch the full plain text and metadata of a single NSW Act.

    Use this when you have a specific Act URL and need its content
    for extraction (obligations, penalties, cross-references).

    Args:
        url: Full URL of the Act e.g. https://legislation.nsw.gov.au/view/html/inforce/current/act-1997-154

    Returns:
        Dict with keys:
          - title: Act title
          - text:  Plain text content
          - url:   The URL that was fetched
    """
    html = _get_page_html(url)
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.select_one("h1, .act-title, title")
    title = title_el.get_text(strip=True) if title_el else ""

    # Remove nav, header, footer noise before extracting text
    for tag in soup.select("nav, header, footer, .toc, script, style"):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    return {"title": title, "text": text, "url": url}
