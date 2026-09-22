"""
Tool: fetch_ministers_list

Fetches the Administrative Arrangements Order page and extracts
the list of minister names from Schedule 1.

Uses Playwright (headless browser) to bypass Cloudflare protection
on the NSW legislation website.
"""

from playwright.sync_api import sync_playwright


def fetch_ministers_list(url: str) -> list[str]:
    """Fetch the admin arrangements page and return all minister names in Schedule 1."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)

        # Minister names live in <span class="frag-heading"> inside each
        # Schedule 1 occurrence block: <div id="sch.1-sec-oc.*">
        minister_elements = page.query_selector_all(
            '[id^="sch.1-sec-oc"] span.frag-heading'
        )

        ministers = []
        for el in minister_elements:
            text = el.inner_text().strip()
            if text:
                ministers.append(text)

        browser.close()

    return ministers
