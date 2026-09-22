"""
Debug script — prints what Playwright actually sees on the page.

Run from the LangGraphLearning directory:
    .venv/bin/python tests/debug_fetch.py
"""

import sys
sys.path.insert(0, ".")

from playwright.sync_api import sync_playwright

URL = "https://legislation.nsw.gov.au/view/whole/html/inforce/2025-05-16/sl-2023-0139"

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("Navigating...")
        page.goto(URL, wait_until="domcontentloaded", timeout=30000)

        # Wait a bit more in case JS renders content after domcontentloaded
        page.wait_for_timeout(3000)

        title = page.title()
        print(f"Page title: {title}\n")

        html_len = len(page.content())
        print(f"HTML length: {html_len} chars\n")

        # Try our selector
        found = page.query_selector_all('[id^="sch.1-sec-oc"] span.frag-heading')
        print(f'Selector [id^="sch.1-sec-oc"] span.frag-heading → {len(found)} elements\n')

        # Check if any sch.1-sec-oc elements exist at all
        oc_divs = page.query_selector_all('[id^="sch.1-sec-oc"]')
        print(f'Elements with id starting "sch.1-sec-oc" → {len(oc_divs)}\n')

        # Check for frag-heading anywhere
        frag_headings = page.query_selector_all('span.frag-heading')
        print(f'Any span.frag-heading on page → {len(frag_headings)}')
        for el in frag_headings[:5]:
            print(f'  "{el.inner_text().strip()}"')

        # Print first 500 chars of body text so we know what the page shows
        body_text = page.inner_text("body")
        print(f'\nFirst 300 chars of body text:\n{body_text[:300]}')

        browser.close()

if __name__ == "__main__":
    main()
