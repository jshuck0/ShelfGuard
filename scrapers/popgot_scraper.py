import re
import pandas as pd
from bs4 import BeautifulSoup
from utils.http_utils import build_session

SESSION = build_session()

def scrape_category(url):
    r = SESSION.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []

    for a in soup.select("a[href]"):
        href = a["href"]
        if not any(x in href for x in ["amazon.com", "walmart.com", "target.com"]):
            continue

        txt = a.parent.get_text(" ", strip=True)

        m_unit = re.search(r"(\d+(?:\.\d+)?)\s*¢", txt)
        m_price = re.search(r"\$(\d+(?:\.\d+)?)", txt)
        m_ct = re.search(r"(\d+)\s*ct", txt, re.I)

        rows.append({
            "product_name": a.get_text(strip=True),
            "unit_cents": float(m_unit.group(1)) if m_unit else None,
            "count": int(m_ct.group(1)) if m_ct else None,
            "total_price": float(m_price.group(1)) if m_price else None,
            "retailer_url": href
        })

    return pd.DataFrame(rows)
