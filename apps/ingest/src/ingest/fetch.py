from pathlib import Path

import httpx

from ingest.documents import DocumentSource

CELLAR_URL_TEMPLATE = "http://publications.europa.eu/resource/celex/{celex_id}"

# Not a spoofed browser UA — this is a legitimate public machine-readable endpoint,
# not a scraping workaround (the eur-lex.europa.eu ELI URL is what's WAF-blocked for
# non-browser clients; CELLAR is the documented content-negotiation alternative).
_USER_AGENT = "ai-lab-ingest/0.1 (portfolio project; contact via repo)"

_HEADERS = {
    "Accept": "application/xhtml+xml",
    "Accept-Language": "eng",
    "User-Agent": _USER_AGENT,
}


def fetch_document_xhtml(
    source: DocumentSource, cache_dir: Path, *, force_refresh: bool = False
) -> str:
    """Fetches a document's XHTML from the CELLAR API, disk-caching the raw response
    so a resumed run never refetches a document it already has."""
    cache_path = cache_dir / f"{source.slug}.xhtml"

    if not force_refresh and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    url = CELLAR_URL_TEMPLATE.format(celex_id=source.celex_id)
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        response = client.get(url, headers=_HEADERS)
        response.raise_for_status()
        text = response.text

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text
