from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentSource:
    """One legal instrument to ingest. `celex_id` drives the actual fetch (via the
    CELLAR content-negotiation API — see fetch.py); `eli_uri` is retained for
    citation/provenance display only, since the literal ELI URL is blocked by
    EUR-Lex's WAF for non-browser clients."""

    slug: str
    name: str
    celex_id: str
    eli_uri: str


DOCUMENTS: list[DocumentSource] = [
    DocumentSource(
        slug="eu_ai_act",
        name="Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
        celex_id="32024R1689",
        eli_uri="http://data.europa.eu/eli/reg/2024/1689/oj",
    ),
    DocumentSource(
        slug="gdpr",
        name="Regulation (EU) 2016/679 (General Data Protection Regulation)",
        celex_id="32016R0679",
        eli_uri="http://data.europa.eu/eli/reg/2016/679/oj",
    ),
]
