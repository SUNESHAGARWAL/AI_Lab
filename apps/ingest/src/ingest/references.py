import re

# Neither the AI Act nor GDPR cites a Recital by number anywhere in its operative
# text (verified by grepping both full fetched documents — zero matches) — normal EU
# legislative drafting; recitals explain intent, they aren't cross-cited by number
# from articles. This pattern is still included so the mechanism is real and would
# catch a Recital citation if one existed (see the parser test's synthetic fixture
# sentence, which exercises this branch directly since the real corpus never does).
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("article", re.compile(r"\bArticle\s+(\d+)(?:\((\d+)\))?")),
    ("chapter", re.compile(r"\bChapter\s+([IVXLCDM]+)\b")),
    ("annex", re.compile(r"\bAnnex\s+([IVXLCDM]+)\b")),
    ("recital", re.compile(r"\bRecital\s+(\d+)\b")),
]


def extract_cross_references(text: str) -> list[str]:
    """Extracts explicit cross-references to Articles, Chapters, Annexes, and
    Recitals from legal text, returning normalized, deduplicated strings like
    "article:9", "article:72.3", "chapter:iii", "recital:9"."""
    seen: dict[str, None] = {}
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if kind == "article":
                number, sub = match.group(1), match.group(2)
                ref = f"article:{number}.{sub}" if sub else f"article:{number}"
            else:
                ref = f"{kind}:{match.group(1).lower()}"
            seen[ref] = None
    return list(seen)
