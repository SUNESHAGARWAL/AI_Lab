from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from ingest.documents import DocumentSource

_CHAPTER_ID_RE = re.compile(r"^cpt_[IVXLCDM]+$")
_SECTION_ID_RE = re.compile(r"^cpt_[IVXLCDM]+\.sct_(\d+)$")
_ARTICLE_ID_RE = re.compile(r"^art_(\d+)$")
_RECITAL_ID_RE = re.compile(r"^rct_(\d+)$")
_ARTICLE_PARAGRAPH_ID_RE = re.compile(r"^(\d+)\.(\d+)$")


@dataclass(frozen=True)
class Point:
    label: str
    text: str


@dataclass(frozen=True)
class Paragraph:
    """`number` is None for articles with a single, unnumbered body (e.g. a
    "Definitions" article whose text isn't split into "1.", "2." sub-paragraphs) —
    real EU legislative drafting doesn't guarantee every article is paragraph-numbered."""

    number: str | None
    text: str
    points: list[Point] = field(default_factory=list)


@dataclass(frozen=True)
class Article:
    number: str
    title: str
    chapter_number: str | None
    chapter_title: str | None
    section_number: str | None
    section_title: str | None
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs if p.text)


@dataclass(frozen=True)
class Recital:
    number: str
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    source: DocumentSource
    articles: list[Article] = field(default_factory=list)
    recitals: list[Recital] = field(default_factory=list)


def _text(tag: Tag | None) -> str:
    if tag is None:
        return ""
    return " ".join(tag.get_text(" ", strip=True).split())


def _title_text(div: Tag) -> str:
    title_div = div.find("div", class_="eli-title", recursive=False)
    return _text(title_div)


def _parse_points(container: Tag) -> list[Point]:
    """Numbered/lettered points render as one <table> per point, first cell the
    label ("(a)", "(1)", ...), second cell the point text."""
    points: list[Point] = []
    for table in container.find_all("table", recursive=False):
        cells = table.find_all("td")
        if len(cells) < 2:
            continue
        points.append(Point(label=_text(cells[0]), text=_text(cells[1])))
    return points


def _paragraph_text(intro: str, points: list[Point]) -> str:
    parts = [intro] if intro else []
    parts.extend(f"{pt.label} {pt.text}" for pt in points)
    return " ".join(parts).strip()


def _paragraph_from_container(container: Tag, number: str | None) -> Paragraph:
    intro = " ".join(
        _text(p) for p in container.find_all("p", class_="oj-normal", recursive=False)
    )
    points = _parse_points(container)
    return Paragraph(number=number, text=_paragraph_text(intro, points), points=points)


def _parse_article_paragraphs(article_div: Tag, article_number: str) -> list[Paragraph]:
    padded = article_number.zfill(3)
    paragraph_divs = [
        child
        for child in article_div.find_all("div", recursive=False)
        if child.get("id", "").startswith(f"{padded}.")
    ]
    if paragraph_divs:
        paragraphs = []
        for div in paragraph_divs:
            match = _ARTICLE_PARAGRAPH_ID_RE.match(div["id"])
            para_number = str(int(match.group(2))) if match else None
            paragraphs.append(_paragraph_from_container(div, para_number))
        return paragraphs

    # No numbered paragraph divs: the article body's direct <p class="oj-normal">/
    # <table> children form one implicit, unnumbered paragraph.
    return [_paragraph_from_container(article_div, None)]


def parse_document(xhtml: str, source: DocumentSource) -> ParsedDocument:
    """Parses ELI-compliant XHTML (as served by the CELLAR API — see fetch.py)
    preserving Chapter -> Section -> Article -> Paragraph -> Point nesting, plus
    Recitals collected separately. Walks every <div id="..."> in document order,
    tracking current chapter/section as state — a flat preorder scan is sufficient
    since chapter/section divs always appear before the articles they contain."""
    soup = BeautifulSoup(xhtml, "lxml-xml")

    articles: list[Article] = []
    recitals: list[Recital] = []

    current_chapter_number: str | None = None
    current_chapter_title: str | None = None
    current_section_number: str | None = None
    current_section_title: str | None = None

    for div in soup.find_all("div", id=True):
        div_id = div["id"]
        classes = div.get("class") or []

        if _CHAPTER_ID_RE.match(div_id):
            current_chapter_number = div_id.removeprefix("cpt_")
            current_chapter_title = _title_text(div)
            current_section_number = None
            current_section_title = None
        elif section_match := _SECTION_ID_RE.match(div_id):
            current_section_number = section_match.group(1)
            current_section_title = _title_text(div)
        elif "eli-subdivision" in classes and (article_match := _ARTICLE_ID_RE.match(div_id)):
            number = article_match.group(1)
            title = _text(div.find("p", class_="oj-sti-art")) or _title_text(div)
            articles.append(
                Article(
                    number=number,
                    title=title,
                    chapter_number=current_chapter_number,
                    chapter_title=current_chapter_title,
                    section_number=current_section_number,
                    section_title=current_section_title,
                    paragraphs=_parse_article_paragraphs(div, number),
                )
            )
        elif "eli-subdivision" in classes and (recital_match := _RECITAL_ID_RE.match(div_id)):
            number = recital_match.group(1)
            points = _parse_points(div)
            text = points[0].text if points else _text(div)
            recitals.append(Recital(number=number, text=text))

    return ParsedDocument(source=source, articles=articles, recitals=recitals)
