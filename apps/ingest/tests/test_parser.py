from pathlib import Path

import pytest
from ingest.chunks import build_chunks
from ingest.documents import DocumentSource
from ingest.parser import parse_document
from ingest.references import extract_cross_references

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample.xhtml"
TEST_SOURCE = DocumentSource(
    slug="test_doc", name="Test Document", celex_id="00000000", eli_uri="http://example.com"
)


@pytest.fixture
def fixture_xhtml() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def test_parse_document_extracts_chapter_section_article_structure(fixture_xhtml: str) -> None:
    doc = parse_document(fixture_xhtml, TEST_SOURCE)

    assert len(doc.articles) == 2

    article_1, article_2 = doc.articles
    assert article_1.number == "1"
    assert article_1.title == "Subject matter"
    assert article_1.chapter_number == "I"
    assert article_1.section_number == "1"

    assert article_2.number == "2"
    assert article_2.title == "Scope"
    assert article_2.chapter_number == "I"
    assert article_2.section_number == "1"


def test_parse_document_extracts_paragraphs(fixture_xhtml: str) -> None:
    doc = parse_document(fixture_xhtml, TEST_SOURCE)
    article_1 = doc.articles[0]

    assert len(article_1.paragraphs) == 2
    assert article_1.paragraphs[0].number == "1"
    assert "harmonised rules" in article_1.paragraphs[0].text
    assert article_1.paragraphs[1].number == "2"
    # points fold into the owning paragraph's text
    assert "a first illustrative point" in article_1.paragraphs[1].text


def test_parse_document_handles_single_unnumbered_paragraph_article(fixture_xhtml: str) -> None:
    doc = parse_document(fixture_xhtml, TEST_SOURCE)
    article_2 = doc.articles[1]

    assert len(article_2.paragraphs) == 1
    assert article_2.paragraphs[0].number is None
    assert "applies to providers" in article_2.paragraphs[0].text


def test_parse_document_extracts_recitals_separately(fixture_xhtml: str) -> None:
    doc = parse_document(fixture_xhtml, TEST_SOURCE)

    assert len(doc.recitals) == 2
    assert doc.recitals[0].number == "8"
    assert "background intent" in doc.recitals[0].text
    assert doc.recitals[1].number == "9"
    assert "transparency intent" in doc.recitals[1].text


def test_extract_cross_references_captures_article_and_chapter_citations() -> None:
    refs = extract_cross_references(
        "see Article 2 for scope. Applies without prejudice to Chapter III."
    )
    assert "article:2" in refs
    assert "chapter:iii" in refs


def test_extract_cross_references_captures_synthetic_recital_citation() -> None:
    # The real corpus never cites a Recital by number — this proves the mechanism
    # itself works using the fixture's one deliberately synthetic sentence.
    refs = extract_cross_references("As noted in Recital 9, transparency matters.")
    assert refs == ["recital:9"]


def test_build_chunks_creates_parent_and_leaf_chunks_for_multi_paragraph_article(
    fixture_xhtml: str,
) -> None:
    doc = parse_document(fixture_xhtml, TEST_SOURCE)
    chunks = build_chunks(doc)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}

    assert "test_doc:article:1" in by_id
    assert "test_doc:article:1:paragraph:1" in by_id
    assert "test_doc:article:1:paragraph:2" in by_id
    assert by_id["test_doc:article:1:paragraph:1"].metadata["parent_chunk_id"] == (
        "test_doc:article:1"
    )
    assert by_id["test_doc:article:1:paragraph:2"].metadata["parent_chunk_id"] == (
        "test_doc:article:1"
    )


def test_build_chunks_skips_redundant_leaf_chunk_for_single_paragraph_article(
    fixture_xhtml: str,
) -> None:
    doc = parse_document(fixture_xhtml, TEST_SOURCE)
    chunks = build_chunks(doc)
    chunk_ids = {chunk.chunk_id for chunk in chunks}

    assert "test_doc:article:2" in chunk_ids
    assert "test_doc:article:2:paragraph:1" not in chunk_ids


def test_build_chunks_captures_cross_references_in_metadata(fixture_xhtml: str) -> None:
    doc = parse_document(fixture_xhtml, TEST_SOURCE)
    chunks = build_chunks(doc)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}

    article_1_refs = by_id["test_doc:article:1"].metadata["cross_references"]
    assert "article:2" in article_1_refs
    assert "chapter:iii" in article_1_refs
    assert "recital:9" in article_1_refs


def test_build_chunks_metadata_values_are_all_strings(fixture_xhtml: str) -> None:
    doc = parse_document(fixture_xhtml, TEST_SOURCE)
    chunks = build_chunks(doc)

    for chunk in chunks:
        for value in chunk.metadata.values():
            assert isinstance(value, str)
