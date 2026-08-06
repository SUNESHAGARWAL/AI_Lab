from dataclasses import dataclass

from evals.golden import Difficulty, QuestionType

_EASY = Difficulty.EASY
_MEDIUM = Difficulty.MEDIUM
_HARD = Difficulty.HARD

_FACTUAL = QuestionType.FACTUAL_LOOKUP
_ROLE = QuestionType.ROLE_SCOPED
_XREF = QuestionType.CROSS_REFERENCE
_OOS = QuestionType.OUT_OF_SCOPE


@dataclass(frozen=True)
class SeedQuestion:
    id: str
    question: str
    difficulty: Difficulty
    question_type: QuestionType
    grounding_note: str


# Authored by reading the real ingested corpus first (see the live Postgres queries
# run during planning: `SELECT chunk_id, left(content, 200) FROM chunks WHERE ...`)
# — every grounding_note names the real chunk(s) the question was built from. Only
# the question TEXTS are assistant-authored here; the relevant_chunk_ids ground truth
# is never filled in by generation code (see candidates.py) — that's the reviewer's
# job after reading the real candidate_pool this module's questions produce.
SEED_QUESTIONS: list[SeedQuestion] = [
    # --- factual_lookup (14): direct lookups answerable from one real chunk ---
    SeedQuestion(
        "cand-001",
        "What is the subject matter of the AI Act according to Article 1?",
        _EASY,
        _FACTUAL,
        "eu_ai_act:article:1",
    ),
    SeedQuestion(
        "cand-002",
        "How does Article 3 of the AI Act define an 'AI system'?",
        _EASY,
        _FACTUAL,
        "eu_ai_act:article:3",
    ),
    SeedQuestion(
        "cand-003",
        "What AI practices does Article 5 of the AI Act prohibit?",
        _EASY,
        _FACTUAL,
        "eu_ai_act:article:5",
    ),
    SeedQuestion(
        "cand-004",
        "What two conditions determine whether an AI system is classified as "
        "high-risk under Article 6 of the AI Act?",
        _MEDIUM,
        _FACTUAL,
        "eu_ai_act:article:6",
    ),
    SeedQuestion(
        "cand-005",
        "What must a risk management system for high-risk AI systems establish "
        "according to Article 9 of the AI Act?",
        _MEDIUM,
        _FACTUAL,
        "eu_ai_act:article:9",
    ),
    SeedQuestion(
        "cand-006",
        "What data governance requirements does Article 10 of the AI Act impose on "
        "training data for high-risk AI systems?",
        _MEDIUM,
        _FACTUAL,
        "eu_ai_act:article:10",
    ),
    SeedQuestion(
        "cand-007",
        "What human oversight measures does Article 14 of the AI Act require for "
        "high-risk AI systems?",
        _EASY,
        _FACTUAL,
        "eu_ai_act:article:14",
    ),
    SeedQuestion(
        "cand-008",
        "How does Article 4 of the GDPR define 'personal data'?",
        _EASY,
        _FACTUAL,
        "gdpr:article:4",
    ),
    SeedQuestion(
        "cand-009",
        "What are the core principles relating to the processing of personal data "
        "listed in Article 5 of the GDPR?",
        _EASY,
        _FACTUAL,
        "gdpr:article:5",
    ),
    SeedQuestion(
        "cand-010",
        "What conditions must be met for consent to be valid under Article 7 of the "
        "GDPR?",
        _MEDIUM,
        _FACTUAL,
        "gdpr:article:7",
    ),
    SeedQuestion(
        "cand-011",
        "What special categories of personal data are covered by Article 9 of the "
        "GDPR?",
        _MEDIUM,
        _FACTUAL,
        "gdpr:article:9",
    ),
    SeedQuestion(
        "cand-012",
        "What is the right to erasure under Article 17 of the GDPR (the 'right to be "
        "forgotten')?",
        _EASY,
        _FACTUAL,
        "gdpr:article:17",
    ),
    SeedQuestion(
        "cand-013",
        "Under what specific timeframe and to whom must a controller report a "
        "personal data breach according to Article 33 of the GDPR?",
        _HARD,
        _FACTUAL,
        "gdpr:article:33",
    ),
    SeedQuestion(
        "cand-014",
        "What criteria in Article 35 of the GDPR trigger the requirement to carry "
        "out a data protection impact assessment?",
        _HARD,
        _FACTUAL,
        "gdpr:article:35",
    ),
    # --- role_scoped (10): explicit role + obligation, grounded in real
    # provider/deployer (AI Act) and controller/processor (GDPR) articles ---
    SeedQuestion(
        "cand-015",
        "As a provider of a high-risk AI system, what compliance obligations do I "
        "have under Article 16 of the AI Act?",
        _MEDIUM,
        _ROLE,
        "eu_ai_act:article:16",
    ),
    SeedQuestion(
        "cand-016",
        "As a deployer of a high-risk AI system, what technical and organisational "
        "measures must I take under Article 26 of the AI Act?",
        _MEDIUM,
        _ROLE,
        "eu_ai_act:article:26",
    ),
    SeedQuestion(
        "cand-017",
        "As a provider of a general-purpose AI model, what documentation "
        "obligations do I have under Article 53 of the AI Act?",
        _HARD,
        _ROLE,
        "eu_ai_act:article:53",
    ),
    SeedQuestion(
        "cand-018",
        "As a provider, what post-market monitoring obligations do I have for a "
        "high-risk AI system under Article 72 of the AI Act?",
        _MEDIUM,
        _ROLE,
        "eu_ai_act:article:72",
    ),
    SeedQuestion(
        "cand-019",
        "As a data controller, what responsibilities do I have under Article 24 of "
        "the GDPR?",
        _EASY,
        _ROLE,
        "gdpr:article:24",
    ),
    SeedQuestion(
        "cand-020",
        "As a data processor engaged by a controller, what guarantees must I "
        "provide under Article 28 of the GDPR?",
        _MEDIUM,
        _ROLE,
        "gdpr:article:28",
    ),
    SeedQuestion(
        "cand-021",
        "As a controller, what records of processing activities am I required to "
        "maintain under Article 30 of the GDPR?",
        _MEDIUM,
        _ROLE,
        "gdpr:article:30",
    ),
    SeedQuestion(
        "cand-022",
        "As a controller or processor, when am I required to designate a data "
        "protection officer under Article 37 of the GDPR?",
        _HARD,
        _ROLE,
        "gdpr:article:37",
    ),
    SeedQuestion(
        "cand-023",
        "As a provider or deployer, what transparency obligations do I have when my "
        "AI system interacts directly with natural persons under Article 50 of the "
        "AI Act?",
        _MEDIUM,
        _ROLE,
        "eu_ai_act:article:50",
    ),
    SeedQuestion(
        "cand-024",
        "As a controller, what conditions must I meet before transferring personal "
        "data to a third country under Article 44 of the GDPR?",
        _HARD,
        _ROLE,
        "gdpr:article:44",
    ),
    # --- cross_reference (10): grounded in the real metadata->>'cross_references'
    # column populated by ingest/references.py during ingestion ---
    SeedQuestion(
        "cand-025",
        "Article 5 of the AI Act prohibits certain practices — which other AI Act "
        "articles does it reference for exceptions or related conditions?",
        _MEDIUM,
        _XREF,
        "eu_ai_act:article:5 cross_references -> article:9,27,49,annex:ii",
    ),
    SeedQuestion(
        "cand-026",
        "How does Article 40 of the AI Act's harmonised standards provision connect "
        "to the data governance and human oversight requirements elsewhere in the "
        "Regulation?",
        _HARD,
        _XREF,
        "eu_ai_act:article:40 cross_references -> article:10,article:24,chapter:v,annex:i",
    ),
    SeedQuestion(
        "cand-027",
        "Article 15 of the GDPR grants a right of access — what other GDPR "
        "articles does it reference regarding automated decision-making and "
        "international transfers?",
        _HARD,
        _XREF,
        "gdpr:article:15 cross_references -> article:22.1,article:46",
    ),
    SeedQuestion(
        "cand-028",
        "What other GDPR articles does Article 37's data protection officer "
        "designation requirement reference?",
        _MEDIUM,
        _XREF,
        "gdpr:article:37 cross_references -> article:9,article:10,article:39",
    ),
    SeedQuestion(
        "cand-029",
        "Article 66 of the AI Act references numerous other articles and annexes — "
        "what is Article 66 about, and how does it relate to Article 5 and "
        "Chapter III?",
        _HARD,
        _XREF,
        "eu_ai_act:article:66 cross_references -> article:5,chapter:iii,annex:i,annex:iii "
        "(and others)",
    ),
    SeedQuestion(
        "cand-030",
        "Article 108 of the AI Act references several other articles and "
        "Chapter III — what is Article 108 about?",
        _MEDIUM,
        _XREF,
        "eu_ai_act:article:108 cross_references -> article:17,19,43,47,57,58,chapter:iii",
    ),
    SeedQuestion(
        "cand-031",
        "How does Article 61 of the GDPR (mutual assistance) relate to Articles 55 "
        "and 66?",
        _MEDIUM,
        _XREF,
        "gdpr:article:61 cross_references -> article:55.1,article:66.1,article:66.2,article:93.2",
    ),
    SeedQuestion(
        "cand-032",
        "Recital 93 of the AI Act references Article 13 — what does Article 13 "
        "require?",
        _EASY,
        _XREF,
        "eu_ai_act:recital:93 cross_references -> article:13",
    ),
    SeedQuestion(
        "cand-033",
        "Article 27 of the GDPR references Articles 3, 9, and 10 — what does "
        "Article 27 itself cover?",
        _MEDIUM,
        _XREF,
        "gdpr:article:27 cross_references -> article:3.2,article:9.1,article:10",
    ),
    SeedQuestion(
        "cand-034",
        "Article 57 of the GDPR cross-references a large number of other articles "
        "including 28, 35, 36, and 40 — what role does Article 57 play, and how "
        "does it connect to those provisions?",
        _HARD,
        _XREF,
        "gdpr:article:57 cross_references -> article:28.8,article:35.4,article:36.2,"
        "article:40.1 (and many others)",
    ),
    # --- out_of_scope (6): clearly outside AI Act/GDPR subject matter, no
    # grounding read needed — constructed to test correct low-relevance behavior ---
    SeedQuestion(
        "cand-035",
        "What is the standard corporate income tax rate in Germany?",
        _EASY,
        _OOS,
        "n/a — outside AI Act/GDPR subject matter",
    ),
    SeedQuestion(
        "cand-036",
        "What ingredients are needed to bake a sourdough loaf?",
        _EASY,
        _OOS,
        "n/a — outside AI Act/GDPR subject matter",
    ),
    SeedQuestion(
        "cand-037",
        "What is the capital of Australia?",
        _EASY,
        _OOS,
        "n/a — outside AI Act/GDPR subject matter",
    ),
    SeedQuestion(
        "cand-038",
        "How do I file a trademark application with the USPTO?",
        _MEDIUM,
        _OOS,
        "n/a — outside AI Act/GDPR subject matter",
    ),
    SeedQuestion(
        "cand-039",
        "What are the rules of chess castling?",
        _MEDIUM,
        _OOS,
        "n/a — outside AI Act/GDPR subject matter",
    ),
    SeedQuestion(
        "cand-040",
        "What is the current interest rate set by the Federal Reserve?",
        _MEDIUM,
        _OOS,
        "n/a — outside AI Act/GDPR subject matter",
    ),
]
