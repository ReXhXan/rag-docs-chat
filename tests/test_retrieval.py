from app.core.loaders import load_document
from app.core.pipeline import build_prompt


def test_retrieves_most_relevant_source(pipeline):
    hits = pipeline.retrieve("Who created the Python programming language?")
    assert hits[0].chunk.metadata["source"] == "python.txt"
    assert hits[0].score >= hits[-1].score


def test_ask_returns_answer_with_citations(pipeline):
    res = pipeline.ask("Where is the Eiffel Tower?")
    assert res["citations"][0]["source"] == "paris.md"
    assert "[1]" in res["answer"]
    assert {"ref", "source", "page", "score", "text"} <= res["citations"][0].keys()


def test_page_numbers_in_citations(pipeline):
    res = pipeline.ask("How does photosynthesis work in plants?")
    top = res["citations"][0]
    assert top["source"] == "bio.pdf" and top["page"] == 2


def test_reingest_is_idempotent(pipeline):
    before = pipeline.store.count()
    pipeline.ingest_pages([("The Eiffel Tower is located in Paris, France.", {"source": "paris.md"})])
    assert pipeline.store.count() == before


def test_prompt_numbers_context(pipeline):
    prompt = build_prompt("q", pipeline.retrieve("Paris"))
    assert "[1]" in prompt and "Question: q" in prompt


def test_load_txt_and_md(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("# Title\nbody")
    assert load_document(f) == [("# Title\nbody", {"source": "n.md"})]
