from scripts.evaluate import evaluate


def test_evaluate_metrics(pipeline):
    qa = [{"question": "Where is the Eiffel Tower?", "expected_source": "paris.md", "expected_keywords": ["Paris"]}]
    m = evaluate(pipeline, qa, k=3)
    assert m["hit_rate@3"] == 1.0 and m["mrr"] == 1.0 and m["keyword_recall"] == 1.0
