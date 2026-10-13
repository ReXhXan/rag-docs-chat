from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_ingest_and_query():
    r = client.post("/ingest", files=[("files", ("notes.txt", b"The capital of Japan is Tokyo."))])
    assert r.status_code == 200 and r.json()["ingested_chunks"]["notes.txt"] == 1
    res = client.post("/query", json={"question": "What is the capital of Japan?"}).json()
    assert res["citations"][0]["source"] == "notes.txt"
    assert "notes.txt" in client.get("/documents").json()["sources"]


def test_rejects_unsupported_type():
    r = client.post("/ingest", files=[("files", ("x.exe", b"bin"))])
    assert r.status_code == 400
