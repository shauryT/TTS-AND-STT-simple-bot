from fastapi.testclient import TestClient

from backend import main
from backend.safety import find_sensitive, mask

client = TestClient(main.app)


def test_sensitive_detection():
    assert find_sensitive("my card is 4111 1111 1111 1111")
    assert find_sensitive("my otp is 482913")
    assert not find_sensitive("what is the weather")
    assert "[removed]" in mask("cvv 123")


def test_secrets_are_blocked():
    r = client.post("/api/chat", json={"message": "my otp is 482913"}).json()
    assert r["mode"] == "blocked"


def test_no_key_is_unavailable(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", None)
    r = client.post("/api/chat", json={"message": "hello"}).json()
    assert r["mode"] == "unavailable"


def test_llm_path_and_failure(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "test-key")
    monkeypatch.setattr(main, "general_reply", lambda h, m: "Hi there")
    assert client.post("/api/chat", json={"message": "hello"}).json() == {"reply": "Hi there", "mode": "llm"}

    def boom(h, m):
        raise RuntimeError("down")
    monkeypatch.setattr(main, "general_reply", boom)
    assert client.post("/api/chat", json={"message": "hello"}).json()["mode"] == "error"


def test_history_is_masked_and_alternates():
    hist = [main.Turn(role="user", content="my otp is 123456"),
            main.Turn(role="assistant", content="ok")]
    msgs = main.build_msgs(hist, "next")
    assert "123456" not in msgs[0]["content"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
