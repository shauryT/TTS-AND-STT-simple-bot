from backend.retrieval import load_banks, search
from backend.safety import find_sensitive, mask, urgent, wants_human

BANKS = load_banks()


def test_same_code_differs_per_bank():
    titles = {b: search(BANKS[b], "I got error E104")[0]["title"] for b in BANKS}
    assert len(set(titles.values())) == 3


def test_spoken_code_variants():
    assert search(BANKS["sunrise"], "error e 104")[0]["code"] == "E104"
    assert search(BANKS["sunrise"], "error 104")[0]["code"] == "E104"


def test_keyword_search_stays_in_bank():
    assert search(BANKS["lotus"], "atm cash not dispensed but debited")[0]["code"] == "E405"
    assert search(BANKS["sunrise"], "atm cash not dispensed but debited") == []


def test_no_match():
    assert search(BANKS["meridian"], "what is the weather like") == []


def test_sensitive_detection():
    assert find_sensitive("my card is 4111 1111 1111 1111")
    assert find_sensitive("my otp is 482913")
    assert not find_sensitive("I got error E104")
    assert "[removed]" in mask("cvv 123")


def test_escalation_flags():
    assert wants_human("let me talk to an agent")
    assert urgent("my card was stolen")
    assert not urgent("upi limit exceeded")


def test_general_mode_listed_first_and_blocks_secrets():
    from fastapi.testclient import TestClient
    from backend.main import app
    c = TestClient(app)
    assert c.get("/api/banks").json()[0]["id"] == "general"
    r = c.post("/api/chat", json={"bank_id": "general", "message": "my otp is 482913"}).json()
    assert r["mode"] == "blocked"
