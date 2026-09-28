"""Per-bank knowledge lookup. Each bank's entries are searched in isolation."""
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "banks"
STOP = set("i my me is the a an to of and it in on for at with this that was am are have has had "
           "not can you your please tell help what why how when do does did get got but or so".split())
MIN_SCORE = 3


def load_banks() -> dict:
    banks = {}
    for f in sorted(DATA_DIR.glob("*.json")):
        b = json.loads(f.read_text(encoding="utf-8"))
        banks[b["id"]] = b
    return banks


def _norm(text: str) -> str:
    # Speech-to-text writes "E 104" or "e-104"; turn that into "e104".
    return re.sub(r"\b([a-z])\s*-?\s*(\d{2,4})\b", r"\1\2", text.lower())


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1}


def search(bank: dict, query: str, k: int = 3) -> list:
    q = _norm(query)
    codes = set(re.findall(r"\b[a-z]{1,3}\d{2,4}\b", q))
    numbers = set(re.findall(r"\b\d{3,4}\b", q))
    qt = _tokens(q)
    scored = []
    for e in bank["entries"]:
        score = 0
        code = e["code"].lower()
        if code in codes:
            score += 10
        elif re.sub(r"\D", "", code) in numbers:
            score += 6
        score += 2 * len(qt & _tokens(" ".join(e["keywords"])))
        score += len(qt & _tokens(e["title"] + " " + e["cause"]))
        if score >= MIN_SCORE:
            scored.append((score, e))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:k]]
