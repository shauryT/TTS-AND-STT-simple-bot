"""Bank voice assistant backend: safety -> escalation -> per-bank retrieval -> (optional) LLM."""
import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .retrieval import load_banks, search
from .safety import find_sensitive, mask, urgent, wants_human

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ivr")

BANKS = load_banks()
MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
API_KEY = os.getenv("ANTHROPIC_API_KEY")
DAILY_CAP = int(os.getenv("DAILY_LLM_CAP", "200"))
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "20"))

app = FastAPI(title="Bank Voice Assistant")
_hits = defaultdict(deque)
_llm_usage = {"day": time.strftime("%Y-%m-%d"), "n": 0}


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=600)


class ChatIn(BaseModel):
    bank_id: str
    message: str = Field(min_length=1, max_length=300)
    history: list[Turn] = Field(default_factory=list, max_length=12)


def rate_limit(request: Request):
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() or (request.client.host if request.client else "?")
    now, dq = time.time(), _hits[ip]
    while dq and now - dq[0] > 60:
        dq.popleft()
    if len(dq) >= RATE_PER_MIN:
        raise HTTPException(429, "Too many requests, please slow down.")
    dq.append(now)


def llm_allowed() -> bool:
    today = time.strftime("%Y-%m-%d")
    if _llm_usage["day"] != today:
        _llm_usage.update(day=today, n=0)
    return bool(API_KEY) and _llm_usage["n"] < DAILY_CAP


GENERAL = {
    "id": "general",
    "name": "General Assistant",
    "greeting": "Hi! I'm your voice assistant. Ask me anything.",
}


def build_msgs(history: list, message: str) -> list:
    """Turn recent history + the new message into alternating user/assistant turns."""
    msgs = []
    for t in history[-6:]:
        text = mask(t.content)
        if msgs and msgs[-1]["role"] == t.role:
            msgs[-1]["content"] += " " + text
        elif msgs or t.role == "user":
            msgs.append({"role": t.role, "content": text})
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] += " " + message
    else:
        msgs.append({"role": "user", "content": message})
    return msgs


def general_reply(history: list, message: str) -> str:
    import anthropic

    system = (
        "You are a friendly voice assistant. Your replies are spoken aloud: use 1 to 3 short "
        "plain sentences, no markdown, lists or emojis. Be accurate and say so when you are "
        "unsure. Never ask for or repeat card numbers, PINs, CVVs, OTPs or passwords. "
        "Reply in the user's language."
    )
    client = anthropic.Anthropic(api_key=API_KEY, timeout=20)
    r = client.messages.create(model=MODEL, max_tokens=300, system=system,
                               messages=build_msgs(history, message))
    _llm_usage["n"] += 1
    return r.content[0].text.strip()


def llm_reply(bank: dict, hits: list, history: list, message: str) -> str:
    import anthropic

    knowledge = "\n\n".join(
        f"[{h['code']}] {h['title']}\nCause: {h['cause']}\nFix: {h['fix']}" for h in hits
    )
    system = (
        f"You are the phone support assistant for {bank['name']}, a bank. Your replies are "
        "spoken aloud: use 1 to 3 short plain sentences, no markdown, lists or emojis. "
        f"Answer ONLY from the KNOWLEDGE below, which is specific to {bank['name']}. "
        "If it does not answer the question, say you are not sure and offer to connect the "
        "caller to an agent. Never ask for or repeat card numbers, PINs, CVVs, OTPs or "
        "passwords. Never invent error codes, limits or policies. Reply in the caller's language.\n\n"
        f"KNOWLEDGE:\n{knowledge}"
    )
    msgs = build_msgs(history, message)

    client = anthropic.Anthropic(api_key=API_KEY, timeout=20)
    r = client.messages.create(model=MODEL, max_tokens=300, system=system, messages=msgs)
    _llm_usage["n"] += 1
    return r.content[0].text.strip()


def plain_reply(top: dict) -> str:
    return (f"For error {top['code']}: {top['title']}. {top['cause']} {top['fix']} "
            "Say agent at any time to reach a person.")


@app.get("/health")
def health():
    return {"ok": True, "llm": bool(API_KEY)}


@app.get("/api/banks")
def banks():
    return [GENERAL] + [{"id": b["id"], "name": b["name"], "greeting": b["greeting"]} for b in BANKS.values()]


def chat_general(body: ChatIn):
    msg = body.message.strip()
    if find_sensitive(msg):
        mode, reply = "blocked", ("For your security, please don't share card numbers, PINs, "
                                  "OTPs or passwords here. What else can I help with?")
    elif not llm_allowed():
        mode, reply = "unavailable", ("General mode needs the language model, which is "
                                      "unavailable right now. Bank modes still work.")
    else:
        try:
            mode, reply = "llm", general_reply(body.history, msg)
        except Exception as e:
            log.warning("LLM failed: %s", type(e).__name__)
            mode, reply = "error", "Sorry, I had trouble answering that. Please try again."
    log.info("bank=general mode=%s", mode)
    return {"reply": reply, "mode": mode, "escalate": False, "codes": []}


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    rate_limit(request)
    if body.bank_id == "general":
        return chat_general(body)
    bank = BANKS.get(body.bank_id)
    if not bank:
        raise HTTPException(404, "Unknown bank")
    msg, mode, codes, escalate = body.message.strip(), "retrieval", [], False

    if find_sensitive(msg):
        mode = "blocked"
        reply = ("For your security, please don't share card numbers, PINs, OTPs or passwords "
                 "here. I can help without them. What error or problem are you seeing?")
    elif urgent(msg):
        mode, escalate = "urgent", True
        reply = (f"If your card or account may be compromised, please block it right away by "
                 f"calling {bank['urgent_number']}. That line is open 24 by 7.")
    elif wants_human(msg):
        mode, escalate = "human", True
        reply = f"Sure. Please call {bank['name']} support on {bank['escalation_number']}, {bank['support_hours']}."
    else:
        hits = search(bank, msg)
        codes = [h["code"] for h in hits]
        if not hits:
            mode, escalate = "no_match", True
            reply = (f"I couldn't find that in {bank['name']}'s records. I can connect you to an "
                     f"agent on {bank['escalation_number']}.")
        else:
            reply = plain_reply(hits[0])
            if llm_allowed():
                try:
                    reply, mode = llm_reply(bank, hits, body.history, msg), "llm"
                except Exception as e:  # fall back to the KB answer if the LLM fails
                    log.warning("LLM failed: %s", type(e).__name__)

    # Log metadata only, never what the caller said.
    log.info("bank=%s mode=%s codes=%s escalate=%s", bank["id"], mode, codes, escalate)
    return {"reply": reply, "mode": mode, "escalate": escalate, "codes": codes}


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="ui")
