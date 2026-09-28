"""Voice assistant backend: safety check -> Gemini reply."""
import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .safety import find_sensitive, mask

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voice")

MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY")
DAILY_CAP = int(os.getenv("DAILY_LLM_CAP", "200"))
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "20"))

SYSTEM = (
    "You are a friendly voice assistant. Your replies are spoken aloud: use 1 to 3 short "
    "plain sentences, no markdown, lists or emojis. Be accurate and say so when you are "
    "unsure. Never ask for or repeat card numbers, PINs, CVVs, OTPs or passwords. "
    "Reply in the user's language."
)

app = FastAPI(title="Voice Assistant")
_hits = defaultdict(deque)
_llm_usage = {"day": time.strftime("%Y-%m-%d"), "n": 0}
_client = None


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=600)


class ChatIn(BaseModel):
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
    global _client
    from google import genai
    from google.genai import types

    if _client is None:
        _client = genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=20000))
    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user",
                      parts=[types.Part(text=m["content"])])
        for m in build_msgs(history, message)
    ]
    r = _client.models.generate_content(
        model=MODEL, contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM, max_output_tokens=500),
    )
    _llm_usage["n"] += 1
    return (r.text or "").strip() or "Sorry, I didn't catch that. Could you say it again?"


@app.get("/health")
def health():
    return {"ok": True, "llm": bool(API_KEY)}


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    rate_limit(request)
    msg = body.message.strip()
    if find_sensitive(msg):
        mode, reply = "blocked", ("For your security, please don't share card numbers, PINs, "
                                  "OTPs or passwords here. What else can I help with?")
    elif not llm_allowed():
        mode, reply = "unavailable", "I'm unavailable right now. Please try again later."
    else:
        try:
            mode, reply = "llm", general_reply(body.history, msg)
        except Exception as e:
            log.warning("LLM failed: %s", type(e).__name__)
            mode, reply = "error", "I'm a bit busy right now. Please try again in a moment."
    # Log metadata only, never what the user said.
    log.info("mode=%s", mode)
    return {"reply": reply, "mode": mode}


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="ui")
