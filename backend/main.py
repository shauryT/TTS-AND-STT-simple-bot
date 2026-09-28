import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .safety import find_sensitive, mask


# =========================================================
# CONFIGURATION
# =========================================================

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Valid Gemini model IDs (as of late 2025):
#   gemini-2.0-flash-lite   <- fast + cheap, good default
#   gemini-2.0-flash
#   gemini-2.5-flash-lite
#   gemini-2.5-flash
MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash-lite").strip()

DAILY_CAP = int(os.getenv("DAILY_LLM_CAP", "200"))
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "20"))


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("voice")


# =========================================================
# SYSTEM INSTRUCTION
# =========================================================

SYSTEM = """
You are a friendly AI voice assistant.
Your responses are usually spoken aloud.

Keep responses natural and conversational.
Normally answer in 1 to 3 short sentences.

Do not use markdown.
Do not use bullet points.
Do not use emojis unless the user specifically asks for them.

Answer the user's question directly.
Be accurate.
If you are unsure, say that you are unsure.

Never ask for or repeat card numbers, PINs, CVVs, OTPs,
passwords, or other sensitive authentication information.

Always reply in the same language as the user.
""".strip()


# =========================================================
# APP
# =========================================================

app = FastAPI(title="AI Voice Assistant")

# Allow the frontend to be served from a different origin during dev
# (e.g. you open index.html directly, or run a separate static server).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# RUNTIME STATE
# =========================================================

_hits = defaultdict(deque)

_llm_usage = {
    "day": time.strftime("%Y-%m-%d"),
    "count": 0,
}

_client = None


# =========================================================
# REQUEST MODELS
# =========================================================

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=600)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=300)
    history: list[Turn] = Field(default_factory=list, max_length=12)


# =========================================================
# RATE LIMIT
# =========================================================

def rate_limit(request: Request):
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = (
        forwarded.split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )

    now = time.time()
    requests = _hits[ip]

    while requests and now - requests[0] > 60:
        requests.popleft()

    if len(requests) >= RATE_PER_MIN:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down.",
        )

    requests.append(now)


# =========================================================
# DAILY GEMINI LIMIT
# =========================================================

def llm_allowed() -> bool:
    today = time.strftime("%Y-%m-%d")

    if _llm_usage["day"] != today:
        _llm_usage["day"] = today
        _llm_usage["count"] = 0

    return bool(API_KEY) and _llm_usage["count"] < DAILY_CAP


# =========================================================
# BUILD CONVERSATION
# =========================================================

def build_messages(history: list[Turn], message: str):
    messages = []

    for turn in history[-6:]:
        text = mask(turn.content).strip()
        if not text:
            continue

        role = "model" if turn.role == "assistant" else "user"

        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += " " + text
        else:
            messages.append({"role": role, "content": text})

    current = mask(message).strip()

    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] += " " + current
    else:
        messages.append({"role": "user", "content": current})

    return messages


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_client():
    global _client

    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    if _client is None:
        from google import genai
        log.info("Initializing Gemini client")
        _client = genai.Client(api_key=API_KEY)

    return _client


# =========================================================
# GEMINI RESPONSE
# =========================================================

def generate_reply(history: list[Turn], message: str) -> str:
    from google.genai import types

    client = get_client()
    messages = build_messages(history, message)

    contents = [
        types.Content(
            role=item["role"],
            parts=[types.Part(text=item["content"])],
        )
        for item in messages
    ]

    log.info("Calling Gemini | model=%s", MODEL)

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            max_output_tokens=300,
            temperature=0.7,
        ),
    )

    _llm_usage["count"] += 1

    reply = (response.text or "").strip()

    if not reply:
        return "I didn't get a response from the AI. Could you try again?"

    return reply


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "ok": True,
        "gemini_configured": bool(API_KEY),
        "model": MODEL,
        "requests_today": _llm_usage["count"],
    }


# =========================================================
# BANKS (kept so /api/banks doesn't 404 if called)
# =========================================================

@app.get("/api/banks")
def banks():
    return {
        "banks": [
            {"id": "general", "name": "General Assistant"},
        ]
    }


# =========================================================
# CHAT API
# =========================================================

@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    rate_limit(request)

    message = body.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # SAFETY CHECK
    if find_sensitive(message):
        log.info("Blocked sensitive message")
        return {
            "reply": (
                "For your security, please don't share "
                "card numbers, PINs, OTPs or passwords here. "
                "What else can I help with?"
            ),
            "mode": "blocked",
        }

    # API KEY CHECK
    if not API_KEY:
        log.error("GEMINI_API_KEY is missing")
        return {
            "reply": "The AI service is not configured yet.",
            "mode": "unavailable",
        }

    # DAILY LIMIT
    if not llm_allowed():
        log.warning("Daily Gemini limit reached")
        return {
            "reply": (
                "The AI service has reached its daily "
                "limit. Please try again later."
            ),
            "mode": "unavailable",
        }

    # GEMINI
    try:
        reply = generate_reply(body.history, message)
        log.info("Gemini request successful")
        return {"reply": reply, "mode": "llm"}

    except Exception as error:
        log.exception("Gemini request failed: %s", error)
        # Surface a hint in dev so you know what actually broke.
        return {
            "reply": (
                "I couldn't get a response from the AI right now. "
                "Please try again."
            ),
            "mode": "error",
            "detail": str(error)[:200],  # remove if you don't want this public
        }


# =========================================================
# FRONTEND (mounted last so /health and /api/* win)
# =========================================================

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIR, html=True),
        name="frontend",
    )
else:
    log.warning("Frontend directory does not exist: %s", FRONTEND_DIR)
