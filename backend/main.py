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


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voice")


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

DAILY_CAP = int(os.getenv("DAILY_LLM_CAP", "200"))
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "20"))


SYSTEM = """
You are a friendly AI voice assistant.

Keep your responses short and natural because they may be spoken aloud.
Use 1 to 3 short sentences.
Do not use markdown, bullet points, emojis, or complicated formatting.

Answer the user's question directly.
Be accurate and say when you are unsure.

Never ask for or repeat:
- card numbers
- PINs
- CVVs
- OTPs
- passwords

Reply in the same language as the user.
"""


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI(title="AI Voice Assistant")


# ---------------------------------------------------------
# Runtime state
# ---------------------------------------------------------

_hits = defaultdict(deque)

_llm_usage = {
    "day": time.strftime("%Y-%m-%d"),
    "n": 0,
}

_client = None


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=600)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=300)
    history: list[Turn] = Field(default_factory=list, max_length=12)


# ---------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------

def rate_limit(request: Request):
    forwarded = request.headers.get("x-forwarded-for", "")

    ip = (
        forwarded.split(",")[0].strip()
        or (request.client.host if request.client else "?")
    )

    now = time.time()
    requests = _hits[ip]

    while requests and now - requests[0] > 60:
        requests.popleft()

    if len(requests) >= RATE_PER_MIN:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down."
        )

    requests.append(now)


# ---------------------------------------------------------
# Daily Gemini limit
# ---------------------------------------------------------

def llm_allowed() -> bool:
    today = time.strftime("%Y-%m-%d")

    if _llm_usage["day"] != today:
        _llm_usage["day"] = today
        _llm_usage["n"] = 0

    return bool(API_KEY) and _llm_usage["n"] < DAILY_CAP


# ---------------------------------------------------------
# Build Gemini conversation
# ---------------------------------------------------------

def build_messages(history: list[Turn], message: str) -> list:
    messages = []

    for turn in history[-6:]:
        text = mask(turn.content)

        if not text:
            continue

        role = "model" if turn.role == "assistant" else "user"

        # Merge consecutive messages from the same role
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += " " + text
        else:
            messages.append({
                "role": role,
                "content": text
            })

    # Add current user message
    current_message = mask(message)

    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] += " " + current_message
    else:
        messages.append({
            "role": "user",
            "content": current_message
        })

    return messages


# ---------------------------------------------------------
# Gemini request
# ---------------------------------------------------------

def ask_gemini(history: list[Turn], message: str) -> str:
    global _client

    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    from google import genai
    from google.genai import types

    # Create client only once
    if _client is None:
        log.info("Creating Gemini client...")

        _client = genai.Client(
            api_key=API_KEY
        )

    conversation = build_messages(history, message)

    contents = []

    for item in conversation:
        contents.append(
            types.Content(
                role=item["role"],
                parts=[
                    types.Part(
                        text=item["content"]
                    )
                ]
            )
        )

    log.info(
        "Sending request to Gemini model=%s",
        MODEL
    )

    response = _client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            max_output_tokens=300,
            temperature=0.7,
        ),
    )

    _llm_usage["n"] += 1

    reply = (response.text or "").strip()

    if not reply:
        return "Sorry, I couldn't generate a response."

    return reply


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "llm_configured": bool(API_KEY),
        "model": MODEL,
        "requests_today": _llm_usage["n"],
    }


# ---------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------

@app.post("/api/chat")
def chat(body: ChatIn, request: Request):

    rate_limit(request)

    message = body.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if find_sensitive(message):

        reply = (
            "For your security, please don't share card numbers, "
            "PINs, OTPs or passwords here. What else can I help with?"
        )

        log.info("mode=blocked")

        return {
            "reply": reply,
            "mode": "blocked"
        }

    # -----------------------------------------------------
    # Gemini availability
    # -----------------------------------------------------

    if not API_KEY:

        log.error("GEMINI_API_KEY is missing.")

        return {
            "reply": (
                "The AI service isn't configured yet. "
                "Please check the Gemini API key."
            ),
            "mode": "unavailable"
        }

    if not llm_allowed():

        log.warning("Daily Gemini request limit reached.")

        return {
            "reply": (
                "The AI service has reached its daily limit. "
                "Please try again later."
            ),
            "mode": "unavailable"
        }

    # -----------------------------------------------------
    # Gemini
    # -----------------------------------------------------

    try:

        reply = ask_gemini(
            body.history,
            message
        )

        log.info("mode=llm")

        return {
            "reply": reply,
            "mode": "llm"
        }

    except Exception as error:

        # IMPORTANT:
        # Print the real Gemini error in the backend terminal.
        log.exception(
            "Gemini request failed: %s",
            error
        )

        return {
            "reply": (
                "I couldn't connect to the AI service right now. "
                "Please try again."
            ),
            "mode": "error"
        }


# ---------------------------------------------------------
# Frontend
# ---------------------------------------------------------

frontend_path = (
    Path(__file__).resolve().parent.parent / "frontend"
)

app.mount(
    "/",
    StaticFiles(
        directory=frontend_path,
        html=True
    ),
    name="frontend"
)
