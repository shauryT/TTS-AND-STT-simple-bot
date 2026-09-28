import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .safety import find_sensitive, mask


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

log = logging.getLogger("voice")


# =========================================================
# GEMINI CONFIG
# =========================================================

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

MODEL = os.getenv(
    "LLM_MODEL",
    "gemini-3.5-flash-lite"
).strip()

DAILY_CAP = int(
    os.getenv("DAILY_LLM_CAP", "200")
)

RATE_PER_MIN = int(
    os.getenv("RATE_PER_MIN", "20")
)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM = """
You are a friendly AI voice assistant.

Your responses will usually be spoken aloud.

Keep replies natural, short and conversational.
Use 1 to 3 short sentences unless the user asks for more detail.

Do not use markdown.
Do not use bullet points.
Do not use emojis.

Answer the user's question directly.

Be accurate.
If you are unsure about something, say that you are unsure.

Never ask for or repeat:
card numbers,
PINs,
CVVs,
OTPs,
passwords,
or other sensitive authentication information.

Reply in the same language that the user uses.
"""


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="AI Voice Assistant",
    version="1.0.0"
)


# =========================================================
# STATE
# =========================================================

_hits = defaultdict(deque)

_llm_usage = {
    "day": time.strftime("%Y-%m-%d"),
    "count": 0
}

_client = None


# =========================================================
# DATA MODELS
# =========================================================

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(
        max_length=600
    )


class ChatIn(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=300
    )

    history: list[Turn] = Field(
        default_factory=list,
        max_length=12
    )


# =========================================================
# RATE LIMITING
# =========================================================

def rate_limit(request: Request):

    forwarded = request.headers.get(
        "x-forwarded-for",
        ""
    )

    ip = (
        forwarded.split(",")[0].strip()
        or (
            request.client.host
            if request.client
            else "unknown"
        )
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


# =========================================================
# DAILY LIMIT
# =========================================================

def llm_allowed():

    today = time.strftime("%Y-%m-%d")

    if _llm_usage["day"] != today:

        _llm_usage["day"] = today
        _llm_usage["count"] = 0

    return (
        bool(API_KEY)
        and
        _llm_usage["count"] < DAILY_CAP
    )


# =========================================================
# BUILD CHAT HISTORY
# =========================================================

def build_messages(
    history: list[Turn],
    message: str
):

    messages = []

    for turn in history[-6:]:

        content = mask(
            turn.content
        ).strip()

        if not content:
            continue

        role = (
            "model"
            if turn.role == "assistant"
            else "user"
        )

        if (
            messages
            and
            messages[-1]["role"] == role
        ):

            messages[-1]["content"] += (
                " " + content
            )

        else:

            messages.append(
                {
                    "role": role,
                    "content": content
                }
            )

    current = mask(
        message
    ).strip()

    if (
        messages
        and
        messages[-1]["role"] == "user"
    ):

        messages[-1]["content"] += (
            " " + current
        )

    else:

        messages.append(
            {
                "role": "user",
                "content": current
            }
        )

    return messages


# =========================================================
# CREATE GEMINI CLIENT
# =========================================================

def get_gemini_client():

    global _client

    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in .env"
        )

    if _client is None:

        log.info(
            "Creating Gemini client"
        )

        from google import genai

        _client = genai.Client(
            api_key=API_KEY
        )

    return _client


# =========================================================
# ASK GEMINI
# =========================================================

def ask_gemini(
    history: list[Turn],
    message: str
):

    from google.genai import types

    client = get_gemini_client()

    messages = build_messages(
        history,
        message
    )

    contents = []

    for item in messages:

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
        "Sending request to Gemini | model=%s",
        MODEL
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            max_output_tokens=300,
            temperature=0.7
        )
    )

    _llm_usage["count"] += 1

    reply = (
        response.text or ""
    ).strip()

    if not reply:
        return (
            "I couldn't generate a response. "
            "Could you try asking that again?"
        )

    return reply


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "ok": True,
        "gemini_configured": bool(API_KEY),
        "model": MODEL,
        "requests_today": _llm_usage["count"]
    }


# =========================================================
# CHAT
# =========================================================

@app.post("/api/chat")
def chat(
    body: ChatIn,
    request: Request
):

    rate_limit(request)

    message = body.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # -----------------------------------------------------
    # SAFETY
    # -----------------------------------------------------

    if find_sensitive(message):

        reply = (
            "For your security, please don't share "
            "card numbers, PINs, OTPs or passwords here. "
            "What else can I help with?"
        )

        log.info(
            "Chat blocked by safety check"
        )

        return {
            "reply": reply,
            "mode": "blocked"
        }

    # -----------------------------------------------------
    # API KEY
    # -----------------------------------------------------

    if not API_KEY:

        log.error(
            "GEMINI_API_KEY is missing."
        )

        return {
            "reply": (
                "The AI service is not configured. "
                "Please check the Gemini API key."
            ),
            "mode": "unavailable"
        }

    # -----------------------------------------------------
    # DAILY LIMIT
    # -----------------------------------------------------

    if not llm_allowed():

        log.warning(
            "Daily Gemini limit reached."
        )

        return {
            "reply": (
                "The AI service has reached "
                "its daily limit. Please try again later."
            ),
            "mode": "unavailable"
        }

    # -----------------------------------------------------
    # GEMINI
    # -----------------------------------------------------

    try:

        reply = ask_gemini(
            body.history,
            message
        )

        log.info(
            "Gemini response successful"
        )

        return {
            "reply": reply,
            "mode": "llm"
        }

    except Exception as error:

        # Print the REAL error to the terminal.
        log.exception(
            "GEMINI REQUEST FAILED"
        )

        return {
            "reply": (
                "I couldn't get a response from "
                "the AI right now. Please try again."
            ),
            "mode": "error"
        }


# =========================================================
# FRONTEND
# =========================================================

FRONTEND_DIR = (
    BASE_DIR / "frontend"
)

if FRONTEND_DIR.exists():

    app.mount(
        "/",
        StaticFiles(
            directory=FRONTEND_DIR,
            html=True
        ),
        name="frontend"
    )

else:

    log.warning(
        "Frontend directory not found: %s",
        FRONTEND_DIR
    )
