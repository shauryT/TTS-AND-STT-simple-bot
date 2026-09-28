# Bank Voice Assistant

A voice chatbot: a **General Assistant** mode for open conversation, plus per-bank modes that troubleshoot errors specific to one bank. Speech in, AI answer, speech out.

> Demo only. All banks, phone numbers and error codes are fictional. Do not use real customer data.

## How it works

```
Browser mic -> Speech-to-text (Web Speech API)
    -> POST /api/chat {bank_id, message, history}
        1. Safety check   card/Aadhaar/PAN/OTP/PIN detected -> refuse, never stored or sent to the LLM
        2. Escalation     fraud/stolen -> urgent block number; "agent" -> human support number
        3. Retrieval      search ONLY the selected bank's knowledge base (data/banks/<bank>.json)
        4. No match       say so and escalate, never guess
        5. Answer         LLM rephrases the retrieved text (if API key set), else the KB text is returned
    -> Text-to-speech (browser speechSynthesis)
```

Key idea: the same error code (for example `E104`) means something different at each bank. The assistant only ever sees the selected bank's records, so answers cannot leak between banks.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # optional: add ANTHROPIC_API_KEY
export $(grep -v '^#' .env | xargs)                   # or set variables your own way
uvicorn backend.main:app --reload
```

Open http://localhost:8000 in **Chrome or Edge** (the mic works on localhost).
With no API key it runs in retrieval-only mode, which is still fully functional.

Run tests: `pip install pytest && pytest`

## Deploy on Render

1. Push this repo to GitHub (make sure `.env` is NOT committed; it is in `.gitignore`).
2. On render.com: **New +** -> **Blueprint** -> pick your repo. Render reads `render.yaml`.
   (Or **New Web Service** with build `pip install -r requirements.txt` and start `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.)
3. In the service's **Environment** tab, set `ANTHROPIC_API_KEY` (optional).
4. Open the `.onrender.com` URL. It is HTTPS, so the mic works.

Free tier note: the service sleeps when idle, so the first request after a pause can take 30+ seconds. Open it once before a demo.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | empty | Enables LLM answers. Empty means retrieval-only |
| `LLM_MODEL` | `claude-haiku-4-5-20251001` | Model used for replies |
| `DAILY_LLM_CAP` | `200` | Max LLM calls per day, then falls back to retrieval-only |
| `RATE_PER_MIN` | `20` | Requests per minute per IP |

## Add a bank

Copy any file in `data/banks/`, change `id`, `name`, contacts and `entries`, and restart. No code changes needed.

## Safety and privacy

- Sensitive data is detected in code, not only in the prompt.
- Logs record bank, mode and matched codes only, never what the caller said.
- The API key lives only in server environment variables.
- Rate limit and daily LLM cap protect your credits on a public URL.

## Known limitations

- Browser speech recognition is weaker for Indian languages and accents. Whisper or Google/Azure STT is the upgrade path.
- Keyword retrieval is simple. Embeddings-based retrieval would handle paraphrases better.
- Spoken digits ("four one one one...") may slip past the digit filter.
- No telephony or caller authentication yet.

## Roadmap

1. Telephony via Twilio or Exotel with server-side STT/TTS
2. Embeddings-based retrieval (RAG)
3. Caller authentication before any account-specific action
4. Call analytics and audit logging that follows bank and RBI requirements

## Structure

```
backend/   main.py (API), retrieval.py (per-bank search), safety.py (filters)
frontend/  index.html (voice UI)
data/banks/  one JSON knowledge base per bank
tests/     pytest tests
render.yaml  Render blueprint
```
