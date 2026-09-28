"""Safety checks that run in code, because a prompt alone is not reliable."""
import re

CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN = re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b")
SECRET_WITH_DIGITS = re.compile(
    r"\b(otp|cvv|cvc|pin|password|passcode)\b\D{0,20}\d{3,8}\b", re.I
)
HUMAN = re.compile(
    r"\b(human|agent|representative|operator|real person|customer care|"
    r"speak to someone|talk to someone|complaint)\b", re.I
)
URGENT = re.compile(
    r"\b(fraud|scam|stolen|hacked|unauthori[sz]ed|lost my card|card (is )?lost)\b", re.I
)


def find_sensitive(text: str) -> bool:
    return any(p.search(text) for p in (CARD, AADHAAR, PAN, SECRET_WITH_DIGITS))


def mask(text: str) -> str:
    for p in (SECRET_WITH_DIGITS, CARD, AADHAAR, PAN):
        text = p.sub("[removed]", text)
    return text


def wants_human(text: str) -> bool:
    return bool(HUMAN.search(text))


def urgent(text: str) -> bool:
    return bool(URGENT.search(text))
