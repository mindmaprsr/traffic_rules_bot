import re

_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|"
    "eleven|twelve|thirteen|fourteen|fifteen|"
    "twenty|thirty|forty|fifty|hundred"
)
_THE_BEFORE_NUMBER = re.compile(
    rf"\bthe\s+(?=\d+|(?:{_NUMBER_WORDS})\b)", re.IGNORECASE
)

def normalize_query(question: str) -> str:
    normalized = _THE_BEFORE_NUMBER.sub("", question).strip()
    normalized = re.sub(r" {2,}", " ", normalized)
    if normalized != question:
        print(f"Original  : {question}")
        print(f"Normalized: {normalized}")
    return normalized
