"""
Query Router: classifies a user query into a retrieval strategy.

Query types and their routing rationale:
  Penalty      → LQE   (vocabulary: "fine" → "penalty units / guilty of an offence")
  Prohibition  → StepBack (reverse-semantic: Act writes what CAN be done, not what cannot)
  CrossSection → MultiQuery (meta question: try multiple angles across chapters)
  Other        → HyDE  (already works well for Definitional/Authority/Procedural/Eligibility/Timeframe)
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=".env")

_client = OpenAI(
    base_url=os.environ["Z_AI_BASE_URL"],
    api_key=os.environ["Z_AI_API_KEY"],
)

_CLASSIFY_PROMPT = """\
Classify the question into exactly ONE of these categories. Reply with the category name only.

Categories:
- Penalty       : asks about fines, penalties, offences, consequences for breaching a rule
- Prohibition   : asks what is prohibited, not allowed, must not be done, or what restrictions exist
- CrossSection  : asks about interaction, relationship, or connection between different parts of the Act
- Other         : anything else (definitions, authority, procedures, eligibility, timeframes)

Examples:
Q: What is the fine for operating without a licence?     → Penalty
Q: What must not be done when dissolving an organisation? → Prohibition
Q: How do Chapter 4 provisions interact with Chapter 6?  → CrossSection
Q: What is the definition of a public hospital?          → Other
Q: Who appoints a board member?                          → Other
Q: How long does a board member hold office?             → Other

Question: {query}
Category:"""


def classify_query(query: str) -> str:
    """Return one of: Penalty | Prohibition | CrossSection | Other"""
    resp = _client.chat.completions.create(
        model=os.environ["Z_AI_CHAT_MODEL"],
        messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(query=query)}],
        temperature=0,
        max_tokens=2000,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Extract just the first word to guard against verbose responses
    first_word = raw.split()[0].rstrip(".:,") if raw else "Other"
    if first_word in ("Penalty", "Prohibition", "CrossSection"):
        return first_word
    return "Other"
