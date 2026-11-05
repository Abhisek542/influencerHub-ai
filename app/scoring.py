"""Deterministic checks. The model judges text relevance; code decides scores, labels and proposed actions.

Rubric v1 from docs/IMPLEMENTATION_DESIGN.md, provisional. PROFILE_CONTEXT is not scored because brands
cannot read applicant bios through the current API.
"""
import re

NICHE_WEIGHT = 35
APPLICATION_WEIGHT = 40
MIN_MESSAGE_WORDS = 3

LIMITATIONS = [
    "Niche, platform and name are self-declared by the applicant.",
    "No follower, engagement or past-performance data exists on the platform.",
    "Applicant bios are not available to brands through the current API, so they were not used.",
    "Platform fit was not assessed because campaigns have no platform field.",
]

_INJECTION = re.compile(
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above)\s+(instructions|prompts?)"
    r"|disregard\s+(all\s+|the\s+)?(previous|prior|above)"
    r"|(rank|score|rate)\s+me\s+(first|highest|\d+)"
    r"|(accept|approve)\s+(this|my)\s+application\s+(immediately|now|automatically)"
    r"|system\s+prompt",
    re.IGNORECASE,
)
_METRIC_CLAIM = re.compile(
    r"\b(followers?|subscribers?|engagement|reach|impressions|conversion rates?|audience size)\b", re.IGNORECASE
)


def normalize(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().strip("\"'“”‘’").lower())


def looks_like_injection(text: str | None) -> bool:
    return bool(_INJECTION.search(text or ""))


def word_count(text: str | None) -> int:
    return len((text or "").split())


def grounded_quotes(quotes: list[str], source: str | None) -> list[str]:
    """Keeps only quotes that actually appear in the source text."""
    haystack = normalize(source)
    return [q for q in quotes if len(normalize(q)) >= 3 and normalize(q) in haystack]


def without_invented_metrics(statements: list[str], source: str | None) -> list[str]:
    """Drops blank statements and any that mention a metric the source text never mentioned."""
    source_text = (source or "").lower()
    return [
        s
        for s in statements
        if s.strip() and all(match.group(0).lower() in source_text for match in _METRIC_CLAIM.finditer(s))
    ]


def niche_score(campaign_niche: str | None, applicant_niche: str | None) -> int | None:
    if not normalize(campaign_niche) or not normalize(applicant_niche):
        return None
    return 100 if normalize(campaign_niche) == normalize(applicant_niche) else 0


def overall_score(niche: int | None, relevance_0_to_5: int | None) -> int | None:
    """Weighted average over scored criteria. Both are required, per the 'fewer than two' rule."""
    if niche is None or relevance_0_to_5 is None:
        return None
    application = relevance_0_to_5 * 20
    return round((niche * NICHE_WEIGHT + application * APPLICATION_WEIGHT) / (NICHE_WEIGHT + APPLICATION_WEIGHT))


def label_for(score: int | None) -> str:
    if score is None:
        return "INSUFFICIENT_EVIDENCE"
    if score >= 75:
        return "STRONG_MATCH"
    if score >= 50:
        return "POTENTIAL_MATCH"
    return "WEAK_MATCH"


def proposed_action(label: str, flagged: bool) -> str:
    """HOLD proposes no status change. Flagged or unclear applicants always go to a person."""
    if flagged:
        return "HOLD"
    return {"STRONG_MATCH": "ACCEPT", "WEAK_MATCH": "REJECT"}.get(label, "HOLD")
