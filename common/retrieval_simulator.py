"""A tiny deterministic retriever and a fictional knowledge base ("Acme Cloud docs").

Real retrievers (dense, hybrid, rerankers) behave differently in the details, but they share
the property these examples depend on: they always return the top-k nearest chunks, whether or
not any of them answers the question. Keeping the retriever lexical and dependency-free makes
every example reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .utils import content_tokens


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str = "acme-docs"


ACME_CHUNKS: list[Chunk] = [
    Chunk("refund", "Refund policy: Annual plans can be refunded in full within 30 days of purchase. "
          "Monthly plans are not refundable, but you can cancel at any time and keep access until the end of the billing period."),
    Chunk("pricing", "Pricing: The Starter plan costs $29 per month. The Pro plan costs $99 per month. "
          "Enterprise pricing is negotiated per contract."),
    Chunk("sla", "Uptime SLA: The Pro plan includes a 99.9% uptime SLA. Enterprise plans include a 99.99% uptime SLA. "
          "Starter has no SLA."),
    Chunk("rate-limits", "API rate limits: Starter allows 100 requests per minute. Pro allows 1,000 requests per minute. "
          "Exceeding the limit returns HTTP 429."),
    Chunk("retention", "Log retention: Logs are retained for 30 days on Starter and 90 days on Pro. "
          "Enterprise retention is configurable up to 365 days."),
    Chunk("support", "Support hours: Support is available Monday to Friday, 9am to 6pm UTC. "
          "Enterprise customers get 24/7 support with a 1 hour response target."),
    Chunk("sso", "Single sign-on: SAML SSO is available on Enterprise plans only. "
          "Starter and Pro use email and password or Google login."),
    Chunk("regions", "Regions: Acme Cloud runs in us-east-1 and eu-west-1. "
          "Data does not leave the region you choose at project creation."),
    Chunk("security", "Security: Data is encrypted at rest with AES-256 and in transit with TLS 1.2 or higher. "
          "Acme Cloud is SOC 2 Type II certified."),
    Chunk("deployments", "Deployments: Each project supports up to 10 concurrent deployments on Pro and 2 on Starter."),
    Chunk("webhooks", "Webhooks: Acme Cloud sends a signed POST request to your webhook URL when a deployment finishes. "
          "Failed deliveries are retried up to 5 times with exponential backoff."),
    # Distractors that share vocabulary with refund/cancel questions but contain no refund terms.
    Chunk("cancel-howto", "How to cancel: To cancel your annual or monthly plan, open Settings, choose Billing, then select Cancel plan. "
          "Back up your data first, because cancelling schedules your project for deletion after the billing period ends."),
    Chunk("annual-marketing", "Annual plans: Choose an annual plan to get priority onboarding and a yearly plan review session. "
          "Annual plans are our most popular option for teams."),
    Chunk("plan-settings", "Plan settings: Open Settings, then Plan, to see your current plan. "
          "Annual and monthly plans are managed from the same page, and you can back up your plan settings as JSON."),
]

NOISE_CHUNKS: list[Chunk] = [
    Chunk(f"noise-{i:02d}", text, source="release-notes")
    for i, text in enumerate(
        [
            "Release 4.1: The dashboard now loads faster and the billing page footer typo was fixed.",
            "Release 4.2: Added dark mode to the console. Keyboard shortcuts are listed under Help.",
            "Release 4.3: The CLI now prints deployment URLs in green. Windows path handling improved.",
            "Release 4.4: Project names may now include underscores. Tag colors were updated.",
            "Release 4.5: The audit view can be sorted by date. Pagination defaults changed to 50 rows.",
            "Release 4.6: Improved empty-state illustrations across the console.",
            "Release 4.7: The status widget now shows the last five incidents.",
            "Release 4.8: Team invitations expire after 7 days and can be resent.",
            "Release 4.9: The metrics chart supports zooming. Export to CSV was fixed.",
            "Release 5.0: New onboarding checklist. The sidebar can be collapsed.",
            "Release 5.1: Added a search box to the project list.",
            "Release 5.2: Fixed a rendering bug in the logs viewer on Safari.",
        ],
        start=1,
    )
]

_BY_ID = {c.id: c for c in ACME_CHUNKS + NOISE_CHUNKS}


def get_chunk(chunk_id: str) -> Chunk:
    return _BY_ID[chunk_id]


class KeywordRetriever:
    """IDF-weighted query-term coverage. Score is in [0, 1]: the share of query weight a chunk covers."""

    def __init__(self, chunks: Optional[Sequence[Chunk]] = None) -> None:
        self.chunks = list(chunks if chunks is not None else ACME_CHUNKS)
        self._tokens = {c.id: set(content_tokens(c.text)) for c in self.chunks}
        self._n = len(self.chunks)
        self._df: dict[str, int] = {}
        for toks in self._tokens.values():
            for tok in toks:
                self._df[tok] = self._df.get(tok, 0) + 1

    def idf(self, token: str) -> float:
        return math.log((self._n + 1) / (self._df.get(token, 0) + 1)) + 1.0

    def score(self, query: str, chunk: Chunk) -> float:
        q = set(content_tokens(query))
        if not q:
            return 0.0
        total = sum(self.idf(t) for t in q)
        hit = sum(self.idf(t) for t in q if t in self._tokens[chunk.id])
        return hit / total

    def retrieve(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        scored = [(c, self.score(query, c)) for c in self.chunks]
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))  # id tie-break keeps runs deterministic
        return scored[:k]


def format_context(chunks: Sequence[Chunk], style: str = "index") -> str:
    """Render chunks as '[1] text' lines (style='index') or '[chunk-id] text' (style='id')."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        label = str(i) if style == "index" else chunk.id
        lines.append(f"[{label}] {chunk.text}")
    return "\n".join(lines)
