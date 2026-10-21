"""Shared helpers: tokenization, lexical support scoring, CLI and output formatting.

The lexical scoring is deliberately simple (no embeddings, no model calls) so every detector
in the catalog runs offline and deterministically. It is a baseline, not a state-of-the-art
grounding checker. Each failure README says where to swap in an NLI model or an LLM judge.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STOPWORDS = frozenset(
    """a an the and or but if then of to in on at by for with from as is are was were be been
    being do does did have has had it its this that these those you your we our they their i me
    my can could should would will may might what which who whom how when where why not no yes
    also than about into over per any all each so there here""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'])")


# --------------------------------------------------------------------------- text helpers
def stem(word: str) -> str:
    """Very small suffix stripper. Good enough to match 'plans'/'plan' and 'refunded'/'refund'."""
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_tokens(text: str) -> list[str]:
    """Lowercased, stemmed, stopword-free tokens. '1,000' and '1000' normalize to the same token."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(_THOUSANDS_RE.sub("", text.lower())):
        if tok in STOPWORDS:
            continue
        if len(tok) < 2 and not tok.isdigit():
            continue
        out.append(stem(tok))
    return out


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def extract_citations(sentence: str) -> list[int]:
    """Return citation indices like [1] or [1, 3] found in a sentence."""
    out: list[int] = []
    for group in _CITATION_RE.findall(sentence):
        out.extend(int(x) for x in re.split(r"\s*,\s*", group))
    return out


def strip_citations(text: str) -> str:
    cleaned = _CITATION_RE.sub("", text)
    return re.sub(r"\s+([.,;!?])", r"\1", cleaned).strip()


def support_detail(claim: str, context: str) -> tuple[float, list[str]]:
    """How much of `claim` is lexically covered by `context`?

    Returns (score in [0, 1], tokens missing from context). Numeric tokens count double:
    a wrong number is the most expensive kind of unsupported content and the easiest to miss.
    """
    claim_toks = set(content_tokens(strip_citations(claim)))
    # Strip '[1]'-style chunk labels from the context so they cannot 'support' a stray digit.
    ctx_toks = set(content_tokens(strip_citations(context)))
    if not claim_toks:
        return 1.0, []

    def weight(tok: str) -> float:
        return 2.0 if any(ch.isdigit() for ch in tok) else 1.0

    total = sum(weight(t) for t in claim_toks)
    hit = sum(weight(t) for t in claim_toks if t in ctx_toks)
    return hit / total, sorted(t for t in claim_toks if t not in ctx_toks)


def support_score(claim: str, context: str) -> float:
    return support_detail(claim, context)[0]


# --------------------------------------------------------------------------- CLI / output
def base_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call a real model (needs OPENAI_API_KEY or ANTHROPIC_API_KEY) instead of the offline stub.",
    )
    return parser


def banner(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def status(ok: bool, label: str, detail: str = "") -> None:
    tag = "[ OK ]" if ok else "[FAIL]"
    print(f"{tag} {label}" + (f"  -- {detail}" if detail else ""))


def verdict(reproduced: bool, message: str) -> None:
    """Uniform end-of-run line. The smoke tests look for 'FAILURE REPRODUCED'."""
    if reproduced:
        print(f"\nFAILURE REPRODUCED: {message}")
    else:
        print(f"\nFailure not reproduced in this run: {message}")


# --------------------------------------------------------------------------- module loading
def load_sibling(current_file: str, name: str):
    """Import `name.py` from the same folder as `current_file`.

    Every catalog folder has a `reproduce.py`, so a plain `import reproduce` would collide in
    sys.modules when several entries are loaded in one process (e.g. a notebook). Unique module
    names avoid that.
    """
    path = Path(current_file).resolve().with_name(f"{name}.py")
    mod_name = f"catalog_{path.parent.name}_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def load_catalog_module(folder: str, name: str):
    """Load catalog/<folder>/<name>.py from anywhere (used by the quickstart notebook)."""
    anchor = REPO_ROOT / "catalog" / folder / f"{name}.py"
    return load_sibling(str(anchor), name)
