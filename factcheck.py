"""Core pipeline for the Fact-Check Agent ("Truth Layer").

Steps:  PDF -> text -> claim extraction (LLM) -> live web search -> verdict (LLM)

Everything uses plain HTTPS calls (``requests``) so there are no provider SDKs to
install and the LLM / search back-ends can be swapped through environment
variables or Streamlit secrets.
"""
from __future__ import annotations

import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Callable, Optional

import requests

VERDICTS = ("Verified", "Inaccurate", "False")

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Claim:
    id: int
    claim: str                 # normalised, self-contained statement
    quote: str = ""            # exact sentence from the PDF
    category: str = "other"    # statistic | date | financial | technical | other
    search_query: str = ""
    page: Optional[int] = None


@dataclass
class Result:
    claim: Claim
    verdict: str = "False"             # Verified | Inaccurate | False
    correct_fact: str = ""             # what the live web says is right
    explanation: str = ""
    confidence: str = "medium"         # low | medium | high
    sources: list = field(default_factory=list)   # [{"title":..., "url":...}]

    def to_row(self) -> dict:
        return {
            "#": self.claim.id,
            "Page": self.claim.page,
            "Claim": self.claim.claim,
            "Verdict": self.verdict,
            "Correct / real fact": self.correct_fact,
            "Why": self.explanation,
            "Confidence": self.confidence,
            "Sources": " | ".join(s.get("url", "") for s in self.sources),
        }


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    provider: str = ""          # gemini | openai | anthropic  (auto-detected if empty)
    llm_key: str = ""
    model: str = ""
    tavily_key: str = ""
    max_claims: int = 15
    timeout: int = 60

    @classmethod
    def from_env(cls, getter: Callable[[str], Optional[str]] = os.environ.get) -> "Config":
        cfg = cls()
        for provider, var in (("gemini", "GEMINI_API_KEY"), ("openai", "OPENAI_API_KEY"),
                              ("anthropic", "ANTHROPIC_API_KEY")):
            if getter(var):
                cfg.provider, cfg.llm_key = provider, getter(var)
                break
        cfg.tavily_key = getter("TAVILY_API_KEY") or ""
        cfg.model = getter("LLM_MODEL") or ""
        return cfg

    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, "")


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def extract_pdf_pages(data: bytes) -> list[str]:
    """Return text per page. Tries pdfplumber first, then pypdf."""
    pages: list[str] = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [(p.extract_text() or "") for p in pdf.pages]
    except Exception:
        pages = []
    if not any(p.strip() for p in pages):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
    return pages


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
class LLMError(RuntimeError):
    pass


def call_llm(cfg: Config, system: str, user: str) -> str:
    """Send one prompt to the configured provider and return raw text."""
    if not cfg.llm_key:
        raise LLMError("No LLM API key configured (GEMINI_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY).")
    model = cfg.resolved_model()
    try:
        if cfg.provider == "gemini":
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": cfg.llm_key, "Content-Type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
                },
                timeout=cfg.timeout,
            )
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        if cfg.provider == "openai":
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {cfg.llm_key}"},
                json={
                    "model": model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                },
                timeout=cfg.timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        if cfg.provider == "anthropic":
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": cfg.llm_key, "anthropic-version": "2023-06-01"},
                json={
                    "model": model,
                    "max_tokens": 4096,
                    "temperature": 0,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                timeout=cfg.timeout,
            )
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"])
    except requests.HTTPError as e:
        raise LLMError(f"{cfg.provider} API error {e.response.status_code}: {e.response.text[:300]}") from e
    except requests.RequestException as e:
        raise LLMError(f"{cfg.provider} request failed: {e}") from e
    raise LLMError(f"Unknown provider '{cfg.provider}'.")


def parse_json(text: str):
    """Parse JSON from an LLM reply, tolerating code fences and stray prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if m:
            return json.loads(m.group(1))
        raise


# --------------------------------------------------------------------------- #
# Step 1: extract claims
# --------------------------------------------------------------------------- #
EXTRACT_SYSTEM = """You are a meticulous fact-checking analyst. From a document, extract every \
CHECKABLE factual claim: statistics, percentages, dates, rankings, market sizes, financial figures, \
prices, user counts, technical specs/benchmarks, and named records ("first", "largest").
Ignore opinions, slogans and vague statements.
Return JSON: {"claims":[{"claim": "<one self-contained sentence naming the subject, figure and year>",
"quote": "<exact sentence from the document>", "category": "statistic|date|financial|technical|other",
"page": <int or null>, "search_query": "<a precise web search query that would confirm or refute it>"}]}
Make each claim understandable on its own (name the company/product, the number, the time period).
Order by how likely the claim is to be wrong or outdated."""


def extract_claims(cfg: Config, pages: list[str]) -> list[Claim]:
    doc = "\n\n".join(f"[Page {i + 1}]\n{t}" for i, t in enumerate(pages) if t.strip())
    if not doc.strip():
        raise ValueError("No extractable text found in the PDF (it may be a scanned image).")
    user = (f"Today's date: {date.today().isoformat()}\nMax claims: {cfg.max_claims}\n\n"
            f"DOCUMENT:\n{doc[:60000]}")
    data = parse_json(call_llm(cfg, EXTRACT_SYSTEM, user))
    items = data.get("claims", data) if isinstance(data, dict) else data
    claims: list[Claim] = []
    for i, it in enumerate(items[: cfg.max_claims], start=1):
        if not isinstance(it, dict) or not it.get("claim"):
            continue
        claims.append(Claim(
            id=i, claim=str(it["claim"]).strip(), quote=str(it.get("quote", "")).strip(),
            category=str(it.get("category", "other")).lower(),
            search_query=str(it.get("search_query") or it["claim"]).strip(),
            page=it.get("page") if isinstance(it.get("page"), int) else None,
        ))
    return claims


# --------------------------------------------------------------------------- #
# Step 2: live web search
# --------------------------------------------------------------------------- #
def search_web(cfg: Config, query: str, n: int = 6) -> list[dict]:
    """Return [{title,url,content}] from Tavily, or DuckDuckGo HTML as a keyless fallback."""
    if cfg.tavily_key:
        try:
            r = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {cfg.tavily_key}"},
                json={"query": query, "max_results": n, "search_depth": "advanced",
                      "topic": "general", "include_answer": False},
                timeout=cfg.timeout,
            )
            r.raise_for_status()
            return [{"title": x.get("title", ""), "url": x.get("url", ""),
                     "content": (x.get("content") or "")[:1200]} for x in r.json().get("results", [])]
        except requests.RequestException:
            pass  # fall through to DuckDuckGo
    return _duckduckgo(query, n, cfg.timeout)


def _duckduckgo(query: str, n: int, timeout: int) -> list[dict]:
    from bs4 import BeautifulSoup
    from urllib.parse import parse_qs, unquote, urlparse
    try:
        r = requests.post("https://html.duckduckgo.com/html/", data={"q": query},
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for res in soup.select(".result")[:n]:
        a = res.select_one(".result__a")
        if not a:
            continue
        href = a.get("href", "")
        if "uddg=" in href:
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        snip = res.select_one(".result__snippet")
        out.append({"title": a.get_text(strip=True), "url": href,
                    "content": snip.get_text(" ", strip=True) if snip else ""})
    return out


# --------------------------------------------------------------------------- #
# Step 3: verdict
# --------------------------------------------------------------------------- #
VERIFY_SYSTEM = """You are a strict fact-checker. You get ONE claim from a document and fresh web search \
evidence. Decide using ONLY the evidence plus well-established knowledge; prefer the most recent, \
authoritative sources (official filings, company/government sites, reputable press).
Verdicts:
- "Verified": evidence matches the claim (allow trivial rounding).
- "Inaccurate": the claim is partly right or was once right, but the figure/date is wrong or OUTDATED. \
You MUST give the correct current value and its date in "correct_fact".
- "False": evidence contradicts the claim outright or no credible evidence exists that it is true. If the \
evidence shows the real fact, put it in "correct_fact"; otherwise say none was found.
Never mark Verified without supporting evidence. Be specific with numbers and dates.
Return JSON: {"verdict":"Verified|Inaccurate|False","correct_fact":"...","explanation":"<=2 sentences",
"confidence":"low|medium|high","source_urls":["<urls actually used, from the evidence>"]}"""


def verify_claim(cfg: Config, claim: Claim) -> Result:
    evidence = search_web(cfg, claim.search_query or claim.claim)
    if not evidence:  # try once more with the raw claim
        evidence = search_web(cfg, claim.claim)
    ev_text = "\n\n".join(
        f"[{i + 1}] {e['title']}\nURL: {e['url']}\n{e['content']}" for i, e in enumerate(evidence)
    ) or "NO SEARCH RESULTS RETURNED."
    user = (f"Today's date: {date.today().isoformat()}\n\nCLAIM: {claim.claim}\n"
            f"ORIGINAL SENTENCE: {claim.quote}\n\nWEB EVIDENCE:\n{ev_text}")
    try:
        data = parse_json(call_llm(cfg, VERIFY_SYSTEM, user))
    except Exception as e:  # never crash the whole report on one claim
        return Result(claim, "False", "", f"Could not verify automatically ({e}).", "low", [])
    verdict = str(data.get("verdict", "False")).strip().capitalize()
    if verdict not in VERDICTS:
        verdict = "False"
    urls = data.get("source_urls") or []
    by_url = {e["url"]: e["title"] for e in evidence}
    sources = [{"title": by_url.get(u, u), "url": u} for u in urls if isinstance(u, str)] or \
              [{"title": e["title"], "url": e["url"]} for e in evidence[:2]]
    if not evidence and verdict == "Verified":
        verdict = "False"  # no evidence can never mean verified
    return Result(claim, verdict, str(data.get("correct_fact", "")).strip(),
                  str(data.get("explanation", "")).strip(),
                  str(data.get("confidence", "medium")).lower(), sources)


def run_pipeline(cfg: Config, pdf_bytes: bytes,
                 progress: Optional[Callable[[str, float], None]] = None) -> list[Result]:
    say = progress or (lambda msg, frac: None)
    say("Reading PDF…", 0.05)
    pages = extract_pdf_pages(pdf_bytes)
    say("Extracting claims…", 0.15)
    claims = extract_claims(cfg, pages)
    if not claims:
        return []
    results: list[Optional[Result]] = [None] * len(claims)
    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(verify_claim, cfg, c): i for i, c in enumerate(claims)}
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            results[futs[f]] = f.result()
            done += 1
            say(f"Verified {done}/{len(claims)} claims…", min(1.0, 0.2 + 0.8 * done / len(claims)))
    return [r for r in results if r]


def summarize(results: list[Result]) -> dict:
    return {v: sum(1 for r in results if r.verdict == v) for v in VERDICTS}


def results_to_json(results: list[Result]) -> str:
    return json.dumps([{**asdict(r.claim), **{k: v for k, v in asdict(r).items() if k != "claim"}}
                       for r in results], indent=2)
