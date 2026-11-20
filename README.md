# 🔎 Fact-Check Agent — the "Truth Layer"

Upload a marketing PDF → the agent extracts every checkable claim (stats, dates, financial & technical
figures), verifies each against **live web search**, and flags it as:

| Verdict | Meaning |
|---|---|
| ✅ **Verified** | Evidence on the live web matches the claim |
| ⚠️ **Inaccurate** | Partly right / outdated / wrong figure — the report shows the **correct, current fact** |
| ❌ **False** | Evidence contradicts the claim, or no credible evidence exists |

Every result carries an explanation, a confidence level and source links. Reports download as CSV/JSON.

**Live app:** `<paste your deployed URL here>`

## How it works

```
PDF ──pdfplumber/pypdf──▶ text per page
     ──LLM──▶ structured claims (claim, exact quote, category, page, precise search query)
     ──Tavily live web search (DuckDuckGo fallback)──▶ fresh evidence per claim (4 parallel workers)
     ──LLM judge (today's date injected)──▶ Verified | Inaccurate | False + correct fact + sources
     ──Streamlit──▶ colour-coded report + downloads
```

Design choices that matter for "trap" documents:
- **Today's date is injected** in every prompt, so stale figures (e.g. last year's user counts) are flagged as *Inaccurate*.
- The judge **must** return the correct current value for Inaccurate claims.
- **No evidence can never be "Verified"** — enforced in code, not just in the prompt.
- One failing claim (API hiccup) never crashes the report; it is surfaced as *False / could not verify*.
- Provider-agnostic: Gemini, OpenAI or Anthropic via plain HTTPS — no SDK lock-in.

## Project layout

```
app.py              Streamlit UI
factcheck.py        pipeline: PDF → claims → search → verdicts
tests/              pytest suite (LLM + search stubbed)
samples/            make_trap_pdf.py → trap_document.pdf for testing
requirements.txt
.streamlit/secrets.toml.example
```

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...        # or OPENAI_API_KEY / ANTHROPIC_API_KEY
export TAVILY_API_KEY=...        # optional but recommended (free tier at tavily.com)
streamlit run app.py
```

Generate a test PDF with deliberate lies/outdated stats: `pip install reportlab && python samples/make_trap_pdf.py`

Run tests: `pip install pytest && pytest`

## Deploy (Streamlit Community Cloud — free)

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io> → **New app** → pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings → Secrets**, paste (see `.streamlit/secrets.toml.example`):
   ```toml
   GEMINI_API_KEY = "..."
   TAVILY_API_KEY = "tvly-..."
   ```
4. Deploy, open the URL, upload a PDF.

Configuration: `LLM_MODEL` overrides the default model
(`gemini-2.5-flash`, `gpt-4o-mini`, `claude-haiku-5-5`).

## Limitations
- Scanned/image-only PDFs have no extractable text (no OCR).
- Verdict quality depends on the search evidence returned; low-confidence results are labelled.
- Very long PDFs are truncated to the first ~60k characters.
