"""Streamlit front-end for the Fact-Check Agent."""
import csv
import io

import streamlit as st

import factcheck as fc

st.set_page_config(page_title="Fact-Check Agent · Truth Layer", page_icon="🔎", layout="wide")


def secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def get_cfg() -> fc.Config:
    import os
    cfg = fc.Config.from_env(lambda k: secret(k) or os.environ.get(k))
    with st.sidebar:
        st.header("Settings")
        st.caption("Keys are read from deployment secrets. Override here only if needed.")
        provider = st.selectbox("LLM provider", ["gemini", "openai", "anthropic"],
                                index=["gemini", "openai", "anthropic"].index(cfg.provider or "gemini"))
        key = st.text_input("LLM API key (optional override)", type="password")
        tav = st.text_input("Tavily search key (optional override)", type="password")
        cfg.max_claims = st.slider("Max claims to check", 5, 30, 15)
        if key:
            cfg.provider, cfg.llm_key = provider, key
        elif cfg.provider != provider:
            # a different provider was chosen but its secret may still exist
            alt = secret({"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
                          "anthropic": "ANTHROPIC_API_KEY"}[provider])
            if alt:
                cfg.provider, cfg.llm_key = provider, alt
        if tav:
            cfg.tavily_key = tav
        st.markdown(f"**Model:** `{cfg.resolved_model() or 'n/a'}`")
        st.markdown(f"**Search:** {'Tavily' if cfg.tavily_key else 'DuckDuckGo fallback'}")
    return cfg


ICON = {"Verified": "✅", "Inaccurate": "⚠️", "False": "❌"}
COLOR = {"Verified": "#e6f4ea", "Inaccurate": "#fff4e0", "False": "#fce8e6"}

st.title("🔎 Fact-Check Agent — the Truth Layer")
st.write("Upload a marketing PDF. The agent extracts every checkable claim (stats, dates, financial and "
         "technical figures), verifies each one against **live web data**, and flags it as "
         "**Verified**, **Inaccurate** (outdated / wrong figure) or **False** (no evidence), "
         "with the correct fact and sources.")

cfg = get_cfg()
upload = st.file_uploader("Upload a PDF", type=["pdf"])

if upload and st.button("Run fact-check", type="primary"):
    if not cfg.llm_key:
        st.error("No LLM API key configured. Add one in the sidebar or in the app secrets.")
        st.stop()
    bar = st.progress(0.0, text="Starting…")
    try:
        results = fc.run_pipeline(cfg, upload.getvalue(),
                                  lambda msg, frac: bar.progress(min(frac, 1.0), text=msg))
        st.session_state["results"] = results
    except Exception as e:
        bar.empty()
        st.error(f"Something went wrong: {e}")
        st.stop()
    bar.empty()

results = st.session_state.get("results")
if results is not None:
    if not results:
        st.warning("No checkable claims were found in this PDF.")
        st.stop()
    counts = fc.summarize(results)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claims checked", len(results))
    c2.metric("✅ Verified", counts["Verified"])
    c3.metric("⚠️ Inaccurate", counts["Inaccurate"])
    c4.metric("❌ False", counts["False"])

    show = st.multiselect("Filter", list(fc.VERDICTS), default=list(fc.VERDICTS))
    for r in results:
        if r.verdict not in show:
            continue
        with st.container(border=True):
            st.markdown(
                f"<div style='background:{COLOR[r.verdict]};padding:6px 10px;border-radius:6px;"
                f"font-weight:600'>{ICON[r.verdict]} {r.verdict} — claim #{r.claim.id}"
                f"{f' (page {r.claim.page})' if r.claim.page else ''}</div>",
                unsafe_allow_html=True)
            st.markdown(f"**Claim:** {r.claim.claim}")
            if r.claim.quote and r.claim.quote != r.claim.claim:
                st.caption(f"In the PDF: “{r.claim.quote}”")
            if r.verdict != "Verified" or r.correct_fact:
                st.markdown(f"**Correct / real fact:** {r.correct_fact or 'No supporting evidence found.'}")
            st.markdown(f"**Why:** {r.explanation}  \n*Confidence: {r.confidence}*")
            if r.sources:
                st.markdown("**Sources:** " + " · ".join(
                    f"[{(s['title'] or s['url'])[:60]}]({s['url']})" for s in r.sources if s.get("url")))

    buf = io.StringIO()
    rows = [r.to_row() for r in results]
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download report (CSV)", buf.getvalue(), "factcheck_report.csv", "text/csv")
    d2.download_button("⬇️ Download report (JSON)", fc.results_to_json(results),
                       "factcheck_report.json", "application/json")
