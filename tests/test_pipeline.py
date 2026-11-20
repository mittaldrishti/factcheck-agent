import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import factcheck as fc

PDF = pathlib.Path(__file__).resolve().parents[1] / "samples" / "trap_document.pdf"

def fake_llm(cfg, system, user):
    if "extract every" in system or "CHECKABLE" in system:
        return "```json\n" + json.dumps({"claims": [
            {"claim": "ChatGPT has 100 million weekly active users", "quote": "ChatGPT has 100 million weekly active users", "category": "statistic", "page": 1, "search_query": "ChatGPT weekly active users"},
            {"claim": "The Eiffel Tower is 450 metres tall", "quote": "x", "category": "technical", "page": 1, "search_query": "Eiffel Tower height"},
            {"claim": "Google was founded in 1998", "quote": "x", "category": "date", "page": 1, "search_query": "Google founded"},
        ]}) + "\n```"
    if "Eiffel" in user:
        return json.dumps({"verdict": "Inaccurate", "correct_fact": "About 330 m", "explanation": "e", "confidence": "high", "source_urls": ["http://a"]})
    if "ChatGPT" in user:
        return json.dumps({"verdict": "verified", "correct_fact": "", "explanation": "e", "source_urls": []})  # no evidence -> must flip
    return json.dumps({"verdict": "Verified", "correct_fact": "1998", "explanation": "e", "confidence": "high", "source_urls": ["http://b"]})

def fake_search(cfg, q, n=6):
    if "ChatGPT" in q: return []
    return [{"title": "T", "url": "http://a", "content": "c"}, {"title": "U", "url": "http://b", "content": "d"}]

def test_pdf_text():
    pages = fc.extract_pdf_pages(PDF.read_bytes())
    assert "Eiffel Tower" in pages[0]

def test_parse_json_fenced():
    assert fc.parse_json("```json\n{\"a\":1}\n```") == {"a": 1}
    assert fc.parse_json("Here: {\"a\":2} thanks") == {"a": 2}

def test_pipeline(monkeypatch):
    monkeypatch.setattr(fc, "call_llm", fake_llm)
    monkeypatch.setattr(fc, "search_web", fake_search)
    cfg = fc.Config(provider="gemini", llm_key="k")
    prog = []
    res = fc.run_pipeline(cfg, PDF.read_bytes(), lambda m, f: prog.append(f))
    assert [r.verdict for r in res] == ["False", "Inaccurate", "Verified"]  # order kept, no-evidence "Verified" -> False
    assert res[1].correct_fact == "About 330 m"
    assert fc.summarize(res) == {"Verified": 1, "Inaccurate": 1, "False": 1}
    assert prog[-1] == 1.0
    assert json.loads(fc.results_to_json(res))[1]["verdict"] == "Inaccurate"

def test_llm_failure_does_not_crash(monkeypatch):
    def boom(cfg, s, u):
        if "CHECKABLE" in s: return json.dumps({"claims": [{"claim": "x"}]})
        raise fc.LLMError("rate limited")
    monkeypatch.setattr(fc, "call_llm", boom); monkeypatch.setattr(fc, "search_web", fake_search)
    res = fc.run_pipeline(fc.Config(provider="gemini", llm_key="k"), PDF.read_bytes())
    assert res[0].verdict == "False" and "rate limited" in res[0].explanation
