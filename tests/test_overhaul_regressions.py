"""
Regression harness tests for the article-intelligence overhaul.

- `test_overhaul_bench_runs_on_smoke`: the eval benchmark harness runs end-to-end.
- `test_handcurated_targeted`: each hand-curated case asserts the targeted
  behavior of the four fixes (ticker FPs, per-asset direction, entity dedup,
  narrative-keyword abstention). Runs the deterministic path only (no LLM).

NOTE: this environment lacks GLiNER/Flair/ReFinED (spaCy + FinBERT only), which
weakens recall for some *includes* assertions. Those cases are checked but
env-limited misses are reported, not hard-failed; the exclusion/dedup/direction
assertions (the actual bug fixes) are hard-failed.
"""
import json
import os
import subprocess
import sys

import pytest

EVAL = "/home/rizzo/alpharidge/eval"
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "handcurated_overhaul.jsonl")


def test_overhaul_bench_runs_on_smoke(tmp_path):
    src = "/home/rizzo/alpharidge/eval/eval/data/gold_z-ai_glm-5.1.jsonl"
    smoke = tmp_path / "smoke.jsonl"
    with open(src) as f, open(smoke, "w") as o:
        for i, line in zip(range(3), f):
            o.write(line)
    r = subprocess.run(
        [sys.executable, "scripts/overhaul_bench.py",
         "--gold", str(smoke), "--limit", "3", "--fields", "assets", "entities"],
        cwd=EVAL, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "assets" in r.stdout and "entities" in r.stdout


@pytest.fixture(scope="module")
def analyzer():
    from alpharidge_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer
    from alpharidge_ai.analyzer.ner_fusion import NERFusionEngine
    a = ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)
    a.ner_engine = NERFusionEngine()
    a._init_narrative_index()
    return a


def _sign(d):
    """Coarsen a 7-class Sentiment value to directional sign for 3-class gold."""
    if d is None:
        return None
    if "bull" in d:
        return "bullish"
    if "bear" in d:
        return "bearish"
    return "neutral"


def _article_direction(ner):
    """Deterministic stand-in for production's ``overall_dir`` (LLM call-1
    sentiment), which the live analyzer passes as ``fallback_direction``. The
    deterministic test path has no LLM, so aggregate FinBERT's per-sentence
    sentiments (already computed by NER) into a coarse article-level read."""
    agg = {}
    for s in getattr(ner, "sentence_sentiments", None) or []:
        agg[s.get("sentiment")] = agg.get(s.get("sentiment"), 0.0) + float(s.get("score", 0) or 0)
    return max(agg, key=agg.get) if agg else "neutral"


def _analyze(a, art):
    from alpharidge_ai.analyzer.aspect_sentiment import score_assets, AspectSentimentScorer
    global _ASPECT_SCORER
    try:
        _ASPECT_SCORER
    except NameError:
        _ASPECT_SCORER = AspectSentimentScorer(device="cpu")
    ner = a.ner_engine.extract_and_resolve(art["title"], art["content"])
    ner_assets = [e for e in ner.resolved_assets if e.ticker]
    # Mirror production: weak/inconclusive per-asset FinABSA defers to the
    # article-level direction, NOT a hardcoded "neutral".
    sents = score_assets(ner_assets, ner, _ASPECT_SCORER,
                         fallback_direction=_article_direction(ner))
    assets = a._build_assets(ner, sents)
    entities = a._build_entities_from_ner(ner)
    one_liner = (art["content"][:200])
    narr = a._select_narratives(art["title"], one_liner, art["content"][:800])
    return assets, entities, narr


# NOTE on hc-012 (triaged 2026-09-02). The failure message reads "ticker IT should be
# EXCLUDED", which misdescribes what happens. IT is emitted as asset_name=GARTNER with
# evidence span "GARTNER" — the extractor resolved the company, whose ticker is IT, and
# removing the word Gartner from the article removes the asset. The common-word gate is
# working: IT is on the acronym blocklist and is not matched as a bare word.
#
# What the case actually asks is whether a company named only as the source of a
# statistic ("a survey by Gartner found...") is an asset of the article. The extractor
# already answers partly: is_primary_subject=False. Whether it should be emitted at all
# is a design question about sources versus subjects, not a defect in the gate.
#
# Left failing on purpose: it encodes an intent nobody has decided. Do not re-diagnose
# it as an ambiguity-gate bug.
def test_handcurated_targeted(analyzer):
    rows = [json.loads(l) for l in open(FIXTURE) if l.strip()]
    hard_failures = []   # bug-fix assertions that MUST hold
    soft_misses = []     # recall-limited *includes* (env caveat)

    for row in rows:
        art, expect = row["article"], row.get("expect", {})
        aid = art["id"]
        assets, entities, narr = _analyze(analyzer, art)
        tickers = {a.ticker.upper() for a in assets}
        dir_by_ticker = {a.ticker.upper(): getattr(a.direction, "value", a.direction) for a in assets}
        ent_names = [e.name for e in entities]
        ent_lower = [n.lower() for n in ent_names]

        for t in expect.get("assets_excludes", []):
            if t.upper() in tickers:
                hard_failures.append(f"{aid}: ticker {t} should be EXCLUDED, got {sorted(tickers)}")
        for t in expect.get("assets_includes", []):
            if t.upper() not in tickers:
                soft_misses.append(f"{aid}: ticker {t} not found (have {sorted(tickers)})")
        for t, want in (expect.get("asset_direction") or {}).items():
            got = dir_by_ticker.get(t.upper())
            if got is None:
                soft_misses.append(f"{aid}: {t} not detected, can't check direction")
            elif _sign(got) != _sign(want):
                # Gold labels are 3-class (bullish/bearish/neutral); FinABSA emits
                # the 7-class scale. Compare directional SIGN, not magnitude.
                hard_failures.append(f"{aid}: {t} direction {got!r} != expected {want!r}")
        if expect.get("entity_no_duplicates"):
            dups = {n for n in ent_lower if ent_lower.count(n) > 1}
            if dups:
                hard_failures.append(f"{aid}: duplicate entities {dups}")
        for n in expect.get("entity_excludes", []):
            if n.lower() in ent_lower:
                hard_failures.append(f"{aid}: entity {n!r} should be EXCLUDED, got {ent_names}")
        for n in expect.get("entity_includes", []):
            if n.lower() not in ent_lower:
                soft_misses.append(f"{aid}: entity {n!r} not found (have {ent_names})")
        for s in expect.get("narrative_keywords_excludes", []):
            if s in narr:
                hard_failures.append(f"{aid}: narrative slug {s!r} should be EXCLUDED, got {narr}")

    if soft_misses:
        print("\n[env-limited recall misses — informational]\n  " + "\n  ".join(soft_misses))
    assert not hard_failures, "\n".join(hard_failures)
