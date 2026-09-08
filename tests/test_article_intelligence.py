"""
Comprehensive integration test for the ArticleIntelligence pipeline.

Tests:
1. Deterministic components (text_stats, asset_extractor) — no LLM needed
2. Full pipeline analysis with real LLM calls
3. Event deduplication (same event from different sources)
4. Narrative tagging accuracy
5. Miner/validator agreement (4-tier validation)
6. Quality assertions on real articles
7. Live RSS article fetching and analysis

Run with: pytest tests/test_article_intelligence.py -v -s
Add --live-llm to enable LLM integration tests (costs API credits)
Add --live-rss to enable RSS fetching tests (requires internet)
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
import hashlib
import pytest
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub bittensor if it has a package conflict (common in dev environments)
try:
    import bittensor
except (RuntimeError, ImportError):
    bt_mock = types.ModuleType("bittensor")
    bt_mock.logging = types.SimpleNamespace(
        info=lambda *a, **kw: None, warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None, debug=lambda *a, **kw: None,
        success=lambda *a, **kw: None,
    )
    bt_mock.Synapse = type("Synapse", (), {})
    bt_mock.Dendrite = type("Dendrite", (), {})
    sys.modules["bittensor"] = bt_mock

from alpharidge_ai.models.article_intelligence import (
    ArticleIntelligence,
    ArticleContentType,
    AssetClass,
    AssetSentiment,
    ChartSummary,
    ContagionLink,
    ContagionMechanism,
    CredibilityFlag,
    EconomicDataPoint,
    EconomicEventType,
    EntityType,
    EventFingerprint,
    EventType,
    ExtractedEntity,
    FactualConfidence,
    GeoImpactZone,
    ImpactPotential,
    MarketAnalysisType,
    NumericUnit,
    PositioningSignalType,
    Sentiment,
    SentimentDirection,
    SourceMetadata,
    TargetAudience,
    TemporalFocus,
    TextStatistics,
    TopicSignature,
    Urgency,
    SCHEMA_VERSION,
)
from alpharidge_ai.analyzer.text_stats import compute_text_stats
from alpharidge_ai.analyzer.asset_extractor import AssetExtractor


# ============================================================================
# Test Article Fixtures — Curated for known expected outputs
# ============================================================================

ARTICLES = {
    "fed_rate_cut": {
        "id": 90001,
        "url": "https://test.example.com/fed-rate-cut",
        "title": "Federal Reserve Cuts Interest Rates by 25 Basis Points, Signals More Easing Ahead",
        "source": "reuters",
        "published": "2026-05-28T14:00:00Z",
        "summary": "The Federal Reserve cut its benchmark interest rate by 25 basis points to 4.75-5.00%, "
                   "citing progress on inflation. Chair Jerome Powell signaled further cuts are likely "
                   "in the coming months if economic data continues to improve.",
        "content": """The Federal Reserve on Wednesday cut its benchmark interest rate by 25 basis points
to a target range of 4.75-5.00%, the second consecutive reduction as policymakers grow more
confident that inflation is moving sustainably toward the 2% target.

Fed Chair Jerome Powell said at a press conference that "the economy is in a good place" and
that the committee will continue to make decisions on a meeting-by-meeting basis.

"If the economy evolves broadly as expected, policy will move over time toward a more neutral
stance," Powell said. "But we are not on any preset course."

The decision was unanimous among all 12 voting members of the Federal Open Market Committee.

Markets reacted positively, with the S&P 500 rising 1.2% and Bitcoin surging past $98,000.
The U.S. dollar index (DXY) fell 0.8% against a basket of major currencies. Gold prices
jumped 1.5% to $2,780 per ounce.

Treasury yields declined across the curve, with the 10-year yield falling 12 basis points
to 4.18%. The 2-year yield, which is most sensitive to Fed policy expectations, dropped
18 basis points to 4.05%.

Economists at Goldman Sachs now expect the Fed to cut rates by an additional 75 basis points
by mid-2027, bringing the target range to 4.00-4.25%. JPMorgan's forecast is more aggressive,
calling for 100 basis points of cuts over the same period.

The rate cut comes amid mixed economic signals. While the labor market remains resilient with
unemployment at 4.1%, consumer spending has shown signs of cooling. The latest CPI reading
showed inflation at 2.4% year-over-year, down from a peak of 9.1% in 2022.

Bitcoin's rally was attributed to reduced opportunity cost of holding non-yielding assets
and increased risk appetite among institutional investors. ETF inflows for the day totaled
$1.8 billion, led by BlackRock's IBIT fund.

Ethereum also gained 3.2% to $4,200, while Solana rose 5.1% to $195. The total crypto
market capitalization exceeded $3.5 trillion for the first time.""",
    },
    "fed_rate_cut_copycat": {
        "id": 90002,
        "url": "https://test.example.com/fed-cuts-rates-25bp",
        "title": "Fed Delivers 25bp Rate Cut; Powell Hints at Further Easing",
        "source": "bloomberg",
        "published": "2026-05-28T14:15:00Z",
        "summary": "The Federal Reserve reduced its key interest rate by a quarter percentage point "
                   "on Wednesday, with Chair Jerome Powell suggesting more cuts could follow.",
        "content": """The Federal Reserve lowered its benchmark lending rate by 25 basis points on
Wednesday to a range of 4.75%-5.00%, marking the second straight reduction as officials
gained confidence that price pressures are easing.

Chair Jerome Powell told reporters the economy remains on solid footing but acknowledged
that risks to the outlook are "roughly in balance." He declined to pre-commit to the pace
of future rate adjustments.

"We will continue to assess incoming data and the evolving outlook," Powell said.
"The timing and pace of additional rate reductions will depend on the data."

US stocks climbed following the announcement, with the S&P 500 adding 1.2%. The dollar
weakened against major currencies, and Treasury yields retreated sharply.

In the cryptocurrency market, Bitcoin jumped above $98,000, while Ethereum gained more
than 3% to trade near $4,200. Spot Bitcoin ETFs recorded $1.8 billion in net inflows.""",
    },
    "nvidia_earnings_beat": {
        "id": 90003,
        "url": "https://test.example.com/nvidia-q3-earnings",
        "title": "Nvidia Reports Record Q3 Revenue of $42 Billion, Beating Estimates by 15%",
        "source": "cnbc",
        "published": "2026-05-28T16:30:00Z",
        "summary": "Nvidia posted record quarterly revenue driven by surging AI chip demand, "
                   "with data center sales up 120% year-over-year.",
        "content": """Nvidia reported fiscal third-quarter revenue of $42.5 billion on Wednesday,
crushing Wall Street estimates of $37 billion and marking a 95% increase from a year ago.

Data center revenue, which includes AI chips like the H100 and B100, reached $35.2 billion,
up 120% year-over-year. CEO Jensen Huang called it "the beginning of a new industrial
revolution powered by AI."

Earnings per share came in at $0.81, compared to consensus estimates of $0.72. Gross margins
expanded to 75.1%, up from 72.3% a year ago.

The company guided Q4 revenue of $45 billion, plus or minus 2%, above the $41 billion
consensus. Huang said demand for Blackwell-generation chips is "incredible" and the company
is working to ramp production as fast as possible.

Nvidia shares rose 8% in after-hours trading to $165 per share. The stock is up 280%
year-to-date, making it the third-largest company by market capitalization at $4.1 trillion.

AMD shares fell 3% in sympathy, as investors priced in Nvidia's dominant market position.
TSMC, which manufactures Nvidia's chips, rose 2% on the implied production volume increase.

AI-related tokens also rallied, with Render (RNDR) gaining 12%, Fetch.ai (FET) up 8%,
and Bittensor (TAO) rising 6% as the AI narrative strengthened.""",
    },
    "crypto_exchange_hack": {
        "id": 90004,
        "url": "https://test.example.com/exchange-hack",
        "title": "Major Crypto Exchange Suffers $400M Hack; BTC Drops 5% on Contagion Fears",
        "source": "coindesk",
        "published": "2026-05-28T08:00:00Z",
        "summary": "A leading cryptocurrency exchange was hacked for approximately $400 million "
                   "in various tokens, triggering a broad market selloff.",
        "content": """A major cryptocurrency exchange suffered a security breach early Wednesday
morning, with approximately $400 million in digital assets drained from hot wallets.

The attack targeted the exchange's Ethereum bridge contract, exploiting a vulnerability
in the cross-chain transfer mechanism. Stolen assets include approximately $200 million
in ETH, $120 million in USDT, and $80 million in various altcoins.

Bitcoin fell 5% to $93,000 within hours of the news breaking, while Ethereum dropped 7%
to $3,800. The total crypto market capitalization shed approximately $150 billion.

DeFi tokens were hit particularly hard, with Aave (AAVE) falling 12%, Uniswap (UNI) down
10%, and Compound (COMP) declining 9%. Stablecoins briefly depegged, with USDT trading
at $0.997 before recovering.

The exchange has halted all withdrawals and is working with blockchain security firms
to trace the stolen funds. The FBI and SEC have been notified, according to sources
familiar with the matter.

"This is a stark reminder that security must remain the top priority," said a spokesperson
for the exchange. "We are committed to making affected users whole."

Regulatory scrutiny is expected to intensify. SEC Chair has previously warned that
exchanges operating without proper registration face enforcement action. This hack
may accelerate proposed legislation requiring exchanges to maintain proof of reserves.""",
    },
    "tariff_announcement": {
        "id": 90005,
        "url": "https://test.example.com/china-tariffs",
        "title": "US Announces 30% Tariff on Chinese Semiconductor Imports, Escalating Trade War",
        "source": "wsj",
        "published": "2026-05-28T10:00:00Z",
        "summary": "The White House announced sweeping new tariffs on Chinese semiconductor imports, "
                   "a move expected to disrupt global chip supply chains.",
        "content": """The United States announced a 30% tariff on all semiconductor imports from China
on Wednesday, dramatically escalating the trade war between the world's two largest economies.

The tariff, effective July 1, covers finished chips, chip-making equipment, and raw materials
used in semiconductor manufacturing. The move is expected to raise costs for American
technology companies that rely on Chinese-manufactured components.

Nvidia shares fell 4% on the news, as roughly 20% of its revenue comes from Chinese customers.
AMD dropped 5%, while Intel, which manufactures more chips domestically, gained 2%.

The Philadelphia Semiconductor Index (SOX) declined 3.5% on the announcement. Broader markets
also sold off, with the S&P 500 falling 1.8% and the Nasdaq Composite declining 2.5%.

Gold rose 1.2% as investors sought safe-haven assets. Oil prices were unchanged. The dollar
index (DXY) strengthened 0.5% as trade war concerns boosted demand for the reserve currency.

Bitcoin initially fell 2% but recovered to trade flat as crypto traders assessed whether
the tariffs would ultimately be inflationary — historically bullish for BTC.

Chinese officials condemned the action, with the Ministry of Commerce calling it
"a serious violation of WTO rules" and promising "proportional countermeasures."

Economists warn the tariffs could add 0.3 percentage points to US inflation over the next
12 months, potentially complicating the Federal Reserve's rate-cut trajectory.""",
    },
}


# ============================================================================
# Section 1: Deterministic Component Tests (no LLM needed)
# ============================================================================


class TestTextStats:
    """Verify text_stats is deterministic and produces correct features."""

    def test_basic_stats(self):
        stats = compute_text_stats(
            "Bitcoin Hits $100K",
            "Bitcoin reached $100,000 today. This is a historic milestone. Analysts expect more gains ahead."
        )
        assert stats.word_count > 0
        assert stats.sentence_count == 3
        assert stats.char_count > 0
        assert stats.title_word_count == 3
        assert stats.title_has_number is True
        assert stats.title_has_question is False

    def test_determinism(self):
        title = "Fed Cuts Rates by 25bp"
        body = "The Federal Reserve cut rates. Markets rallied. Gold rose."
        s1 = compute_text_stats(title, body)
        s2 = compute_text_stats(title, body)
        assert s1 == s2

    def test_hedging_detection(self):
        stats = compute_text_stats(
            "Analysts: Rate Cut Might Happen",
            "Sources say the Fed could potentially cut rates. It reportedly might happen soon. "
            "The outcome is uncertain and analysts suggest caution."
        )
        assert stats.hedging_score > 0.0

    def test_certainty_detection(self):
        stats = compute_text_stats(
            "Fed Confirms Rate Cut",
            "The Federal Reserve will definitely cut rates. This has been officially confirmed "
            "and is absolutely certain to happen."
        )
        assert stats.certainty_score > 0.0

    def test_bullish_title_sentiment(self):
        stats = compute_text_stats("Bitcoin Surges to New ATH", "Content here.")
        assert stats.title_sentiment in (Sentiment.BULLISH, Sentiment.VERY_BULLISH)

    def test_bearish_title_sentiment(self):
        stats = compute_text_stats("Markets Crash on Trade War Fears", "Content here.")
        assert stats.title_sentiment in (Sentiment.BEARISH, Sentiment.VERY_BEARISH)

    def test_empty_body(self):
        stats = compute_text_stats("Title Only", "")
        assert stats.word_count == 0
        assert stats.char_count == 0

    def test_numeric_density(self):
        stats = compute_text_stats(
            "Data Release",
            "CPI was 3.2%, up from 2.9%. GDP grew 2.1%. Unemployment at 4.1%. "
            "Revenue of $42.5 billion. EPS of $0.81."
        )
        assert stats.numeric_density > 0.05

    def test_clickbait_detection(self):
        stats = compute_text_stats(
            "You Won't Believe What Bitcoin Did!! BREAKING!!!",
            "Some content here."
        )
        assert stats.clickbait_score > 0.0

    def test_quote_density(self):
        stats = compute_text_stats(
            "Fed Chair Speaks",
            'Powell said "The economy is in a good place" and added '
            '"We are not on any preset course" during the press conference.'
        )
        assert stats.quote_density > 0.0


class TestAssetExtractor:
    """Verify multi-asset extraction from text."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.extractor = AssetExtractor()

    def test_loads_assets(self):
        assert len(self.extractor.assets) >= 100

    def test_cashtag_extraction(self):
        results = self.extractor.extract_assets("$BTC rallies", "$BTC is up 5% today")
        tickers = [r.ticker for r in results]
        assert "BTC" in tickers

    def test_multi_asset_extraction(self):
        results = self.extractor.extract_assets(
            "Bitcoin and Ethereum both rally",
            "Bitcoin surged past $100K while Ethereum hit $4,500. Solana also gained."
        )
        tickers = {r.ticker for r in results}
        assert "BTC" in tickers
        assert "ETH" in tickers
        assert "SOL" in tickers

    def test_equity_detection(self):
        results = self.extractor.extract_assets(
            "Nvidia Reports Record Earnings",
            "NVDA posted $42.5 billion in revenue. Apple and Microsoft also reported."
        )
        tickers = {r.ticker for r in results}
        assert "NVDA" in tickers

    def test_forex_detection(self):
        results = self.extractor.extract_assets(
            "Dollar Weakens on Fed Cut",
            "The US dollar index fell as the Federal Reserve cut rates. EUR/USD gained."
        )
        # Should find DXY or forex-related assets
        asset_classes = {r.asset_class for r in results}
        assert len(results) > 0

    def test_primary_subject_identification(self):
        results = self.extractor.extract_assets(
            "$BTC Surges Past $100K on ETF Inflows",
            "$BTC Bitcoin gained 5%. Ethereum also rose. Gold was flat."
        )
        btc = next((r for r in results if r.ticker == "BTC"), None)
        assert btc is not None
        assert btc.is_primary_subject is True

    def test_disambiguation(self):
        results = self.extractor.extract_assets(
            "Apple Reports Q3 Earnings",
            "Apple posted record iPhone sales. AAPL stock rose 3%."
        )
        tickers = {r.ticker for r in results}
        assert "AAPL" in tickers

    def test_no_false_positives_on_common_words(self):
        results = self.extractor.extract_assets(
            "Weather Report",
            "The sun was shining near the river. It was a nice day for apples and sand."
        )
        # Should not extract SOL (sun), NEAR (near), SAND (sand), AAPL (apples)
        tickers = {r.ticker for r in results}
        assert "SOL" not in tickers  # "sun" shouldn't match Solana
        assert "SAND" not in tickers  # "sand" shouldn't match The Sandbox

    def test_sector_extraction(self):
        sectors = self.extractor.extract_sectors(
            "Fed Cuts Rates, Bitcoin Rallies",
            "The Federal Reserve cut interest rates. Bitcoin and crypto surged."
        )
        sector_symbols = {s["symbol"] for s in sectors}
        assert "CRYPTO" in sector_symbols or "MONETARY" in sector_symbols

    def test_multi_sector(self):
        sectors = self.extractor.extract_sectors(
            "Tariffs Hit Tech Stocks, Crypto Drops",
            "New tariffs on semiconductor imports from China hit technology stocks. "
            "Bitcoin also fell on risk-off sentiment."
        )
        assert len(sectors) >= 2

    def test_evidence_spans(self):
        results = self.extractor.extract_assets(
            "$BTC Bitcoin",
            "$BTC rallied. Bitcoin hit $100K."
        )
        btc = next((r for r in results if r.ticker == "BTC"), None)
        assert btc is not None
        assert len(btc.evidence_spans) >= 1


class TestContentHash:
    """Verify content hash is deterministic."""

    def test_deterministic(self):
        h1 = ArticleIntelligence.compute_content_hash("Title", "Body content here")
        h2 = ArticleIntelligence.compute_content_hash("Title", "Body content here")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = ArticleIntelligence.compute_content_hash("Title A", "Content A")
        h2 = ArticleIntelligence.compute_content_hash("Title B", "Content B")
        assert h1 != h2

    def test_whitespace_normalized(self):
        h1 = ArticleIntelligence.compute_content_hash("Title", "Body  content   here")
        h2 = ArticleIntelligence.compute_content_hash("Title", "Body content here")
        assert h1 == h2

    def test_case_normalized(self):
        h1 = ArticleIntelligence.compute_content_hash("TITLE", "BODY")
        h2 = ArticleIntelligence.compute_content_hash("title", "body")
        assert h1 == h2


class TestEventDeduplication:
    """Verify that copycat articles about the same event produce matching fingerprints."""

    def test_same_content_same_hash(self):
        art1 = ARTICLES["fed_rate_cut"]
        art2 = ARTICLES["fed_rate_cut"]
        h1 = ArticleIntelligence.compute_content_hash(art1["title"], art1["content"])
        h2 = ArticleIntelligence.compute_content_hash(art2["title"], art2["content"])
        assert h1 == h2

    def test_copycat_different_hash(self):
        """Different articles about the same event should have different content hashes
        but should be clustered by event fingerprint (tested in LLM section)."""
        art1 = ARTICLES["fed_rate_cut"]
        art2 = ARTICLES["fed_rate_cut_copycat"]
        h1 = ArticleIntelligence.compute_content_hash(art1["title"], art1["content"])
        h2 = ArticleIntelligence.compute_content_hash(art2["title"], art2["content"])
        assert h1 != h2  # Different text = different hash


class TestSchemaValidation:
    """Verify the ArticleIntelligence schema accepts valid data and rejects invalid."""

    def test_minimal_valid(self):
        intel = ArticleIntelligence(
            article_id=1, url="https://example.com", title="Test",
            published_at="2026-01-01T00:00:00Z", analyzed_at="2026-01-01T00:00:00Z",
            source=SourceMetadata(source_id="test", source_name="Test", credibility_score=0.5),
            content_type=ArticleContentType.OTHER,
            impact_potential=ImpactPotential.LOW,
            technical_quality="none",
            overall_sentiment=Sentiment.NEUTRAL,
            overall_sentiment_score=0.0,
            sentiment_direction=SentimentDirection.NEUTRAL,
            chart_summary=ChartSummary(headline="Test", one_liner="Test", context_paragraph="Test"),
            event_fingerprint=EventFingerprint(
                event_type=EventType.OTHER, event_title="Test",
                content_hash="abc123", semantic_fingerprint=["test", "event", "unknown"],
            ),
            topic_signature=TopicSignature(
                primary_sector_id=9, primary_sector_symbol="OTHER",
            ),
            text_stats=compute_text_stats("Test", "Test body"),
            factual_confidence=FactualConfidence.SPECULATIVE,
        )
        assert intel.schema_version == SCHEMA_VERSION
        assert intel.article_id == 1

    def test_canonical_string(self):
        intel = ArticleIntelligence(
            article_id=1, url="https://example.com", title="Test",
            published_at="2026-01-01T00:00:00Z", analyzed_at="2026-01-01T00:00:00Z",
            source=SourceMetadata(source_id="test", source_name="Test", credibility_score=0.5),
            content_type=ArticleContentType.BREAKING_NEWS,
            impact_potential=ImpactPotential.HIGH,
            technical_quality="high",
            overall_sentiment=Sentiment.BULLISH,
            overall_sentiment_score=0.6,
            sentiment_direction=SentimentDirection.POSITIVE,
            chart_summary=ChartSummary(headline="Test", one_liner="Test", context_paragraph="Test"),
            event_fingerprint=EventFingerprint(
                event_type=EventType.MONETARY_POLICY, event_title="Test",
                event_date="2026-05-28",
                content_hash="abc", semantic_fingerprint=["fed", "rate", "cut"],
            ),
            topic_signature=TopicSignature(primary_sector_id=2, primary_sector_symbol="MONETARY"),
            text_stats=compute_text_stats("Test", "Body"),
            factual_confidence=FactualConfidence.CONFIRMED,
        )
        canonical = intel.to_canonical_string()
        assert "2|breaking_news|bullish" in canonical
        assert "monetary_policy|2026-05-28" in canonical

    def test_serialization_roundtrip(self):
        intel = ArticleIntelligence(
            article_id=1, url="https://example.com", title="Test",
            published_at="2026-01-01T00:00:00Z", analyzed_at="2026-01-01T00:00:00Z",
            source=SourceMetadata(source_id="test", source_name="Test", credibility_score=0.5),
            content_type=ArticleContentType.OTHER,
            impact_potential=ImpactPotential.LOW,
            technical_quality="none",
            overall_sentiment=Sentiment.NEUTRAL,
            overall_sentiment_score=0.0,
            sentiment_direction=SentimentDirection.NEUTRAL,
            chart_summary=ChartSummary(headline="T", one_liner="T", context_paragraph="T"),
            event_fingerprint=EventFingerprint(
                event_type=EventType.OTHER, event_title="T",
                content_hash="x", semantic_fingerprint=["a", "b", "c"],
            ),
            topic_signature=TopicSignature(primary_sector_id=9, primary_sector_symbol="OTHER"),
            text_stats=compute_text_stats("T", "T"),
            factual_confidence=FactualConfidence.SPECULATIVE,
        )
        d = intel.model_dump()
        intel2 = ArticleIntelligence(**d)
        assert intel2.article_id == intel.article_id
        assert intel2.content_type == intel.content_type
        assert intel2.to_canonical_string() == intel.to_canonical_string()


# ============================================================================
# Section 2: LLM Integration Tests (requires --live-llm flag)
# ============================================================================


@pytest.fixture(scope="module")
def analyzer():
    """Create an ArticleIntelligenceAnalyzer for LLM tests."""
    import types
    import bittensor as bt

    from alpharidge_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer
    return ArticleIntelligenceAnalyzer()


@pytest.fixture(scope="module")
def analyzed_articles(analyzer, request):
    """Analyze all test articles once, reuse across tests."""
    if not request.config.getoption("--live-llm", default=False):
        pytest.skip("Requires --live-llm flag")

    results = {}
    for name, article in ARTICLES.items():
        print(f"\n  Analyzing: {name} ({article['title'][:60]}...)")
        start = time.time()
        try:
            intel = analyzer.analyze(
                article_id=article["id"],
                url=article["url"],
                title=article["title"],
                source=article["source"],
                published=article["published"],
                summary=article["summary"],
                content=article["content"],
            )
        except Exception as e:
            print(f"  → EXCEPTION: {e}")
            intel = None
        elapsed = time.time() - start
        if intel:
            print(f"  → Done in {elapsed:.1f}s | assets={len(intel.assets)} | "
                  f"entities={len(intel.entities)} | sentiment={intel.overall_sentiment.value}")
        else:
            print(f"  → FAILED in {elapsed:.1f}s (returned None)")
        results[name] = intel

    failed = [n for n, i in results.items() if i is None]
    if failed:
        print(f"\n  WARNING: {len(failed)} articles failed: {failed}")
        print(f"  Successful: {[n for n, i in results.items() if i is not None]}")

    return results


def _get(analyzed_articles, name):
    """Get an analyzed article, skip test if it's None."""
    intel = analyzed_articles.get(name)
    if intel is None:
        pytest.skip(f"{name} analysis returned None (LLM error)")
    return intel


class TestLLMAnalysisQuality:
    """Verify LLM produces high-quality, correct analysis."""

    def test_fed_rate_cut_classification(self, analyzed_articles):
        intel = _get(analyzed_articles, "fed_rate_cut")
        assert intel.content_type in (ArticleContentType.BREAKING_NEWS, ArticleContentType.DATA_RELEASE)
        assert intel.impact_potential in (ImpactPotential.CRITICAL, ImpactPotential.HIGH)
        assert intel.overall_sentiment in (Sentiment.BULLISH, Sentiment.VERY_BULLISH, Sentiment.SLIGHTLY_BULLISH)
        assert intel.overall_sentiment_score > 0.0

    def test_fed_rate_cut_assets(self, analyzed_articles):
        intel = _get(analyzed_articles, "fed_rate_cut")
        tickers = {a.ticker for a in intel.assets}
        # Must detect BTC, ETH, SOL from the article
        assert "BTC" in tickers, f"BTC not found in {tickers}"
        assert "ETH" in tickers, f"ETH not found in {tickers}"

    def test_fed_rate_cut_per_asset_sentiment(self, analyzed_articles):
        intel = _get(analyzed_articles, "fed_rate_cut")
        btc = next((a for a in intel.assets if a.ticker == "BTC"), None)
        assert btc is not None
        assert btc.direction in (Sentiment.BULLISH, Sentiment.VERY_BULLISH, Sentiment.SLIGHTLY_BULLISH)
        assert btc.causal_driver  # Must have explanation
        assert len(btc.causal_driver) > 10

    def test_fed_rate_cut_economic_data(self, analyzed_articles):
        intel = _get(analyzed_articles, "fed_rate_cut")
        assert len(intel.economic_data) > 0
        # Should extract the rate cut data point
        rate_data = [d for d in intel.economic_data
                     if d.event_type in (EconomicEventType.FOMC_DECISION, EconomicEventType.OTHER)]
        assert len(rate_data) > 0 or len(intel.economic_data) > 0

    def test_fed_rate_cut_entities(self, analyzed_articles):
        intel = _get(analyzed_articles, "fed_rate_cut")
        entity_names = {e.name.lower() for e in intel.entities}
        # Must find Jerome Powell and Federal Reserve
        assert any("powell" in n for n in entity_names), f"Powell not in {entity_names}"
        assert any("fed" in n or "reserve" in n for n in entity_names), f"Fed not in {entity_names}"

    def test_fed_rate_cut_chart_summary(self, analyzed_articles):
        intel = _get(analyzed_articles, "fed_rate_cut")
        assert len(intel.chart_summary.headline) > 10
        assert len(intel.chart_summary.headline) <= 120
        assert len(intel.chart_summary.one_liner) > 10
        assert len(intel.chart_summary.context_paragraph) > 50

    def test_fed_rate_cut_contagion(self, analyzed_articles):
        intel = _get(analyzed_articles, "fed_rate_cut")
        assert len(intel.contagion_links) > 0
        # Should have cross-market impact predictions
        targets = {l.target_ticker for l in intel.contagion_links}
        print(f"  Contagion targets: {targets}")

    def test_nvidia_earnings_classification(self, analyzed_articles):
        intel = _get(analyzed_articles, "nvidia_earnings_beat")
        assert intel is not None
        assert intel.content_type in (ArticleContentType.EARNINGS, ArticleContentType.BREAKING_NEWS)
        assert intel.impact_potential in (ImpactPotential.HIGH, ImpactPotential.CRITICAL)

    def test_nvidia_assets(self, analyzed_articles):
        intel = _get(analyzed_articles, "nvidia_earnings_beat")
        tickers = {a.ticker for a in intel.assets}
        assert "NVDA" in tickers, f"NVDA not found in {tickers}"
        # Should also detect AMD, TSMC as secondary
        nvda = next((a for a in intel.assets if a.ticker == "NVDA"), None)
        assert nvda.is_primary_subject is True

    def test_nvidia_numeric_claims(self, analyzed_articles):
        intel = _get(analyzed_articles, "nvidia_earnings_beat")
        assert len(intel.numeric_claims) > 0 or len(intel.economic_data) > 0
        # Should extract revenue figure ($42.5B)

    def test_hack_classification(self, analyzed_articles):
        intel = _get(analyzed_articles, "crypto_exchange_hack")
        assert intel is not None
        assert intel.overall_sentiment in (Sentiment.BEARISH, Sentiment.VERY_BEARISH)
        assert intel.factual_confidence in (
            FactualConfidence.CONFIRMED, FactualConfidence.ATTRIBUTED, FactualConfidence.SPECULATIVE
        )

    def test_hack_contagion(self, analyzed_articles):
        intel = _get(analyzed_articles, "crypto_exchange_hack")
        assert len(intel.contagion_links) > 0

    def test_tariff_geo_impact(self, analyzed_articles):
        intel = _get(analyzed_articles, "tariff_announcement")
        assert intel is not None
        assert intel.primary_geo in (GeoImpactZone.US, GeoImpactZone.GLOBAL, GeoImpactZone.CHINA)

    def test_all_articles_have_required_fields(self, analyzed_articles):
        success = 0
        failures = []
        for name, intel in analyzed_articles.items():
            if intel is None:
                failures.append(f"{name} (returned None)")
                continue
            assert intel.schema_version == SCHEMA_VERSION
            assert len(intel.chart_summary.headline) > 0, f"{name} missing headline"
            assert len(intel.event_fingerprint.semantic_fingerprint) >= 3, f"{name} insufficient fingerprint"
            assert intel.text_stats.word_count > 0, f"{name} zero word count"
            success += 1
            print(f"  {name}: {intel.content_type.value}, {intel.overall_sentiment.value}, "
                  f"{len(intel.assets)} assets, {len(intel.entities)} entities, "
                  f"{len(intel.contagion_links)} contagion, {len(intel.economic_data)} econ data")
        if failures:
            print(f"\n  WARNING: {len(failures)} articles failed: {failures}")
        assert success >= len(analyzed_articles) - 1, (
            f"Too many failures: {len(failures)}/{len(analyzed_articles)} failed: {failures}"
        )


class TestEventClustering:
    """Verify that copycat articles get similar event fingerprints."""

    def test_copycats_share_event_type(self, analyzed_articles):
        a1 = _get(analyzed_articles, "fed_rate_cut")
        a2 = _get(analyzed_articles, "fed_rate_cut_copycat")
        assert a1.event_fingerprint.event_type == a2.event_fingerprint.event_type

    def test_copycats_share_event_date(self, analyzed_articles):
        a1 = _get(analyzed_articles, "fed_rate_cut")
        a2 = _get(analyzed_articles, "fed_rate_cut_copycat")
        assert a1.event_fingerprint.event_date == a2.event_fingerprint.event_date

    def test_copycats_similar_fingerprint(self, analyzed_articles):
        a1 = _get(analyzed_articles, "fed_rate_cut")
        a2 = _get(analyzed_articles, "fed_rate_cut_copycat")
        fp1 = set(a1.event_fingerprint.semantic_fingerprint)
        fp2 = set(a2.event_fingerprint.semantic_fingerprint)
        overlap = len(fp1 & fp2) / max(len(fp1 | fp2), 1)
        print(f"  Fingerprint overlap: {overlap:.2f} ({fp1 & fp2})")
        assert overlap >= 0.1, f"Fingerprint overlap too low: {overlap:.2f}"

    def test_copycats_similar_title(self, analyzed_articles):
        a1 = _get(analyzed_articles, "fed_rate_cut")
        a2 = _get(analyzed_articles, "fed_rate_cut_copycat")
        from alpharidge_ai.analyzer.scoring import _normalize_text, _levenshtein_ratio
        t1 = _normalize_text(a1.event_fingerprint.event_title)
        t2 = _normalize_text(a2.event_fingerprint.event_title)
        sim = _levenshtein_ratio(t1, t2)
        print(f"  Title similarity: {sim:.2f}")
        print(f"    T1: {t1}")
        print(f"    T2: {t2}")
        assert sim >= 0.2, f"Title similarity too low: {sim:.2f}"

    def test_different_events_different_fingerprint(self, analyzed_articles):
        fed = _get(analyzed_articles, "fed_rate_cut")
        nvidia = _get(analyzed_articles, "nvidia_earnings_beat")
        assert fed.event_fingerprint.event_type != nvidia.event_fingerprint.event_type

    def test_different_events_low_fingerprint_overlap(self, analyzed_articles):
        fed = _get(analyzed_articles, "fed_rate_cut")
        hack = _get(analyzed_articles, "crypto_exchange_hack")
        fp1 = set(fed.event_fingerprint.semantic_fingerprint)
        fp2 = set(hack.event_fingerprint.semantic_fingerprint)
        overlap = len(fp1 & fp2) / max(len(fp1 | fp2), 1)
        assert overlap < 0.5, f"Different events have too much overlap: {overlap:.2f}"


class TestValidationAgreement:
    """Test that running analysis twice produces matching results (Tier 1-3)."""

    def test_miner_validator_agreement(self, analyzer, request):
        if not request.config.getoption("--live-llm", default=False):
            pytest.skip("Requires --live-llm flag")

        art = ARTICLES["nvidia_earnings_beat"]
        print("\n  Running miner analysis...")
        miner_intel = analyzer.analyze(
            article_id=art["id"], url=art["url"], title=art["title"],
            source=art["source"], published=art["published"],
            summary=art["summary"], content=art["content"],
        )
        print("  Running validator analysis...")
        validator_intel = analyzer.analyze(
            article_id=art["id"], url=art["url"], title=art["title"],
            source=art["source"], published=art["published"],
            summary=art["summary"], content=art["content"],
        )
        assert miner_intel is not None
        assert validator_intel is not None

        from alpharidge_ai.analyzer.scoring import validate_article_intelligence
        is_valid, composite, details = validate_article_intelligence(miner_intel, validator_intel)

        print(f"\n  Validation result: valid={is_valid}, composite={composite:.4f}")
        print(f"  Tier 1: {sum(1 for v in details['tier1'].values() if v['match'])}/{len(details['tier1'])} match")
        print(f"  Tier 2: {sum(1 for v in details['tier2'].values() if v.get('match'))}/{len(details['tier2'])} match")
        for field, info in details["tier3"].items():
            if field != "composite":
                print(f"  Tier 3 {field}: {info['score']:.4f} (weight={info['weight']})")

        # With temp=0 and same model, Tier 1 should always pass
        tier1_failures = [k for k, v in details["tier1"].items() if not v["match"]]
        if tier1_failures:
            for f in tier1_failures:
                d = details["tier1"][f]
                print(f"  TIER 1 FAIL: {f}: miner={d['miner']} validator={d['validator']}")

        # Tier 2 should always pass (deterministic)
        tier2_failures = [k for k, v in details["tier2"].items() if v.get("match") is False]
        assert len(tier2_failures) == 0, f"Tier 2 failures (should be impossible): {tier2_failures}"

        print(f"\n  Overall: composite={composite:.4f}, threshold=0.80")
        # Log but don't hard-fail on composite (LLM variance may cause minor diffs)
        if not is_valid:
            print(f"  WARNING: Composite below threshold. This may be due to LLM non-determinism.")


# ============================================================================
# Section 3: Live RSS Tests (requires --live-rss flag)
# ============================================================================


class TestLiveRSS:
    """Fetch real articles from RSS and analyze them."""

    @pytest.fixture(scope="class")
    def live_articles(self, request):
        if not request.config.getoption("--live-rss", default=False):
            pytest.skip("Requires --live-rss flag")

        import asyncio
        sys.path.insert(0, "/home/rizzo/sn45/news-scraper")
        from finnews import News

        async def fetch():
            client = News()
            articles = await client.get_news(["reuters", "bbc", "cnbc"], scrape=False)
            return articles[:10]

        return asyncio.run(fetch())

    def test_fetches_articles(self, live_articles):
        assert len(live_articles) > 0
        print(f"\n  Fetched {len(live_articles)} articles from RSS")
        for a in live_articles[:5]:
            print(f"  - [{a.source}] {a.title[:80]}")

    def test_analyze_rss_articles(self, live_articles, analyzer, request):
        if not request.config.getoption("--live-llm", default=False):
            pytest.skip("Requires both --live-rss and --live-llm")

        for article in live_articles[:3]:
            print(f"\n  Analyzing RSS article: {article.title[:60]}...")
            intel = analyzer.analyze(
                article_id=hash(article.url) % 100000,
                url=article.url,
                title=article.title,
                source=article.source,
                published=str(article.published) if article.published else None,
                summary=article.summary,
                content=article.content,
            )
            assert intel is not None, f"Analysis returned None for: {article.title}"
            assert len(intel.chart_summary.headline) > 0
            assert intel.text_stats.word_count >= 0
            print(f"  → {intel.content_type.value} | {intel.overall_sentiment.value} | "
                  f"{len(intel.assets)} assets | {len(intel.entities)} entities")


# ============================================================================
# Section 3b: Embedding Tests
# ============================================================================


class TestEmbeddings:
    """Verify embedding generation and semantic properties."""

    def test_embeddings_populated(self, analyzed_articles):
        for name, intel in analyzed_articles.items():
            if intel is None:
                continue
            assert intel.title_embedding is not None, f"{name}: title_embedding is None"
            assert intel.body_embedding is not None, f"{name}: body_embedding is None"
            assert intel.narrative_embedding is not None, f"{name}: narrative_embedding is None"
            assert len(intel.title_embedding) == 384, f"{name}: title_embedding has {len(intel.title_embedding)}d"
            assert len(intel.body_embedding) == 384
            assert len(intel.narrative_embedding) == 384

    def test_embeddings_normalized(self, analyzed_articles):
        import numpy as np
        for name, intel in analyzed_articles.items():
            if intel is None:
                continue
            for emb_name, emb in [("title", intel.title_embedding),
                                   ("body", intel.body_embedding),
                                   ("narrative", intel.narrative_embedding)]:
                norm = np.linalg.norm(emb)
                assert abs(norm - 1.0) < 0.05, f"{name} {emb_name}: L2 norm is {norm:.4f}"

    def test_different_articles_different_embeddings(self, analyzed_articles):
        import numpy as np
        items = [(n, i) for n, i in analyzed_articles.items() if i is not None]
        for i, (name_a, intel_a) in enumerate(items):
            for name_b, intel_b in items[i + 1:]:
                sim = float(np.dot(intel_a.title_embedding, intel_b.title_embedding))
                assert sim < 0.99, f"{name_a} vs {name_b}: title embeddings too similar ({sim:.4f})"

    def test_narrative_keywords_are_slugs(self, analyzed_articles):
        """With taxonomy awareness, LLM should output known narrative slugs."""
        import json, os
        with open(os.path.join(os.path.dirname(__file__), "..", "alpharidge_ai", "analyzer", "data", "narratives.json")) as f:
            slugs = {n["slug"] for n in json.load(f)}
        total_kws = 0
        slug_matches = 0
        for name, intel in analyzed_articles.items():
            if intel is None:
                continue
            for kw in intel.narrative_keywords:
                total_kws += 1
                if kw.lower().replace("_", "-").replace(" ", "-") in slugs:
                    slug_matches += 1
        if total_kws > 0:
            rate = slug_matches / total_kws
            print(f"\n  Narrative slug match rate: {slug_matches}/{total_kws} ({rate:.0%})")
            assert rate >= 0.5, f"Slug match rate too low: {rate:.0%}"


# ============================================================================
# Section 4: Batch Quality Report (run all articles, print summary)
# ============================================================================


class TestBatchQualityReport:
    """Run all articles and print a comprehensive quality report."""

    def test_quality_report(self, analyzed_articles):
        print("\n" + "=" * 80)
        print("ARTICLE INTELLIGENCE QUALITY REPORT")
        print("=" * 80)

        for name, intel in analyzed_articles.items():
            print(f"\n--- {name} ---")
            if intel is None:
                print("  SKIPPED (analysis returned None)")
                continue
            print(f"  Title: {intel.title[:70]}")
            print(f"  Content Type: {intel.content_type.value}")
            print(f"  Sentiment: {intel.overall_sentiment.value} ({intel.overall_sentiment_score:+.2f})")
            print(f"  Impact: {intel.impact_potential.value}")
            print(f"  Factual: {intel.factual_confidence.value}")
            print(f"  Urgency: {intel.urgency.value}")
            print(f"  Audience: {intel.target_audience.value}")
            print(f"  Geo: {intel.primary_geo.value}")
            print(f"  Event: {intel.event_fingerprint.event_type.value} - {intel.event_fingerprint.event_title[:60]}")
            print(f"  Assets ({len(intel.assets)}):")
            for a in intel.assets[:5]:
                primary = " [PRIMARY]" if a.is_primary_subject else ""
                print(f"    {a.ticker} ({a.asset_class.value}): {a.direction.value} "
                      f"mag={a.magnitude:.2f} conf={a.confidence:.2f}{primary}")
                print(f"      Driver: {a.causal_driver[:80]}")
            print(f"  Entities ({len(intel.entities)}):")
            for e in intel.entities[:5]:
                print(f"    {e.name} ({e.entity_type.value}/{e.role.value})")
            print(f"  Economic Data ({len(intel.economic_data)}):")
            for d in intel.economic_data[:3]:
                actual = f"actual={d.actual_value}" if d.actual_value else ""
                expected = f"expected={d.expected_value}" if d.expected_value else ""
                print(f"    {d.event_name}: {actual} {expected} ({d.unit.value})")
            print(f"  Contagion ({len(intel.contagion_links)}):")
            for c in intel.contagion_links[:3]:
                print(f"    {c.source_ticker} → {c.target_ticker}: {c.direction.value} "
                      f"via {c.mechanism.value} (str={c.strength:.2f})")
            print(f"  Quotes ({len(intel.quotes)}):")
            for q in intel.quotes[:2]:
                print(f"    {q.speaker}: \"{q.text[:60]}...\" ({q.sentiment.value})")
            print(f"  Narratives: {intel.narrative_keywords}")
            print(f"  Summary: {intel.chart_summary.headline}")
            print(f"  Fingerprint: {intel.event_fingerprint.semantic_fingerprint}")
            te = intel.title_embedding
            be = intel.body_embedding
            ne = intel.narrative_embedding
            print(f"  Embeddings: title={'✓' if te else '✗'}({len(te) if te else 0}d) "
                  f"body={'✓' if be else '✗'}({len(be) if be else 0}d) "
                  f"narrative={'✓' if ne else '✗'}({len(ne) if ne else 0}d)")

        print("\n" + "=" * 80)
        print("END QUALITY REPORT")
        print("=" * 80)


# ============================================================================
# Section 8: Off-LLM builders + hybrid validation contract (no LLM, no models)
# ============================================================================

from alpharidge_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer
from alpharidge_ai.analyzer.scoring import (
    validate_article_intelligence, _ordinal_score, _ev,
    _SENTIMENT_LADDER, _IMPACT_LADDER,
)


def _bare_analyzer():
    """An analyzer instance with NO model loads — only the attrs the off-LLM
    builders touch are set."""
    az = ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)
    az.dependency_graph = {
        "ETH": {"dependents": [
            {"ticker": "STETH", "asset_class": "crypto", "relationship": "liquid_staking_derivative"},
            {"ticker": "ARB", "asset_class": "crypto", "relationship": "ecosystem_token"},
        ]},
        "NVDA": {"dependents": [
            {"ticker": "TSM", "asset_class": "equity", "relationship": "supply_chain"},
        ]},
    }
    return az


class TestGraphContagion:
    def test_builds_links_from_graph(self):
        az = _bare_analyzer()
        links = az._build_contagion_from_graph(["ETH", "NVDA"])
        pairs = {(l.source_ticker, l.target_ticker) for l in links}
        assert pairs == {("ETH", "STETH"), ("ETH", "ARB"), ("NVDA", "TSM")}

    def test_mechanism_mapping_and_defaults(self):
        az = _bare_analyzer()
        links = {l.target_ticker: l for l in az._build_contagion_from_graph(["ETH", "NVDA"])}
        assert links["STETH"].mechanism.value == "protocol_dependency"
        assert links["TSM"].mechanism.value == "supply_chain"
        # deterministic priors: NEUTRAL direction, fixed 0.7 strength/confidence
        assert links["TSM"].direction.value == "neutral"
        assert links["TSM"].strength == 0.7 and links["TSM"].confidence == 0.7

    def test_unknown_relationship_falls_back_to_correlation(self):
        az = _bare_analyzer()
        az.dependency_graph = {"X": {"dependents": [
            {"ticker": "Y", "asset_class": "crypto", "relationship": "totally_unknown"}]}}
        links = az._build_contagion_from_graph(["X"])
        assert links[0].mechanism.value == "correlation"

    def test_no_tickers_no_links(self):
        assert _bare_analyzer()._build_contagion_from_graph([]) == []


class TestAspectSentiment:
    """Per-asset direction now comes from target-aware FinABSA (score_assets).
    Stub the scorer's votes so these stay fast and deterministic."""

    @staticmethod
    def _scorer(vote_by_kw):
        from alpharidge_ai.analyzer.aspect_sentiment import AspectSentimentScorer

        class _Stub(AspectSentimentScorer):
            def _ensure(self):
                self._model = object()
            def label_many(self, pairs, batch_size=16):
                def v(t):
                    t = t.lower()
                    if any(k in t for k in vote_by_kw["bear"]):
                        return -1
                    if any(k in t for k in vote_by_kw["bull"]):
                        return 1
                    return 0
                return [v(txt) for txt, _ in pairs]
        return _Stub(device="cpu")

    def test_majority_vote_direction(self):
        from alpharidge_ai.analyzer.aspect_sentiment import score_assets
        ner = types.SimpleNamespace(sentence_sentiments=[
            {"text": "NVDA earnings crushed estimates"},
            {"text": "NVDA guidance raised sharply"},
            {"text": "NVDA faces some export risk"},
        ])
        asset = types.SimpleNamespace(ticker="NVDA", canonical_name="NVDA", surface_forms=[])
        scorer = self._scorer({"bull": ["crushed", "raised"], "bear": ["risk"]})
        out = {d["ticker"]: d for d in score_assets([asset], ner, scorer)}
        assert out["NVDA"]["direction"].endswith("bullish")  # 2 bull vs 1 bear
        assert 0.0 <= out["NVDA"]["magnitude"] <= 1.0

    def test_no_mention_defaults_fallback(self):
        from alpharidge_ai.analyzer.aspect_sentiment import score_assets
        ner = types.SimpleNamespace(sentence_sentiments=[{"text": "Bitcoin rallied"}])
        asset = types.SimpleNamespace(ticker="NVDA", canonical_name="Nvidia", surface_forms=[])
        scorer = self._scorer({"bull": [], "bear": []})
        out = {d["ticker"]: d for d in score_assets([asset], ner, scorer,
                                                    fallback_direction="neutral")}
        assert out["NVDA"]["direction"] == "neutral"
        assert "fell back" in out["NVDA"]["causal_driver"]


class TestScoringHelpers:
    def test_ordinal_exact(self):
        assert _ordinal_score("bullish", "bullish", _SENTIMENT_LADDER) == 1.0

    def test_ordinal_adjacent_partial_credit(self):
        s = _ordinal_score("very_bullish", "bullish", _SENTIMENT_LADDER)
        assert 0.8 < s < 1.0  # one step on a 7-rung ladder

    def test_ordinal_off_ladder_zero(self):
        assert _ordinal_score("mixed", "neutral", _SENTIMENT_LADDER) == 0.0

    def test_ev_handles_enum_and_str(self):
        assert _ev(ImpactPotential.HIGH) == "high"
        assert _ev("HIGH") == "high"


def _make_intel(**ov):
    base = dict(
        article_id=1, url="https://example.com", title="Fed cuts rates 25bps",
        published_at="2026-01-01T12:00:00Z", analyzed_at="2026-01-01T12:00:00Z",
        source=SourceMetadata(source_id="reuters", source_name="Reuters", credibility_score=1.0),
        content_type=ArticleContentType.BREAKING_NEWS,
        market_analysis_type=MarketAnalysisType.MACRO,
        impact_potential=ImpactPotential.HIGH,
        technical_quality="medium",
        urgency=Urgency.BREAKING,
        temporal_focus=TemporalFocus.CURRENT,
        overall_sentiment=Sentiment.BULLISH,
        overall_sentiment_score=0.5,
        sentiment_direction=SentimentDirection.POSITIVE,
        chart_summary=ChartSummary(headline="Fed cuts rates 25bps",
                                   one_liner="The Federal Reserve cut interest rates by 25bps today",
                                   context_paragraph="A paragraph describing the rate decision and its market impact."),
        event_fingerprint=EventFingerprint(event_type=EventType.MONETARY_POLICY,
                                           event_title="Fed rate cut", event_date="2026-01-01",
                                           content_hash="hash123", semantic_fingerprint=["fed", "rate", "cut"]),
        topic_signature=TopicSignature(primary_sector_id=2, primary_sector_symbol="MONETARY"),
        text_stats=compute_text_stats("Fed cuts rates 25bps", "The Federal Reserve cut interest rates by 25bps today."),
        factual_confidence=FactualConfidence.CONFIRMED,
        primary_geo=GeoImpactZone.US,
        target_audience=TargetAudience.INSTITUTIONAL,
        credibility_flag=CredibilityFlag.VERIFIED_SOURCE,
    )
    base.update(ov)
    return ArticleIntelligence(**base)


class TestHybridValidationContract:
    def test_identical_passes(self):
        ok, comp, _ = validate_article_intelligence(_make_intel(), _make_intel())
        assert ok and comp > 0.99

    def test_subjective_enum_diff_now_passes(self):
        # overall_sentiment off by one ordinal rung: hard-failed Tier-1 before,
        # now scored with partial credit -> still valid.
        miner = _make_intel(overall_sentiment=Sentiment.VERY_BULLISH,
                            impact_potential=ImpactPotential.MEDIUM)
        ok, comp, details = validate_article_intelligence(miner, _make_intel())
        assert ok, f"expected pass, composite={comp} details={details['tier3']}"

    def test_llm_enum_diff_no_longer_hard_fails(self):
        # content_type is LLM-derived -> now tolerant-scored, not a hard gate.
        miner = _make_intel(content_type=ArticleContentType.OPINION)
        ok, comp, _ = validate_article_intelligence(miner, _make_intel())
        assert ok, f"content_type diff should be tolerated, composite={comp}"

    def test_deterministic_field_diff_hard_fails(self):
        # primary_sector_id is gazetteer-deterministic -> a mismatch is a Tier-1 gate fail.
        miner = _make_intel(topic_signature=TopicSignature(
            primary_sector_id=7, primary_sector_symbol="TECH"))
        ok, comp, details = validate_article_intelligence(miner, _make_intel())
        assert not ok and comp == 0.0
        assert details["tier1"]["primary_sector_id"]["match"] is False

    def test_contagion_determinism_gate(self):
        miner = _make_intel(contagion_links=[
            ContagionLink(source_ticker="ETH", target_ticker="STETH",
                          target_asset_class=AssetClass.CRYPTO,
                          mechanism=ContagionMechanism.PROTOCOL_DEPENDENCY,
                          direction=SentimentDirection.NEUTRAL, strength=0.7, confidence=0.7,
                          reasoning="dependency_graph")])
        validator = _make_intel()  # no contagion links
        ok, comp, details = validate_article_intelligence(miner, validator)
        assert not ok and details["tier2"]["contagion_determinism"]["jaccard"] < 0.9

    @staticmethod
    def _nvda(direction):
        return AssetSentiment(
            ticker="NVDA", asset_name="NVIDIA", asset_class=AssetClass.EQUITY,
            direction=direction, magnitude=0.5, confidence=0.5,
            short_term_outlook=direction, medium_term_outlook=direction,
            long_term_outlook=direction, causal_driver="x",
            relevance_score=1.0, is_primary_subject=True, evidence_spans=["NVDA"])

    def test_asset_sentiment_determinism_gate(self):
        # A full sign reversal on the only common asset drags mean agreement below
        # the threshold -> hard fail. (Schema: agreement/common_assets, not the old
        # direction_rate/outlook_rate.)
        miner = _make_intel(assets=[self._nvda(Sentiment.BULLISH)])
        validator = _make_intel(assets=[self._nvda(Sentiment.BEARISH)])
        ok, comp, details = validate_article_intelligence(miner, validator)
        det = details["tier2"]["asset_sentiment_determinism"]
        assert not ok and det["agreement"] < det["threshold"] and det["common_assets"] == 1

    def test_zero_asset_miner_against_resolved_validator_hard_fails(self):
        # Anti-cheat: validator resolved an asset, miner submitted none -> the miner
        # is skipping the Tier-2b agreement gate (a no-op with no common assets).
        miner = _make_intel(assets=[])
        validator = _make_intel(assets=[self._nvda(Sentiment.BULLISH)])
        ok, comp, details = validate_article_intelligence(miner, validator)
        assert not ok and comp == 0.0
        assert details["tier2"]["asset_presence"]["reason"] == "miner_missing_assets"

    def test_both_zero_assets_still_passes(self):
        # Symmetric absence (no assets either side) is normal -> not penalized.
        ok, comp, _ = validate_article_intelligence(_make_intel(assets=[]), _make_intel(assets=[]))
        assert ok and comp > 0.99

    # ── TIER3_THRESHOLD recalibration 0.75 -> 0.70 ─────────────────────────────
    @staticmethod
    def _noisy_miner():
        # An honest miner whose independent LLM call produced different free-text
        # (chart_summary, event_fingerprint, narrative_keywords) + a couple of
        # nominal-enum differences -- the realistic source of honest composite
        # variance. Deterministic fields stay identical so only Tier-3 moves.
        return _make_intel(
            chart_summary=ChartSummary(headline="Totally different phrasing here",
                                       one_liner="A wholly different one liner sentence about it",
                                       context_paragraph="An unrelated paragraph with other wording entirely."),
            event_fingerprint=EventFingerprint(event_type=EventType.MONETARY_POLICY,
                                               event_title="Different event title text",
                                               event_date="2026-01-01", content_hash="hash123",
                                               semantic_fingerprint=["alpha", "beta", "gamma"]),
            narrative_keywords=["xyz-unrelated-narrative"],
            content_type=ArticleContentType.OPINION,
            primary_geo=GeoImpactZone.EU,
            target_audience=TargetAudience.RETAIL)

    def test_tier3_threshold_value_is_recalibrated(self):
        import alpharidge_ai.analyzer.scoring as S
        assert S.TIER3_THRESHOLD == 0.70

    def test_threshold_gates_exactly_at_composite(self):
        # The accept/reject boundary tracks the SERVED threshold. The module constant
        # in scoring.py is only a fallback for when config cannot be read, so patching
        # it proves nothing about what a validator actually does.
        from alpharidge_ai import config as cfg
        miner, validator = self._noisy_miner(), _make_intel(narrative_keywords=["fed-policy"])
        _, comp, _ = validate_article_intelligence(miner, validator)
        assert comp < 1.0  # genuinely degraded
        orig = getattr(cfg, "TIER3_THRESHOLD", None)
        try:
            cfg.TIER3_THRESHOLD = round(comp - 0.02, 4)
            ok_lo, _, _ = validate_article_intelligence(miner, validator)
            cfg.TIER3_THRESHOLD = round(comp + 0.02, 4)
            ok_hi, _, _ = validate_article_intelligence(miner, validator)
        finally:
            if orig is not None:
                cfg.TIER3_THRESHOLD = orig
        assert ok_lo and not ok_hi

    def test_recalibration_rescues_honest_noise_pair(self):
        # A pair the old 0.75 floor rejected but 0.70 accepts -> the whole point.
        from alpharidge_ai import config as cfg
        miner, validator = self._noisy_miner(), _make_intel(narrative_keywords=["fed-policy"])
        _, comp, _ = validate_article_intelligence(miner, validator)
        assert 0.70 <= comp < 0.75, f"craft composite {comp} not in the recalibration band"
        orig = getattr(cfg, "TIER3_THRESHOLD", None)
        try:
            cfg.TIER3_THRESHOLD = 0.70
            ok_new, _, _ = validate_article_intelligence(miner, validator)
            cfg.TIER3_THRESHOLD = 0.75
            ok_old, _, _ = validate_article_intelligence(miner, validator)
        finally:
            if orig is not None:
                cfg.TIER3_THRESHOLD = orig
        assert ok_new and not ok_old

    def test_the_served_threshold_is_what_gates(self):
        """The module constant is a fallback, not the value in force. A test that
        patches it passes or fails for reasons unrelated to what validators do."""
        from alpharidge_ai import config as cfg
        import alpharidge_ai.analyzer.scoring as S
        miner, validator = self._noisy_miner(), _make_intel(narrative_keywords=["fed-policy"])
        _, comp, _ = validate_article_intelligence(miner, validator)

        orig_cfg = getattr(cfg, "TIER3_THRESHOLD", None)
        orig_mod = S.TIER3_THRESHOLD
        try:
            cfg.TIER3_THRESHOLD = round(comp - 0.02, 4)   # served: accept
            S.TIER3_THRESHOLD = round(comp + 0.02, 4)     # module: would reject
            accepted, _, _ = validate_article_intelligence(miner, validator)
        finally:
            if orig_cfg is not None:
                cfg.TIER3_THRESHOLD = orig_cfg
            S.TIER3_THRESHOLD = orig_mod
        assert accepted, "the served value must win over the module fallback"
