"""Orchestration: turn a (submission, reference, article) triple into one continuous
graded observation. Owns the multilingual embedder, the rolling rarity table, and the
faithfulness scorer. Kept out of the analyzer scoring module so that stays dict-pure.
"""
from __future__ import annotations

from alpharidge_ai import config
from alpharidge_ai.validator import quality
from alpharidge_ai.validator.embedder import MultilingualEmbedder
from alpharidge_ai.validator.rarity import RarityTable


def _substantive(ref: dict) -> bool:
    return (str((ref.get("market_analysis_type") or "none")).lower() != "none"
            or bool(ref.get("assets")))


class GradedScorer:
    def __init__(self):
        self.emb = MultilingualEmbedder()
        self.faith = quality.Faithfulness(self.emb)
        self.rarity = RarityTable()
        self.rarity.load()
        self._since_save = 0

    def score(self, miner_intel, validator_intel, article):
        """Return (graded in [0,1], weight). `*_intel` are ArticleIntelligence objects."""
        ref = validator_intel.model_dump(mode="json")
        sub = miner_intel.model_dump(mode="json")
        # warm the rarity table from the reference analysis stream
        self.rarity.observe(ref)
        self._since_save += 1
        if self._since_save >= 100:
            self.rarity.save()
            self._since_save = 0
        art = {"title": getattr(article, "title", None), "content": getattr(article, "content", None)}
        g = quality.graded_score(sub, ref, art, self.emb, self.rarity.weights())
        w = config.SAMPLING_SUBSTANTIVE_WEIGHT if _substantive(ref) else 1.0
        return g, w

    def faithfulness(self, miner_intel, article) -> float:
        """Reference-free faithfulness of the analysis against the article. `article` must be
        the validator's own reference copy, not the submitted object."""
        sub = miner_intel.model_dump(mode="json")
        art = {"title": getattr(article, "title", None), "content": getattr(article, "content", None)}
        return float(self.faith.score(sub, art))
