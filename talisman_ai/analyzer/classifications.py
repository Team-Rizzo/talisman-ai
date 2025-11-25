"""
Categorical Enums for X-Post Classification

All classification decisions use discrete categories (no floats) to enable
deterministic validation via exact matching of canonical strings.
"""

from enum import Enum

class ContentType(str, Enum):
    """Type of X-Post content"""
    TECHNICAL_INSIGHT = "technical_insight"      # Technical analysis, architecture, code
    ANNOUNCEMENT = "announcement"                # Product launches, releases
    PARTNERSHIP = "partnership"                  # Collaborations, integrations
    MILESTONE = "milestone"                      # Achievements, metrics
    TUTORIAL = "tutorial"                        # How-to, guides
    SECURITY = "security"                        # Exploits, vulnerabilities
    GOVERNANCE = "governance"                    # Proposals, voting
    MARKET_DISCUSSION = "market_discussion"      # Price, trading, speculation
    HIRING = "hiring"                            # Job postings
    MEME = "meme"                               # Jokes, entertainment
    HYPE = "hype"                               # Excitement, enthusiasm
    OPINION = "opinion"                         # Personal views, analysis
    COMMUNITY = "community"                      # General chatter, engagement
    FUD = "fud"                                  # Fear, uncertainty, doubt
    OTHER = "other"


class MarketAnalysis(str, Enum):
    """Market analysis type"""
    TECHNICAL = "technical"     # Indicators, order flow, patterns
    ECONOMIC = "economic"       # Fundamentals, costs, revenue
    POLITICAL = "political"     # Regulatory, governance
    SOCIAL = "social"          # Narrative, virality
    OTHER = "other"


class ImpactPotential(str, Enum):
    """Expected community impact"""
    HIGH = "HIGH"       # Major release/mainnet/security incident
    MEDIUM = "MEDIUM"   # Notable update/launch/partnership
    LOW = "LOW"        # Minor info
    NONE = "NONE"      # Chatty/irrelevant


class Sentiment(str, Enum):
    """Market sentiment tone"""
    VERY_BULLISH = "very_bullish"   # 🚀, moon, ATH, pump
    BULLISH = "bullish"             # Positive price/growth signals
    NEUTRAL = "neutral"             # Default, factual
    BEARISH = "bearish"             # Concerns, dump, issues
    VERY_BEARISH = "very_bearish"   # Crash, exploit, major failure


class TechnicalQuality(str, Enum):
    """Quality of technical information"""
    HIGH = "high"       # ≥2 specifics: APIs, versions, repos, endpoints, metrics
    MEDIUM = "medium"   # 1 specific detail
    LOW = "low"        # Claims without specifics
    NONE = "none"      # No technical content

