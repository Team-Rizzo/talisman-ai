"""
Pydantic models for Talisman AI API.

These models correspond to the Prisma schema and are used for
request/response validation and serialization.
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field


# ============================================================================
# Account Models (Twitter/X user accounts)
# ============================================================================

class AccountBase(BaseModel):
    """Base account model with common fields."""
    id: int  # BigInt in Prisma
    name: Optional[str] = None
    screen_name: str = Field(alias="screenName")
    user_name: Optional[str] = Field(None, alias="userName")
    location: Optional[str] = None
    description: Optional[str] = None
    verified: bool = False
    is_blue_verified: bool = Field(False, alias="isBlueVerified")
    followers_count: int = Field(0, alias="followersCount")
    following_count: int = Field(0, alias="followingCount")
    statuses_count: int = Field(0, alias="statusesCount")
    profile_image_url: Optional[str] = Field(None, alias="profileImageUrl")
    
    class Config:
        populate_by_name = True


class AccountCreate(BaseModel):
    """Model for creating a new account."""
    id: int
    screen_name: str
    name: Optional[str] = None
    user_name: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    verified: bool = False
    is_blue_verified: bool = False
    followers_count: int = 0
    following_count: int = 0
    statuses_count: int = 0
    profile_image_url: Optional[str] = None


class Account(AccountBase):
    """Full account model for responses."""
    # NOTE: Keep timestamps as ISO strings to avoid JSON serialization issues when
    # TweetWithAuthor objects are embedded inside bittensor synapses.
    created_at: Optional[str] = Field(None, alias="createdAt")
    
    class Config:
        populate_by_name = True


# ============================================================================
# Tweet Analysis Models (Sentiment/classification - separate from raw tweet)
# ============================================================================

class TweetAnalysisBase(BaseModel):
    """Base tweet analysis model."""
    sentiment: Optional[str] = None  # very_bullish, bullish, neutral, bearish, very_bearish
    subnet_id: Optional[int] = Field(None, alias="subnetId")
    subnet_name: Optional[str] = Field(None, alias="subnetName")
    content_type: Optional[str] = Field(None, alias="contentType")
    technical_quality: Optional[str] = Field(None, alias="technicalQuality")
    market_analysis: Optional[str] = Field(None, alias="marketAnalysis")
    impact_potential: Optional[str] = Field(None, alias="impactPotential")
    
    class Config:
        populate_by_name = True
        # Allow API-side analysis objects (which include id/tweetId/analyzedAt) to be
        # accepted where we only need the base classification fields.
        extra = "allow"


class TweetAnalysisCreate(BaseModel):
    """Model for creating tweet analysis."""
    tweet_id: int
    sentiment: Optional[str] = None
    subnet_id: Optional[int] = None
    subnet_name: Optional[str] = None
    content_type: Optional[str] = None
    analysis_data: Optional[dict] = None


class TweetAnalysis(TweetAnalysisBase):
    """Full tweet analysis model for responses."""
    id: int
    tweet_id: int = Field(alias="tweetId")
    analyzed_at: str = Field(alias="analyzedAt")
    
    class Config:
        populate_by_name = True


# ============================================================================
# Tweet Models
# ============================================================================

class TweetBase(BaseModel):
    """Base tweet model with common fields."""
    id: int  # BigInt in Prisma
    type: str = "tweet"
    url: Optional[str] = None
    text: Optional[str] = None
    lang: Optional[str] = None
    
    # Engagement metrics
    retweet_count: int = Field(0, alias="retweetCount")
    reply_count: int = Field(0, alias="replyCount")
    like_count: int = Field(0, alias="likeCount")
    quote_count: int = Field(0, alias="quoteCount")
    view_count: int = Field(0, alias="viewCount")
    bookmark_count: int = Field(0, alias="bookmarkCount")
    
    # Reply/conversation info
    is_reply: bool = Field(False, alias="isReply")
    in_reply_to_id: Optional[int] = Field(None, alias="inReplyToId")
    conversation_id: Optional[int] = Field(None, alias="conversationId")
    
    # Author
    author_id: Optional[int] = Field(None, alias="authorId")
    
    # Timestamps
    created_at: Optional[str] = Field(None, alias="createdAt")
    received_at: str = Field(alias="receivedAt")
    
    class Config:
        populate_by_name = True


class TweetCreate(BaseModel):
    """Model for creating a new tweet."""
    id: int
    type: str = "tweet"
    url: Optional[str] = None
    text: Optional[str] = None
    lang: Optional[str] = None
    author_id: Optional[int] = None
    created_at: Optional[str] = None
    retweet_count: int = 0
    reply_count: int = 0
    like_count: int = 0
    quote_count: int = 0
    view_count: int = 0
    bookmark_count: int = 0
    is_reply: bool = False
    in_reply_to_id: Optional[int] = None
    conversation_id: Optional[int] = None


class Tweet(TweetBase):
    """Full tweet model for responses."""
    pass


class TweetWithAuthor(Tweet):
    """Tweet model with nested author (account) information."""
    author: Optional[Account] = None
    # Miner responses only include base analysis fields (no DB ids/timestamps),
    # so this must be the base type (or parsing will fail).
    analysis: Optional[TweetAnalysisBase] = None


# ============================================================================
# Scoring Models
# ============================================================================

class ScoringBase(BaseModel):
    """Base scoring model with common fields."""
    id: int
    tweet_id: int = Field(alias="tweetId")
    status: str = "pending"  # pending, in_progress, completed
    
    class Config:
        populate_by_name = True


class ScoringCreate(BaseModel):
    """Model for creating a scoring entry."""
    tweet_id: int
    status: str = "pending"
    validator_hotkey: Optional[str] = None


class ScoringUpdate(BaseModel):
    """Model for updating scoring status."""
    status: str
    validator_hotkey: Optional[str] = None


class Scoring(ScoringBase):
    """Full scoring model for responses."""
    start_time: Optional[str] = Field(None, alias="startTime")
    validator_hotkey: Optional[str] = Field(None, alias="validatorHotkey")
    score: Optional[float] = None
    created_at: str = Field(alias="createdAt")
    
    class Config:
        populate_by_name = True


class ScoringWithTweet(Scoring):
    """Scoring model with nested tweet information."""
    tweet: TweetWithAuthor


# ============================================================================
# Penalty Models
# ============================================================================

class PenaltyBase(BaseModel):
    """Base penalty model with common fields."""
    hotkey: str
    reason: Optional[str] = None  # reason is optional in the schema
    
    class Config:
        populate_by_name = True


class PenaltyCreate(BaseModel):
    """Model for creating a penalty."""
    hotkey: str
    reason: Optional[str] = None


class Penalty(PenaltyBase):
    """Full penalty model for responses."""
    id: int
    timestamp: str


class PenaltyBulkCreate(BaseModel):
    """Model for creating multiple penalties at once."""
    penalties: List[PenaltyCreate]


# ============================================================================
# Reward Models
# ============================================================================

class RewardBase(BaseModel):
    """Base reward model with common fields."""
    start_block: int = Field(alias="startBlock")
    stop_block: int = Field(alias="stopBlock")
    hotkey: str
    points: float
    
    class Config:
        populate_by_name = True


class RewardCreate(BaseModel):
    """Model for creating a reward."""
    start_block: int
    stop_block: int
    hotkey: str
    points: float


class Reward(RewardBase):
    """Full reward model for responses."""
    id: int
    created_at: str = Field(alias="createdAt")
    
    class Config:
        populate_by_name = True


class RewardBulkCreate(BaseModel):
    """Model for creating multiple rewards at once."""
    rewards: List[RewardCreate]


# ============================================================================
# Blacklisted Hotkey Models
# ============================================================================

class BlacklistedHotkeyBase(BaseModel):
    """Base blacklisted hotkey model."""
    hotkey: str
    reason: Optional[str] = None


class BlacklistedHotkeyCreate(BaseModel):
    """Model for creating a blacklisted hotkey."""
    hotkey: str
    reason: Optional[str] = None


class BlacklistedHotkey(BlacklistedHotkeyBase):
    """Full blacklisted hotkey model for responses."""
    created_at: str = Field(alias="createdAt")
    
    class Config:
        populate_by_name = True


class BlacklistedHotkeyBulkCreate(BaseModel):
    """Model for creating multiple blacklisted hotkeys at once."""
    hotkeys: List[str]
    reason: Optional[str] = None


# ============================================================================
# Response Models
# ============================================================================

class TweetsForScoringResponse(BaseModel):
    """Response model for getting tweets for scoring."""
    tweets: List[TweetWithAuthor]
    count: int


class CompletedTweetSubmission(BaseModel):
    """Model for submitting a completed scored tweet."""
    tweet_id: int
    sentiment: str
    # Optional richer classification fields (backward compatible with sentiment-only submissions)
    subnet_id: Optional[int] = None
    subnet_name: Optional[str] = None
    content_type: Optional[str] = None
    technical_quality: Optional[str] = None
    market_analysis: Optional[str] = None
    impact_potential: Optional[str] = None
    relevance_confidence: Optional[str] = None


class CompletedTweetsSubmission(BaseModel):
    """Model for submitting multiple completed scored tweets."""
    completed_tweets: List[CompletedTweetSubmission]


class SubmissionResponse(BaseModel):
    """Generic response for submission endpoints."""
    success: bool
    message: str
    count: int = 0


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str


# ============================================================================
# TAO Price Models
# ============================================================================

class TaoPriceResponse(BaseModel):
    """Response model for TAO/USD price endpoint."""
    price_usd: float
    last_updated: str
    source: str
    stale: bool
