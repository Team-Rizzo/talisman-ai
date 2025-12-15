"""
Pydantic models for Talisman AI API.

These models correspond to the Prisma schema and are used for
request/response validation and serialization.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ============================================================================
# User Models
# ============================================================================

class UserBase(BaseModel):
    """Base user model with common fields."""
    id: str
    username: str
    screen_name: str = Field(alias="screenName")
    following: int = 0
    followers: int = 0
    
    class Config:
        populate_by_name = True


class UserCreate(BaseModel):
    """Model for creating a new user."""
    id: str
    username: str
    screen_name: str
    following: int = 0
    followers: int = 0


class User(UserBase):
    """Full user model for responses."""
    pass


# ============================================================================
# Tweet Models
# ============================================================================

class TweetBase(BaseModel):
    """Base tweet model with common fields."""
    id: str
    created_at: datetime = Field(alias="createdAt")
    text: str
    user_id: str = Field(alias="userId")
    timestamp: datetime
    sentiment: Optional[str] = None
    
    class Config:
        populate_by_name = True


class TweetCreate(BaseModel):
    """Model for creating a new tweet."""
    id: str
    created_at: datetime
    text: str
    user_id: str
    timestamp: datetime
    sentiment: Optional[str] = None


class Tweet(TweetBase):
    """Full tweet model for responses."""
    insertion_timestamp: datetime = Field(alias="insertionTimestamp")
    
    class Config:
        populate_by_name = True


class TweetWithUser(Tweet):
    """Tweet model with nested user information."""
    user: User


# ============================================================================
# Scoring Models
# ============================================================================

class ScoringBase(BaseModel):
    """Base scoring model with common fields."""
    id: str
    tweet_id: str = Field(alias="tweetId")
    status: str = "pending"
    
    class Config:
        populate_by_name = True


class ScoringCreate(BaseModel):
    """Model for creating a scoring entry."""
    tweet_id: str
    status: str = "pending"
    validator_hotkey: Optional[str] = None


class ScoringUpdate(BaseModel):
    """Model for updating scoring status."""
    status: str
    validator_hotkey: Optional[str] = None


class Scoring(ScoringBase):
    """Full scoring model for responses."""
    start_time: Optional[datetime] = Field(None, alias="startTime")
    validator_hotkey: Optional[str] = Field(None, alias="validatorHotkey")


class ScoringWithTweet(Scoring):
    """Scoring model with nested tweet information."""
    tweet: Tweet


# ============================================================================
# Penalty Models
# ============================================================================

class PenaltyBase(BaseModel):
    """Base penalty model with common fields."""
    hotkey: str
    reason: str
    
    class Config:
        populate_by_name = True


class PenaltyCreate(BaseModel):
    """Model for creating a penalty."""
    hotkey: str
    reason: str


class Penalty(PenaltyBase):
    """Full penalty model for responses."""
    id: str
    timestamp: datetime


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
    id: str


class RewardBulkCreate(BaseModel):
    """Model for creating multiple rewards at once."""
    rewards: List[RewardCreate]


# ============================================================================
# Blacklisted Hotkey Models
# ============================================================================

class BlacklistedHotkeyBase(BaseModel):
    """Base blacklisted hotkey model."""
    hotkey: str


class BlacklistedHotkeyCreate(BlacklistedHotkeyBase):
    """Model for creating a blacklisted hotkey."""
    pass


class BlacklistedHotkey(BlacklistedHotkeyBase):
    """Full blacklisted hotkey model for responses."""
    pass


class BlacklistedHotkeyBulkCreate(BaseModel):
    """Model for creating multiple blacklisted hotkeys at once."""
    hotkeys: List[str]


# ============================================================================
# Response Models
# ============================================================================

class TweetsForScoringResponse(BaseModel):
    """Response model for getting tweets for scoring."""
    tweets: List[TweetWithUser]
    count: int


class CompletedTweetSubmission(BaseModel):
    """Model for submitting a completed scored tweet."""
    tweet_id: str
    sentiment: str


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

