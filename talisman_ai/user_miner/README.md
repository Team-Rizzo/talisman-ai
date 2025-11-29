# User Miner Module

The user miner is the core component that orchestrates the post processing pipeline: **scrape → analyze → submit**. It runs continuously in a background thread, fetching posts from X/Twitter, analyzing them for subnet relevance and sentiment, and submitting high-quality posts to the coordination API v2.

## Overview

The miner follows a **block-based approach** aligned with API v2's rate limiting system:

1. **Wait** for new block window (every 100 blocks, ~20 minutes)
2. **Scrape** exactly 5 posts from X/Twitter API (matches API rate limit)
3. **Analyze** each post for subnet relevance and sentiment using LLM-based analysis
4. **Score** posts using a weighted combination of relevance, value, and recency
5. **Submit** all analyzed posts to the API v2 server (`/v2/submit`)

The miner runs in a single background thread for simplicity and reliability. It synchronizes with the API's block number to ensure accurate window alignment and respects server-side rate limits.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MyMiner (Background Thread)            │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ PostScraper  │───▶│   Analyzer   │───▶│ APIClient   │  │
│  │              │    │              │    │             │  │
│  │ - Fetches    │    │ - Relevance  │    │ - Submits   │  │
│  │   tweets     │    │ - Sentiment  │    │   to API    │  │
│  │ - Filters    │    │ - Scoring    │    │ - Retries   │  │
│  │   duplicates │    │              │    │             │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Components

### MyMiner (`my_miner.py`)

The main orchestrator class that runs the processing loop. It:

- Manages the background thread lifecycle (`start()`, `stop()`)
- Coordinates scraping, analysis, and submission
- Tracks processed posts to avoid duplicates
- Uses **block-based windows** (every 100 blocks) aligned with API v2 rate limits
- Synchronizes block number with API (source of truth)
- Respects configuration limits (`BLOCKS_PER_WINDOW`, `MAX_SUBMISSIONS_PER_WINDOW`)
- Handles errors gracefully with logging (429 rate limits, 409 conflicts)

**Key Methods:**
- `start()`: Starts the background processing thread, syncs with API block number
- `stop()`: Stops the thread gracefully (waits up to 5 seconds)
- `_run()`: Main processing loop (runs in background thread)
- `_should_scrape()`: Determines if it's time to scrape based on window info from the API
- `_update_window_from_response()`: Updates window tracking from API responses

### PostScraper (`post_scraper.py`)

Fetches posts from X/Twitter API using the Tweepy client.

**Features:**
- **On-demand fetching**: Fetches fresh posts each time `scrape_posts()` is called (no pool maintained)
- Searches for tweets matching configurable keywords (default: `["bittensor"]`)
- Fetches tweets from the last 72 hours
- Excludes retweets and filters by English language
- Fetches at least 10 tweets (X API minimum), then randomly samples down to requested count
- Designed for block-based workflow: fetches exactly `count` posts per window

**Configuration:**
- Keywords are configurable in `post_scraper.py` (line 25): `self.keywords = ["bittensor"]`
- Uses `X_BEARER_TOKEN` and `X_API_BASE` from `.miner_env` configuration

**Data Format:**
Each post includes:
- `id`: Post ID
- `content`: Post text
- `author`: Author username
- `timestamp`: Unix timestamp
- `account_age`: Account age in days
- `retweets`, `likes`, `responses`: Engagement metrics
- `followers`: Author follower count

### Analyzer (`analyzer/`)

Uses LLM-based analysis to determine subnet relevance and sentiment.

**Subnet Relevance Analysis:**
- Analyzes how relevant a post is to each registered BitTensor subnet
- Uses hybrid approach: LLM understanding + deterministic scoring
- Returns relevance scores (0.0 to 1.0) for each subnet
- Subnets are loaded from `analyzer/data/subnets.json`

**Sentiment Analysis:**
- Classifies sentiment on a scale from -1.0 (very bearish) to +1.0 (very bullish)
- Categories:
  - `1.0`: Very bullish (excitement, major positive developments)
  - `0.5`: Moderately positive (optimistic, good news)
  - `0.0`: Neutral (factual, informative, balanced)
  - `-0.5`: Moderately negative (concerns, skepticism)
  - `-1.0`: Very bearish (major issues, strong criticism)

**Scoring:**
Posts are scored using `score_post_entry()` which combines three components:

1. **Relevance** (50% weight): Mean relevance score of top-k subnets
   - Uses LLM-based analyzer to determine subnet relevance
   - Considers top 5 most relevant subnets by default

2. **Value** (40% weight): Signal quality based on engagement and author credibility
   - Normalizes: likes, retweets, quotes, replies, followers, account age
   - Uses caps: 5k likes, 1k retweets, 300 quotes, 600 replies, 200k followers, 7 years account age

3. **Recency** (10% weight): How recent the post is
   - Linear decay from 1.0 (just posted) to 0.0 (older than 24 hours)
   - Formula: `1.0 - (age_hours / 24.0)`

**Final Score:**
```
post_score = 0.50 × relevance + 0.40 × value + 0.10 × recency
```

### APIClient (`api_client.py`)

Handles HTTP communication with the coordination API v2 server.

**Features:**
- Submits posts to `/v2/submit` endpoint (API v2)
- Uses Bittensor wallet authentication (`X-Auth-*` headers)
- Implements retry logic with exponential backoff:
  - Up to 3 attempts total
  - Wait times: 3s after first failure, 6s after second failure
  - 10 second timeout per request
- Handles rate limit errors (429): Extracts block/window info, doesn't retry immediately
- Handles conflict errors (409): Permanent failure, doesn't retry
- Treats "duplicate" status as success (API is idempotent per `(miner_hotkey, post_id)`)
- Extracts block/window info from API responses for synchronization
- Logs validation selection status (when posts are selected for validation)

**Submission Format:**
```python
{
    "miner_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    "post_id": "1234567890",
    "content": "Normalized post text...",
    "date": 1234567890,  # Unix timestamp
    "author": "username",
    "account_age": 365,
    "retweets": 10,
    "likes": 50,
    "responses": 5,
    "followers": 1000,
    "tokens": {  # Subnet relevance scores
        "subnet_name": 0.85,
        ...
    },
    "sentiment": 0.5,  # -1.0 to +1.0
    "score": 0.75  # Final weighted score
}
```

## Processing Flow

1. **Initialization:**
   - Miner starts with hotkey from parent neuron
   - Initializes `PostScraper`, `Analyzer`, and `APIClient`
   - Loads subnet registry from `analyzer/data/subnets.json`
   - Attempts to synchronize with API's block number via `/v2/status` endpoint

2. **Block Window Detection:**
   - Polls current block number every ~12 seconds (block time)
   - Uses API's block number as source of truth (synchronized from API responses)
   - Detects new block window when `current_block // BLOCKS_PER_WINDOW` changes
   - Each window is `BLOCKS_PER_WINDOW` blocks (default: 100 blocks ≈ 20 minutes)

3. **Scrape Cycle (per window):**
   - When new window detected, scrapes exactly `MAX_SUBMISSIONS_PER_WINDOW` posts (default: 5)
   - Matches API v2's rate limit: `MAX_SUBMISSION_RATE` per `BLOCKS_PER_WINDOW`
   - Filters out posts already seen (tracked in `_seen_post_ids`)

4. **Analysis:**
   - For each new post:
     - Normalizes content using `norm_text()` (ensures consistency with validator)
     - Analyzes for subnet relevance and sentiment using LLM
     - Calculates post score using `score_post_entry()`
     - Prepares submission data with all required fields

5. **Submission:**
   - Submits all analyzed posts to `/v2/submit` endpoint
   - Synchronizes block number with API from response (source of truth)
   - Handles errors appropriately:
     - **429 (Rate Limit)**: Waits for next window, doesn't retry immediately
     - **409 (Conflict)**: Permanent failure, marks post as seen, doesn't retry
     - **Other errors**: Retries up to 3 times with exponential backoff
   - Tracks successful submissions in `_seen_post_ids`
   - Increments `posts_processed` counter

6. **Limits:**
   - Respects API's server-side rate limits (5 posts per 100-block window)
   - Continues until stopped explicitly

## Configuration

The miner respects the following environment variables from `.miner_env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `BLOCKS_PER_WINDOW` | Number of blocks per rate limit window | `100` (~20 minutes) |
| `MAX_SUBMISSIONS_PER_WINDOW` | Max posts to scrape/submit per window | `5` (matches API rate limit) |
| `X_BEARER_TOKEN` | X/Twitter API bearer token | Required |
| `X_API_BASE` | X/Twitter API base URL | `https://api.twitter.com/2` |
| `MINER_API_URL` | Coordination API v2 server URL | `http://127.0.0.1:8001` (v2 port) |
| `BATCH_HTTP_TIMEOUT` | HTTP timeout for API requests | `30.0` |
| `MODEL` | LLM model identifier | `deepseek-ai/DeepSeek-V3-0324` |
| `API_KEY` | LLM API key | Required |
| `LLM_BASE` | LLM API base URL | `https://llm.chutes.ai/v1` |

**Note:** Legacy variables (`SCRAPE_INTERVAL_SECONDS`, `POSTS_PER_SCRAPE`, `POSTS_TO_SUBMIT`) are no longer used. The miner now uses block-based windows aligned with API v2's rate limiting system.

### Rate Limiting Configuration

**IMPORTANT:** The miner's rate limit configuration (`BLOCKS_PER_WINDOW` and `MAX_SUBMISSIONS_PER_WINDOW`) **must match** the API server's configuration. These values are not independent tuning knobs - they must be changed together with the API server's settings to ensure proper synchronization.

- The miner scrapes and attempts to submit at most `MAX_SUBMISSIONS_PER_WINDOW` posts per `BLOCKS_PER_WINDOW` block window.
- The API server enforces these same limits server-side and returns 429 (rate limit exceeded) if exceeded.
- The miner relies on the API as the source of truth for window boundaries and rate limit status.
- Changing these values without updating the API server will cause rate limit mismatches and submission failures.

## Text Normalization

The miner normalizes post content using `norm_text()` to ensure consistency with validator analysis:

- Unicode normalization (NFC)
- Line ending normalization (`\r\n` → `\n`)
- Whitespace collapse (multiple spaces → single space)
- Trim leading/trailing whitespace

This ensures that minor formatting differences don't cause false mismatches during validation.

## Error Handling

The miner handles errors gracefully:

- **Scraping errors**: Logs warning and continues to next window
- **Analysis errors**: Logs warning, uses default values (score = 0.0), continues
- **API submission errors**:
  - **429 (Rate Limit)**: Extracts block/window info, waits for next window, doesn't retry immediately
  - **409 (Conflict)**: Permanent failure (duplicate/conflict), marks post as seen, doesn't retry
  - **Other errors**: Retries up to 3 times with exponential backoff (3s, 6s, 12s)
- **Block synchronization errors**: Waits for API synchronization (via startup sync or submission responses)
- **Thread errors**: Logs error, waits ~12 seconds (block time), continues loop

## Statistics

The miner tracks `posts_processed` internally (number of posts successfully submitted), which is logged when the miner stops.

## API v2 Integration

The miner is designed to work with **API v2's probabilistic validation system**:

- **Rate Limiting**: Submits exactly 5 posts per 100-block window, matching API's `MAX_SUBMISSION_RATE`
- **Block Synchronization**: Uses API's block number as source of truth for window calculations
- **Validation Selection**: Posts have a 20% chance (configurable) of being selected for validation (handled server-side)
- **Idempotency**: API is idempotent per `(miner_hotkey, post_id)` - duplicate submissions return "duplicate" status
- **Error Responses**: API returns block/window info in all responses for synchronization

The miner doesn't need to know about validation selection or reward calculation - these are handled entirely by the API server.

## Integration

The miner is integrated into the main neuron (`neurons/miner.py`):

```python
class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        hotkey = self.wallet.hotkey.ss58_address
        self.my_miner = MyMiner(hotkey=hotkey)
        self.my_miner.start()
```

The miner runs independently in a background thread and doesn't interfere with the neuron's synapse processing.

## Customization

### Changing Search Keywords

Edit `post_scraper.py` line 24:
```python
self.keywords = ["your", "keywords", "here"]
```