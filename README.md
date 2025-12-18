# Talisman AI 🪬 The Perception Subnet for On-Chain Trading Insights  

## 🪬 Vision (Why this subnet exists)

We are building an AI financial reasoning agent that:

- Watches what’s happening across the crypto world - markets, chain activity, and social sentiment
- Spots meaningful signals as they are happening
- Explains what’s happening in plain language
- Converts insights into recommended trading or staking actions
- Surfaces those decisions directly to your Talisman wallets

The subnet doesn’t “decide” what to trade, it feeds the agent with validated, high-quality signal data.

Think of it as your AI assistant for crypto decisions. To achieve this, the system needs situational awareness across multiple data streams.

That awareness starts with SN45, which serves as the eyes and ears of the agent.

| Layer      | What it does                                 | Delivered by       |
|------------|----------------------------------------------|--------------------|
| Perception | Collect signals (markets, chains, sentiment) | SN45               |
| Reasoning  | Analyze signals, generate insights           | Talisman Agent     |
| Action     | Recommend / execute trading or staking       | Wallet Integration |

![architecture phase 1](./architecture_p1.png)

---

## Phase Roadmap

| Phase              | Data Source             | Goal |
|--------------------|--------------------------|------|
| ✅ Phase 1 (current) | Social media sentiment  | Identify conversations affecting Bittensor ecosystem, starting with X |
| 🔜 Phase 2          | Chain activity + market data | Detect real on-chain money flow + market shifts, subnet tokenomics, and subnet identity changes |
| 🔜 Phase 3          | Agent insights to wallets | Actionable personalized staking/trading suggestions |

---

## 🪬 Overview

For Phase 1, Talisman AI (Subnet 45) continuously analyzes social media for Bittensor-relevant activity, starting with X.

Miners collectively search for high value posts that are relevant to specific subnets; validators verify accuracy and enforce quality.

A coordination API leases tweet work items to validators and stores completed analysis back into Postgres for downstream consumption.

---

## 🪬 How It Works

### 🪬 Miner (V3)

- Receives TweetBatch requests from validators over the Bittensor network
- Analyzes each tweet using LLM to determine:
  - Subnet relevance (which subnet the tweet is about)
  - Sentiment (very_bullish, bullish, neutral, bearish, very_bearish)
  - Content type (technical_insight, announcement, etc.)
- Returns enriched tweets with analysis data for validator verification

---

### 🪬 Validation

The validator re-analyzes posts independently.  
If any post fails validation, that miner batch is labeled INVALID and discarded.  
Only if all posts pass does the miner receive VALID and the batch proceeds to the next step in the pipeline.

In V3, validation is performed by re-running the same analyzer on a sampled subset of the miner batch and requiring an **exact match** on the key categorical fields:

- `subnet_id`
- `sentiment`
- `content_type`
- `technical_quality`
- `market_analysis`
- `impact_potential`



---

## 🪬 Rewards

V3 rewards are tracked as **epoch-bucketed points** inside validators:

- ✅ **Valid batch**: miner earns points (currently +1 per accepted tweet)
- ❌ **Invalid batch**: miner is penalized for that epoch

Validators combine:

- their **local** rewards/penalties, plus
- **validator↔validator broadcasts** (compact epoch snapshots)

Then, for a delayed epoch window (typically **E-2**), validators compute weights and set them on-chain. Penalized miners have their reward zeroed for that epoch.

---

## 🪬 Architecture (V3)

```

┌──────────────┐        ┌───────────┐        ┌──────────────┐
│ API Server   │  --->  │ Validator │  --->  │   Miner      │
│ (lease queue)│        │           │        │ (analysis)   │
└──────────────┘        └─────┬─────┘        └──────┬───────┘
                              │                     │
                              │  TweetBatch         │
                              │  (with analysis)    │
                              │<────────────────────┘
                              │
                              v
                        ┌──────────────┐
                        │ Set Weights  │
                        │  (on-chain)  │
                        └──────────────┘

```

---

## 🪬 Project Structure

```
talisman-ai/
├── neurons/                    # Miner and validator nodes
│   ├── miner.py               # Miner entry point
│   ├── validator.py           # Validator entry point
└── talisman_ai/               # Core library
    ├── protocol.py            # Bittensor protocol definitions
    ├── config.py              # Configuration
    ├── analyzer/              # Analysis utilities
    ├── validator/             # Validator logic
    └── utils/                 # Utility functions

talisman-ai-api/
├── main.py                     # FastAPI app + routes
├── prisma/schema.prisma        # Postgres schema (scoring lease queue + tweet_analysis)
└── utils/                      # Auth + whitelist utilities


```

---

## 🪬 Configuration

Before running miners or validators, you need to set up your environment configuration files. Template files are provided that you must rename and fill in with your credentials.

### Miner Configuration (`.miner_env`)

Copy `.miner_env_tmpl` to `.miner_env` and configure the following variables:

| Variable | Description |
|----------|-------------|
| `MODEL` | LLM model identifier for analysis (e.g., `deepseek-ai/DeepSeek-V3-0324`) |
| `API_KEY` | API key for the LLM service |
| `LLM_BASE` | Base URL for the LLM API endpoint |

**Note**: V3 miners do not need X/Twitter API credentials. They receive tweets from validators over the network.

### Validator Configuration (`.vali_env`)

Copy `.vali_env_tmpl` to `.vali_env` and configure the following variables:

| Variable | Description |
|----------|-------------|
| `MODEL` | LLM model identifier for re-analysis (e.g., `deepseek-ai/DeepSeek-V3-0324`) |
| `API_KEY` | API key for the LLM service |
| `LLM_BASE` | Base URL for the LLM API endpoint |
| `MINER_API_URL` | Base URL of the coordination API server (e.g., `http://localhost:8000`) |
| `BATCH_HTTP_TIMEOUT` | HTTP timeout in seconds for API requests (default: `30.0`) |
| `VALIDATION_POLL_SECONDS` | Seconds between poll attempts (default: `10`) |
| `MINER_BATCH_SIZE` | Tweets per miner batch (default: `3`) |
| `TWEET_MAX_PROCESS_TIME` | Local processing timeout in seconds before requeue (default: `300.0`) |
| `LOCAL_MINER_UID` | Optional: force the validator to query one miner uid (local testing) |
| `VALIDATOR_BROADCAST_MAX_TARGETS` | Max validators to broadcast epoch snapshots to (default: `32`) |

---

## 🪬 Running on Mainnet

**Run Miner**



```bash
pip install -r requirements.txt
pip install -e .
cp .miner_env_tmpl .miner_env
# edit .miner_env to include your LLM information (MODEL, API_KEY, LLM_BASE)
.venv/bin/python -m neurons.miner \
  --netuid 45 \
  --wallet.name your_coldkey_here \
  --wallet.hotkey your_hotkey_here \
  --logging.info
```

*Optional: Add `--axon.external_port` and `--axon.external_ip`

**Run Validator**

```bash
pip install -r requirements.txt
pip install -e .
cp .vali_env_tmpl .vali_env
# edit .vali_env to include your LLM information (MODEL, API_KEY, LLM_BASE)
.venv/bin/python -m neurons.validator \
  --netuid 45 \
  --subtensor.network <finney/local/test> \
  --wallet.name <your wallet> \
  --wallet.hotkey <your hotkey> \
  --logging.info
```

*Optional*: Run the validator under PM2 with the auto-updater:

```bash
python3 scripts/start_validator.py --pm2_name sn45vali -- --netuid 45 --logging.info
```

If you run into a pip error like “packages do not match the hashes…”, it can be caused by a stale pip wheel cache.
Try:

```bash
.venv/bin/python -m pip cache purge
```

---

## 🪬 License

MIT
