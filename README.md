# Alpharidge AI (Subnet 45)

Subnet 45 reads the world's financial news and turns it into structured market data. About 290,000 articles a day, from 12,997 sources. It's the data layer behind **[AlphaRidge](https://alpharidge.ai)**, which is where that data goes to be used.

Almost all of that work is done by miners. A miner reads an article and produces a full analysis of it: sentiment per asset, entities resolved to Wikidata IDs and tradeable tickers, economic data points, contagion chains, narrative tags and embeddings.

Validators hand out the work in batches and spot-check a sample from each one, re-running the pipeline themselves on those few articles without seeing what the miner sent back. If the sample doesn't reproduce, the whole batch is thrown out and earns nothing. That keeps the volume honest without paying to do it twice.

| | |
|---|---|
| Articles analyzed to date | **16.9 M** |
| Analyzed per day | **~290,000** |
| Distinct source domains | **12,997** |
| Real-world events clustered | **2.6 M** |
| Tickers with linked coverage | **2,138** |
| Assets tracked by the extractor | **2,314** |

<sub>Measured 2026-08-29. Live throughput, the analysis stream and the miner leaderboard run continuously on the subnet observatory at [alpharidge.ai](https://alpharidge.ai).</sub>

**New miner?** Go straight to the [Quickstart](#quickstart-run-a-miner).

---

## Table of contents

- [What this powers](#what-this-powers)
- [Quickstart: run a miner](#quickstart-run-a-miner)
- [How you earn](#how-you-earn)
- [Where the rest of the emission goes](#where-the-rest-of-the-emission-goes)
- [What a miner actually does](#what-a-miner-actually-does)
- [How validation works](#how-validation-works)
- [Dispatch](#dispatch)
- [Configuration](#configuration)
- [Running a validator](#running-a-validator)
- [Architecture](#architecture)

---

## What this powers

SN45 does two things: it reads every article, and it works out how those articles relate to each other. AlphaRidge is the product built on both.

| Layer | What it does | Where |
|---|---|---|
| **Perception** | Read and analyze every article; verify every analysis | **SN45** |
| **Cross-article intelligence** | Cluster articles into real-world events; match events to long-running market narratives | **SN45** |
| **Product** | Chart, Screener, Watchlists, Alerts, Backtesting, AI assistant, public API | AlphaRidge |

In the product, miner output shows up as:

- **News marks on the price chart**: every analyzed article plotted against the tape, ranked by impact.
- **Screener columns**: 7-day sentiment, article count, max impact, bull/bear split and 30-day event count, screenable across the whole equity universe.
- **A per-symbol news feed** built on real ticker linkage from entity resolution, not a text match on the company name.
- **An events browser**: articles grouped into the real-world event behind them, so one story is one row instead of forty.
- **A public sentiment and events API**, and an AI assistant that can read the same feed.
- **A live subnet observatory**: throughput, the analysis stream, trending events and a miner leaderboard.

---

## Quickstart: run a miner

### 1. Hardware

Miner and validator run the same analyzer, so they need the same box.

| | Minimum | Comfortable |
|---|---|---|
| GPU | 8 GB VRAM | 24 GB+ |
| RAM | 16 GB | 32 GB |
| Disk | 60 GB free | 100 GB |
| CPU | 8 cores | 16+ cores |

About 44 GB of models download on first run. The NER stage is CPU- and Python-heavy, not purely GPU-bound. Core count is what usually sets your throughput, and throughput is what you are paid for.

### 2. Install

```bash
git clone https://github.com/Team-Rizzo/alpharidge-ai.git
cd alpharidge-ai
python3.12 -m venv .venv && source .venv/bin/activate
./install.sh          # CUDA 12.8 default
```

For a different CUDA build: `TORCH_INDEX=https://download.pytorch.org/whl/cuXXX ./install.sh`

<details>
<summary>Manual install (equivalent, if you would rather not run the script)</summary>

```bash
python -m pip install --upgrade pip setuptools wheel

# 1. PyTorch: match the CUDA build to your driver (see https://pytorch.org):
pip install "torch>=2" --index-url https://download.pytorch.org/whl/cu128

# 2. The rest of the stack (the spaCy en_core_web_trf model is pinned in requirements.txt):
pip install -r requirements.txt
pip install -e .

# 3. ReFinED (Amazon entity linker) is not on PyPI. Install it with --no-deps so it
#    does not downgrade torch/transformers, then add its small runtime deps:
pip install --no-deps "git+https://github.com/amazon-science/ReFinED.git@V1"
pip install ujson nltk Unidecode lmdb prettyprint
```
</details>

### 3. Configure

```bash
cp miner_env.example .miner_env
```

**For a standard setup you only need to set one value: `API_KEY`.** Everything else in the template is pre-filled.

`API_KEY` is your own [OpenRouter](https://openrouter.ai/keys) key (`sk-or-...`). Two of the four pipeline stages are LLM calls, so you pay for your own inference. It's your main running cost after hardware and it scales directly with how many articles you get through, so budget for it.

### 4. Register and run

```bash
btcli subnet register --netuid 45 --wallet.name <coldkey> --wallet.hotkey <hotkey>

.venv/bin/python -m neurons.miner \
  --netuid 45 \
  --wallet.name <coldkey> \
  --wallet.hotkey <hotkey> \
  --logging.info
```

Optional: `--axon.external_ip` and `--axon.external_port` if you are behind NAT or a proxy.

> **If your axon sits behind nginx**, set `underscores_in_headers on;`. Bittensor sends `bt_header_*` request headers, and nginx silently strips underscored headers by default. Your miner will look healthy and never receive a single batch.

### 5. Confirm it is working

First run downloads ~44 GB of models, so give it time before you judge anything.

- Your logs should show batches arriving, then an analysis time of roughly 12–18 seconds per article.
- Batches arrive on a lease. If you do not return one in time it goes back in the queue for someone else, so a slow box loses work it was already given.
- Your first batches are small on purpose. Batch size ramps with demonstrated throughput, so returning good work promptly grows your allocation.
- Watch the miner leaderboard on the subnet observatory to see your accepted volume and reputation as validators see it.

**Common first-day problems**

| Symptom | Usual cause |
|---|---|
| No batches at all | Axon unreachable, or an nginx/proxy stripping `bt_header_*` |
| Batches arrive, everything is INVALID | Wrong or missing `API_KEY`, or an exhausted OpenRouter balance. Empty LLM output fails validation on every tier |
| Batches arrive, most time out | Box too slow for the batch size, or too few CPU cores for the NER stage |
| `packages do not match the hashes` on install | Stale pip cache. Run `.venv/bin/python -m pip cache purge` |

---

## How you earn

In one line:

```
pay  =  verified volume  ×  reputation multiplier
```

**Verified volume** is the count of items you analyzed in batches that passed their audit. A validator does not re-derive every article you send; it samples a few from each batch and re-runs the pipeline on those. Reproduce the sample and the whole batch counts. Fail it and the whole batch is discarded, however good the rest of it was. Rejected and timed-out items earn nothing and count against you.

That split is the whole arrangement. Miner throughput is the network's throughput; the audit is just the cheap check that keeps it worth something.

**Reputation** is a slow-moving score in `[0, 1]`, an exponential moving average of your graded results. It becomes a multiplier through a logistic gate: below the midpoint the multiplier collapses toward zero, above it, it approaches 1.0. Volume sets the size of your claim. Reputation decides how much of it you keep.

A few consequences:

1. **You're paid for verified volume, scaled by quality.** Uptime and effort don't enter into it.
2. **Failing validation costs you twice.** The item earns nothing, *and* the failure counts against you in the epoch's net test. A miner whose penalties outweigh its points over the window is zeroed for that window.
3. **A single bad batch won't sink you.** The test is a net one, so a stray timeout or mismatch in an otherwise productive window gets absorbed.
4. **Running more UIDs does not multiply your reward.** Each UID's share is computed only from its own verified work. Ten UIDs doing one UID's work earn one UID's pay, split ten ways, and pay ten registration costs to do it.

Your revenue per unit of work is pegged in USD, not in alpha. A point is worth a fixed dollar amount of emission, so your income per analyzed article does not swing with the token price.

---

## Where the rest of the emission goes

Every epoch, each miner's verified points are priced in USD and converted into a percentage of the subnet's miner emission ([`alpharidge_ai/utils/burn.py`](alpharidge_ai/utils/burn.py)). Add those percentages up. Whatever is left over, the share of emission that no verified work claimed, is assigned to **UID 189, the burn UID**.

```python
weights[BURN_UID] = 1 - min(total_percent_needed, 100) / 100   # BURN_UID = 189
```

UID 189 is a burn address. Weight that lands there isn't paid to anyone: it's emission the network doesn't issue, because no verified work claimed it that epoch. Every validator runs the line above and gets the same number.

So nothing is being deducted from what you earn. The burn is emission still on the table, and every verified article you produce takes some of it. What the split looks like depends on how much the network claims in total:

| Condition | Regime | What happens |
|---|---|---|
| Total claimed < 100% | **Absolute** | Each miner is paid exactly what its points are worth at the USD price. The unclaimed remainder burns. |
| Total claimed > 100% | **Share-split** | All shares scale by `100 / total`. Miners split **100% of miner emission** between them, and **burn is exactly zero.** |

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./emission-bar-dark.svg">
    <img alt="How miner emission splits each epoch: miners claim their own share, the remainder burns at UID 189" src="./emission-bar.svg">
  </picture>
</div>

Nobody sets the burn level directly. It's the residual, and it falls as the network's verified output rises. Once the subnet produces enough verified work to be worth the whole of miner emission, it sits at zero and everything goes to miners.

It also changes what competition looks like here. While there's still a residual, your share comes from your own points alone, so another miner earning more doesn't cost you anything. You're both taking from the burn rather than from each other.

---

## What a miner actually does

Miners receive batches over Bittensor and run a staged pipeline on every item. The protocol defines `ArticleBatch`, `TweetBatch` and `TelegramBatch`, and the analyzers for all three ship in this repo. News articles run the deepest of the three pipelines.

The Article Intelligence pipeline has four stages:

| Stage | What runs |
|---|---|
| **1. Deterministic + NER** | Text statistics, keyword asset extraction across 2,300+ tracked assets, and a 4-engine NER fusion (spaCy-trf + GLiNER + Flair + ReFinED) resolving entities to Wikidata QIDs and tradeable tickers, plus FinBERT sentence-level sentiment |
| **2. LLM, Extract & Classify** | Classification enums, economic data points, quotes and an event fingerprint, with the NER output passed in as hints |
| **3. LLM, Reason & Summarize** | Chart summaries and narrative keywords, built from a compact fact sheet rather than the raw article. Per-asset sentiment and contagion chains are computed deterministically off-LLM during assembly |
| **4. Embeddings** | Title, body and narrative vectors (`all-MiniLM-L6-v2`, 384-d, L2-normalized) for downstream clustering |

The output is a structured analysis with dozens of fields across many feature groups, returned to the validator for verification.

The deterministic parts of stage 3 are deliberate. Anything reproducible can be checked exactly by a validator, and anything a validator can check exactly is something you can be paid for reliably.

---

## How validation works

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./audit-dark.svg">
    <img alt="How validation works: a miner analyses a whole batch, the validator re-runs a sample, and the sample decides whether the whole batch is paid or discarded" src="./audit.svg">
  </picture>
</div>

Validators audit a batch rather than reproducing it. A validator draws `VALIDATION_SAMPLE_SIZE` items from the batch, re-runs the same analyzer on those, and grades the result through a four-tier ladder, implemented in [`alpharidge_ai/analyzer/scoring.py`](alpharidge_ai/analyzer/scoring.py). **Fail any tier and the whole batch is INVALID and earns nothing.**

| Tier | Check | Bar |
|---|---|---|
| **1** | Exact enum match | Content type, sentiment, impact, urgency, primary-asset sentiment direction and other categorical fields must match exactly |
| **2** | Deterministic match | Content hash and text statistics (word / sentence / character counts, ticker mentions) must be byte-identical |
| **2.5** | Embedding verification | Title cosine ≥ 0.90, narrative cosine ≥ 0.80, plus dimension and L2-norm sanity checks |
| **3** | Weighted composite | Asset extraction, asset sentiment, entities, economic data, event fingerprint, contagion and narrative keywords, scored with Jaccard / text similarity against a weighted threshold |

Batch validation also runs cross-article adversarial checks, such as flagging a miner whose embeddings come back near-identical for different articles.

Tier 2 catches most honest mistakes. Because it's byte-identical, the article you analyze has to be exactly the article you were sent. Truncating it, re-fetching it from the source or normalizing the text will all fail the hash.

---

## Dispatch

Work is leased to individual miners rather than broadcast. Each validator keeps a window per miner and adjusts it from what it sees:

- **Ack timeout**: a short acknowledgement replaces a long blocking send, so an unreachable axon stops holding a dispatch slot.
- **Liveness**: miners age out of the roster without a heartbeat and stop receiving work.
- **Adaptive batch size**: your batch size ramps up with demonstrated throughput and shrinks when you fail.
- **Cooldowns**: consecutive invalid batches back a miner off with an escalating delay.
- **Lease TTL**: an item you do not return in time is released back into the queue.

A lease you sit on isn't lost to the network, since it goes back in the pool and gets dispatched again. It's only lost to you.

---

## Configuration

### Miner (`.miner_env`)

Copy `miner_env.example` to `.miner_env`. Only `API_KEY` normally needs changing.

| Variable | Description |
|---|---|
| `API_KEY` | **Your OpenRouter API key (`sk-or-...`)**, from https://openrouter.ai/keys |
| `MODEL` | LLM model for analysis (pre-filled, e.g. `deepseek/deepseek-v4-flash`) |
| `LLM_BASE` | LLM API base URL (pre-filled: `https://openrouter.ai/api/v1`) |
| `MINER_API_URL` | Coordination API base URL (pre-filled: `https://api.alpharidge.ai`) |
| `BATCH_HTTP_TIMEOUT` | HTTP timeout in seconds for API requests (default `30.0`) |

### Validator (`.vali_env`)

Copy `vali_env.example` to `.vali_env`. Same rule: `API_KEY` is normally the only required edit.

| Variable | Description |
|---|---|
| `API_KEY` | **Your OpenRouter API key (`sk-or-...`)** |
| `MODEL` / `LLM_BASE` / `MINER_API_URL` | Pre-filled, as above |
| `VALIDATION_POLL_SECONDS` | Seconds between polls for new work (default `10`) |
| `VALIDATION_FETCH_LIMIT` | Items fetched per poll, split into miner batches (default `24`) |
| `VALIDATION_MAX_WORKERS` | Max concurrent validation threads making LLM calls (default `8`) |
| `MINER_SEND_TIMEOUT` | Validator → miner dendrite timeout in seconds (default `6`) |
| `SCORES_BLOCK_INTERVAL` | Blocks between score fetches (default `100`) |
| `LLM_CACHE_TTL` / `LLM_CACHE_MAX_SIZE` | TTL (s) and max size of the in-memory LLM result cache (defaults `300` / `1024`) |
| `BATCH_HTTP_TIMEOUT` | HTTP timeout in seconds (default `30.0`) |

> Verifiable-points settings (`API_ATTESTATION_PUBKEY`, `ENFORCE_SIGNED_ATTESTATIONS`, `DEEP_VERIFY_SAMPLE_RATE`) have safe defaults in `config.py` and do not normally need setting.

---

## Running a validator

Same install and same hardware as a miner.

```bash
cp vali_env.example .vali_env
# Set API_KEY. MODEL / LLM_BASE / MINER_API_URL are pre-filled.

.venv/bin/python -m neurons.validator \
  --netuid 45 \
  --subtensor.network finney \
  --wallet.name <coldkey> \
  --wallet.hotkey <hotkey> \
  --logging.info
```

Optional: run under PM2 with the auto-updater:

```bash
python3 scripts/start_validator.py --pm2_name sn45vali -- --netuid 45 --logging.info
```

Validators don't take each other's word about a miner. Each pools its own points and penalties with compact, signed epoch snapshots from its peers. Those snapshots carry a Merkle root over the sender's verdicts, and the receiver re-derives that root from the underlying records rather than accepting the claim. A snapshot that does not reconcile is reported.

---

## Architecture

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./architecture-dark.svg">
    <img alt="SN45 architecture: coordination API leases batches to miners, the validator audits a sample and sets weights on chain, and verified analysis feeds the cross-article layer and AlphaRidge" src="./architecture.svg">
  </picture>
</div>

Two independent things happen once a batch is audited.

**Scoring stays on the validator.** It keeps its own points and penalties, pools them with signed epoch snapshots from the other validators, and sets weights from that total. Nothing is read back from the coordination API to do it, and no component outside the validators has any say in what a miner is paid.

**The analysis goes on to be used.** Verified output is stored and feeds the layer no single miner can produce alone:

- **Event clustering**: group items covering the same real-world occurrence, via event fingerprint → content hash → embedding cosine → title overlap.
- **Narrative matching**: match those events to long-running market themes by slug, embedding and keyword signal, with background jobs maintaining narrative lifecycle, centroid drift and discovery. The narrative set is curated and deliberately narrow, at 38 themes rather than an open-ended list.

<details>
<summary><strong>Project structure</strong></summary>

```
alpharidge-ai/
├── neurons/                          # Miner and validator nodes
│   ├── miner.py                      # Miner entry point
│   └── validator.py                  # Validator entry point
└── alpharidge_ai/                    # Core library
    ├── protocol.py                   # Bittensor synapses (ArticleBatch, TweetBatch, TelegramBatch, ...)
    ├── config.py                     # Configuration
    ├── models/                       # Data models (article_intelligence.py, reward.py)
    ├── analyzer/                     # Analysis pipeline
    │   ├── article_intelligence_analyzer.py   # Staged article pipeline orchestrator
    │   ├── ner_fusion.py             # spaCy + GLiNER + Flair + ReFinED + FinBERT fusion
    │   ├── asset_extractor.py        # Keyword multi-asset extraction
    │   ├── text_stats.py             # Deterministic text features
    │   ├── scoring.py                # Multi-tier validation / scoring
    │   ├── llm_cache.py              # TTL cache for deterministic LLM calls
    │   └── data/                     # Asset / narrative / contagion / entity reference data
    ├── validator/                    # Validation, grading, reputation, reward + penalty broadcasts
    │   ├── validation_client.py      # Window aggregation, weights, broadcasts, API submission
    │   ├── reputation.py             # EMA update + emission gate
    │   └── reputation_store.py       # Durable per-hotkey reputation state
    └── utils/
        ├── burn.py                   # Points -> percent -> weights, burn residual
        ├── reward.py                 # Epoch-bucketed point store
        └── penalty.py                # Epoch-bucketed penalty store
```
</details>

---

## License

MIT
