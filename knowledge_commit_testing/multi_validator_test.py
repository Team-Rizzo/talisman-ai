#!/usr/bin/env python3
"""
Multi-validator commit/read test for Bittensor.

This demonstrates the flow where:
1. Multiple validators commit scores for miners
2. Other validators read all commits
3. Scores are aggregated to calculate consensus weights

Since we only have one validator wallet, this script:
- Commits scores for multiple miners in one JSON blob
- Shows how to read and parse all validator commits
- Demonstrates weight calculation from aggregated scores
"""

import argparse
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-validator commit test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--wallet-name", type=str, default="cold", help="Wallet name")
    parser.add_argument("--wallet-hotkey", type=str, default="hot", help="Hotkey name")
    parser.add_argument("--wallet-path", type=str, default="~/.bittensor/wallets", help="Wallet path")
    parser.add_argument("--network", type=str, default="test", help="Network")
    parser.add_argument("--netuid", type=int, default=76, help="Subnet UID")
    parser.add_argument("--action", type=str, choices=["commit", "read", "aggregate"], 
                        default="read", help="Action to perform")
    return parser.parse_args()


def create_multi_score_commitment(scores: dict) -> str:
    """
    Create a commitment with multiple miner scores.
    
    Args:
        scores: Dict of {miner_uid: score} where score is 0 or 1
    
    Returns:
        JSON string for commitment
    """
    data = {
        "scores": scores,  # {miner_uid: score, ...}
        "version": "1.0"
    }
    return json.dumps(data)


def parse_commitment(data: str) -> dict:
    """Parse a commitment string."""
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"raw": data}


def aggregate_scores(all_commitments: dict) -> dict:
    """
    Aggregate scores from all validator commits.
    
    Args:
        all_commitments: Dict of {validator_hotkey: commitment_data}
    
    Returns:
        Dict with aggregated scores per miner
    """
    # Track votes per miner: {miner_uid: {"yes": count, "no": count, "validators": [...]}}
    miner_votes = defaultdict(lambda: {"yes": 0, "no": 0, "validators": []})
    
    for validator_hotkey, commitment_data in all_commitments.items():
        parsed = parse_commitment(commitment_data)
        
        # Handle both formats: single score and multi-score
        if "scores" in parsed:
            # Multi-score format: {"scores": {uid: score, ...}}
            for miner_uid, score in parsed["scores"].items():
                miner_uid = int(miner_uid)  # JSON keys are strings
                if score == 1:
                    miner_votes[miner_uid]["yes"] += 1
                else:
                    miner_votes[miner_uid]["no"] += 1
                miner_votes[miner_uid]["validators"].append(validator_hotkey[:16])
        elif "miner_uid" in parsed:
            # Single score format: {"miner_uid": uid, "score": score}
            miner_uid = parsed["miner_uid"]
            score = parsed["score"]
            if score == 1:
                miner_votes[miner_uid]["yes"] += 1
            else:
                miner_votes[miner_uid]["no"] += 1
            miner_votes[miner_uid]["validators"].append(validator_hotkey[:16])
    
    return dict(miner_votes)


def calculate_weights(miner_votes: dict, total_validators: int) -> dict:
    """
    Calculate weights based on aggregated votes.
    
    A miner gets weight proportional to yes votes / total validators who voted.
    
    Args:
        miner_votes: Output from aggregate_scores
        total_validators: Total number of validators on subnet
    
    Returns:
        Dict of {miner_uid: weight}
    """
    weights = {}
    for miner_uid, votes in miner_votes.items():
        total_votes = votes["yes"] + votes["no"]
        if total_votes > 0:
            # Weight = yes_votes / total_votes (validators who scored this miner)
            weights[miner_uid] = votes["yes"] / total_votes
        else:
            weights[miner_uid] = 0.0
    return weights


def main():
    args = parse_args()
    
    import bittensor as bt
    
    print("=" * 70)
    print("Multi-Validator Commit Test")
    print("=" * 70)
    
    # Setup
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey, path=args.wallet_path)
    subtensor = bt.Subtensor(network=args.network)
    
    print(f"\nWallet: {args.wallet_name}/{args.wallet_hotkey}")
    print(f"Hotkey: {wallet.hotkey.ss58_address}")
    print(f"Network: {subtensor.network}")
    print(f"Subnet: {args.netuid}")
    
    # Check registration
    is_registered = subtensor.is_hotkey_registered(
        netuid=args.netuid,
        hotkey_ss58=wallet.hotkey.ss58_address
    )
    if not is_registered:
        print(f"\n❌ Not registered on subnet {args.netuid}")
        return
    
    my_uid = subtensor.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=wallet.hotkey.ss58_address,
        netuid=args.netuid
    )
    print(f"Your UID: {my_uid}")
    
    # Get metagraph for validator info
    metagraph = subtensor.metagraph(args.netuid)
    total_validators = sum(1 for vp in metagraph.validator_permit if vp)
    total_neurons = metagraph.n
    print(f"Total neurons: {total_neurons}")
    print(f"Validators with permit: {total_validators}")
    
    if args.action == "commit":
        print("\n" + "-" * 70)
        print("COMMITTING SCORES")
        print("-" * 70)
        
        # Example: Score multiple miners
        # In real usage, these would be miners you've validated
        scores = {
            0: 1,   # Miner UID 0: good (score=1)
            1: 1,   # Miner UID 1: good
            2: 0,   # Miner UID 2: bad (score=0)
            3: 1,   # Miner UID 3: good
        }
        
        print(f"\nScores to commit:")
        for uid, score in scores.items():
            status = "✓ GOOD" if score == 1 else "✗ BAD"
            print(f"  Miner UID {uid}: {status}")
        
        commitment_data = create_multi_score_commitment(scores)
        print(f"\nCommitment JSON: {commitment_data}")
        
        try:
            result = subtensor.set_commitment(
                wallet=wallet,
                netuid=args.netuid,
                data=commitment_data,
                wait_for_inclusion=True,
                wait_for_finalization=True
            )
            if result.success:
                print(f"\n✅ Commitment set successfully!")
            else:
                print(f"\n❌ Failed: {result.error_message}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    elif args.action == "read":
        print("\n" + "-" * 70)
        print("READING ALL VALIDATOR COMMITS")
        print("-" * 70)
        
        try:
            all_commits = subtensor.get_all_commitments(netuid=args.netuid)
            
            if not all_commits:
                print("\nNo commitments found on subnet.")
                return
            
            print(f"\nFound {len(all_commits)} validator commitment(s):\n")
            
            for hotkey, data in all_commits.items():
                # Check if this is our commit
                is_me = " (YOU)" if hotkey == wallet.hotkey.ss58_address else ""
                print(f"Validator: {hotkey[:20]}...{hotkey[-8:]}{is_me}")
                print(f"  Raw: {data}")
                
                parsed = parse_commitment(data)
                if "scores" in parsed:
                    print(f"  Scores:")
                    for uid, score in parsed["scores"].items():
                        status = "✓" if score == 1 else "✗"
                        print(f"    Miner {uid}: {status}")
                elif "miner_uid" in parsed:
                    status = "✓" if parsed["score"] == 1 else "✗"
                    print(f"  Miner {parsed['miner_uid']}: {status}")
                print()
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif args.action == "aggregate":
        print("\n" + "-" * 70)
        print("AGGREGATING SCORES & CALCULATING WEIGHTS")
        print("-" * 70)
        
        try:
            all_commits = subtensor.get_all_commitments(netuid=args.netuid)
            
            if not all_commits:
                print("\nNo commitments found.")
                return
            
            print(f"\nProcessing {len(all_commits)} validator commitment(s)...")
            
            # Aggregate votes
            miner_votes = aggregate_scores(all_commits)
            
            if not miner_votes:
                print("No parseable scores found in commitments.")
                return
            
            print(f"\n{'='*70}")
            print("AGGREGATED VOTES BY MINER")
            print(f"{'='*70}")
            
            for miner_uid in sorted(miner_votes.keys()):
                votes = miner_votes[miner_uid]
                total = votes["yes"] + votes["no"]
                print(f"\nMiner UID {miner_uid}:")
                print(f"  YES votes: {votes['yes']}")
                print(f"  NO votes:  {votes['no']}")
                print(f"  Total validators scored: {total}")
                print(f"  Validators: {', '.join(votes['validators'])}")
            
            # Calculate weights
            weights = calculate_weights(miner_votes, len(all_commits))
            
            print(f"\n{'='*70}")
            print("CALCULATED WEIGHTS (for consensus)")
            print(f"{'='*70}")
            
            for miner_uid in sorted(weights.keys()):
                weight = weights[miner_uid]
                bar = "█" * int(weight * 20) + "░" * (20 - int(weight * 20))
                print(f"  Miner {miner_uid}: {weight:.2%} [{bar}]")
            
            print(f"\n{'='*70}")
            print("WEIGHT ARRAY (ready for set_weights)")
            print(f"{'='*70}")
            
            # Create weight array for set_weights
            import numpy as np
            weight_array = np.zeros(total_neurons, dtype=np.float32)
            for miner_uid, weight in weights.items():
                if miner_uid < total_neurons:
                    weight_array[miner_uid] = weight
            
            # Show non-zero weights
            nonzero = [(i, w) for i, w in enumerate(weight_array) if w > 0]
            print(f"  UIDs with weight: {[uid for uid, _ in nonzero]}")
            print(f"  Weights: {[f'{w:.2f}' for _, w in nonzero]}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    main()

