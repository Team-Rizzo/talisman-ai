#!/usr/bin/env python3
"""
Track new commits and only process changes.

This demonstrates how to:
1. Detect when a validator has made a NEW commit
2. Only process/reward based on new commits
3. Avoid double-counting the same commit
"""

import json
import os

# Parse args before importing bittensor
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wallet-name", default="cold")
parser.add_argument("--wallet-hotkey", default="hot")
parser.add_argument("--network", default="test")
parser.add_argument("--netuid", type=int, default=76)
parser.add_argument("--state-file", default="commit_state.json", help="File to track seen commits")
args = parser.parse_args()

import bittensor as bt


def load_state(state_file: str) -> dict:
    """Load the last-seen commit blocks for each validator."""
    if os.path.exists(state_file):
        with open(state_file) as f:
            return json.load(f)
    return {}


def save_state(state_file: str, state: dict):
    """Save the state to disk."""
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def main():
    print("=" * 70)
    print("Commit Tracker - Detect New Commits")
    print("=" * 70)
    
    # Setup
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    subtensor = bt.Subtensor(network=args.network)
    
    print(f"\nNetwork: {subtensor.network}")
    print(f"Subnet: {args.netuid}")
    print(f"State file: {args.state_file}")
    
    # Load previous state
    state = load_state(args.state_file)
    last_seen_blocks = state.get("last_seen_blocks", {})
    print(f"Tracking {len(last_seen_blocks)} validators from previous run")
    
    # Get all commits
    all_commits = subtensor.get_all_commitments(netuid=args.netuid)
    
    if not all_commits:
        print("\nNo commits found on subnet.")
        return
    
    print(f"\nFound {len(all_commits)} total commits. Checking for new ones...\n")
    print("-" * 70)
    
    new_commits = []
    
    for hotkey, commit_data in all_commits.items():
        # Get metadata to find block number
        metadata = subtensor.get_commitment_metadata(
            netuid=args.netuid,
            hotkey_ss58=hotkey
        )
        
        if not metadata or not isinstance(metadata, dict):
            print(f"⚠️  {hotkey[:20]}... - no metadata")
            continue
        
        commit_block = metadata.get("block", 0)
        last_block = last_seen_blocks.get(hotkey, 0)
        
        short_hotkey = f"{hotkey[:16]}...{hotkey[-8:]}"
        
        if commit_block > last_block:
            # NEW COMMIT!
            print(f"🆕 {short_hotkey}")
            print(f"   Block: {commit_block} (was: {last_block})")
            print(f"   Data: {commit_data}")
            
            # Parse and collect
            try:
                parsed = json.loads(commit_data)
                new_commits.append({
                    "hotkey": hotkey,
                    "block": commit_block,
                    "data": parsed
                })
            except json.JSONDecodeError:
                print(f"   ⚠️  Could not parse JSON")
            
            # Update our tracking
            last_seen_blocks[hotkey] = commit_block
            print()
        else:
            print(f"⏭️  {short_hotkey} - no change (block {commit_block})")
    
    print("-" * 70)
    
    # Summary
    if new_commits:
        print(f"\n✅ Found {len(new_commits)} NEW commit(s)!\n")
        
        print("NEW SCORES TO PROCESS:")
        for commit in new_commits:
            print(f"\n  From: {commit['hotkey'][:20]}...")
            print(f"  Block: {commit['block']}")
            
            data = commit["data"]
            if "scores" in data:
                for miner_uid, score in data["scores"].items():
                    action = "REWARD" if score == 1 else "PENALIZE"
                    print(f"    → Miner {miner_uid}: {action}")
            elif "miner_uid" in data:
                action = "REWARD" if data["score"] == 1 else "PENALIZE"
                print(f"    → Miner {data['miner_uid']}: {action}")
    else:
        print(f"\n⏸️  No new commits since last check.")
    
    # Save updated state
    state["last_seen_blocks"] = last_seen_blocks
    save_state(args.state_file, state)
    print(f"\n💾 State saved to {args.state_file}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

