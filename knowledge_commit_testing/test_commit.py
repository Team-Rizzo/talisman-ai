#!/usr/bin/env python3
"""
Test script for Bittensor's knowledge commit functionality.

This script demonstrates:
1. Setting a commitment (miner UID + score) on the chain
2. Reading the commitment back from the chain
3. Reading all commitments for a subnet

Usage:
    python test_commit.py --wallet-name <name> --wallet-hotkey <hotkey>
    
Example:
    python test_commit.py --wallet-name validator --wallet-hotkey default --network test
"""

# IMPORTANT: Parse args BEFORE importing bittensor to avoid argparse hijacking
import argparse
import json


def parse_args():
    """Parse command line arguments before bittensor import."""
    parser = argparse.ArgumentParser(
        description="Test Bittensor commit functionality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Wallet args
    parser.add_argument("--wallet-name", type=str, default="default", help="Wallet name")
    parser.add_argument("--wallet-hotkey", type=str, default="default", help="Hotkey name")
    parser.add_argument("--wallet-path", type=str, default="~/.bittensor/wallets", help="Wallet path")
    
    # Network args
    parser.add_argument("--network", type=str, default="test", help="Network (test, finney, local)")
    
    # Custom args
    parser.add_argument("--netuid", type=int, default=76, help="Subnet UID")
    parser.add_argument("--miner-uid", type=int, default=0, help="Miner UID to commit score for")
    parser.add_argument("--score", type=int, choices=[0, 1], default=1, help="Score to commit (0 or 1)")
    parser.add_argument("--action", type=str, choices=["commit", "read", "read-all", "both"], 
                        default="both", help="Action to perform")
    
    return parser.parse_args()


def create_commitment_data(miner_uid: int, score: int) -> str:
    """
    Create a JSON-formatted commitment string containing miner UID and score.
    
    Args:
        miner_uid: The UID of the miner being scored
        score: The score (0 or 1)
    
    Returns:
        JSON string containing the commitment data
    """
    data = {
        "miner_uid": miner_uid,
        "score": score,
        "version": "1.0"
    }
    return json.dumps(data)


def parse_commitment_data(data: str) -> dict:
    """
    Parse a commitment string back into structured data.
    
    Args:
        data: JSON string from the chain
    
    Returns:
        Dictionary with miner_uid and score
    """
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"raw": data}


def main(args):
    # Import bittensor after argparse
    import bittensor as bt
    
    print("=" * 60)
    print("Bittensor Commitment Test")
    print("=" * 60)
    
    # Initialize wallet
    print(f"\n[1] Initializing wallet: {args.wallet_name}/{args.wallet_hotkey}")
    wallet = bt.Wallet(
        name=args.wallet_name,
        hotkey=args.wallet_hotkey,
        path=args.wallet_path
    )
    print(f"    Hotkey SS58: {wallet.hotkey.ss58_address}")
    
    # Initialize subtensor connection
    print(f"\n[2] Connecting to {args.network} network...")
    subtensor = bt.Subtensor(network=args.network)
    print(f"    Connected to: {subtensor.network}")
    print(f"    Chain endpoint: {subtensor.chain_endpoint}")
    
    netuid = args.netuid
    
    # Check if registered
    print(f"\n[3] Checking registration on subnet {netuid}...")
    is_registered = subtensor.is_hotkey_registered(
        netuid=netuid,
        hotkey_ss58=wallet.hotkey.ss58_address
    )
    if not is_registered:
        print(f"    ERROR: Wallet is not registered on subnet {netuid}")
        print("    Please register using: btcli subnets register")
        return
    
    # Get our UID
    uid = subtensor.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=wallet.hotkey.ss58_address,
        netuid=netuid
    )
    print(f"    Registered! UID: {uid}")
    
    action = args.action
    miner_uid = args.miner_uid
    score = args.score
    
    # Perform actions
    if action in ["commit", "both"]:
        print(f"\n[4] Setting commitment...")
        print(f"    Miner UID: {miner_uid}")
        print(f"    Score: {score}")
        
        # Create commitment data
        commitment_data = create_commitment_data(miner_uid, score)
        print(f"    Data: {commitment_data}")
        
        try:
            result = subtensor.set_commitment(
                wallet=wallet,
                netuid=netuid,
                data=commitment_data,
                wait_for_inclusion=True,
                wait_for_finalization=True
            )
            if result.success:
                print(f"    ✓ Commitment set successfully!")
                # Try to get block hash from receipt if available
                if hasattr(result, 'extrinsic_receipt') and result.extrinsic_receipt:
                    print(f"    Receipt: {result.extrinsic_receipt}")
            else:
                print(f"    ✗ Failed to set commitment: {result.error_message}")
        except Exception as e:
            print(f"    ✗ Error setting commitment: {e}")
    
    if action in ["read", "both"]:
        print(f"\n[5] Reading commitment for UID {uid}...")
        try:
            commitment = subtensor.get_commitment(
                netuid=netuid,
                uid=uid
            )
            if commitment:
                print(f"    Raw data: {commitment}")
                parsed = parse_commitment_data(commitment)
                print(f"    Parsed: {json.dumps(parsed, indent=4)}")
            else:
                print(f"    No commitment found for UID {uid}")
        except Exception as e:
            print(f"    ✗ Error reading commitment: {e}")
    
    if action in ["read-all", "both"]:
        print(f"\n[6] Reading all commitments on subnet {netuid}...")
        try:
            all_commitments = subtensor.get_all_commitments(netuid=netuid)
            if all_commitments:
                print(f"    Found {len(all_commitments)} commitment(s):")
                for hotkey, data in all_commitments.items():
                    print(f"\n    Hotkey: {hotkey[:16]}...{hotkey[-8:]}")
                    print(f"    Raw: {data}")
                    parsed = parse_commitment_data(data)
                    if "miner_uid" in parsed:
                        print(f"    Miner UID: {parsed.get('miner_uid')}, Score: {parsed.get('score')}")
            else:
                print(f"    No commitments found on subnet {netuid}")
        except Exception as e:
            print(f"    ✗ Error reading all commitments: {e}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
