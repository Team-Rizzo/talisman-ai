#!/usr/bin/env python3
"""
Quick interactive test for Bittensor commitment functionality.

This is a simpler script for quick testing - just modify the variables below.
"""

import json

# ============= CONFIGURATION =============
# Modify these values as needed

WALLET_NAME = "validator"      # Your wallet name
WALLET_HOTKEY = "default"      # Your hotkey name  
WALLET_PATH = "~/.bittensor/wallets"  # Path to wallets

NETWORK = "test"               # test, finney, or local
NETUID = 76                    # Subnet UID

# For testing commits
MINER_UID_TO_SCORE = 0         # The miner UID to commit a score for
SCORE = 1                      # Score: 0 or 1

# =========================================


def main():
    # Import bittensor after config to avoid argparse issues
    import bittensor as bt
    
    # Make these global so the menu can modify them
    global MINER_UID_TO_SCORE, SCORE
    
    print("\n🔗 Connecting to Bittensor...")
    
    # Setup
    wallet = bt.Wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY, path=WALLET_PATH)
    subtensor = bt.Subtensor(network=NETWORK)
    
    print(f"   Wallet: {WALLET_NAME}/{WALLET_HOTKEY}")
    print(f"   Hotkey: {wallet.hotkey.ss58_address}")
    print(f"   Network: {subtensor.network}")
    print(f"   Subnet: {NETUID}")
    
    # Check registration
    is_registered = subtensor.is_hotkey_registered(
        netuid=NETUID,
        hotkey_ss58=wallet.hotkey.ss58_address
    )
    if not is_registered:
        print(f"\n❌ Wallet not registered on subnet {NETUID}!")
        return
    
    uid = subtensor.get_uid_for_hotkey_on_subnet(
        hotkey_ss58=wallet.hotkey.ss58_address,
        netuid=NETUID
    )
    print(f"   Your UID: {uid}")
    
    # Menu
    while True:
        print("\n" + "="*50)
        print("Options:")
        print("  1. Set a commitment (miner score)")
        print("  2. Read my commitment")
        print("  3. Read all commitments")
        print("  4. Read commitment for specific UID")
        print("  5. Change miner UID/score settings")
        print("  q. Quit")
        print("="*50)
        
        choice = input("\nChoice: ").strip().lower()
        
        if choice == "1":
            # Set commitment
            data = json.dumps({
                "miner_uid": MINER_UID_TO_SCORE,
                "score": SCORE,
                "version": "1.0"
            })
            print(f"\n📝 Setting commitment: {data}")
            try:
                result = subtensor.set_commitment(
                    wallet=wallet,
                    netuid=NETUID,
                    data=data,
                    wait_for_inclusion=True,
                    wait_for_finalization=True
                )
                if result.success:
                    print(f"✅ Success! Block: {result.block_hash}")
                else:
                    print(f"❌ Failed: {result.error_message}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "2":
            # Read my commitment
            print(f"\n📖 Reading commitment for UID {uid}...")
            try:
                commitment = subtensor.get_commitment(netuid=NETUID, uid=uid)
                if commitment:
                    print(f"   Raw: {commitment}")
                    try:
                        parsed = json.loads(commitment)
                        print(f"   Parsed: {json.dumps(parsed, indent=4)}")
                    except:
                        print("   (Could not parse as JSON)")
                else:
                    print("   No commitment found")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "3":
            # Read all commitments
            print(f"\n📖 Reading all commitments on subnet {NETUID}...")
            try:
                all_commits = subtensor.get_all_commitments(netuid=NETUID)
                if all_commits:
                    print(f"   Found {len(all_commits)} commitment(s):\n")
                    for hotkey, data in all_commits.items():
                        print(f"   {hotkey[:20]}...{hotkey[-8:]}")
                        print(f"      → {data}")
                        try:
                            parsed = json.loads(data)
                            print(f"      → miner_uid={parsed.get('miner_uid')}, score={parsed.get('score')}")
                        except:
                            pass
                        print()
                else:
                    print("   No commitments found")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "4":
            # Read specific UID
            try:
                target_uid = int(input("Enter UID to read: ").strip())
                print(f"\n📖 Reading commitment for UID {target_uid}...")
                commitment = subtensor.get_commitment(netuid=NETUID, uid=target_uid)
                if commitment:
                    print(f"   Raw: {commitment}")
                    try:
                        parsed = json.loads(commitment)
                        print(f"   Parsed: {json.dumps(parsed, indent=4)}")
                    except:
                        print("   (Could not parse as JSON)")
                else:
                    print("   No commitment found")
            except ValueError:
                print("   Invalid UID")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "5":
            # Change settings
            try:
                new_uid = input(f"Enter miner UID [{MINER_UID_TO_SCORE}]: ").strip()
                if new_uid:
                    MINER_UID_TO_SCORE = int(new_uid)
                new_score = input(f"Enter score (0 or 1) [{SCORE}]: ").strip()
                if new_score:
                    SCORE = int(new_score)
                print(f"   Settings updated: miner_uid={MINER_UID_TO_SCORE}, score={SCORE}")
            except ValueError:
                print("   Invalid input")
        
        elif choice == "q":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("   Invalid choice")


if __name__ == "__main__":
    main()
