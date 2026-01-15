"""
Analyzer module for subnet relevance and sentiment analysis.
"""

import json
from pathlib import Path
from typing import List
from talisman_ai import config  # Loads .miner_env and .vali_env
from .relevance import SubnetRelevanceAnalyzer
from .telegram_relevance import TelegramRelevanceAnalyzer

def create_subnet_entry(
    subnet_id: int,
    name: str,
    description: str,
    primary_functions: List[str] = [],
    unique_identifiers: List[str] = [],
    distinguishing_features: List[str] = []
):
    """Create a consistent subnet registry entry"""
    return {
        "id": subnet_id,
        "name": name,
        "description": description,
        "primary_functions": primary_functions,
        "unique_identifiers": unique_identifiers,
        "distinguishing_features": distinguishing_features
    }

def setup_analyzer(subnets_file: str = None) -> SubnetRelevanceAnalyzer:
    """
    Setup analyzer with subnets from JSON file.
    
    Args:
        subnets_file: Path to subnets.json file. If None, uses default location.
        
    Returns:
        Configured SubnetRelevanceAnalyzer instance
    """
    if subnets_file is None:
        # Default to data/subnets.json relative to this file
        analyzer_dir = Path(__file__).parent
        subnets_file = analyzer_dir / "data" / "subnets.json"
    
    analyzer = SubnetRelevanceAnalyzer()
    
    # Load subnets from JSON file
    with open(subnets_file, 'r') as f:
        subnets_data = json.load(f)
    
    # Register each subnet
    for subnet_data in subnets_data:
        analyzer.register_subnet(subnet_data)
    
    return analyzer

def setup_telegram_analyzer(subnets_file: str = None) -> TelegramRelevanceAnalyzer:
    """
    Setup telegram analyzer with subnets from JSON file.
    
    Args:
        subnets_file: Path to subnets.json file. If None, uses default location.
        
    Returns:
        Configured TelegramRelevanceAnalyzer instance
    """
    if subnets_file is None:
        # Default to data/subnets.json relative to this file
        analyzer_dir = Path(__file__).parent
        subnets_file = analyzer_dir / "data" / "subnets.json"
    
    analyzer = TelegramRelevanceAnalyzer()
    
    # Load subnets from JSON file
    with open(subnets_file, 'r') as f:
        subnets_data = json.load(f)
    
    # Register each subnet
    for subnet_data in subnets_data:
        analyzer.register_subnet(subnet_data)
    
    return analyzer
