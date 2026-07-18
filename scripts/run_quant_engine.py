#!/usr/bin/env python3
"""Run the Quant Analysis & Execution Engine.

This script loads the stock universe from config.yaml and runs the complete
Phase-based screening system.

Usage:
    python run_quant_engine.py
    python run_quant_engine.py --tickers AAPL MSFT GOOGL
    python run_quant_engine.py --config custom_config.yaml
"""

import argparse
import logging
import sys
from datetime import datetime
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml

from src.screening.quant_engine import QuantAnalysisEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Config dict
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def save_results(report: str, output_dir: str = './data/results'):
    """Save screening results to file.

    Args:
        report: Report string
        output_dir: Output directory
    """
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"quant_screen_{timestamp}.txt"
        filepath = Path(output_dir) / filename

        # Save report
        with open(filepath, 'w') as f:
            f.write(report)

        logger.info(f"Results saved to {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None


def save_json_result(payload: dict, output_dir: str = './data/results'):
    """Save structured report payload as JSON.

    Args:
        payload: Report payload
        output_dir: Output directory
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"quant_screen_{timestamp}.json"
        filepath = Path(output_dir) / filename

        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2)

        latest_path = Path(output_dir) / "latest_report.json"
        with open(latest_path, 'w') as f:
            json.dump(payload, f, indent=2)

        logger.info(f"Structured results saved to {filepath} and {latest_path}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving structured results: {e}")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run Quant Analysis & Execution Engine'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Override config with specific tickers'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./data/cache',
        help='Cache directory (default: ./data/cache)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get stock universe
    if args.tickers:
        tickers = args.tickers
        logger.info(f"Using command-line tickers: {tickers}")
    elif config and 'stock_universe' in config:
        tickers = config['stock_universe']
        logger.info(f"Using {len(tickers)} tickers from config")
    else:
        logger.error("No stock universe specified")
        sys.exit(1)

    # Initialize engine
    risk_policy = config.get("risk_control", {}) if isinstance(config, dict) else {}
    holdings = config.get("holdings", []) if isinstance(config, dict) else []
    params = config.get("parameters", {}) if isinstance(config, dict) else {}
    engine = QuantAnalysisEngine(
        cache_dir=args.cache_dir,
        risk_policy=risk_policy,
        min_buy_score=params.get("min_buy_score", 70),
        min_sell_score=params.get("min_sell_score", 60),
        min_phase2_pct=params.get("min_phase2_pct", 15.0),
        holdings=holdings,
    )

    # Run screening
    try:
        logger.info("Starting Quant Analysis Engine...")
        report, payload = engine.run_report(tickers)

        # Print report
        print(report)

        # Save to file if requested
        if not args.no_save:
            output_dir = config.get('output', {}).get('output_dir', './data/results')
            save_results(report, output_dir)
            save_json_result(payload, output_dir)

    except KeyboardInterrupt:
        logger.info("\nScreening interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running engine: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
