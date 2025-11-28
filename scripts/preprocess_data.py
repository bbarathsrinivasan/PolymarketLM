"""
Preprocess Polymarket CSVs into a unified JSONL for fine-tuning.

Inputs (expected under data/raw/):
- Nested directory structure with meta/, prices/, trades/ subdirectories

Outputs:
- data/fine_tune.jsonl

This script:
1) Loads raw CSVs from nested directory structure
2) Normalizes & merges by market_id
3) Produces instruction-style JSONL samples for three tasks:
   - Market outcome prediction
   - Manipulation detection
   - User classification
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def discover_market_directories(raw_data_dir):
    """Discover all market directories in the raw data folder."""
    market_dirs = []
    for item in os.listdir(raw_data_dir):
        item_path = os.path.join(raw_data_dir, item)
        if os.path.isdir(item_path):
            meta_dir = os.path.join(item_path, 'meta')
            if os.path.isdir(meta_dir):
                market_dirs.append(item_path)
    logger.info(f"Found {len(market_dirs)} market directories")
    return market_dirs


def load_market_metadata(market_dir):
    """Load market metadata from meta CSV file."""
    meta_dir = os.path.join(market_dir, 'meta')
    meta_files = glob.glob(os.path.join(meta_dir, 'meta_*.csv'))
    
    if not meta_files:
        logger.warning(f"No meta CSV found in {market_dir}")
        return None
    
    try:
        df = pd.read_csv(meta_files[0])
        return df
    except Exception as e:
        logger.error(f"Error loading meta CSV from {market_dir}: {e}")
        return None


def load_price_data(market_dir):
    """Load and aggregate all price CSV files for a market directory."""
    prices_dir = os.path.join(market_dir, 'prices')
    if not os.path.isdir(prices_dir):
        return {}
    
    price_files = glob.glob(os.path.join(prices_dir, '*_price.csv'))
    prices_by_market = defaultdict(list)
    
    for price_file in price_files:
        try:
            df = pd.read_csv(price_file)
            if len(df) == 0:
                continue
                
            if 'market_slug' in df.columns and 'price' in df.columns and 'timestamp' in df.columns:
                # Extract market_slug from the data
                market_slug = df['market_slug'].iloc[0] if len(df) > 0 else None
                if market_slug and pd.notna(market_slug):
                    # Store price records with all relevant fields
                    for _, row in df.iterrows():
                        prices_by_market[market_slug].append({
                            'timestamp': row.get('timestamp'),
                            'price': row.get('price'),
                            'market_slug': market_slug
                        })
        except Exception as e:
            logger.warning(f"Error loading price file {price_file}: {e}")
    
    return prices_by_market


def load_trade_data(market_dir):
    """Load and aggregate all trade CSV files for a market directory."""
    trades_dir = os.path.join(market_dir, 'trades')
    if not os.path.isdir(trades_dir):
        return []
    
    trade_files = glob.glob(os.path.join(trades_dir, '*_trades.csv'))
    all_trades = []
    
    for trade_file in trade_files:
        try:
            df = pd.read_csv(trade_file)
            if len(df) > 0:
                all_trades.append(df)
        except Exception as e:
            logger.warning(f"Error loading trade file {trade_file}: {e}")
    
    if all_trades:
        return pd.concat(all_trades, ignore_index=True)
    return pd.DataFrame()


def merge_market_data(meta_df, prices_dict, trades_df):
    """Merge metadata, prices, and trades by market_id and market_slug."""
    merged_data = []
    
    if meta_df is None or len(meta_df) == 0:
        return []
    
    for _, market_row in meta_df.iterrows():
        market_id = market_row.get('market_id')
        market_slug = market_row.get('market_slug')
        
        if pd.isna(market_id):
            continue
        
        # Get prices for this market (match by market_slug)
        market_prices = []
        if market_slug and market_slug in prices_dict:
            market_prices = prices_dict[market_slug]
        else:
            # Try to find prices by matching slug patterns
            for slug, prices in prices_dict.items():
                if market_slug and slug and (market_slug in slug or slug in market_slug):
                    market_prices.extend(prices)
        
        # Filter trades for this market (by slug)
        market_trades = pd.DataFrame()
        if trades_df is not None and len(trades_df) > 0:
            if 'slug' in trades_df.columns and market_slug:
                market_trades = trades_df[trades_df['slug'] == market_slug].copy()
            elif 'title' in trades_df.columns:
                # Try to match by title/question
                market_question = str(market_row.get('market_question', ''))
                if market_question and market_question != 'nan':
                    # Match by question text
                    market_trades = trades_df[
                        trades_df['title'].str.contains(market_question[:100], na=False, case=False)
                    ].copy()
        
        merged_data.append({
            'market_id': market_id,
            'market_slug': market_slug,
            'market_row': market_row,
            'prices': market_prices,
            'trades': market_trades
        })
    
    return merged_data


def derive_market_outcome(prices, threshold_high=0.95, threshold_low=0.05):
    """Derive market outcome from final price."""
    if not prices:
        return None
    
    # Sort by timestamp and get final price
    sorted_prices = sorted(prices, key=lambda x: x.get('timestamp', 0))
    if not sorted_prices:
        return None
    
    final_price = sorted_prices[-1].get('price', 0)
    
    try:
        final_price = float(final_price)
        if final_price >= threshold_high:
            return "Yes"
        elif final_price <= threshold_low:
            return "No"
        else:
            # Use majority of recent prices
            recent_prices = [p.get('price', 0) for p in sorted_prices[-10:]]
            avg_recent = np.mean([float(p) for p in recent_prices if p])
            return "Yes" if avg_recent > 0.5 else "No"
    except (ValueError, TypeError):
        return None


def format_price_history(prices, max_points=50):
    """Format price history as a readable string."""
    if not prices:
        return "No price data available"
    
    sorted_prices = sorted(prices, key=lambda x: x.get('timestamp', 0))
    
    # Sample prices if too many
    if len(sorted_prices) > max_points:
        step = len(sorted_prices) // max_points
        sorted_prices = sorted_prices[::step]
    
    formatted = []
    for p in sorted_prices:
        timestamp = p.get('timestamp', 0)
        price = p.get('price', 0)
        try:
            dt = datetime.fromtimestamp(int(timestamp))
            formatted.append(f"{dt.strftime('%Y-%m-%d %H:%M')}: {price:.4f}")
        except (ValueError, TypeError, OSError):
            formatted.append(f"Timestamp {timestamp}: {price:.4f}")
    
    return "\n".join(formatted[:max_points])


def format_trade_summary(trades_df):
    """Format trade summary statistics."""
    if trades_df is None or len(trades_df) == 0:
        return "No trades available"
    
    try:
        total_trades = len(trades_df)
        buy_trades = trades_df[trades_df['side'] == 'BUY'] if 'side' in trades_df.columns else pd.DataFrame()
        sell_trades = trades_df[trades_df['side'] == 'SELL'] if 'side' in trades_df.columns else pd.DataFrame()
        
        buy_volume = buy_trades['size'].sum() if 'size' in buy_trades.columns and len(buy_trades) > 0 else 0
        sell_volume = sell_trades['size'].sum() if 'size' in sell_trades.columns and len(sell_trades) > 0 else 0
        
        avg_price = trades_df['price'].mean() if 'price' in trades_df.columns else 0
        price_range = f"{trades_df['price'].min():.4f} - {trades_df['price'].max():.4f}" if 'price' in trades_df.columns else "N/A"
        
        return (f"Total trades: {total_trades}, "
                f"Buy volume: {buy_volume:.2f}, "
                f"Sell volume: {sell_volume:.2f}, "
                f"Average price: {avg_price:.4f}, "
                f"Price range: {price_range}")
    except Exception as e:
        logger.warning(f"Error formatting trade summary: {e}")
        return f"Total trades: {len(trades_df)}"


def create_outcome_prediction_examples(merged_markets):
    """Create training examples for market outcome prediction task."""
    examples = []
    
    for market_data in merged_markets:
        market_row = market_data['market_row']
        market_id = market_data['market_id']
        prices = market_data['prices']
        trades = market_data['trades']
        
        outcome = derive_market_outcome(prices)
        if outcome is None:
            continue
        
        market_question = market_row.get('market_question', 'N/A')
        market_outcomes = market_row.get('market_outcomes', '[]')
        volume = market_row.get('volume', 0)
        
        price_history = format_price_history(prices)
        trade_summary = format_trade_summary(trades)
        
        input_text = (
            f"Market ID: {market_id}\n"
            f"Question: {market_question}\n"
            f"Outcomes: {market_outcomes}\n"
            f"Volume: {volume}\n"
            f"Price History:\n{price_history}\n"
            f"Trade Summary: {trade_summary}"
        )
        
        examples.append({
            "instruction": "Predict the market outcome given the historical data.",
            "input": input_text,
            "output": outcome
        })
    
    logger.info(f"Created {len(examples)} outcome prediction examples")
    return examples


def detect_manipulation_indicators(prices, trades_df):
    """Detect manipulation patterns in market data."""
    indicators = []
    
    # 1. Price volatility
    if prices and len(prices) > 1:
        try:
            price_values = [float(p.get('price', 0)) for p in prices if p.get('price')]
            if len(price_values) > 1:
                price_changes = np.diff(price_values)
                volatility = np.std(price_changes) if len(price_changes) > 0 else 0
                if volatility > 0.3:  # High volatility threshold
                    indicators.append("high_volatility")
        except Exception:
            pass
    
    # 2. Unusual price spikes
    if prices and len(prices) > 2:
        try:
            sorted_prices = sorted(prices, key=lambda x: x.get('timestamp', 0))
            price_values = [float(p.get('price', 0)) for p in sorted_prices if p.get('price')]
            for i in range(1, len(price_values)):
                change = abs(price_values[i] - price_values[i-1])
                if change > 0.5:  # 50% jump
                    indicators.append("price_spike")
                    break
        except Exception:
            pass
    
    # 3. Volume anomalies
    if trades_df is not None and len(trades_df) > 0 and 'size' in trades_df.columns:
        try:
            volumes = trades_df['size'].values
            if len(volumes) > 10:
                mean_vol = np.mean(volumes)
                std_vol = np.std(volumes)
                # Check for outliers (3+ standard deviations)
                outliers = volumes[np.abs(volumes - mean_vol) > 3 * std_vol]
                if len(outliers) > len(volumes) * 0.1:  # More than 10% outliers
                    indicators.append("volume_anomaly")
        except Exception:
            pass
    
    # 4. Wash trading indicators (same user rapid buy/sell)
    if trades_df is not None and len(trades_df) > 0:
        try:
            if 'proxyWallet' in trades_df.columns and 'side' in trades_df.columns and 'timestamp' in trades_df.columns:
                # Group by user and check for rapid opposite trades
                for wallet in trades_df['proxyWallet'].unique()[:100]:  # Limit to avoid performance issues
                    user_trades = trades_df[trades_df['proxyWallet'] == wallet].sort_values('timestamp')
                    if len(user_trades) > 1:
                        sides = user_trades['side'].values
                        timestamps = user_trades['timestamp'].values
                        # Check for rapid BUY-SELL or SELL-BUY sequences
                        for i in range(len(sides) - 1):
                            if sides[i] != sides[i+1] and abs(timestamps[i+1] - timestamps[i]) < 3600:  # Within 1 hour
                                indicators.append("wash_trading")
                                break
                        if "wash_trading" in indicators:
                            break
        except Exception:
            pass
    
    return indicators


def create_manipulation_detection_examples(merged_markets):
    """Create training examples for manipulation detection task."""
    examples = []
    
    for market_data in merged_markets:
        market_row = market_data['market_row']
        market_id = market_data['market_id']
        prices = market_data['prices']
        trades = market_data['trades']
        
        indicators = detect_manipulation_indicators(prices, trades)
        is_manipulated = len(indicators) >= 2
        
        market_question = market_row.get('market_question', 'N/A')
        volume = market_row.get('volume', 0)
        
        price_history = format_price_history(prices, max_points=30)
        trade_summary = format_trade_summary(trades)
        
        input_text = (
            f"Market ID: {market_id}\n"
            f"Question: {market_question}\n"
            f"Volume: {volume}\n"
            f"Price History:\n{price_history}\n"
            f"Trade Summary: {trade_summary}\n"
            f"Manipulation Indicators Detected: {', '.join(indicators) if indicators else 'None'}"
        )
        
        examples.append({
            "instruction": "Detect if the following market experienced manipulation (Yes or No).",
            "input": input_text,
            "output": "Yes" if is_manipulated else "No"
        })
    
    logger.info(f"Created {len(examples)} manipulation detection examples")
    return examples


def calculate_user_statistics(all_trades_by_user):
    """Calculate statistics for user classification."""
    user_stats = {}
    
    for wallet, user_trades in all_trades_by_user.items():
        if len(user_trades) == 0:
            continue
        
        try:
            total_trades = len(user_trades)
            total_volume = user_trades['size'].sum() if 'size' in user_trades.columns else 0
            avg_trade_size = total_volume / total_trades if total_trades > 0 else 0
            
            # Market diversity
            unique_markets = user_trades['slug'].nunique() if 'slug' in user_trades.columns else 0
            
            # Trading frequency (trades per day)
            if 'timestamp' in user_trades.columns:
                timestamps = user_trades['timestamp'].dropna()
                if len(timestamps) > 1:
                    time_span_days = (timestamps.max() - timestamps.min()) / 86400  # Convert to days
                    trades_per_day = total_trades / max(time_span_days, 1)
                else:
                    trades_per_day = 0
            else:
                trades_per_day = 0
            
            # Profit calculation (simplified - would need actual market outcomes)
            # For now, we'll use a placeholder that will be calculated later
            user_stats[wallet] = {
                'total_trades': total_trades,
                'total_volume': total_volume,
                'avg_trade_size': avg_trade_size,
                'unique_markets': unique_markets,
                'trades_per_day': trades_per_day,
                'trades': user_trades
            }
        except Exception as e:
            logger.warning(f"Error calculating stats for user {wallet}: {e}")
            continue
    
    return user_stats


def calculate_user_profit(user_trades, market_outcomes_by_id, market_outcomes_by_slug):
    """Calculate user profit based on market outcomes."""
    total_profit = 0
    profitable_trades = 0
    total_trades = 0
    
    if 'slug' not in user_trades.columns or 'side' not in user_trades.columns:
        return 0, 0, 0
    
    for _, trade in user_trades.iterrows():
        market_slug = trade.get('slug')
        side = trade.get('side')
        size = trade.get('size', 0)
        price = trade.get('price', 0)
        outcome = trade.get('outcome', '')
        
        # Find market outcome by slug first, then by trade outcome
        market_outcome = None
        if market_slug and market_slug in market_outcomes_by_slug:
            market_outcome = market_outcomes_by_slug[market_slug]
        elif outcome and str(outcome).upper() in ['YES', 'NO']:
            # Use trade outcome as proxy (Yes/No from the trade record)
            market_outcome = str(outcome).capitalize() if str(outcome).upper() == 'YES' else 'No'
        
        # Skip trades where we can't determine outcome
        if market_outcome is None:
            continue
        
        try:
            size = float(size)
            price = float(price)
            
            if side == 'BUY':
                if market_outcome == 'Yes':
                    # Profit if outcome is Yes and price was low
                    profit = size * (1.0 - price)
                else:
                    # Loss if outcome is No
                    profit = -size * price
            else:  # SELL
                if market_outcome == 'No':
                    # Profit if outcome is No and price was high
                    profit = size * price
                else:
                    # Loss if outcome is Yes
                    profit = -size * (1.0 - price)
            
            total_profit += profit
            total_trades += 1
            if profit > 0:
                profitable_trades += 1
        except (ValueError, TypeError):
            continue
    
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    return total_profit, profitable_trades, win_rate


def classify_user(user_stats, profit, win_rate):
    """Classify user as Noise Trader or Informed Trader."""
    if profit > 0 and win_rate > 60 and user_stats['unique_markets'] >= 3:
        return "Informed Trader"
    elif profit < 0 and win_rate < 40 and user_stats['avg_trade_size'] < 100:
        return "Noise Trader"
    elif profit > 0 and win_rate > 50:
        return "Informed Trader"
    else:
        return "Noise Trader"


def create_user_classification_examples(all_trades_df, market_outcomes_by_id, market_outcomes_by_slug):
    """Create training examples for user classification task."""
    examples = []
    
    if all_trades_df is None or len(all_trades_df) == 0:
        logger.warning("No trades data available for user classification")
        return examples
    
    if 'proxyWallet' not in all_trades_df.columns:
        logger.warning("proxyWallet column not found in trades data")
        return examples
    
    # Group trades by user
    all_trades_by_user = {wallet: group for wallet, group in all_trades_df.groupby('proxyWallet')}
    
    # Limit to 500 users
    MAX_USERS = 500
    if len(all_trades_by_user) > MAX_USERS:
        logger.info(f"Limiting user classification to {MAX_USERS} users (found {len(all_trades_by_user)} total)")
        # Take first 500 users (you could also sample randomly if preferred)
        all_trades_by_user = dict(list(all_trades_by_user.items())[:MAX_USERS])
    
    # Calculate statistics
    user_stats = calculate_user_statistics(all_trades_by_user)
    
    # Calculate profits and classify
    for wallet, stats in user_stats.items():
        user_trades = stats['trades']
        profit, profitable_trades, win_rate = calculate_user_profit(
            user_trades, market_outcomes_by_id, market_outcomes_by_slug
        )
        
        classification = classify_user(stats, profit, win_rate)
        
        input_text = (
            f"User ID: {wallet}\n"
            f"Total Trades: {stats['total_trades']}\n"
            f"Total Volume: {stats['total_volume']:.2f}\n"
            f"Average Trade Size: {stats['avg_trade_size']:.2f}\n"
            f"Active Markets: {stats['unique_markets']}\n"
            f"Trades per Day: {stats['trades_per_day']:.2f}\n"
            f"Profit: {profit:.2f}\n"
            f"Win Rate: {win_rate:.1f}%"
        )
        
        examples.append({
            "instruction": "Classify the trader based on their history (Noise Trader or Informed Trader).",
            "input": input_text,
            "output": classification
        })
    
    logger.info(f"Created {len(examples)} user classification examples")
    return examples


def main():
    """Main preprocessing pipeline."""
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    raw_data_dir = project_root / 'data' / 'raw'
    output_file = project_root / 'data' / 'fine_tune.jsonl'
    
    logger.info("Starting Polymarket data preprocessing")
    logger.info(f"Raw data directory: {raw_data_dir}")
    logger.info(f"Output file: {output_file}")
    
    # Discover market directories
    market_dirs = discover_market_directories(raw_data_dir)
    
    if not market_dirs:
        logger.error("No market directories found!")
        return
    
    # Collect all data
    all_merged_markets = []
    all_trades_list = []
    market_outcomes_by_id = {}
    market_outcomes_by_slug = {}
    
    for market_dir in market_dirs:
        logger.info(f"Processing {os.path.basename(market_dir)}")
        
        # Load data
        meta_df = load_market_metadata(market_dir)
        prices_dict = load_price_data(market_dir)
        trades_df = load_trade_data(market_dir)
        
        if meta_df is not None and len(meta_df) > 0:
            merged = merge_market_data(meta_df, prices_dict, trades_df)
            all_merged_markets.extend(merged)
            
            # Store market outcomes by both ID and slug
            for market_data in merged:
                market_id = market_data['market_id']
                market_slug = market_data.get('market_slug')
                outcome = derive_market_outcome(market_data['prices'])
                if outcome:
                    market_outcomes_by_id[market_id] = outcome
                    if market_slug:
                        market_outcomes_by_slug[market_slug] = outcome
        
        if trades_df is not None and len(trades_df) > 0:
            all_trades_list.append(trades_df)
    
    # Combine all trades
    if all_trades_list:
        all_trades_df = pd.concat(all_trades_list, ignore_index=True)
    else:
        all_trades_df = pd.DataFrame()
    
    logger.info(f"Total markets processed: {len(all_merged_markets)}")
    logger.info(f"Total trades: {len(all_trades_df)}")
    
    # Create training examples
    outcome_examples = create_outcome_prediction_examples(all_merged_markets)
    manipulation_examples = create_manipulation_detection_examples(all_merged_markets)
    user_examples = create_user_classification_examples(
        all_trades_df, market_outcomes_by_id, market_outcomes_by_slug
    )
    
    # Combine all examples
    all_examples = outcome_examples + manipulation_examples + user_examples
    
    logger.info(f"Total examples created: {len(all_examples)}")
    logger.info(f"  - Outcome prediction: {len(outcome_examples)}")
    logger.info(f"  - Manipulation detection: {len(manipulation_examples)}")
    logger.info(f"  - User classification: {len(user_examples)}")
    
    # Write to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    written_count = 0
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in all_examples:
                try:
                    json_line = json.dumps(example, ensure_ascii=False)
                    f.write(json_line + '\n')
                    written_count += 1
                except Exception as e:
                    logger.error(f"Error writing example: {e}")
                    continue
        
        logger.info(f"Successfully wrote {written_count} examples to {output_file}")
        
        # Verify output file
        verify_output_file(output_file, written_count)
        
        # Print sample examples
        logger.info("\n=== Sample Examples ===")
        if len(outcome_examples) > 0:
            logger.info(f"\n[Outcome Prediction Example]")
            logger.info(f"Instruction: {outcome_examples[0]['instruction']}")
            logger.info(f"Input (first 200 chars): {outcome_examples[0]['input'][:200]}...")
            logger.info(f"Output: {outcome_examples[0]['output']}")
        
        if len(manipulation_examples) > 0:
            logger.info(f"\n[Manipulation Detection Example]")
            logger.info(f"Instruction: {manipulation_examples[0]['instruction']}")
            logger.info(f"Input (first 200 chars): {manipulation_examples[0]['input'][:200]}...")
            logger.info(f"Output: {manipulation_examples[0]['output']}")
        
        if len(user_examples) > 0:
            logger.info(f"\n[User Classification Example]")
            logger.info(f"Instruction: {user_examples[0]['instruction']}")
            logger.info(f"Input (first 200 chars): {user_examples[0]['input'][:200]}...")
            logger.info(f"Output: {user_examples[0]['output']}")
    except Exception as e:
        logger.error(f"Error writing to output file: {e}")
        raise


def verify_output_file(output_file, expected_count):
    """Verify the output JSONL file is valid."""
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            actual_count = len(lines)
            
            if actual_count != expected_count:
                logger.warning(f"Expected {expected_count} lines, found {actual_count}")
            
            # Validate JSON format
            valid_count = 0
            for i, line in enumerate(lines[:100]):  # Check first 100 lines
                try:
                    json.loads(line.strip())
                    valid_count += 1
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON on line {i+1}: {e}")
            
            if valid_count == min(100, actual_count):
                logger.info(f"Output file validation: All checked lines are valid JSON")
            else:
                logger.warning(f"Output file validation: {valid_count}/{min(100, actual_count)} lines are valid")
    except Exception as e:
        logger.error(f"Error verifying output file: {e}")


if __name__ == "__main__":
    main()
