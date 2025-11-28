"""
Create a small sample dataset for quick testing of the fine-tuning pipeline.

Usage:
    python scripts/create_sample_dataset.py --input data/fine_tune.jsonl --output data/fine_tune_sample.jsonl --num_samples 50
"""

import argparse
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_dataset(input_path, output_path, num_samples, seed=42):
    """Create a sample dataset by selecting random samples."""
    import random
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    logger.info(f"Reading dataset from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        all_examples = [json.loads(line) for line in f]
    
    logger.info(f"Total examples: {len(all_examples)}")
    
    # Sample random examples
    random.seed(seed)
    if num_samples >= len(all_examples):
        logger.warning(f"Requested {num_samples} samples but only {len(all_examples)} available. Using all examples.")
        sampled = all_examples
    else:
        sampled = random.sample(all_examples, num_samples)
    
    # Write sample dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {len(sampled)} samples to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in sampled:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Sample dataset created successfully!")


def main():
    parser = argparse.ArgumentParser(description="Create a sample dataset for testing")
    parser.add_argument(
        "--input",
        type=str,
        default="data/fine_tune.jsonl",
        help="Input JSONL dataset path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/fine_tune_sample.jsonl",
        help="Output sample dataset path"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to include"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    
    args = parser.parse_args()
    create_sample_dataset(args.input, args.output, args.num_samples, args.seed)


if __name__ == "__main__":
    main()

