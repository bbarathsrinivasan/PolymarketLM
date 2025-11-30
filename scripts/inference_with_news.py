"""
Inference script with RAG-style news augmentation.

Loads the fine-tuned model and generates responses with and without news context.
"""

import argparse
import torch
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
import logging

from integration.prompt_augmenter import augment_prompt_with_news
from integration.config import load_config, get_rss_feeds, get_num_articles, get_cache_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_with_adapter(base_model_name, adapter_path, use_4bit=True):
    """Load base model and apply LoRA adapter."""
    logger.info(f"Loading base model: {base_model_name}")
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def format_prompt(instruction, input_text=None):
    """Format prompt in Mistral Instruct format."""
    user_prompt = instruction
    if input_text:
        user_prompt += "\n" + input_text
    return f"<s>[INST] {user_prompt.strip()} [/INST]"


def generate_response(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7):
    """Generate response from model."""
    # Determine device
    if hasattr(model, "base_model"):
        base_model = model.base_model.model if hasattr(model.base_model, "model") else model.base_model
    else:
        base_model = model
    
    if hasattr(base_model, "hf_device_map"):
        device = next(base_model.parameters()).device
    else:
        device = next(base_model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt")
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with RAG-style news augmentation"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model name"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="models/checkpoints/Polymarket-7B-LoRA",
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Predict the market outcome given the historical data and relevant news. Explain your reasoning, especially how the news articles inform your prediction.",
        help="Instruction for the task"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input text with market data (or use --market_question and other flags)"
    )
    parser.add_argument(
        "--market_question",
        type=str,
        default=None,
        help="Market question (e.g., 'Will Bitcoin reach $100k by 2024?')"
    )
    parser.add_argument(
        "--market_id",
        type=str,
        default=None,
        help="Market ID"
    )
    parser.add_argument(
        "--price_history",
        type=str,
        default=None,
        help="Price history (formatted text)"
    )
    parser.add_argument(
        "--num_news",
        type=int,
        default=None,
        help="Number of news articles to include (overrides config)"
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--no_news",
        action="store_true",
        help="Skip news augmentation (baseline comparison)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="integration/news_config.yaml",
        help="Path to news config file"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )
    
    args = parser.parse_args()
    
    # Validate adapter path
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        logger.error(f"Adapter path not found: {adapter_path}")
        logger.error("Make sure you have trained the model first using finetune_qlora.py")
        return
    
    # Load config
    config = load_config(args.config)
    feed_urls = get_rss_feeds(config)
    num_articles = args.num_news or get_num_articles(config)
    cache_dir = get_cache_dir(config)
    
    # Prepare input text
    if args.input:
        input_text = args.input
    else:
        # Build input from components
        input_parts = []
        if args.market_id:
            input_parts.append(f"Market ID: {args.market_id}")
        if args.market_question:
            input_parts.append(f"Question: {args.market_question}")
        if args.price_history:
            input_parts.append(f"Price History:\n{args.price_history}")
        
        if not input_parts:
            logger.error("Must provide either --input or --market_question")
            return
        
        input_text = "\n".join(input_parts)
    
    # Load model
    logger.info("=" * 80)
    logger.info("Loading Model")
    logger.info("=" * 80)
    try:
        model, tokenizer = load_model_with_adapter(
            args.base_model,
            str(adapter_path),
            use_4bit=not args.no_4bit
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Generate baseline response (without news)
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE RESPONSE (Without News)")
    logger.info("=" * 80)
    baseline_prompt = format_prompt(args.instruction, input_text)
    logger.info(f"Prompt: {baseline_prompt[:300]}...")
    
    baseline_response = generate_response(
        model, tokenizer, baseline_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    logger.info(f"\nResponse: {baseline_response}")
    
    # Generate news-augmented response
    if not args.no_news:
        logger.info("\n" + "=" * 80)
        logger.info("NEWS-AUGMENTED RESPONSE")
        logger.info("=" * 80)
        
        # Augment prompt with news
        logger.info(f"Retrieving {num_articles} relevant news articles...")
        augmented_input, articles = augment_prompt_with_news(
            args.instruction,
            input_text,
            feed_urls,
            num_articles=num_articles,
            cache_dir=cache_dir
        )
        
        if articles:
            logger.info(f"\nRetrieved {len(articles)} relevant articles:")
            for i, article in enumerate(articles, 1):
                logger.info(f"  {i}. {article.get('title', 'Untitled')} - {article.get('source', 'Unknown')}")
        else:
            logger.info("No relevant articles found, using original prompt")
            augmented_input = input_text
        
        # Generate response with news
        augmented_prompt = format_prompt(args.instruction, augmented_input)
        logger.info(f"\nAugmented Prompt: {augmented_prompt[:500]}...")
        
        news_response = generate_response(
            model, tokenizer, augmented_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        logger.info(f"\nResponse: {news_response}")
        
        # Comparison
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON")
        logger.info("=" * 80)
        logger.info(f"Baseline Response: {baseline_response}")
        logger.info(f"News-Augmented Response: {news_response}")
    else:
        logger.info("\nSkipping news augmentation (--no_news flag set)")
    
    logger.info("\n" + "=" * 80)
    logger.info("Inference completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

