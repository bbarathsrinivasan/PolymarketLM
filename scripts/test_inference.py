"""
Test inference with the fine-tuned LoRA adapter.

This script loads the base model with the LoRA adapter and tests it
on sample prompts from each task type.
"""

import argparse
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
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
    
    return model, tokenizer


def format_prompt(instruction, input_text=None):
    """Format prompt in Mistral Instruct format."""
    user_prompt = instruction
    if input_text:
        user_prompt += "\n" + input_text
    return f"<s>[INST] {user_prompt.strip()} [/INST]"


def generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7):
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()


def test_sample_prompts(model, tokenizer):
    """Test model on sample prompts from each task."""
    test_cases = [
        {
            "task": "Outcome Prediction",
            "instruction": "Predict the market outcome given the historical data. Answer in a short paragraph. Explain your reasoning for the answer.",
            "input": "Market ID: 12345\nQuestion: Will Candidate X win the election?\nOutcomes: [\"Yes\", \"No\"]\nVolume: 1000\nPrice History:\n2024-01-01: 0.45\n2024-01-02: 0.52\n2024-01-03: 0.58\nTrade Summary: Total trades: 50, Buy volume: 600, Sell volume: 400"
        },
        {
            "task": "Manipulation Detection",
            "instruction": "Detect if the following market experienced manipulation (Yes or No). Answer in a short paragraph. Explain your reasoning for the answer.",
            "input": "Market ID: 12345\nQuestion: Will Event Y happen?\nVolume: 5000\nPrice History:\n2024-01-01: 0.30\n2024-01-02: 0.85\n2024-01-03: 0.20\nTrade Summary: Total trades: 200, Buy volume: 4000, Sell volume: 1000\nManipulation Indicators Detected: price_spike, wash_trading"
        },
        {
            "task": "User Classification",
            "instruction": "Classify the trader based on their history (Noise Trader or Informed Trader). Answer in a short paragraph. Explain your reasoning for the answer.",
            "input": "User ID: 0xabc123\nTotal Trades: 150\nTotal Volume: 5000.5\nAverage Trade Size: 33.34\nActive Markets: 10\nTrades per Day: 5.0\nProfit: 250.3\nWin Rate: 65.0%"
        }
    ]
    
    logger.info("=" * 60)
    logger.info("Testing Model on Sample Prompts")
    logger.info("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n[Test {i}] {test_case['task']}")
        logger.info("-" * 60)
        
        prompt = format_prompt(test_case["instruction"], test_case.get("input"))
        logger.info(f"Prompt: {prompt[:200]}...")
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=128)
        logger.info(f"Response: {response}")
        logger.info("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test inference with fine-tuned LoRA adapter")
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
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--custom_prompt",
        type=str,
        default=None,
        help="Test with a custom prompt (instruction and input separated by '|')"
    )
    
    args = parser.parse_args()
    
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        logger.error(f"Adapter path not found: {adapter_path}")
        logger.error("Make sure you have trained the model first using finetune_qlora.py")
        return
    
    # Load model with adapter
    try:
        model, tokenizer = load_model_with_adapter(
            args.base_model,
            str(adapter_path),
            use_4bit=not args.no_4bit
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Test with custom prompt or sample prompts
    if args.custom_prompt:
        parts = args.custom_prompt.split("|")
        instruction = parts[0].strip()
        input_text = parts[1].strip() if len(parts) > 1 else None
        
        logger.info("Testing custom prompt...")
        prompt = format_prompt(instruction, input_text)
        logger.info(f"Prompt: {prompt}")
        
        response = generate_response(model, tokenizer, prompt)
        logger.info(f"Response: {response}")
    else:
        test_sample_prompts(model, tokenizer)
    
    logger.info("\n" + "=" * 60)
    logger.info("Inference testing completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

