"""
Evaluate fine-tuned Gemma model on test set.

This script loads the fine-tuned Gemma model (with LoRA adapter) and evaluates it.
Outputs results to JSON file for report generation.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_model_with_adapter(base_model_name: str, adapter_path: str, use_4bit: bool = True):
    """Load base model and apply LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    
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
    
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def format_prompt(instruction: str, input_text: str = None) -> str:
    """Format prompt in Gemma-IT format."""
    user_prompt = instruction
    if input_text:
        user_prompt += "\n" + input_text
    
    formatted = (
        f"<start_of_turn>user\n"
        f"{user_prompt.strip()}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return formatted


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 0.0):
    """Generate response from model."""
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
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()


def extract_prediction(response: str, task_type: str) -> str:
    """Extract prediction from model response."""
    response_lower = response.lower()
    
    if task_type in ["outcome_prediction", "manipulation_detection"]:
        m = re.search(r'\b(yes|no)\b', response_lower)
        if m:
            return m.group(1).capitalize()
        return response.strip().split()[0].capitalize() if response.strip() else ""
    
    elif task_type == "user_classification":
        if "informed trader" in response_lower:
            return "Informed Trader"
        elif "noise trader" in response_lower:
            return "Noise Trader"
        return ""
    
    return ""


def infer_task_type(instruction: str) -> str:
    """Infer task type from instruction."""
    instruction_lower = instruction.lower()
    if "predict the market outcome" in instruction_lower or "outcome" in instruction_lower:
        return "outcome_prediction"
    elif "manipulation" in instruction_lower:
        return "manipulation_detection"
    elif "classify the trader" in instruction_lower or "noise trader" in instruction_lower:
        return "user_classification"
    return "unknown"


def calculate_loss(model, tokenizer, instruction: str, input_text: str, target_text: str):
    """Calculate loss for a prompt-target pair."""
    if hasattr(model, "base_model"):
        base_model = model.base_model.model if hasattr(model.base_model, "model") else model.base_model
    else:
        base_model = model
    
    if hasattr(base_model, "hf_device_map"):
        device = next(base_model.parameters()).device
    else:
        device = next(base_model.parameters()).device
    
    user_prompt = instruction
    if input_text:
        user_prompt += "\n" + input_text
    
    full_text = (
        "<bos><start_of_turn>user\n"
        f"{user_prompt.strip()}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{target_text.strip()}\n"
        "<end_of_turn>"
    )
    
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        return loss
    except Exception:
        return None


def evaluate_model(model, tokenizer, test_dataset):
    """Evaluate model on test set."""
    results = []
    losses = []
    task_results = defaultdict(list)
    
    for ex in tqdm(test_dataset, desc="Evaluating"):
        instruction = ex["instruction"]
        input_text = ex.get("input", "")
        ground_truth = ex["output"]
        task_type = infer_task_type(instruction)
        
        # Format prompt
        prompt = format_prompt(instruction, input_text)
        
        # Calculate loss
        loss = calculate_loss(model, tokenizer, instruction, input_text, ground_truth)
        if loss is not None:
            losses.append(loss)
        
        # Generate response
        try:
            response = generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0)
            prediction = extract_prediction(response, task_type)
        except Exception as e:
            print(f"Error generating response: {e}")
            response = ""
            prediction = ""
        
        # Check correctness
        correct = (prediction.strip() == ground_truth.strip())
        
        result = {
            "instruction": instruction[:100],
            "input": input_text[:200] if input_text else "",
            "expected": ground_truth,
            "prediction": prediction,
            "response": response,
            "correct": correct,
            "task": task_type,
            "loss": loss
        }
        results.append(result)
        task_results[task_type].append(result)
    
    return results, losses, task_results


def calculate_metrics(results, losses, task_results):
    """Calculate accuracy metrics and perplexity."""
    import math
    
    metrics = {
        "overall": {
            "total": len(results),
            "correct": sum(r["correct"] for r in results),
            "accuracy": 0.0,
            "loss": None,
            "perplexity": None
        },
        "by_task": {}
    }
    
    # Overall accuracy
    if metrics["overall"]["total"] > 0:
        metrics["overall"]["accuracy"] = metrics["overall"]["correct"] / metrics["overall"]["total"]
    
    # Calculate overall perplexity
    valid_losses = [l for l in losses if l is not None]
    if valid_losses:
        avg_loss = sum(valid_losses) / len(valid_losses)
        metrics["overall"]["loss"] = avg_loss
        try:
            metrics["overall"]["perplexity"] = math.exp(avg_loss)
        except OverflowError:
            metrics["overall"]["perplexity"] = float("inf")
    
    # Per-task metrics
    for task, task_res in task_results.items():
        total = len(task_res)
        correct = sum(r["correct"] for r in task_res)
        accuracy = correct / total if total > 0 else 0.0
        
        task_loss_list = [r.get("loss") for r in task_res if r.get("loss") is not None]
        task_loss = None
        task_perplexity = None
        if task_loss_list:
            task_loss = sum(task_loss_list) / len(task_loss_list)
            try:
                task_perplexity = math.exp(task_loss)
            except OverflowError:
                task_perplexity = float("inf")
        
        metrics["by_task"][task] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "loss": task_loss,
            "perplexity": task_perplexity
        }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Gemma model")
    parser.add_argument("--base_model", type=str, default="google/gemma-7b-it")
    parser.add_argument("--adapter_path", type=str, default="models/checkpoints/Polymarket-Gemma-7B-LoRA")
    parser.add_argument("--dataset_path", type=str, default="data/fine_tune.jsonl")
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="report_generation/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test samples to evaluate (for faster testing)")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load test dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    if args.test_split > 0:
        dataset = dataset.train_test_split(test_size=args.test_split, seed=args.seed)
        test_dataset = dataset["test"]
    else:
        test_dataset = dataset
    
    # Limit test samples if specified
    if args.max_samples is not None and args.max_samples > 0:
        if len(test_dataset) > args.max_samples:
            print(f"Limiting evaluation to {args.max_samples} samples (from {len(test_dataset)})")
            test_dataset = test_dataset.select(range(args.max_samples))
    
    print(f"\n{'='*60}")
    print(f"Evaluating Fine-tuned Gemma Model")
    print(f"{'='*60}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"{'='*60}\n")
    
    # Load model
    model, tokenizer = load_model_with_adapter(args.base_model, args.adapter_path, use_4bit=not args.no_4bit)
    
    # Evaluate
    results, losses, task_results = evaluate_model(model, tokenizer, test_dataset)
    metrics = calculate_metrics(results, losses, task_results)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: FINE-TUNED GEMMA")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%} "
          f"({metrics['overall']['correct']}/{metrics['overall']['total']})")
    if metrics['overall']['perplexity'] is not None:
        print(f"Perplexity: {metrics['overall']['perplexity']:.2f}")
    print(f"\nPer-Task Metrics:")
    for task, task_metrics in metrics['by_task'].items():
        print(f"  {task}:")
        print(f"    Accuracy: {task_metrics['accuracy']:.2%} ({task_metrics['correct']}/{task_metrics['total']})")
        if task_metrics['perplexity'] is not None:
            print(f"    Perplexity: {task_metrics['perplexity']:.2f}")
    print(f"{'='*60}\n")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "finetuned_gemma.json"
    with open(output_file, 'w') as f:
        json.dump({
            "base_model": args.base_model,
            "adapter_path": args.adapter_path,
            "seed": args.seed,
            "metrics": metrics,
            "results": results[:10]  # Save first 10 examples for error analysis
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

