"""
Compare base model vs fine-tuned (LoRA adapter) model on test dataset for Llama 7B.

This script uses the adapter approach:
- Loads base model once (14GB)
- Loads fine-tuned model by applying LoRA adapter to base model (200MB adapter)
- Evaluates both on the same test dataset
- Calculates accuracy, per-task metrics, and improvement
- Provides detailed comparison report
"""

import argparse
import json
import torch
import sys
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
import logging
from collections import defaultdict
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_base_model(model_name, use_4bit=True):
    """Load base model in FP16 or 4-bit for consistent comparison."""
    logger.info(f"Loading base model: {model_name}")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        if device == "cuda" and use_4bit:
            # Use 4-bit quantization for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Loaded base model with 4-bit quantization")
        else:
            # Load in FP16
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            if device == "cuda" and not hasattr(model, "hf_device_map"):
                model = model.to(device)
            logger.info(f"Loaded base model in FP16/FP32")
        
        model.eval()
        
        # Verify model is on correct device
        if device == "cuda":
            if hasattr(model, "hf_device_map"):
                logger.info(f"Base model uses device_map with devices: {model.hf_device_map}")
            else:
                first_param_device = next(model.parameters()).device
                logger.info(f"Base model loaded on device: {first_param_device}")
        
    except Exception as e:
        logger.error(f"Error loading base model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def load_model_with_adapter(base_model, adapter_path, base_model_name):
    """Load fine-tuned model by applying LoRA adapter to base model."""
    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    
    try:
        # Apply adapter to the base model
        finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
        finetuned_model.eval()
        logger.info("LoRA adapter loaded successfully")
        
        # Verify adapter is applied
        if hasattr(finetuned_model, "peft_config"):
            logger.info(f"Adapter config: {list(finetuned_model.peft_config.keys())}")
        
    except Exception as e:
        logger.error(f"Error loading adapter: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Tokenizer is the same as base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return finetuned_model, tokenizer


def format_prompt(instruction, input_text=None):
    """Format prompt in Llama 2 Chat format."""
    SYSTEM_PROMPT = "You are a helpful AI assistant."
    
    user_prompt = instruction
    if input_text:
        user_prompt += "\n" + input_text
    
    # Llama 2 Chat format: <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]
    formatted = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_prompt.strip()} [/INST]"
    )
    return formatted


def generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.1):
    """Generate response from model with low temperature for more deterministic outputs."""
    # Determine device - handle both single device and device_map cases
    # For PeftModel, get device from base model
    if hasattr(model, "base_model"):
        base_model = model.base_model.model if hasattr(model.base_model, "model") else model.base_model
    else:
        base_model = model
    
    if hasattr(base_model, "hf_device_map"):
        # Model uses device_map, inputs will be automatically placed
        device = next(base_model.parameters()).device
    else:
        device = next(base_model.parameters()).device
    
    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt")
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"CUDA OOM during generation: {e}")
            raise
        else:
            logger.warning(f"Generation error: {e}, returning empty response")
            import traceback
            logger.debug(traceback.format_exc())
            return ""
    except Exception as e:
        logger.warning(f"Unexpected error during generation: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return ""


def parse_response(response, task_type):
    """
    Parse model response to extract prediction/classification.
    
    Handles different task types:
    - Outcome Prediction: Extract "Yes" or "No"
    - Manipulation Detection: Extract "Yes" or "No"
    - User Classification: Extract "Noise Trader" or "Informed Trader"
    """
    response_lower = response.lower()
    
    # For outcome prediction and manipulation detection
    if task_type in ["outcome_prediction", "manipulation_detection"]:
        # Look for explicit Yes/No
        if re.search(r'\byes\b', response_lower):
            return "Yes"
        elif re.search(r'\bno\b', response_lower):
            return "No"
        # Try to extract from common patterns
        if "will resolve to yes" in response_lower or "outcome is yes" in response_lower:
            return "Yes"
        if "will resolve to no" in response_lower or "outcome is no" in response_lower:
            return "No"
        if "manipulated" in response_lower or "manipulation" in response_lower:
            if "no manipulation" in response_lower or "not manipulated" in response_lower:
                return "No"
            return "Yes"
        # Default based on first word
        first_word = response.split()[0].lower() if response.split() else ""
        if first_word in ["yes", "no"]:
            return first_word.capitalize()
        return None
    
    # For user classification
    elif task_type == "user_classification":
        if "informed trader" in response_lower:
            return "Informed Trader"
        elif "noise trader" in response_lower:
            return "Noise Trader"
        # Try to extract from patterns
        if "informed" in response_lower and "noise" not in response_lower:
            return "Informed Trader"
        if "noise" in response_lower and "informed" not in response_lower:
            return "Noise Trader"
        return None
    
    return None


def infer_task_type(instruction):
    """Infer task type from instruction text."""
    instruction_lower = instruction.lower()
    
    if "predict the market outcome" in instruction_lower or "outcome" in instruction_lower:
        return "outcome_prediction"
    elif "manipulation" in instruction_lower or "manipulated" in instruction_lower:
        return "manipulation_detection"
    elif "classify the trader" in instruction_lower or "noise trader" in instruction_lower or "informed trader" in instruction_lower:
        return "user_classification"
    
    return "unknown"


def calculate_loss(model, tokenizer, instruction, input_text, target_text):
    """Calculate loss for a prompt-target pair using Llama format."""
    # Determine device
    if hasattr(model, "base_model"):
        base_model = model.base_model.model if hasattr(model.base_model, "model") else model.base_model
    else:
        base_model = model
    
    if hasattr(base_model, "hf_device_map"):
        device = next(base_model.parameters()).device
    else:
        device = next(base_model.parameters()).device
    
    # Format full text in Llama training format: <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST] response </s>
    SYSTEM_PROMPT = "You are a helpful AI assistant."
    user_prompt = instruction
    if input_text:
        user_prompt += "\n" + input_text
    
    full_text = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_prompt.strip()} [/INST] {target_text.strip()} </s>"
    )
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Calculate loss
    try:
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        return loss
    except Exception as e:
        logger.debug(f"Error calculating loss: {e}")
        return None


def evaluate_model(model, tokenizer, test_dataset, max_samples=None):
    """
    Run evaluation on test set.
    
    Returns list of results with:
    - correct: bool
    - response: str
    - task: str
    - ground_truth: str
    - prediction: str
    - loss: float (for perplexity calculation)
    """
    results = []
    losses = []
    
    dataset_size = len(test_dataset)
    if max_samples:
        dataset_size = min(max_samples, dataset_size)
    
    logger.info(f"Evaluating model on {dataset_size} examples...")
    
    for i, example in enumerate(test_dataset):
        if max_samples and i >= max_samples:
            break
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{dataset_size} examples...", flush=True)
            sys.stdout.flush()
        
        instruction = example["instruction"]
        input_text = example.get("input", "")
        ground_truth = example["output"]
        
        # Infer task type
        task_type = infer_task_type(instruction)
        
        # Format prompt
        prompt = format_prompt(instruction, input_text)
        
        # Calculate loss for perplexity (using training format)
        loss = calculate_loss(model, tokenizer, instruction, input_text, ground_truth)
        if loss is not None:
            losses.append(loss)
        
        # Generate response
        try:
            response = generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.1)
            if not response:
                logger.debug(f"Empty response for example {i}")
        except Exception as e:
            logger.warning(f"Error generating response for example {i}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            response = ""
        
        # Parse prediction
        prediction = parse_response(response, task_type)
        
        # Check correctness
        correct = (prediction is not None and prediction.strip() == ground_truth.strip())
        
        results.append({
            "correct": correct,
            "response": response,
            "task": task_type,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "instruction": instruction[:100],  # First 100 chars for reference
            "loss": loss
        })
    
    return results, losses


def calculate_metrics(results, losses):
    """Calculate accuracy metrics and perplexity per task and overall."""
    import math
    
    metrics = {
        "overall": {
            "total": len(results),
            "correct": sum(r["correct"] for r in results),
            "accuracy": 0.0,
            "loss": 0.0,
            "perplexity": 0.0
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
    task_results = defaultdict(list)
    task_losses = defaultdict(list)
    
    for r in results:
        task_results[r["task"]].append(r)
        if r.get("loss") is not None:
            task_losses[r["task"]].append(r["loss"])
    
    for task, task_res in task_results.items():
        total = len(task_res)
        correct = sum(r["correct"] for r in task_res)
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate perplexity for this task
        task_loss_list = task_losses.get(task, [])
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


def print_comparison_report(base_results, finetuned_results, base_metrics, finetuned_metrics):
    """Print detailed comparison report."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON REPORT")
    logger.info("=" * 80)
    
    # Overall comparison
    logger.info("\n📊 OVERALL PERFORMANCE")
    logger.info("-" * 80)
    base_acc = base_metrics["overall"]["accuracy"]
    finetuned_acc = finetuned_metrics["overall"]["accuracy"]
    improvement = finetuned_acc - base_acc
    
    base_loss = base_metrics["overall"].get("loss")
    finetuned_loss = finetuned_metrics["overall"].get("loss")
    base_ppl = base_metrics["overall"].get("perplexity")
    finetuned_ppl = finetuned_metrics["overall"].get("perplexity")
    
    logger.info(f"Base Model:")
    logger.info(f"  Accuracy: {base_acc:.2%} ({base_metrics['overall']['correct']}/{base_metrics['overall']['total']})")
    if base_loss is not None:
        logger.info(f"  Loss: {base_loss:.4f}")
    if base_ppl is not None:
        logger.info(f"  Perplexity: {base_ppl:.2f}")
    
    logger.info(f"Fine-tuned Model:")
    logger.info(f"  Accuracy: {finetuned_acc:.2%} ({finetuned_metrics['overall']['correct']}/{finetuned_metrics['overall']['total']})")
    if finetuned_loss is not None:
        logger.info(f"  Loss: {finetuned_loss:.4f}")
    if finetuned_ppl is not None:
        logger.info(f"  Perplexity: {finetuned_ppl:.2f}")
    
    logger.info(f"Improvement:")
    logger.info(f"  Accuracy: {improvement:+.2%} ({improvement * 100:.1f} percentage points)")
    if base_loss is not None and finetuned_loss is not None:
        loss_improvement = base_loss - finetuned_loss
        logger.info(f"  Loss: {loss_improvement:+.4f} ({'better' if loss_improvement > 0 else 'worse'})")
    if base_ppl is not None and finetuned_ppl is not None:
        ppl_improvement = base_ppl - finetuned_ppl
        logger.info(f"  Perplexity: {ppl_improvement:+.2f} ({'better' if ppl_improvement > 0 else 'worse'})")
    
    if improvement > 0:
        logger.info(f"✅ Fine-tuned model is {improvement / base_acc * 100:.1f}% better in accuracy")
    elif improvement < 0:
        logger.info(f"⚠️  Fine-tuned model is {abs(improvement) / base_acc * 100:.1f}% worse in accuracy")
    else:
        logger.info(f"➡️  Models perform equally in accuracy")
    
    # Per-task comparison
    logger.info("\n📈 PER-TASK PERFORMANCE")
    logger.info("-" * 80)
    
    all_tasks = set(base_metrics["by_task"].keys()) | set(finetuned_metrics["by_task"].keys())
    
    for task in sorted(all_tasks):
        base_task = base_metrics["by_task"].get(task, {"accuracy": 0.0, "total": 0, "correct": 0})
        finetuned_task = finetuned_metrics["by_task"].get(task, {"accuracy": 0.0, "total": 0, "correct": 0})
        
        task_name = task.replace("_", " ").title()
        base_acc_task = base_task["accuracy"]
        finetuned_acc_task = finetuned_task["accuracy"]
        improvement_task = finetuned_acc_task - base_acc_task
        
        logger.info(f"\n{task_name}:")
        logger.info(f"  Base:      {base_acc_task:.2%} ({base_task['correct']}/{base_task['total']})")
        if base_task.get('perplexity') is not None:
            logger.info(f"            Perplexity: {base_task['perplexity']:.2f}")
        logger.info(f"  Fine-tuned: {finetuned_acc_task:.2%} ({finetuned_task['correct']}/{finetuned_task['total']})")
        if finetuned_task.get('perplexity') is not None:
            logger.info(f"            Perplexity: {finetuned_task['perplexity']:.2f}")
        logger.info(f"  Improvement: {improvement_task:+.2%}")
        if base_task.get('perplexity') is not None and finetuned_task.get('perplexity') is not None:
            ppl_improvement_task = base_task['perplexity'] - finetuned_task['perplexity']
            logger.info(f"            Perplexity: {ppl_improvement_task:+.2f} ({'better' if ppl_improvement_task > 0 else 'worse'})")
    
    # Sample predictions comparison
    logger.info("\n🔍 SAMPLE PREDICTIONS (First 5 examples)")
    logger.info("-" * 80)
    
    for i in range(min(5, len(base_results))):
        base_r = base_results[i]
        finetuned_r = finetuned_results[i]
        
        logger.info(f"\nExample {i + 1} ({base_r['task']}):")
        logger.info(f"  Ground Truth: {base_r['ground_truth']}")
        logger.info(f"  Base Prediction:      {base_r['prediction'] or 'N/A'} {'✓' if base_r['correct'] else '✗'}")
        logger.info(f"  Fine-tuned Prediction: {finetuned_r['prediction'] or 'N/A'} {'✓' if finetuned_r['correct'] else '✗'}")
        logger.info(f"  Base Response:      {base_r['response'][:100]}...")
        logger.info(f"  Fine-tuned Response: {finetuned_r['response'][:100]}...")
    
    logger.info("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare base model vs fine-tuned (LoRA adapter) model for Llama 7B using adapter approach"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="models/checkpoints/Polymarket-Llama-7B-LoRA",
        help="Path to LoRA adapter (not merged model)"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="data/fine_tune.jsonl",
        help="Path to test dataset (JSONL format)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of test samples (for quick testing)"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Use test split from dataset (0.0 to use all data, 0.1 for 10% test split)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: Save detailed results to JSON file"
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization (use FP16 instead)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    test_dataset_path = Path(args.test_dataset)
    if not test_dataset_path.exists():
        logger.error(f"Test dataset not found: {test_dataset_path}")
        return
    
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        logger.error(f"Adapter path not found: {adapter_path}")
        logger.error("Make sure you have trained the model first using finetune_llama.py")
        logger.error("Expected path: models/checkpoints/Polymarket-Llama-7B-LoRA")
        return
    
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON (Adapter Approach) - LLAMA 7B")
    logger.info("=" * 80)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Adapter path: {args.adapter_path}")
    logger.info(f"Test dataset: {args.test_dataset}")
    logger.info(f"Using 4-bit quantization: {not args.no_4bit}")
    logger.info("=" * 80)
    logger.info("Strategy: Load base model once, apply adapter for fine-tuned model")
    logger.info("=" * 80)
    
    # Load test dataset
    logger.info("\nLoading test dataset...")
    dataset = load_dataset("json", data_files=str(test_dataset_path), split="train")
    
    # Split into test set if needed
    if args.test_split > 0:
        dataset = dataset.train_test_split(test_size=args.test_split, seed=42)
        test_dataset = dataset["test"]
        logger.info(f"Using {len(test_dataset)} examples from test split")
    else:
        test_dataset = dataset
        logger.info(f"Using all {len(test_dataset)} examples")
    
    if args.max_samples:
        test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))
        logger.info(f"Limited to {len(test_dataset)} samples for testing")
    
    # Load base model once
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading base model (14GB)")
    logger.info("=" * 80)
    try:
        base_model, base_tokenizer = load_base_model(args.base_model, use_4bit=not args.no_4bit)
        
        # Quick test generation to verify model works
        logger.info("Testing base model with sample generation...")
        test_prompt = format_prompt("Test prompt", "Test input")
        test_response = generate_response(base_model, base_tokenizer, test_prompt, max_new_tokens=10)
        logger.info(f"Base model test generation successful: {test_response[:50]}...")
        
    except Exception as e:
        logger.error(f"Error loading base model: {e}")
        raise
    
    # Evaluate base model first (while it's loaded)
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Evaluating base model")
    logger.info("=" * 80)
    base_results, base_losses = evaluate_model(base_model, base_tokenizer, test_dataset, args.max_samples)
    base_metrics = calculate_metrics(base_results, base_losses)
    
    # Now load fine-tuned model by applying adapter to the same base model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Loading fine-tuned model (applying 200MB adapter)")
    logger.info("=" * 80)
    logger.info("Note: Reusing base model instance - only loading 200MB adapter!")
    try:
        # Apply adapter to the same base model instance
        # PeftModel wraps the model, so we can reuse the base model
        logger.info(f"Applying LoRA adapter from {adapter_path} to base model...")
        finetuned_model, finetuned_tokenizer = load_model_with_adapter(
            base_model, 
            str(adapter_path), 
            args.base_model
        )
        
        # Quick test generation to verify model works
        logger.info("Testing fine-tuned model with sample generation...")
        test_prompt = format_prompt("Test prompt", "Test input")
        test_response = generate_response(finetuned_model, finetuned_tokenizer, test_prompt, max_new_tokens=10)
        logger.info(f"Fine-tuned model test generation successful: {test_response[:50]}...")
        
    except Exception as e:
        logger.error(f"Error loading fine-tuned model: {e}")
        logger.error("Note: If this fails, the adapter might need a fresh base model instance")
        raise
    
    # Evaluate fine-tuned model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Evaluating fine-tuned model")
    logger.info("=" * 80)
    finetuned_results, finetuned_losses = evaluate_model(finetuned_model, finetuned_tokenizer, test_dataset, args.max_samples)
    finetuned_metrics = calculate_metrics(finetuned_results, finetuned_losses)
    
    # Print comparison report
    print_comparison_report(base_results, finetuned_results, base_metrics, finetuned_metrics)
    
    # Save results if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "base_metrics": base_metrics,
            "finetuned_metrics": finetuned_metrics,
            "base_results": base_results,
            "finetuned_results": finetuned_results,
            "comparison": {
                "overall_improvement": {
                    "accuracy": finetuned_metrics["overall"]["accuracy"] - base_metrics["overall"]["accuracy"],
                    "loss": (finetuned_metrics["overall"].get("loss") - base_metrics["overall"].get("loss")) if (finetuned_metrics["overall"].get("loss") is not None and base_metrics["overall"].get("loss") is not None) else None,
                    "perplexity": (finetuned_metrics["overall"].get("perplexity") - base_metrics["overall"].get("perplexity")) if (finetuned_metrics["overall"].get("perplexity") is not None and base_metrics["overall"].get("perplexity") is not None) else None
                },
                "task_improvements": {
                    task: {
                        "accuracy": finetuned_metrics["by_task"].get(task, {}).get("accuracy", 0) - 
                                  base_metrics["by_task"].get(task, {}).get("accuracy", 0),
                        "perplexity": (finetuned_metrics["by_task"].get(task, {}).get("perplexity") - 
                                      base_metrics["by_task"].get(task, {}).get("perplexity")) if (
                                          finetuned_metrics["by_task"].get(task, {}).get("perplexity") is not None and
                                          base_metrics["by_task"].get(task, {}).get("perplexity") is not None
                                      ) else None
                    }
                    for task in set(base_metrics["by_task"].keys()) | set(finetuned_metrics["by_task"].keys())
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📄 Detailed results saved to: {output_path}")
    
    logger.info("\n✅ Comparison completed!")


if __name__ == "__main__":
    main()

