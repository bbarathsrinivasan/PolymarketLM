"""
Compare base model vs fine-tuned (merged) model on test dataset.

This script:
- Loads both base and fine-tuned models in the same way (FP16)
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
    AutoTokenizer
)
from datasets import load_dataset
import logging
from collections import defaultdict
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path, model_name=None):
    """Load model in FP16 for consistent comparison."""
    logger.info(f"Loading model from: {model_path}")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model_path_obj = Path(model_path)
    is_local_model = model_path_obj.exists()
    
    try:
        # Try loading with device_map first (for models that were saved with it)
        if device == "cuda":
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                logger.info("Loaded model with device_map='auto'")
            except Exception as e:
                logger.warning(f"Failed to load with device_map='auto': {e}")
                logger.info("Trying to load without device_map...")
                # Fallback: load to CPU first, then move to GPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map=None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
                logger.info(f"Loaded model and moved to {device}")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        
        model.eval()
        
        # Verify model is on correct device and consolidate if needed
        if device == "cuda":
            # Check if model uses device_map (multi-device)
            if hasattr(model, "hf_device_map"):
                logger.info(f"Model uses device_map with devices: {model.hf_device_map}")
                # For device_map models, we need to ensure inputs go to the right device
                # The first parameter's device is usually the main device
                first_param_device = next(model.parameters()).device
                logger.info(f"First parameter device: {first_param_device}")
            else:
                first_param_device = next(model.parameters()).device
                logger.info(f"Model loaded on device: {first_param_device}")
                if first_param_device.type != "cuda":
                    logger.warning(f"Model parameters are on {first_param_device}, expected cuda!")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Load tokenizer from same path or base model name
    tokenizer_path = model_path if is_local_model else (model_name or model_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception:
        # Fallback to base model name if tokenizer not found in merged model
        logger.info(f"Tokenizer not found in {tokenizer_path}, using {model_name or model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_name or model_path, trust_remote_code=True)
    
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


def generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.1):
    """Generate response from model with low temperature for more deterministic outputs."""
    # Determine device - handle both single device and device_map cases
    if hasattr(model, "hf_device_map"):
        # Model uses device_map, inputs will be automatically placed
        # Find the device of the first parameter (usually the embedding layer)
        device = next(model.parameters()).device
        # For device_map models, we still need to put inputs on a device
        # The model will handle distribution automatically
    else:
        device = next(model.parameters()).device
    
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


def evaluate_model(model, tokenizer, test_dataset, max_samples=None):
    """
    Run evaluation on test set.
    
    Returns list of results with:
    - correct: bool
    - response: str
    - task: str
    - ground_truth: str
    - prediction: str
    """
    results = []
    
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
            "instruction": instruction[:100]  # First 100 chars for reference
        })
    
    return results


def calculate_metrics(results):
    """Calculate accuracy metrics per task and overall."""
    metrics = {
        "overall": {
            "total": len(results),
            "correct": sum(r["correct"] for r in results),
            "accuracy": 0.0
        },
        "by_task": {}
    }
    
    # Overall accuracy
    if metrics["overall"]["total"] > 0:
        metrics["overall"]["accuracy"] = metrics["overall"]["correct"] / metrics["overall"]["total"]
    
    # Per-task metrics
    task_results = defaultdict(list)
    for r in results:
        task_results[r["task"]].append(r)
    
    for task, task_res in task_results.items():
        total = len(task_res)
        correct = sum(r["correct"] for r in task_res)
        accuracy = correct / total if total > 0 else 0.0
        
        metrics["by_task"][task] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy
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
    
    logger.info(f"Base Model:")
    logger.info(f"  Accuracy: {base_acc:.2%} ({base_metrics['overall']['correct']}/{base_metrics['overall']['total']})")
    logger.info(f"Fine-tuned Model:")
    logger.info(f"  Accuracy: {finetuned_acc:.2%} ({finetuned_metrics['overall']['correct']}/{finetuned_metrics['overall']['total']})")
    logger.info(f"Improvement: {improvement:+.2%} ({improvement * 100:.1f} percentage points)")
    
    if improvement > 0:
        logger.info(f"✅ Fine-tuned model is {improvement / base_acc * 100:.1f}% better")
    elif improvement < 0:
        logger.info(f"⚠️  Fine-tuned model is {abs(improvement) / base_acc * 100:.1f}% worse")
    else:
        logger.info(f"➡️  Models perform equally")
    
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
        logger.info(f"  Fine-tuned: {finetuned_acc_task:.2%} ({finetuned_task['correct']}/{finetuned_task['total']})")
        logger.info(f"  Improvement: {improvement_task:+.2%}")
    
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
        description="Compare base model vs fine-tuned (merged) model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model name or path"
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        default="models/checkpoints/Polymarket-7B-MERGED",
        help="Path to fine-tuned merged model"
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
    
    args = parser.parse_args()
    
    # Validate paths
    test_dataset_path = Path(args.test_dataset)
    if not test_dataset_path.exists():
        logger.error(f"Test dataset not found: {test_dataset_path}")
        return
    
    finetuned_model_path = Path(args.finetuned_model)
    if not finetuned_model_path.exists():
        logger.error(f"Fine-tuned model not found: {finetuned_model_path}")
        logger.error("Make sure you have trained and merged the model first using finetune_qlora.py")
        return
    
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Fine-tuned model: {args.finetuned_model}")
    logger.info(f"Test dataset: {args.test_dataset}")
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
    
    # Load models
    logger.info("\nLoading base model...")
    try:
        base_model, base_tokenizer = load_model(args.base_model)
        
        # Quick test generation to verify model works
        logger.info("Testing base model with sample generation...")
        test_prompt = format_prompt("Test prompt", "Test input")
        test_response = generate_response(base_model, base_tokenizer, test_prompt, max_new_tokens=10)
        logger.info(f"Base model test generation successful: {test_response[:50]}...")
        
    except Exception as e:
        logger.error(f"Error loading base model: {e}")
        raise
    
    # Clear GPU cache before loading second model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache before loading fine-tuned model")
    
    logger.info("\nLoading fine-tuned model...")
    try:
        finetuned_model, finetuned_tokenizer = load_model(str(finetuned_model_path), args.base_model)
        
        # Quick test generation to verify model works
        logger.info("Testing fine-tuned model with sample generation...")
        test_prompt = format_prompt("Test prompt", "Test input")
        test_response = generate_response(finetuned_model, finetuned_tokenizer, test_prompt, max_new_tokens=10)
        logger.info(f"Fine-tuned model test generation successful: {test_response[:50]}...")
        
    except Exception as e:
        logger.error(f"Error loading fine-tuned model: {e}")
        raise
    
    # Evaluate both models
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING BASE MODEL")
    logger.info("=" * 80)
    base_results = evaluate_model(base_model, base_tokenizer, test_dataset, args.max_samples)
    base_metrics = calculate_metrics(base_results)
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING FINE-TUNED MODEL")
    logger.info("=" * 80)
    finetuned_results = evaluate_model(finetuned_model, finetuned_tokenizer, test_dataset, args.max_samples)
    finetuned_metrics = calculate_metrics(finetuned_results)
    
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
                "overall_improvement": finetuned_metrics["overall"]["accuracy"] - base_metrics["overall"]["accuracy"],
                "task_improvements": {
                    task: finetuned_metrics["by_task"].get(task, {}).get("accuracy", 0) - 
                          base_metrics["by_task"].get(task, {}).get("accuracy", 0)
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

