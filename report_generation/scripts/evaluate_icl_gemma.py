"""
Evaluate In-Context Learning (Zero-shot and Few-shot) for Gemma-7B-IT.

This script evaluates the base Gemma model on the test set using:
- Zero-shot (no examples)
- Few-shot (k examples, default k=3)

Outputs results to JSON file for report generation.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load base model and tokenizer."""
    print(f"Loading model: {model_name}")
    if use_4bit:
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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_examples(jsonl_path: str) -> List[Dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append({
                    'instruction': obj.get('instruction', '').strip(),
                    'input': obj.get('input', '').strip() if obj.get('input') else '',
                    'output': obj.get('output', '').strip()
                })
            except Exception:
                continue
    return examples


def select_examples_random(examples: List[Dict], k: int, exclude: Dict = None) -> List[Dict]:
    """Select k random examples, excluding the current one."""
    pool = [e for e in examples if e != exclude] if exclude else examples
    if k <= 0:
        return []
    k = min(k, len(pool))
    return random.sample(pool, k)


def select_examples_by_task(user_instruction: str, examples: List[Dict], k: int, exclude: Dict = None) -> List[Dict]:
    """Select examples by task similarity."""
    pool = [e for e in examples if e != exclude] if exclude else examples
    if k <= 0:
        return []

    user_tokens = set([t for t in user_instruction.lower().split() if len(t) > 2])
    scored = []
    for ex in pool:
        tokens = set([t for t in ex['instruction'].lower().split() if len(t) > 2])
        overlap = len(user_tokens & tokens)
        scored.append((overlap, ex))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [ex for score, ex in scored if score > 0]

    if len(selected) < k:
        remaining = [ex for _, ex in scored if ex not in selected]
        need = k - len(selected)
        if remaining:
            selected += random.sample(remaining, min(len(remaining), need))
    return selected[:k]


def format_prompt_gemma(examples: List[Dict], instruction: str, input_text: str = None) -> str:
    """Format prompt in Gemma dialogue format."""
    parts = []
    parts.append("<bos>")
    parts.append("<start_of_turn>system\nYou are a helpful and accurate prediction model. "
                 "Answer ONLY with 'Yes' or 'No' for prediction tasks, or 'Noise Trader'/'Informed Trader' for classification.<end_of_turn>")

    for ex in examples:
        user_prompt = ex["instruction"]
        if ex.get("input"):
            user_prompt += "\n" + ex["input"]

        parts.append(
            f"<start_of_turn>user\n{user_prompt.strip()}<end_of_turn>\n"
            f"<start_of_turn>model\n{ex['output'].strip()}<end_of_turn>"
        )

    user_prompt = instruction
    if input_text:
        user_prompt += "\n" + input_text

    parts.append(
        f"<start_of_turn>user\n{user_prompt.strip()}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    return "\n".join(parts)


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0):
    """Generate response from model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_len:]
    resp = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return resp.strip()


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


def evaluate_icl(model, tokenizer, test_examples: List[Dict], train_examples: List[Dict], 
                 num_shots: int, selection_method: str = "random", max_input_tokens: int = 2048):
    """Evaluate ICL performance."""
    results = []
    correct = 0
    total = 0
    
    task_results = {"outcome_prediction": [], "manipulation_detection": [], "user_classification": []}
    
    for ex in tqdm(test_examples, desc=f"Evaluating {num_shots}-shot"):
        inst = ex["instruction"]
        inp = ex.get("input", "")
        expected = ex["output"].strip()
        task_type = infer_task_type(inst)
        
        # Select examples
        if num_shots > 0:
            if selection_method == "random":
                selected = select_examples_random(train_examples, num_shots, exclude=ex)
            else:
                selected = select_examples_by_task(inst, train_examples, num_shots, exclude=ex)
        else:
            selected = []
        
        # Build prompt
        prompt = format_prompt_gemma(selected, inst, inp)
        
        # Truncate if too long
        inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
        if inputs.input_ids.shape[1] > max_input_tokens:
            # Truncate by removing examples from the end
            while inputs.input_ids.shape[1] > max_input_tokens and selected:
                selected.pop()
                prompt = format_prompt_gemma(selected, inst, inp)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
        
        # Generate
        response = generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0)
        prediction = extract_prediction(response, task_type)
        
        is_correct = (prediction.strip() == expected.strip())
        if is_correct:
            correct += 1
        total += 1
        
        result = {
            "instruction": inst,
            "input": inp,
            "expected": expected,
            "prediction": prediction,
            "response": response,
            "correct": is_correct,
            "task": task_type
        }
        results.append(result)
        task_results[task_type].append(result)
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Per-task accuracy
    task_accuracies = {}
    for task, task_res in task_results.items():
        if task_res:
            task_correct = sum(1 for r in task_res if r["correct"])
            task_accuracies[task] = task_correct / len(task_res)
        else:
            task_accuracies[task] = 0.0
    
    return {
        "overall_accuracy": accuracy,
        "total": total,
        "correct": correct,
        "task_accuracies": task_accuracies,
        "task_counts": {task: len(res) for task, res in task_results.items()},
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICL for Gemma")
    parser.add_argument("--model_name", type=str, default="google/gemma-7b-it")
    parser.add_argument("--dataset_path", type=str, default="data/fine_tune.jsonl")
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--num_shots", type=int, default=3, help="Number of few-shot examples (0 for zero-shot)")
    parser.add_argument("--selection", type=str, default="random", choices=["random", "by_task"])
    parser.add_argument("--output_dir", type=str, default="report_generation/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test samples to evaluate (for faster testing)")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    if args.test_split > 0:
        dataset = dataset.train_test_split(test_size=args.test_split, seed=args.seed)
        test_dataset = dataset["test"]
        train_dataset = dataset["train"]
    else:
        # Use all as test, no train examples for ICL
        test_dataset = dataset
        train_dataset = dataset
    
    # Convert to list format
    test_examples = [{"instruction": ex["instruction"], "input": ex.get("input", ""), "output": ex["output"]} 
                     for ex in test_dataset]
    train_examples = [{"instruction": ex["instruction"], "input": ex.get("input", ""), "output": ex["output"]} 
                     for ex in train_dataset]
    
    # Limit test samples if specified
    if args.max_samples is not None and args.max_samples > 0:
        if len(test_examples) > args.max_samples:
            print(f"Limiting evaluation to {args.max_samples} samples (from {len(test_examples)})")
            test_examples = test_examples[:args.max_samples]
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_name, use_4bit=not args.no_4bit)
    
    # Evaluate
    shot_label = "zero-shot" if args.num_shots == 0 else f"{args.num_shots}-shot"
    print(f"\n{'='*60}")
    print(f"Evaluating {shot_label} ICL for Gemma")
    print(f"{'='*60}\n")
    
    results = evaluate_icl(model, tokenizer, test_examples, train_examples, 
                          args.num_shots, args.selection)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {shot_label.upper()}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2%} ({results['correct']}/{results['total']})")
    print(f"\nPer-Task Accuracies:")
    for task, acc in results['task_accuracies'].items():
        count = results['task_counts'][task]
        print(f"  {task}: {acc:.2%} ({count} examples)")
    print(f"{'='*60}\n")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"icl_gemma_{shot_label.replace('-', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model": args.model_name,
            "num_shots": args.num_shots,
            "selection_method": args.selection,
            "seed": args.seed,
            **results
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

