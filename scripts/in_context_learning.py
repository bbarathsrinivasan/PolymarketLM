"""
In-context learning script for Gemma and Mistral-formatted models.

Usage examples:
  python scripts/in_context_learning.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter_path models/checkpoints/Polymarket-7B-LoRA \
    --examples_path data/fine_tune.jsonl \
    --shots 3 --selection by_task --format mistral

Supported selection methods: random, by_task
Supported formats: gemma, mistral

This script loads examples from a JSONL (instruction,input,output), selects k examples,
prepends them to the user prompt in the requested training format, ensures token budget,
and runs generation on the loaded model (optionally applying a LoRA adapter).
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import re 

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel


def load_model_with_adapter(base_model_name: str, adapter_path: str = None, use_4bit: bool = True):
    """Load base model (optionally 4-bit) and apply LoRA adapter if provided."""
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

    model = base_model
    if adapter_path:
        print(f"Applying LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_examples(jsonl_path: str) -> List[Dict]:
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # Expecting keys: instruction, input (optional), output
            examples.append({
                'instruction': obj.get('instruction', '').strip(),
                'input': obj.get('input', '').strip() if obj.get('input') else '',
                'output': obj.get('output', '').strip()
            })
    return examples


def select_examples_random(examples: List[Dict], k: int) -> List[Dict]:
    if k <= 0:
        return []
    k = min(k, len(examples))
    return random.sample(examples, k)


def select_examples_by_task(user_instruction: str, examples: List[Dict], k: int) -> List[Dict]:
    """Select examples by simple token overlap between instructions.
    Falls back to random sampling if not enough matches."""
    if k <= 0:
        return []

    user_tokens = set([t for t in user_instruction.lower().split() if len(t) > 2])
    scored = []
    for ex in examples:
        tokens = set([t for t in ex['instruction'].lower().split() if len(t) > 2])
        overlap = len(user_tokens & tokens)
        scored.append((overlap, ex))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [ex for score, ex in scored if score > 0]

    if len(selected) < k:
        # include top matches and then random others
        remaining = [ex for _, ex in scored if ex not in selected]
        need = k - len(selected)
        if remaining:
            selected += random.sample(remaining, min(len(remaining), need))
    return selected[:k]


def format_prompt_mistral(examples: List[Dict], instruction: str, input_text: str = None) -> str:
    parts = []
    # Examples include full response
    for ex in examples:
        user_prompt = ex['instruction']
        if ex.get('input'):
            user_prompt += "\n" + ex['input']
        parts.append(f"<s>[INST] {user_prompt.strip()} [/INST] {ex['output'].strip()} </s>")

    # Append user prompt (no response)
    user_prompt = instruction
    if input_text:
        user_prompt += "\n" + input_text
    parts.append(f"<s>[INST] {user_prompt.strip()} [/INST]")
    return "\n".join(parts)


def format_prompt_gemma(examples: List[Dict], instruction: str, input_text: str = None) -> str:
    parts = []

    # Optional but good: BOS-like start
    parts.append("<bos>")

    # Minimal system message
    parts.append("<start_of_turn>system\nYou are a helpful and accurate prediction model. "
                 "Answer ONLY with 'Yes' or 'No'.<end_of_turn>")

    # Few-shot examples
    for ex in examples:
        user_prompt = ex["instruction"]
        if ex.get("input"):
            user_prompt += "\n" + ex["input"]

        parts.append(
            f"<start_of_turn>user\n{user_prompt.strip()}<end_of_turn>\n"
            f"<start_of_turn>model\n{ex['output'].strip()}<end_of_turn>"
        )

    # Final user query
    user_prompt = instruction
    if input_text:
        user_prompt += "\n" + input_text

    parts.append(
        f"<start_of_turn>user\n{user_prompt.strip()}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    return "\n".join(parts)


def build_prompt(format_name: str, examples: List[Dict], instruction: str, input_text: str = None) -> str:
    fmt = format_name.lower()
    if fmt == 'mistral':
        return format_prompt_mistral(examples, instruction, input_text)
    if fmt == 'gemma':
        return format_prompt_gemma(examples, instruction, input_text)
    # default to mistral
    return format_prompt_mistral(examples, instruction, input_text)


def ensure_token_budget(tokenizer, prompt: str, max_input_tokens: int, examples: List[Dict], format_name: str) -> Tuple[str, List[Dict]]:
    """Trim examples (by dropping last) until prompt token length <= max_input_tokens."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=False)
    length = enc.input_ids.shape[1]
    if length <= max_input_tokens:
        return prompt, examples

    # Iteratively drop the least-important example (last in list)
    trimmed = examples.copy()
    while length > max_input_tokens and trimmed:
        trimmed.pop()  # drop last
        prompt = build_prompt(format_name, trimmed, instruction_global, input_global)
        enc = tokenizer(prompt, return_tensors="pt", truncation=False)
        length = enc.input_ids.shape[1]

    # If still too long and no examples left, truncate prompt tokens directly (let model handle)
    if length > max_input_tokens:
        # Truncate using tokenizer to max_input_tokens
        toks = enc.input_ids[0][:max_input_tokens]
        prompt = tokenizer.decode(toks, skip_special_tokens=True)
    return prompt, trimmed


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id
    )

    input_len = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_len:]
    resp = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return resp.strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='In-Context Learning script')
    parser.add_argument('--base_model', type=str, required=True, help='Base model name')
    parser.add_argument('--adapter_path', type=str, default=None, help='Path to LoRA adapter (optional)')
    parser.add_argument('--examples_path', type=str, default='data/fine_tune.jsonl')
    parser.add_argument('--shots', type=int, default=0)
    parser.add_argument('--selection', type=str, default='random', choices=['random', 'by_task'])
    parser.add_argument('--format', type=str, default='mistral', choices=['mistral', 'gemma'])
    parser.add_argument('--no_4bit', action='store_true', help='Disable 4-bit quantization')
    parser.add_argument('--max_input_tokens', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--instruction', type=str, required=True)
    parser.add_argument('--input_text', type=str, default=None)
    parser.add_argument('--expected_output', type=str, default=None)

    args = parser.parse_args()

    # Load dataset
    examples_path = Path(args.examples_path)
    if not examples_path.exists():
        print(f"Examples file not found: {examples_path}")
        sys.exit(1)

    examples = load_examples(str(examples_path))
    if not examples:
        print("No examples loaded.")
        sys.exit(1)

    # Load model/tokenizer once
    model, tokenizer = load_model_with_adapter(
        args.base_model,
        args.adapter_path,
        use_4bit=not args.no_4bit
    )

    global instruction_global, input_global
    instruction_global = args.instruction
    input_global = args.input_text

    def run_eval(num_shots: int, label: str):
        """Run one evaluation mode (zero-shot or few-shot) and compute accuracy."""
        print(f"\n================ {label.upper()} EVALUATION ================")

        # Select examples
        if num_shots <= 0:
            selected = []
        else:
            if args.selection == 'random':
                selected = select_examples_random(examples, num_shots)
            else:
                selected = select_examples_by_task(args.instruction, examples, num_shots)

        # Build prompt
        prompt = build_prompt(args.format, selected, args.instruction, args.input_text)

        # Ensure token limit
        prompt, _ = ensure_token_budget(
            tokenizer, prompt, args.max_input_tokens, selected, args.format
        )

        print("\n--- Prompt (first 1000 chars) ---")
        print(prompt[:1000])

        # Generate
        resp = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )

        print("\n--- Response ---")
        print(resp)

        # Accuracy
        if args.expected_output is None:
            print("\n--- Accuracy ---\nN/A (no expected output provided)")
            return None

        expected = args.expected_output.strip()
        m = re.search(r'\b(yes|no)\b', resp, re.I)
        if m:
            pred = m.group(1).capitalize()
        else:
            pred = resp.strip().split()[0] if resp.strip() else ""

        acc = 1.0 if pred.lower() == expected.lower() else 0.0

        print("\n--- Accuracy ---")
        print(f"Predicted: {pred}")
        print(f"Exact-match accuracy: {acc:.3f}")

        return acc

        # -----------------------------
    # DATASET-LEVEL EVALUATION LOGIC
    # -----------------------------

    def extract_pred(text: str):
        """Extract yes/no prediction from the model output."""
        m = re.search(r'\b(yes|no)\b', text, re.I)
        if m:
            return m.group(1).lower()
        # fallback: first token
        toks = text.strip().split()
        return toks[0].lower() if toks else ""


    def evaluate_dataset(num_shots: int, label: str):

        print(f"\n================ {label.upper()} DATASET EVALUATION ================\n")

        MAX_TEST_EXAMPLES = 15
        eval_examples = examples[:MAX_TEST_EXAMPLES]

        correct = 0
        total = 0

        for idx, ex in enumerate(tqdm(eval_examples, desc=f"{label} eval", unit="example")):

            inst = ex["instruction"]
            inp = ex.get("input", "")
            expected = ex["output"].strip().lower()

            # Few-shot examples (exclude current example)
            if num_shots > 0:
                pool = [e for e in examples if e is not ex]
                if args.selection == "random":
                    selected = select_examples_random(pool, num_shots)
                else:
                    selected = select_examples_by_task(inst, pool, num_shots)
            else:
                selected = []

            # Build prompt
            global instruction_global, input_global
            instruction_global = inst
            input_global = inp

            prompt = build_prompt(args.format, selected, inst, inp)

            # Enforce token limit
            prompt, _ = ensure_token_budget(
                tokenizer, prompt, args.max_input_tokens, selected, args.format
            )

            # Model inference
            resp = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            ).lower()

            # Extract yes/no
            m = re.search(r'\b(yes|no)\b', resp, re.I)
            pred = m.group(1).lower() if m else resp.split()[0] if resp.strip() else ""

            # Score
            is_correct = (pred == expected)
            if is_correct:
                correct += 1
            total += 1

            # Print full example details
            print("\n----------------------------------------")
            print(f"Example {idx+1}/{MAX_TEST_EXAMPLES}")
            print("----------------------------------------")
            print(f"Instruction: {inst}")
            if inp:
                print(f"Input: {inp}")
            print(f"Expected: {expected}")
            print(f"Predicted: {pred}")
            print(f"Correct: {is_correct}")
            print("----------------------------------------\n")

        # Final accuracy
        acc = correct / total if total > 0 else 0.0

        print("\n================ FINAL ACCURACY SUMMARY ================\n")
        print(f"{label} accuracy: {acc*100:.2f}% ({correct}/{total} correct)")
        print("\n========================================================\n")

        return acc



    # Run dataset-level evaluations
    zero_shot_acc = evaluate_dataset(0, "zero-shot")
    few_shot_acc = evaluate_dataset(args.shots, "few-shot")

    print("\n================ FINAL SUMMARY =================\n")
    print(f"Zero-shot accuracy: {zero_shot_acc * 100:.2f}%")
    print(f"Few-shot accuracy: {few_shot_acc * 100:.2f}%")
    print("\n================================================\n")

