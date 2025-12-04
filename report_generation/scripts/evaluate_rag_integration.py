"""
Evaluate fine-tuned models with RAG (Retrieval-Augmented Generation).

This script evaluates both Mistral and Gemma fine-tuned models with and without
RAG augmentation using search/news retrieval. Compares baseline vs RAG performance.
"""

import argparse
import json
import re
import sys
import math
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import integration modules
from integration.prompt_augmenter import augment_prompt_with_search, extract_market_question
from integration.search_retriever import SearchRetriever


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


def format_prompt_mistral(instruction: str, input_text: str = None) -> str:
    """Format prompt in Mistral Instruct format."""
    user_prompt = instruction
    if input_text:
        user_prompt += "\n" + input_text
    return f"<s>[INST] {user_prompt.strip()} [/INST]"


def format_prompt_gemma(instruction: str, input_text: str = None) -> str:
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


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0):
    """Generate response from model."""
    if hasattr(model, "base_model"):
        base_model = model.base_model.model if hasattr(model.base_model, "model") else model.base_model
    else:
        base_model = model
    
    if hasattr(base_model, "hf_device_map"):
        device = next(base_model.parameters()).device
    else:
        device = next(base_model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
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


def calculate_loss(model, tokenizer, instruction: str, input_text: str, target_text: str, model_type: str = "mistral"):
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
    
    if model_type == "mistral":
        full_text = f"<s>[INST] {user_prompt.strip()} [/INST] {target_text.strip()} </s>"
    else:  # gemma
        full_text = (
            "<bos><start_of_turn>user\n"
            f"{user_prompt.strip()}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            f"{target_text.strip()}\n"
            "<end_of_turn>"
        )
    
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        return loss
    except Exception:
        return None


def format_response_with_sources(response: str, search_results: List[Dict]) -> str:
    """Format response to include source citations with links."""
    if not search_results:
        return response
    
    # Build source citations
    sources_text = "\n\nSources:\n"
    for i, result in enumerate(search_results, 1):
        title = result.get('title', 'Untitled')
        link = result.get('link', '')
        source = result.get('source', 'Unknown Source')
        
        if link:
            sources_text += f"[{i}] {title} - {source} ({link})\n"
        else:
            sources_text += f"[{i}] {title} - {source}\n"
    
    # Try to add citations in the response if not already present
    if "[1]" not in response and "[2]" not in response:
        # Simple approach: append sources at the end
        return f"{response}{sources_text}"
    else:
        # Sources already referenced, just append the source list
        return f"{response}{sources_text}"


def load_test_examples(dataset_path: str, num_examples: int = 200, test_split: float = 0.1, seed: int = 42):
    """Load test examples, prioritizing outcome prediction tasks."""
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    if test_split > 0:
        dataset = dataset.train_test_split(test_size=test_split, seed=seed)
        test_dataset = dataset["test"]
    else:
        test_dataset = dataset
    
    # Convert to list
    examples = []
    for ex in test_dataset:
        examples.append({
            "instruction": ex["instruction"],
            "input": ex.get("input", ""),
            "output": ex["output"]
        })
    
    # Prioritize outcome prediction tasks (they benefit most from external context)
    outcome_examples = [ex for ex in examples if "predict the market outcome" in ex["instruction"].lower()]
    other_examples = [ex for ex in examples if "predict the market outcome" not in ex["instruction"].lower()]
    
    # Take up to num_examples, prioritizing outcome prediction
    selected = []
    if len(outcome_examples) >= num_examples:
        selected = outcome_examples[:num_examples]
    else:
        selected = outcome_examples + other_examples[:num_examples - len(outcome_examples)]
    
    print(f"Selected {len(selected)} examples ({len(outcome_examples)} outcome prediction, {len(selected) - len(outcome_examples)} other)")
    return selected


def evaluate_baseline(model, tokenizer, examples: List[Dict], model_type: str = "mistral"):
    """Evaluate baseline (no RAG) performance."""
    results = []
    losses = []
    task_results = defaultdict(list)
    
    format_prompt_fn = format_prompt_mistral if model_type == "mistral" else format_prompt_gemma
    
    for ex in tqdm(examples, desc=f"Evaluating baseline ({model_type})"):
        instruction = ex["instruction"]
        input_text = ex.get("input", "")
        ground_truth = ex["output"]
        task_type = infer_task_type(instruction)
        
        # Format prompt
        prompt = format_prompt_fn(instruction, input_text)
        
        # Calculate loss
        loss = calculate_loss(model, tokenizer, instruction, input_text, ground_truth, model_type)
        if loss is not None:
            losses.append(loss)
        
        # Generate response
        try:
            response = generate_response(model, tokenizer, prompt, max_new_tokens=128, temperature=0.0)
            prediction = extract_prediction(response, task_type)
        except Exception as e:
            print(f"Error generating response: {e}")
            response = ""
            prediction = ""
        
        # Check correctness
        correct = (prediction.strip() == ground_truth.strip())
        
        result = {
            "instruction": instruction,
            "input": input_text,
            "expected": ground_truth,
            "prediction": prediction,
            "response": response,
            "response_with_sources": response,  # Same as response for baseline
            "search_results": [],
            "correct": correct,
            "task": task_type,
            "loss": loss
        }
        results.append(result)
        task_results[task_type].append(result)
    
    return results, losses, task_results


def evaluate_rag(model, tokenizer, examples: List[Dict], search_retriever: SearchRetriever, 
                 num_search_results: int = 5, model_type: str = "mistral"):
    """Evaluate with RAG augmentation."""
    results = []
    losses = []
    task_results = defaultdict(list)
    
    format_prompt_fn = format_prompt_mistral if model_type == "mistral" else format_prompt_gemma
    
    for idx, ex in enumerate(tqdm(examples, desc=f"Evaluating RAG ({model_type})")):
        instruction = ex["instruction"]
        input_text = ex.get("input", "")
        ground_truth = ex["output"]
        task_type = infer_task_type(instruction)
        
        # Add delay between searches to avoid rate limiting (except for first search)
        if idx > 0:
            time.sleep(1)  # 1 second delay between searches
        
        # Augment prompt with search results
        try:
            augmented_input, search_results = augment_prompt_with_search(
                instruction,
                input_text,
                provider=search_retriever.provider,
                api_key=search_retriever.api_key,
                num_results=num_search_results,
                search_retriever=search_retriever,
                cache_dir=str(search_retriever.cache_dir) if search_retriever.cache_dir else None
            )
            
            # If no results found, try a simpler query without "news recent" suffix
            if not search_results:
                market_question = extract_market_question(input_text)
                if market_question:
                    try:
                        # Try without the "news recent" enhancement
                        simple_results = search_retriever.get_relevant_search_results(
                            market_question,
                            num_results=num_search_results,
                            enhance_query=False  # Don't add "news recent"
                        )
                        if simple_results:
                            search_results = simple_results
                            # Format augmented input manually
                            from integration.prompt_augmenter import format_search_section
                            search_section = format_search_section(search_results)
                            augmented_input = f"{input_text}\n\nRelevant Information from Web Search:\n{search_section}\n\nPlease analyze the above search results and explain how they inform your prediction. Consider the credibility of sources, recency of information, and how the information relates to the market question."
                    except Exception:
                        pass  # Keep original augmented_input if this also fails
        except Exception as e:
            print(f"Error retrieving search results: {e}")
            search_results = []
            augmented_input = input_text
        
        # Format prompt with augmented input
        prompt = format_prompt_fn(instruction, augmented_input)
        
        # Calculate loss (use original input for loss calculation to be fair)
        loss = calculate_loss(model, tokenizer, instruction, input_text, ground_truth, model_type)
        if loss is not None:
            losses.append(loss)
        
        # Generate response
        try:
            response = generate_response(model, tokenizer, prompt, max_new_tokens=128, temperature=0.0)
            prediction = extract_prediction(response, task_type)
            
            # Format response with sources
            response_with_sources = format_response_with_sources(response, search_results)
        except Exception as e:
            print(f"Error generating response: {e}")
            response = ""
            prediction = ""
            response_with_sources = ""
        
        # Check correctness
        correct = (prediction.strip() == ground_truth.strip())
        
        result = {
            "instruction": instruction,
            "input": input_text,
            "expected": ground_truth,
            "prediction": prediction,
            "response": response,
            "response_with_sources": response_with_sources,
            "search_results": search_results,
            "correct": correct,
            "task": task_type,
            "loss": loss
        }
        results.append(result)
        task_results[task_type].append(result)
    
    return results, losses, task_results


def calculate_metrics(results, losses, task_results):
    """Calculate accuracy metrics and perplexity."""
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


def serialize_for_json(obj):
    """Recursively convert datetime objects to ISO format strings for JSON serialization."""
    from datetime import datetime
    
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj


def save_results(results, metrics, output_dir: Path, model_name: str, method: str, 
                 base_model: str, adapter_path: str, seed: int, num_examples: int):
    """Save evaluation results to JSON file."""
    output_file = output_dir / f"rag_{method}_{model_name.lower().replace('-', '_')}.json"
    
    # Serialize datetime objects before saving
    serializable_results = serialize_for_json(results[:20])  # Save first 20 examples for analysis
    serializable_metrics = serialize_for_json(metrics)
    
    with open(output_file, 'w') as f:
        json.dump({
            "model": base_model,
            "adapter_path": adapter_path,
            "method": method,
            "num_examples": num_examples,
            "seed": seed,
            "metrics": serializable_metrics,
            "results": serializable_results
        }, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    return output_file


def generate_comparison_tables(baseline_results: Dict, rag_results: Dict, output_dir: Path, model_name: str):
    """Generate comparison tables between baseline and RAG."""
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not installed. Skipping comparison table generation.")
        print("Install with: pip install pandas")
        return
    
    # Overall comparison
    rows = []
    for method, data in [("Baseline", baseline_results), ("RAG", rag_results)]:
        metrics = data.get("metrics", {}).get("overall", {})
        rows.append({
            "Method": method,
            "Accuracy": f"{metrics.get('accuracy', 0.0):.2%}",
            "Correct/Total": f"{metrics.get('correct', 0)}/{metrics.get('total', 0)}",
            "Perplexity": f"{metrics.get('perplexity', 0):.2f}" if metrics.get('perplexity') else "N/A"
        })
    
    df_overall = pd.DataFrame(rows)
    
    # Per-task comparison
    task_rows = []
    all_tasks = set()
    for data in [baseline_results, rag_results]:
        all_tasks.update(data.get("metrics", {}).get("by_task", {}).keys())
    
    for task in sorted(all_tasks):
        baseline_task = baseline_results.get("metrics", {}).get("by_task", {}).get(task, {})
        rag_task = rag_results.get("metrics", {}).get("by_task", {}).get(task, {})
        
        task_rows.append({
            "Task": task,
            "Baseline Accuracy": f"{baseline_task.get('accuracy', 0.0):.2%}",
            "RAG Accuracy": f"{rag_task.get('accuracy', 0.0):.2%}",
            "Baseline Perplexity": f"{baseline_task.get('perplexity', 0):.2f}" if baseline_task.get('perplexity') else "N/A",
            "RAG Perplexity": f"{rag_task.get('perplexity', 0):.2f}" if rag_task.get('perplexity') else "N/A",
            "Improvement": f"{(rag_task.get('accuracy', 0.0) - baseline_task.get('accuracy', 0.0)) * 100:.1f}pp"
        })
    
    df_tasks = pd.DataFrame(task_rows)
    
    # Save CSV
    csv_file = output_dir / f"rag_comparison_{model_name.lower().replace('-', '_')}.csv"
    df_overall.to_csv(csv_file, index=False)
    print(f"Comparison table saved to: {csv_file}")
    
    # Save Markdown
    md_file = output_dir / f"rag_comparison_{model_name.lower().replace('-', '_')}.md"
    with open(md_file, 'w') as f:
        f.write(f"# RAG Comparison: {model_name}\n\n")
        f.write("## Overall Metrics\n\n")
        f.write(df_overall.to_markdown(index=False))
        f.write("\n\n## Per-Task Metrics\n\n")
        f.write(df_tasks.to_markdown(index=False))
    
    print(f"Comparison markdown saved to: {md_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned models with RAG")
    parser.add_argument("--base_model_mistral", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--adapter_path_mistral", type=str, default="models/checkpoints/Polymarket-7B-LoRA")
    parser.add_argument("--base_model_gemma", type=str, default="google/gemma-7b-it")
    parser.add_argument("--adapter_path_gemma", type=str, default="models/checkpoints/Polymarket-Gemma-7B-LoRA")
    parser.add_argument("--dataset_path", type=str, default="data/fine_tune.jsonl",
                       help="Path to dataset JSONL file (can use data/dummy_rag_dataset.jsonl for web-searchable examples)")
    parser.add_argument("--num_examples", type=int, default=200)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--search_provider", type=str, default="duckduckgo", choices=["duckduckgo", "serpapi"])
    parser.add_argument("--num_search_results", type=int, default=5)
    parser.add_argument("--serpapi_key", type=str, default=None, help="SerpAPI key (or set SERPAPI_KEY env var)")
    parser.add_argument("--output_dir", type=str, default="report_generation/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--models", type=str, default="both", choices=["both", "mistral", "gemma"])
    parser.add_argument("--cache_dir", type=str, default="integration/.search_cache")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load test examples
    print(f"\n{'='*60}")
    print("Loading Test Examples")
    print(f"{'='*60}")
    examples = load_test_examples(args.dataset_path, args.num_examples, args.test_split, args.seed)
    print(f"Loaded {len(examples)} examples\n")
    
    # Initialize search retriever
    search_retriever = SearchRetriever(
        provider=args.search_provider,
        api_key=args.serpapi_key or None,
        cache_dir=args.cache_dir
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate models
    models_to_eval = []
    if args.models in ["both", "mistral"]:
        models_to_eval.append(("mistral", args.base_model_mistral, args.adapter_path_mistral))
    if args.models in ["both", "gemma"]:
        models_to_eval.append(("gemma", args.base_model_gemma, args.adapter_path_gemma))
    
    all_results = {}
    
    for model_type, base_model, adapter_path in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type.upper()}")
        print(f"{'='*60}\n")
        
        # Check adapter path
        if not Path(adapter_path).exists():
            print(f"Warning: Adapter path not found: {adapter_path}")
            print(f"Skipping {model_type} evaluation")
            continue
        
        # Load model
        model, tokenizer = load_model_with_adapter(base_model, adapter_path, use_4bit=not args.no_4bit)
        
        # Evaluate baseline
        print(f"\nEvaluating baseline ({model_type})...")
        baseline_results, baseline_losses, baseline_task_results = evaluate_baseline(
            model, tokenizer, examples, model_type
        )
        baseline_metrics = calculate_metrics(baseline_results, baseline_losses, baseline_task_results)
        
        # Save baseline results
        baseline_data = {
            "metrics": baseline_metrics,
            "results": baseline_results
        }
        save_results(
            baseline_results, baseline_metrics, output_dir, model_type, "baseline",
            base_model, adapter_path, args.seed, args.num_examples
        )
        all_results[f"{model_type}_baseline"] = baseline_data
        
        # Print baseline summary
        print(f"\nBaseline Results ({model_type}):")
        print(f"  Accuracy: {baseline_metrics['overall']['accuracy']:.2%}")
        if baseline_metrics['overall']['perplexity']:
            print(f"  Perplexity: {baseline_metrics['overall']['perplexity']:.2f}")
        
        # Evaluate RAG
        print(f"\nEvaluating RAG ({model_type})...")
        rag_results, rag_losses, rag_task_results = evaluate_rag(
            model, tokenizer, examples, search_retriever, args.num_search_results, model_type
        )
        rag_metrics = calculate_metrics(rag_results, rag_losses, rag_task_results)
        
        # Save RAG results
        rag_data = {
            "metrics": rag_metrics,
            "results": rag_results
        }
        save_results(
            rag_results, rag_metrics, output_dir, model_type, "rag",
            base_model, adapter_path, args.seed, args.num_examples
        )
        all_results[f"{model_type}_rag"] = rag_data
        
        # Print RAG summary
        print(f"\nRAG Results ({model_type}):")
        print(f"  Accuracy: {rag_metrics['overall']['accuracy']:.2%}")
        if rag_metrics['overall']['perplexity']:
            print(f"  Perplexity: {rag_metrics['overall']['perplexity']:.2f}")
        
        # Generate comparison tables
        print(f"\nGenerating comparison tables for {model_type}...")
        generate_comparison_tables(baseline_data, rag_data, output_dir, model_type)
        
        # Print improvement
        accuracy_improvement = (rag_metrics['overall']['accuracy'] - baseline_metrics['overall']['accuracy']) * 100
        print(f"\nAccuracy Improvement: {accuracy_improvement:+.2f} percentage points")
        
        # Count how many examples had search results
        examples_with_results = sum(1 for r in rag_results if r.get('search_results'))
        print(f"Examples with search results: {examples_with_results}/{len(rag_results)} ({examples_with_results/len(rag_results)*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

