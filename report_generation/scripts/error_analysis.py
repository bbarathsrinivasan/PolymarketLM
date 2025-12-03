"""
Error Analysis Script

Analyzes errors across all evaluation results to identify:
- Common failure patterns
- Task-specific challenges
- Model-specific weaknesses
- Examples of failures for qualitative analysis
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def load_all_results(results_dir: Path) -> Dict:
    """Load all result JSON files."""
    results = {}
    
    # ICL results
    for model in ["mistral", "gemma"]:
        for shot in ["zero_shot", "3_shot"]:
            file_path = results_dir / f"icl_{model}_{shot}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    results[f"ICL_{model}_{shot}"] = json.load(f)
    
    # Fine-tuned results
    for model in ["mistral", "gemma"]:
        file_path = results_dir / f"finetuned_{model}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                results[f"Finetuned_{model}"] = json.load(f)
    
    return results


def analyze_errors(results: Dict) -> Dict:
    """Analyze errors across all methods."""
    analysis = {
        "overall_stats": {},
        "task_breakdown": defaultdict(lambda: {"total_errors": 0, "error_rate": 0.0}),
        "common_errors": [],
        "method_comparison": {},
        "error_examples": defaultdict(list)
    }
    
    # Collect all errors
    all_errors = defaultdict(list)
    error_by_example = defaultdict(set)  # Track which methods failed on which examples
    
    for method_name, data in results.items():
        if "ICL" in method_name:
            results_list = data.get("results", [])
            total = data.get("total", 0)
            correct = data.get("correct", 0)
        else:  # Fine-tuned
            metrics = data.get("metrics", {})
            total = metrics.get("overall", {}).get("total", 0)
            correct = metrics.get("overall", {}).get("correct", 0)
            results_list = data.get("results", [])
        
        errors = total - correct
        error_rate = errors / total if total > 0 else 0.0
        
        analysis["overall_stats"][method_name] = {
            "total": total,
            "correct": correct,
            "errors": errors,
            "error_rate": error_rate,
            "accuracy": 1.0 - error_rate
        }
        
        # Collect errors by task
        for result in results_list:
            if not result.get("correct", True):
                task = result.get("task", "unknown")
                all_errors[task].append({
                    "method": method_name,
                    "expected": result.get("expected", ""),
                    "predicted": result.get("prediction", ""),
                    "instruction": result.get("instruction", "")[:150],
                    "input": result.get("input", "")[:200] if result.get("input") else "",
                    "response": result.get("response", "")[:200]
                })
                
                # Track which methods failed on this example
                example_key = f"{result.get('instruction', '')[:50]}_{result.get('expected', '')}"
                error_by_example[example_key].add(method_name)
                
                # Store example for qualitative analysis
                if len(analysis["error_examples"][method_name]) < 5:
                    analysis["error_examples"][method_name].append({
                        "task": task,
                        "expected": result.get("expected", ""),
                        "predicted": result.get("prediction", ""),
                        "instruction": result.get("instruction", "")[:100],
                        "input": result.get("input", "")[:150] if result.get("input") else "",
                        "response": result.get("response", "")[:150]
                    })
        
        # Per-task error breakdown
        if "ICL" in method_name:
            task_accs = data.get("task_accuracies", {})
            task_counts = data.get("task_counts", {})
        else:
            task_metrics = data.get("metrics", {}).get("by_task", {})
            task_accs = {task: m.get("accuracy", 0.0) for task, m in task_metrics.items()}
            task_counts = {task: m.get("total", 0) for task, m in task_metrics.items()}
        
        for task, count in task_counts.items():
            acc = task_accs.get(task, 0.0)
            error_count = int(count * (1 - acc))
            analysis["task_breakdown"][task]["total_errors"] += error_count
    
    # Calculate error rates per task
    for task in analysis["task_breakdown"]:
        total_examples = sum(
            stats.get("task_counts", {}).get(task, 0) 
            for stats in results.values() 
            if "ICL" in list(results.keys())[list(results.values()).index(stats)] or 
               task in stats.get("metrics", {}).get("by_task", {})
        )
        if total_examples > 0:
            analysis["task_breakdown"][task]["error_rate"] = (
                analysis["task_breakdown"][task]["total_errors"] / total_examples
            )
    
    # Find common errors (examples where multiple methods failed)
    common_errors = {
        key: list(methods) 
        for key, methods in error_by_example.items() 
        if len(methods) >= 2
    }
    analysis["common_errors"] = {
        "count": len(common_errors),
        "examples": list(common_errors.items())[:10]  # Top 10
    }
    
    # Method comparison
    analysis["method_comparison"] = {
        "most_robust": min(analysis["overall_stats"].items(), key=lambda x: x[1]["error_rate"])[0],
        "least_robust": max(analysis["overall_stats"].items(), key=lambda x: x[1]["error_rate"])[0],
        "hardest_task": max(analysis["task_breakdown"].items(), key=lambda x: x[1]["error_rate"])[0] if analysis["task_breakdown"] else "N/A"
    }
    
    return analysis


def print_analysis(analysis: Dict):
    """Print error analysis summary."""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 80)
    
    print("\n📊 Overall Error Statistics:")
    print("-" * 80)
    for method, stats in analysis["overall_stats"].items():
        print(f"{method}:")
        print(f"  Total: {stats['total']}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Error Rate: {stats['error_rate']:.2%}")
        print(f"  Accuracy: {stats['accuracy']:.2%}")
        print()
    
    print("\n📈 Per-Task Error Breakdown:")
    print("-" * 80)
    for task, stats in analysis["task_breakdown"].items():
        print(f"{task.replace('_', ' ').title()}:")
        print(f"  Total Errors: {stats['total_errors']}")
        print(f"  Error Rate: {stats['error_rate']:.2%}")
        print()
    
    print("\n🔍 Common Errors (Multiple Methods Failed):")
    print("-" * 80)
    print(f"Number of examples where 2+ methods failed: {analysis['common_errors']['count']}")
    print(f"Top examples: {len(analysis['common_errors']['examples'])}")
    
    print("\n🏆 Method Comparison:")
    print("-" * 80)
    print(f"Most Robust: {analysis['method_comparison']['most_robust']}")
    print(f"Least Robust: {analysis['method_comparison']['least_robust']}")
    print(f"Hardest Task: {analysis['method_comparison']['hardest_task']}")
    
    print("\n📝 Error Examples (First 2 per method):")
    print("-" * 80)
    for method, examples in analysis["error_examples"].items():
        print(f"\n{method}:")
        for i, ex in enumerate(examples[:2], 1):
            print(f"  Example {i}:")
            print(f"    Task: {ex['task']}")
            print(f"    Expected: {ex['expected']}")
            print(f"    Predicted: {ex['predicted']}")
            print(f"    Instruction: {ex['instruction']}...")
            if ex.get('input'):
                print(f"    Input: {ex['input']}...")
            print(f"    Response: {ex['response']}...")
            print()
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze errors across all evaluation results")
    parser.add_argument("--results_dir", type=str, default="report_generation/results")
    parser.add_argument("--output_file", type=str, default="report_generation/outputs/error_analysis_detailed.json")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading results...")
    results = load_all_results(results_dir)
    
    if not results:
        print("No results found! Please run evaluation scripts first.")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Analyze errors
    print("Analyzing errors...")
    analysis = analyze_errors(results)
    
    # Print summary
    print_analysis(analysis)
    
    # Save detailed analysis
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {output_file}")


if __name__ == "__main__":
    main()

