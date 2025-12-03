"""
Generate comparison tables and summaries from all evaluation results.

This script reads all JSON result files and generates:
- Overall comparison table
- Per-task comparison table
- Error analysis summary
- Markdown tables for report
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_results(results_dir: Path):
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


def create_overall_comparison_table(results: dict):
    """Create overall comparison table."""
    rows = []
    
    for method_name, data in results.items():
        if "ICL" in method_name:
            metrics = data
            accuracy = metrics.get("overall_accuracy", 0.0)
            total = metrics.get("total", 0)
            correct = metrics.get("correct", 0)
            perplexity = None
        else:  # Fine-tuned
            metrics = data.get("metrics", {}).get("overall", {})
            accuracy = metrics.get("accuracy", 0.0)
            total = metrics.get("total", 0)
            correct = metrics.get("correct", 0)
            perplexity = metrics.get("perplexity")
        
        rows.append({
            "Method": method_name,
            "Accuracy": f"{accuracy:.2%}",
            "Correct/Total": f"{correct}/{total}",
            "Perplexity": f"{perplexity:.2f}" if perplexity else "N/A"
        })
    
    df = pd.DataFrame(rows)
    return df


def create_per_task_table(results: dict):
    """Create per-task comparison table."""
    rows = []
    
    for method_name, data in results.items():
        if "ICL" in method_name:
            task_accs = data.get("task_accuracies", {})
            task_counts = data.get("task_counts", {})
        else:  # Fine-tuned
            task_metrics = data.get("metrics", {}).get("by_task", {})
            task_accs = {task: m.get("accuracy", 0.0) for task, m in task_metrics.items()}
            task_counts = {task: m.get("total", 0) for task, m in task_metrics.items()}
        
        for task in ["outcome_prediction", "manipulation_detection", "user_classification"]:
            acc = task_accs.get(task, 0.0)
            count = task_counts.get(task, 0)
            rows.append({
                "Method": method_name,
                "Task": task.replace("_", " ").title(),
                "Accuracy": f"{acc:.2%}",
                "Count": count
            })
    
    df = pd.DataFrame(rows)
    return df


def create_error_analysis(results: dict):
    """Create error analysis summary."""
    error_summary = defaultdict(lambda: {"total_errors": 0, "error_examples": []})
    
    for method_name, data in results.items():
        if "ICL" in method_name:
            results_list = data.get("results", [])
        else:  # Fine-tuned
            results_list = data.get("results", [])
        
        for result in results_list:
            if not result.get("correct", True):
                task = result.get("task", "unknown")
                error_summary[method_name]["total_errors"] += 1
                if len(error_summary[method_name]["error_examples"]) < 5:
                    error_summary[method_name]["error_examples"].append({
                        "task": task,
                        "expected": result.get("expected", ""),
                        "predicted": result.get("prediction", ""),
                        "instruction": result.get("instruction", "")[:100]
                    })
    
    return error_summary


def generate_markdown_tables(df_overall, df_per_task, output_file: Path):
    """Generate markdown tables for report."""
    with open(output_file, 'w') as f:
        f.write("# Comparison Tables\n\n")
        
        f.write("## Overall Performance\n\n")
        f.write(df_overall.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Per-Task Performance\n\n")
        f.write(df_per_task.to_markdown(index=False))
        f.write("\n\n")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison tables")
    parser.add_argument("--results_dir", type=str, default="report_generation/results")
    parser.add_argument("--output_dir", type=str, default="report_generation/outputs")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading results...")
    results = load_results(results_dir)
    
    if not results:
        print("No results found! Please run evaluation scripts first.")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Create tables
    print("Creating comparison tables...")
    df_overall = create_overall_comparison_table(results)
    df_per_task = create_per_task_table(results)
    
    # Save tables
    df_overall.to_csv(output_dir / "overall_comparison.csv", index=False)
    df_per_task.to_csv(output_dir / "per_task_comparison.csv", index=False)
    
    # Generate markdown
    generate_markdown_tables(df_overall, df_per_task, output_dir / "comparison_tables.md")
    
    # Error analysis
    error_summary = create_error_analysis(results)
    with open(output_dir / "error_analysis.json", 'w') as f:
        json.dump(error_summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    print("\nOverall Comparison:")
    print(df_overall.to_string(index=False))
    print("\nPer-Task Comparison:")
    print(df_per_task.to_string(index=False))


if __name__ == "__main__":
    main()

