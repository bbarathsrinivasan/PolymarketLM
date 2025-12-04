"""
Generate comprehensive comparison table and graph for all methods.

This script aggregates results from:
- ICL Zero-shot (Mistral, Gemma)
- ICL Few-shot (Mistral, Gemma)
- Fine-tuned (Mistral, Gemma)
- Fine-tuned + RAG (Mistral, Gemma)

Creates:
- Comprehensive comparison table (CSV and Markdown)
- Visualization graph (PNG)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Install with: pip install pandas")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib numpy")


def load_all_results(results_dir: Path):
    """Load all result JSON files."""
    results = {}
    
    # ICL results
    for model in ["mistral", "gemma"]:
        for shot in ["zero_shot", "3_shot"]:
            file_path = results_dir / f"icl_{model}_{shot}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Normalize ICL results structure
                    method_name = f"ICL_{model.capitalize()}_Zero-shot" if shot == "zero_shot" else f"ICL_{model.capitalize()}_Few-shot"
                    
                    if "overall_accuracy" in data:
                        # Standard ICL format
                        results[method_name] = data
                    else:
                        # Alternative format - convert to standard
                        metrics = data.get("metrics", {}).get("overall", {})
                        task_metrics = data.get("metrics", {}).get("by_task", {})
                        task_accs = {}
                        task_counts = {}
                        for task, task_data in task_metrics.items():
                            if isinstance(task_data, dict):
                                task_accs[task] = task_data.get("accuracy", 0.0)
                                task_counts[task] = task_data.get("total", 0)
                            else:
                                task_accs[task] = task_data
                                task_counts[task] = 0
                        
                        results[method_name] = {
                            "overall_accuracy": metrics.get("accuracy", 0.0),
                            "total": metrics.get("total", 0),
                            "correct": metrics.get("correct", 0),
                            "perplexity": metrics.get("perplexity"),
                            "task_accuracies": task_accs,
                            "task_counts": task_counts
                        }
    
    # Fine-tuned results
    for model in ["mistral", "gemma"]:
        file_path = results_dir / f"finetuned_{model}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                results[f"Fine-tuned_{model.capitalize()}"] = json.load(f)
    
    # RAG results (baseline and RAG)
    for model in ["mistral", "gemma"]:
        # Baseline (fine-tuned without RAG) - this is same as finetuned, so skip if already loaded
        # We'll use the baseline for comparison, but it's essentially the same as fine-tuned
        
        # RAG (fine-tuned with RAG)
        rag_path = results_dir / f"rag_rag_{model}.json"
        if rag_path.exists():
            with open(rag_path, 'r') as f:
                data = json.load(f)
                results[f"Fine-tuned_{model.capitalize()}_RAG"] = data
    
    return results


def extract_metrics(data: dict, method_name: str):
    """Extract metrics from result data, handling different formats."""
    if "ICL" in method_name:
        # ICL format
        if "overall_accuracy" in data:
            accuracy = data.get("overall_accuracy", 0.0)
            total = data.get("total", 0)
            correct = data.get("correct", 0)
            perplexity = data.get("perplexity")
            task_metrics = {}
            task_accs = data.get("task_accuracies", {})
            task_counts = data.get("task_counts", {})
            
            for task in ["outcome_prediction", "manipulation_detection", "user_classification"]:
                acc = task_accs.get(task, 0.0)
                count = task_counts.get(task, 0)
                task_metrics[task] = {
                    "accuracy": acc if isinstance(acc, (int, float)) else 0.0,
                    "total": count if isinstance(count, int) else 0,
                    "correct": int(acc * count) if isinstance(acc, (int, float)) and isinstance(count, int) else 0
                }
        else:
            # Alternative ICL format
            metrics = data.get("metrics", {}).get("overall", {})
            accuracy = metrics.get("accuracy", 0.0)
            total = metrics.get("total", 0)
            correct = metrics.get("correct", 0)
            perplexity = metrics.get("perplexity")
            task_metrics = {}
            by_task = data.get("metrics", {}).get("by_task", {})
            for task, task_data in by_task.items():
                if isinstance(task_data, dict):
                    task_metrics[task] = {
                        "accuracy": task_data.get("accuracy", 0.0),
                        "total": task_data.get("total", 0),
                        "correct": task_data.get("correct", 0)
                    }
    else:
        # Fine-tuned or RAG format
        metrics = data.get("metrics", {}).get("overall", {})
        accuracy = metrics.get("accuracy", 0.0)
        total = metrics.get("total", 0)
        correct = metrics.get("correct", 0)
        perplexity = metrics.get("perplexity")
        task_metrics = {}
        by_task = data.get("metrics", {}).get("by_task", {})
        for task, task_data in by_task.items():
            if isinstance(task_data, dict):
                task_metrics[task] = {
                    "accuracy": task_data.get("accuracy", 0.0),
                    "total": task_data.get("total", 0),
                    "correct": task_data.get("correct", 0)
                }
    
    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "perplexity": perplexity,
        "task_metrics": task_metrics
    }


def create_comprehensive_table(results: dict):
    """Create comprehensive comparison table."""
    rows = []
    
    # Define order for methods
    method_order = [
        "ICL_Mistral_Zero-shot",
        "ICL_Mistral_Few-shot",
        "ICL_Gemma_Zero-shot",
        "ICL_Gemma_Few-shot",
        "Fine-tuned_Mistral",
        "Fine-tuned_Gemma",
        "Fine-tuned_Mistral_RAG",
        "Fine-tuned_Gemma_RAG"
    ]
    
    for method_name in method_order:
        if method_name not in results:
            continue
        
        data = results[method_name]
        metrics = extract_metrics(data, method_name)
        
        # Format method name for display
        display_name = method_name.replace("_", " ").replace("ICL", "ICL").replace("Fine-tuned", "Fine-tuned")
        
        rows.append({
            "Method": display_name,
            "Accuracy": f"{metrics['accuracy']:.2%}",
            "Correct/Total": f"{metrics['correct']}/{metrics['total']}",
            "Perplexity": f"{metrics['perplexity']:.2f}" if metrics['perplexity'] else "N/A",
            "Accuracy_Value": metrics['accuracy']  # For sorting/plotting
        })
    
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(rows)
        return df
    else:
        # Return as list of dicts if pandas not available
        return rows


def create_per_task_table(results: dict):
    """Create per-task comparison table."""
    rows = []
    
    method_order = [
        "ICL_Mistral_Zero-shot",
        "ICL_Mistral_Few-shot",
        "ICL_Gemma_Zero-shot",
        "ICL_Gemma_Few-shot",
        "Fine-tuned_Mistral",
        "Fine-tuned_Gemma",
        "Fine-tuned_Mistral_RAG",
        "Fine-tuned_Gemma_RAG"
    ]
    
    tasks = ["outcome_prediction", "manipulation_detection", "user_classification"]
    task_display = {
        "outcome_prediction": "Outcome Prediction",
        "manipulation_detection": "Manipulation Detection",
        "user_classification": "User Classification"
    }
    
    for method_name in method_order:
        if method_name not in results:
            continue
        
        data = results[method_name]
        metrics = extract_metrics(data, method_name)
        task_metrics = metrics['task_metrics']
        
        display_name = method_name.replace("_", " ").replace("ICL", "ICL").replace("Fine-tuned", "Fine-tuned")
        
        for task in tasks:
            task_data = task_metrics.get(task, {})
            acc = task_data.get("accuracy", 0.0)
            count = task_data.get("total", 0)
            
            rows.append({
                "Method": display_name,
                "Task": task_display[task],
                "Accuracy": f"{acc:.2%}",
                "Count": count,
                "Accuracy_Value": acc
            })
    
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(rows)
        return df
    else:
        return rows


def create_visualization(results: dict, output_dir: Path):
    """Create visualization graph."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping graph generation: matplotlib not available")
        return
    
    method_order = [
        "ICL_Mistral_Zero-shot",
        "ICL_Mistral_Few-shot",
        "ICL_Gemma_Zero-shot",
        "ICL_Gemma_Few-shot",
        "Fine-tuned_Mistral",
        "Fine-tuned_Gemma",
        "Fine-tuned_Mistral_RAG",
        "Fine-tuned_Gemma_RAG"
    ]
    
    # Extract data for plotting
    methods = []
    accuracies = []
    colors = []
    
    for method_name in method_order:
        if method_name not in results:
            continue
        
        data = results[method_name]
        metrics = extract_metrics(data, method_name)
        
        display_name = method_name.replace("_", " ").replace("ICL", "ICL").replace("Fine-tuned", "Fine-tuned")
        methods.append(display_name)
        accuracies.append(metrics['accuracy'] * 100)  # Convert to percentage
        
        # Color coding
        if "ICL" in method_name and "Zero" in method_name:
            colors.append('#FF6B6B')  # Red for zero-shot
        elif "ICL" in method_name:
            colors.append('#4ECDC4')  # Teal for few-shot
        elif "RAG" in method_name:
            colors.append('#95E1D3')  # Light green for RAG
        else:
            colors.append('#F38181')  # Pink for fine-tuned
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    bars = ax1.bar(range(len(methods)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Accuracy Comparison Across All Methods', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax1.set_ylim(0, max(accuracies) * 1.15 if accuracies else 100)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random Baseline (50%)')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.legend(loc='upper left', fontsize=9)
    
    # Grouped bar chart by task
    tasks = ["outcome_prediction", "manipulation_detection", "user_classification"]
    task_display = {
        "outcome_prediction": "Outcome\nPrediction",
        "manipulation_detection": "Manipulation\nDetection",
        "user_classification": "User\nClassification"
    }
    
    # Prepare data for grouped chart
    task_data = {task: [] for task in tasks}
    for method_name in method_order:
        if method_name not in results:
            for task in tasks:
                task_data[task].append(0)
            continue
        
        data = results[method_name]
        metrics = extract_metrics(data, method_name)
        task_metrics = metrics['task_metrics']
        
        for task in tasks:
            task_acc = task_metrics.get(task, {}).get("accuracy", 0.0)
            task_data[task].append(task_acc * 100)
    
    # Create grouped bar chart
    x = np.arange(len(methods))
    width = 0.25
    
    for i, task in enumerate(tasks):
        offset = (i - 1) * width
        ax2.bar(x + offset, task_data[task], width, label=task_display[task], alpha=0.8, edgecolor='black', linewidth=0.8)
    
    ax2.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Task Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / "comprehensive_comparison_graph.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive comparison table and graph")
    parser.add_argument("--results_dir", type=str, default="report_generation/results",
                       help="Directory containing result JSON files")
    parser.add_argument("--output_dir", type=str, default="report_generation/outputs",
                       help="Output directory for tables and graphs")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    results = load_all_results(results_dir)
    
    print(f"Found {len(results)} result files:")
    for method in sorted(results.keys()):
        print(f"  - {method}")
    
    # Create comprehensive table
    print("\nCreating comprehensive comparison table...")
    df_overall = create_comprehensive_table(results)
    
    if PANDAS_AVAILABLE:
        # Save CSV
        csv_file = output_dir / "comprehensive_comparison.csv"
        df_overall.to_csv(csv_file, index=False)
        print(f"Table saved to: {csv_file}")
        
        # Save Markdown
        md_file = output_dir / "comprehensive_comparison.md"
        with open(md_file, 'w') as f:
            f.write("# Comprehensive Method Comparison\n\n")
            f.write("## Overall Accuracy Comparison\n\n")
            f.write(df_overall[["Method", "Accuracy", "Correct/Total", "Perplexity"]].to_markdown(index=False))
            f.write("\n\n")
        
        # Create per-task table
        print("Creating per-task comparison table...")
        df_tasks = create_per_task_table(results)
        
        # Save per-task CSV
        task_csv_file = output_dir / "comprehensive_per_task_comparison.csv"
        df_tasks.to_csv(task_csv_file, index=False)
        print(f"Per-task table saved to: {task_csv_file}")
        
        # Add per-task to markdown
        with open(md_file, 'a') as f:
            f.write("## Per-Task Accuracy Comparison\n\n")
            f.write(df_tasks[["Method", "Task", "Accuracy", "Count"]].to_markdown(index=False))
            f.write("\n\n")
        
        # Create visualization
        print("Creating visualization graph...")
        create_visualization(results, output_dir)
        
        # Print summary
        print(f"\n{'='*60}")
        print("COMPREHENSIVE COMPARISON SUMMARY")
        print(f"{'='*60}")
        print("\nOverall Accuracy Rankings:")
        df_sorted = df_overall.sort_values('Accuracy_Value', ascending=False)
        for i, row in df_sorted.iterrows():
            print(f"  {row['Method']:40s} {row['Accuracy']:>8s}")
        print(f"\n{'='*60}\n")
    else:
        # Fallback: save as JSON and simple text table
        print("Warning: pandas not available. Saving as JSON instead.")
        json_file = output_dir / "comprehensive_comparison.json"
        with open(json_file, 'w') as f:
            json.dump(df_overall, f, indent=2)
        print(f"Results saved to: {json_file}")
        
        # Create simple text table
        txt_file = output_dir / "comprehensive_comparison.txt"
        with open(txt_file, 'w') as f:
            f.write("COMPREHENSIVE METHOD COMPARISON\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"{'Method':<40} {'Accuracy':>10} {'Correct/Total':>15} {'Perplexity':>12}\n")
            f.write("-" * 60 + "\n")
            for row in sorted(df_overall, key=lambda x: x['Accuracy_Value'], reverse=True):
                f.write(f"{row['Method']:<40} {row['Accuracy']:>10} {row['Correct/Total']:>15} {row['Perplexity']:>12}\n")
        print(f"Text table saved to: {txt_file}")
        
        # Try to create visualization anyway
        create_visualization(results, output_dir)
    
    print(f"All outputs saved to: {output_dir}")
    print(f"  - comprehensive_comparison.csv")
    print(f"  - comprehensive_comparison.md")
    print(f"  - comprehensive_per_task_comparison.csv")
    print(f"  - comprehensive_comparison_graph.png")


if __name__ == "__main__":
    main()

