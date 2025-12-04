"""
Generate comparison graph from the markdown table in `report_generation/outputs/comarison.md`.

This script recreates a graph with the same style and layout as
`comprehensive_comparison_graph.png`, using:
- Overall accuracies from `comarison.md`
- Per-task accuracies from `per_task_comparison.csv` (already generated)
"""

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for script usage
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


METHOD_ORDER = [
    "ICL Mistral Zero-shot",
    "ICL Mistral Few-shot",
    "ICL Gemma Zero-shot",
    "ICL Gemma Few-shot",
    "Fine-tuned Mistral",
    "Fine-tuned Gemma",
    "Fine-tuned Mistral RAG",
    "Fine-tuned Gemma RAG",
]


def load_overall_from_markdown(md_path: Path) -> pd.DataFrame:
    """Parse the overall comparison markdown table from `comarison.md`."""
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Collect only the lines that belong to the main table (skip header separator lines)
    table_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # Skip alignment/separator lines like `|:---|:---|`
        if set(stripped.replace(" ", "")) <= {"|", ":", "-"}:
            continue
        table_lines.append(stripped)

    if not table_lines:
        raise ValueError(f"No markdown table found in {md_path}")

    table_str = "\n".join(table_lines)
    df = pd.read_csv(StringIO(table_str), sep="|")

    # Drop empty columns caused by leading/trailing pipes
    df = df.drop(columns=[col for col in df.columns if col.strip() == ""])

    # Clean column names and values
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Ensure expected columns exist
    expected_cols = {"Method", "Accuracy", "Correct/Total"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in markdown table: {missing}")

    # Convert accuracy string like "7.89%" to numeric value
    df["Accuracy_Value"] = (
        df["Accuracy"].str.rstrip("%").astype(float) / 100.0
    )

    # Reorder rows according to METHOD_ORDER
    df["Method_order"] = df["Method"].apply(
        lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER)
    )
    df = df.sort_values("Method_order").reset_index(drop=True)
    df = df.drop(columns=["Method_order"])

    return df


def load_per_task_csv(csv_path: Path) -> pd.DataFrame:
    """Load per-task comparison data from CSV."""
    df = pd.read_csv(csv_path)
    # Ensure ordering is consistent with METHOD_ORDER
    df["Method_order"] = df["Method"].apply(
        lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER)
    )
    df = df.sort_values(["Method_order", "Task"]).reset_index(drop=True)
    df = df.drop(columns=["Method_order"])
    return df


def create_comarison_graph(
    overall_df: pd.DataFrame, per_task_df: pd.DataFrame, output_path: Path
) -> None:
    """Create a comparison graph matching the style of `comprehensive_comparison_graph.png`."""
    methods = overall_df["Method"].tolist()
    accuracies = (overall_df["Accuracy_Value"] * 100.0).tolist()

    # Color coding consistent with existing script
    colors = []
    for method_name in methods:
        if "ICL" in method_name and "Zero" in method_name:
            colors.append("#FF6B6B")  # Red for zero-shot
        elif "ICL" in method_name:
            colors.append("#4ECDC4")  # Teal for few-shot
        elif "RAG" in method_name:
            colors.append("#95E1D3")  # Light green for RAG
        else:
            colors.append("#F38181")  # Pink for fine-tuned

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left subplot: overall accuracy bar chart
    x = np.arange(len(methods))
    bars = ax1.bar(
        x,
        accuracies,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )
    ax1.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Overall Accuracy Comparison Across All Methods",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=10)
    ax1.set_ylim(0, max(accuracies) * 1.15 if accuracies else 100)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.axhline(
        y=50,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Random Baseline (50%)",
    )

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.legend(loc="upper left", fontsize=9)

    # Right subplot: per-task grouped bar chart
    tasks = ["Outcome Prediction", "Manipulation Detection", "User Classification"]
    task_display = {
        "Outcome Prediction": "Outcome\nPrediction",
        "Manipulation Detection": "Manipulation\nDetection",
        "User Classification": "User\nClassification",
    }

    # Prepare per-task data in same method order
    task_data = {task: [] for task in tasks}
    for method in methods:
        df_method = per_task_df[per_task_df["Method"] == method]
        for task in tasks:
            row = df_method[df_method["Task"] == task]
            if not row.empty:
                acc_str = row.iloc[0]["Accuracy"]
                acc_value = float(str(acc_str).rstrip("%"))
            else:
                acc_value = 0.0
            task_data[task].append(acc_value)

    x = np.arange(len(methods))
    width = 0.25

    for i, task in enumerate(tasks):
        offset = (i - 1) * width
        ax2.bar(
            x + offset,
            task_data[task],
            width,
            label=task_display[task],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.8,
        )

    ax2.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Per-Task Accuracy Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha="right", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate comparison graph from report_generation/outputs/comarison.md "
            "and per_task_comparison.csv"
        )
    )
    parser.add_argument(
        "--md_path",
        type=str,
        default="report_generation/outputs/comarison.md",
        help="Path to the markdown file containing overall comparison table.",
    )
    parser.add_argument(
        "--per_task_csv",
        type=str,
        default="report_generation/outputs/per_task_comparison.csv",
        help="Path to the per-task comparison CSV file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="report_generation/outputs/comarison_graph.png",
        help="Path to save the generated comparison graph PNG.",
    )

    args = parser.parse_args()

    md_path = Path(args.md_path)
    per_task_csv = Path(args.per_task_csv)
    output_path = Path(args.output_path)

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")
    if not per_task_csv.exists():
        raise FileNotFoundError(f"Per-task CSV file not found: {per_task_csv}")

    overall_df = load_overall_from_markdown(md_path)
    per_task_df = load_per_task_csv(per_task_csv)

    create_comarison_graph(overall_df, per_task_df, output_path)
    print(f"Graph saved to: {output_path}")


if __name__ == "__main__":
    main()


