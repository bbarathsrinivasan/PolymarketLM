# Quick Start Guide

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Test setup:
```bash
python report_generation/scripts/test_scripts.py
```

## Complete Workflow (Copy-Paste Ready)

### 1. Fine-tune Models

```bash
# Mistral
python scripts/finetune_qlora.py \
    --config report_generation/configs/finetuning_config_mistral.yaml \
    --output_dir models/checkpoints/Polymarket-7B-LoRA

# Gemma
python scripts/finetune_gemma.py \
    --config report_generation/configs/finetuning_config_gemma.yaml \
    --output_dir models/checkpoints/Polymarket-Gemma-7B-LoRA
```

### 2. Run ICL Evaluations

```bash
# Mistral Zero-shot
python report_generation/scripts/evaluate_icl_mistral.py --num_shots 0 --max_samples 1000

# Mistral Few-shot
python report_generation/scripts/evaluate_icl_mistral.py --num_shots 3 --selection random --max_samples 1000

# Gemma Zero-shot
python report_generation/scripts/evaluate_icl_gemma.py --num_shots 0 --max_samples 1000

# Gemma Few-shot
python report_generation/scripts/evaluate_icl_gemma.py --num_shots 3 --selection random --max_samples 1000
```

### 3. Run Fine-tuned Evaluations

```bash
# Mistral Fine-tuned
python report_generation/scripts/evaluate_finetuned_mistral.py \
    --adapter_path models/checkpoints/Polymarket-7B-LoRA \
    --max_samples 1000

# Gemma Fine-tuned
python report_generation/scripts/evaluate_finetuned_gemma.py \
    --adapter_path models/checkpoints/Polymarket-Gemma-7B-LoRA \
    --max_samples 1000
```

### 4. Generate Dummy RAG Dataset (Optional but Recommended)

Create examples with web-searchable content:

```bash
python report_generation/scripts/generate_dummy_rag_dataset.py \
    --output_path data/dummy_rag_dataset.jsonl \
    --num_examples 50
```

### 5. Run RAG Evaluation (Additional Experiment)

```bash
# Using dummy dataset (recommended - has web-searchable content)
python report_generation/scripts/evaluate_rag_integration.py \
    --dataset_path data/dummy_rag_dataset.jsonl \
    --num_examples 50 \
    --search_provider duckduckgo \
    --num_search_results 5

# Or using real dataset
python report_generation/scripts/evaluate_rag_integration.py \
    --num_examples 50 \
    --search_provider duckduckgo \
    --num_search_results 5

# Evaluate individually
python report_generation/scripts/evaluate_rag_integration.py --models mistral --num_examples 50
python report_generation/scripts/evaluate_rag_integration.py --models gemma --num_examples 50
```

### 6. Generate Tables and Analysis

```bash
# Comparison tables
python report_generation/scripts/generate_comparison_tables.py

# Error analysis
python report_generation/scripts/error_analysis.py
```

### 7. Update Report

1. Open `report_generation/report.md`
2. Fill in all `[TO FILL]` placeholders with results from:
   - `report_generation/outputs/comparison_tables.md`
   - `report_generation/results/rag_comparison_*.md` (RAG experiment)
   - `report_generation/outputs/error_analysis_detailed.json`
   - Training logs (wandb or output.log)

## Expected Output Files

After running all steps, you should have:

```
data/
├── fine_tune.jsonl
└── dummy_rag_dataset.jsonl  # Optional: for RAG evaluation

report_generation/
├── results/
│   ├── icl_mistral_zero_shot.json
│   ├── icl_mistral_3_shot.json
│   ├── icl_gemma_zero_shot.json
│   ├── icl_gemma_3_shot.json
│   ├── finetuned_mistral.json
│   ├── finetuned_gemma.json
│   ├── rag_baseline_mistral.json
│   ├── rag_mistral.json
│   ├── rag_baseline_gemma.json
│   ├── rag_gemma.json
│   ├── rag_comparison_mistral.csv
│   ├── rag_comparison_mistral.md
│   ├── rag_comparison_gemma.csv
│   └── rag_comparison_gemma.md
└── outputs/
    ├── overall_comparison.csv
    ├── per_task_comparison.csv
    ├── comparison_tables.md
    ├── error_analysis.json
    └── error_analysis_detailed.json
```

## Time Estimates

- Fine-tuning (per model): 2-4 hours on GPU
- ICL evaluation (per method): 10-30 minutes
- Fine-tuned evaluation (per model): 10-30 minutes
- RAG evaluation (per model): 30-60 minutes (includes web search)
- Table generation: < 1 minute

**Total time**: ~6-12 hours (mostly fine-tuning)

## Tips

- Run fine-tuning in background or on a server
- ICL evaluations can run in parallel (different terminals)
- Check `report_generation/results/` after each evaluation
- Use `--max_samples 100` for quick testing

