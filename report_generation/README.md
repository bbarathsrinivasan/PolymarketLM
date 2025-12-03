# Report Generation Guide

This folder contains all scripts, configs, and documentation needed to generate the project report.

## 📁 Folder Structure

```
report_generation/
├── scripts/              # Evaluation and analysis scripts
├── configs/              # Configuration files
├── results/              # JSON result files (generated)
├── outputs/               # Tables, graphs, summaries (generated)
├── README.md             # This file
├── report_creation.md     # Process documentation
└── report.md             # Report template (fill with results)
```

## 🚀 Execution Order

### Step 1: Fine-tune Models (if not already done)

**Mistral:**
```bash
python scripts/finetune_qlora.py \
    --config report_generation/configs/finetuning_config_mistral.yaml \
    --output_dir models/checkpoints/Polymarket-7B-LoRA
```

**Gemma:**
```bash
python scripts/finetune_gemma.py \
    --config report_generation/configs/finetuning_config_gemma.yaml \
    --output_dir models/checkpoints/Polymarket-Gemma-7B-LoRA
```

### Step 2: Evaluate In-Context Learning (ICL)

#### Zero-shot ICL

**Mistral Zero-shot:**
```bash
python report_generation/scripts/evaluate_icl_mistral.py \
    --num_shots 0 \
    --max_samples 1000 \
    --output_dir report_generation/results
```

**Gemma Zero-shot:**
```bash
python report_generation/scripts/evaluate_icl_gemma.py \
    --num_shots 0 \
    --max_samples 1000 \
    --output_dir report_generation/results
```

#### Few-shot ICL (3-shot)

**Mistral Few-shot:**
```bash
python report_generation/scripts/evaluate_icl_mistral.py \
    --num_shots 3 \
    --selection random \
    --max_samples 1000 \
    --output_dir report_generation/results
```

**Gemma Few-shot:**
```bash
python report_generation/scripts/evaluate_icl_gemma.py \
    --num_shots 3 \
    --selection random \
    --max_samples 1000 \
    --output_dir report_generation/results
```

### Step 3: Evaluate Fine-tuned Models

**Mistral Fine-tuned:**
```bash
python report_generation/scripts/evaluate_finetuned_mistral.py \
    --adapter_path models/checkpoints/Polymarket-7B-LoRA \
    --max_samples 1000 \
    --output_dir report_generation/results
```

**Gemma Fine-tuned:**
```bash
python report_generation/scripts/evaluate_finetuned_gemma.py \
    --adapter_path models/checkpoints/Polymarket-Gemma-7B-LoRA \
    --max_samples 1000 \
    --output_dir report_generation/results
```

### Step 4: Generate Comparison Tables

```bash
python report_generation/scripts/generate_comparison_tables.py \
    --results_dir report_generation/results \
    --output_dir report_generation/outputs
```

This will generate:
- `overall_comparison.csv` - Overall accuracy comparison
- `per_task_comparison.csv` - Per-task accuracy comparison
- `comparison_tables.md` - Markdown tables for report
- `error_analysis.json` - Error analysis summary

### Step 5: Run Error Analysis

```bash
python report_generation/scripts/error_analysis.py \
    --results_dir report_generation/results \
    --output_file report_generation/outputs/error_analysis_detailed.json
```

This will generate:
- Detailed error analysis with examples
- Common failure patterns
- Method comparison (most/least robust)
- Task-specific error breakdown

### Step 6: Update Report

1. Copy results from `report_generation/outputs/comparison_tables.md` to `report.md`
2. Fill in additional sections in `report.md` with your findings
3. Add training/validation loss curves from wandb logs
4. Add error analysis examples from `error_analysis_detailed.json`
5. Update all `[TO FILL]` placeholders with actual results

## 📊 Understanding ICL Experiments

### Zero-shot vs Few-shot

- **Zero-shot (0-shot)**: No examples provided, model relies on pre-training
- **Few-shot (3-shot)**: 3 examples provided in the prompt to guide the model

### Selection Methods

- **random**: Randomly select examples from training set
- **by_task**: Select examples that match the task type (e.g., outcome prediction examples for outcome prediction queries)

### Expected Output Files

After running all evaluations, you should have:

```
report_generation/results/
├── icl_mistral_zero_shot.json
├── icl_mistral_3_shot.json
├── icl_gemma_zero_shot.json
├── icl_gemma_3_shot.json
├── finetuned_mistral.json
└── finetuned_gemma.json
```

## 🔧 Configuration

All configurations are in `configs/`:
- `icl_config.yaml` - ICL experiment settings
- `finetuning_config_mistral.yaml` - Mistral fine-tuning settings
- `finetuning_config_gemma.yaml` - Gemma fine-tuning settings

## 📝 Results Documentation

After running all scripts:

1. **Check `report_generation/results/`** for JSON result files
2. **Check `report_generation/outputs/`** for comparison tables
3. **Update `report.md`** with your results
4. **Add training curves** from wandb (if using wandb)

## 🧪 Testing Setup

Before running experiments, test your setup:

```bash
python report_generation/scripts/test_scripts.py
```

This will verify:
- All required dependencies are installed
- Scripts can be imported (syntax check)
- Config files exist
- Directories are ready
- Dataset is available

## 🐛 Troubleshooting

### Missing dependencies
- Install with: `pip install -r requirements.txt`
- If using GPU: Ensure CUDA toolkit is installed

### Model not found
- Ensure models are fine-tuned first (Step 1)
- Check adapter paths in evaluation scripts
- Verify checkpoint directories exist

### CUDA out of memory
- Use `--no_4bit` flag to disable 4-bit quantization (uses more memory)
- Reduce batch size in config files
- Reduce max_length in config files

### No results generated
- Check that dataset path is correct
- Ensure test split is properly configured
- Verify JSON result files are created in `results/` folder
- Check that models are loaded successfully (check for error messages)

## 📚 Additional Notes

- All scripts use seed=42 for reproducibility
- Test split is 10% by default (configurable)
- Results are saved in JSON format for easy parsing
- Comparison tables are generated in both CSV and Markdown formats

