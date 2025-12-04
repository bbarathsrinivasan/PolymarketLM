# Report Creation Process Documentation

This document explains the entire process of generating the project report, including all experiments, evaluations, and analysis steps.

## Overview

This project evaluates LLMs (Mistral-7B and Gemma-7B) on Polymarket prediction tasks using two main approaches:
1. **In-Context Learning (ICL)**: Zero-shot and few-shot prompting
2. **Fine-tuning**: QLoRA-based parameter-efficient fine-tuning

## Project Structure

```
PolymarketLM/
├── data/
│   ├── fine_tune.jsonl          # Full training dataset
│   └── fine_tune_sample.jsonl   # Sample dataset for testing
├── scripts/
│   ├── finetune_qlora.py        # Fine-tune Mistral
│   ├── finetune_gemma.py        # Fine-tune Gemma
│   └── preprocess_data.py       # Data preprocessing
├── models/checkpoints/          # Saved model adapters
├── report_generation/
│   ├── scripts/                 # Evaluation scripts
│   ├── configs/                 # Configuration files
│   ├── results/                 # JSON result files
│   └── outputs/                 # Generated tables/graphs
└── wandb/                       # Training logs (if using wandb)
```

## Task Description

### Three Tasks

1. **Market Outcome Prediction**: Predict whether a market will resolve to "Yes" or "No"
2. **Manipulation Detection**: Identify if a market experienced manipulation (Yes/No)
3. **User Classification**: Classify traders as "Noise Trader" or "Informed Trader"

### Dataset

- **Source**: Polymarket prediction market data
- **Format**: JSONL with `instruction`, `input`, `output` fields
- **Split**: 90% train, 10% test (seed=42)
- **Total Examples**: ~1,954 (474 outcome prediction, 474 manipulation detection, ~1,006 user classification)

## Experiment Pipeline

### Phase 1: Data Preparation

1. **Preprocess raw data**:
   ```bash
   python scripts/preprocess_data.py
   ```
   - Loads CSVs from `data/raw/`
   - Creates training examples for all three tasks
   - Outputs to `data/fine_tune.jsonl`

2. **Verify dataset**:
   - Check dataset size and distribution
   - Verify format is correct

### Phase 2: Fine-tuning (if not already done)

#### Mistral Fine-tuning

```bash
python scripts/finetune_qlora.py \
    --config report_generation/configs/finetuning_config_mistral.yaml \
    --output_dir models/checkpoints/Polymarket-7B-LoRA
```

**Hyperparameters**:
- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- Method: QLoRA (4-bit quantization)
- LoRA rank: 32
- LoRA alpha: 32
- Learning rate: 0.0005
- Batch size: 4
- Gradient accumulation: 16
- Epochs: 2
- Max length: 192

#### Gemma Fine-tuning

```bash
python scripts/finetune_gemma.py \
    --config report_generation/configs/finetuning_config_gemma.yaml \
    --output_dir models/checkpoints/Polymarket-Gemma-7B-LoRA
```

**Hyperparameters**:
- Base model: `google/gemma-7b-it`
- Method: QLoRA (4-bit quantization)
- LoRA rank: 32
- LoRA alpha: 32
- Learning rate: 0.0005
- Batch size: 4
- Gradient accumulation: 16
- Epochs: 2
- Max length: 192

**Note**: Training logs are saved to wandb (if configured) or can be extracted from training output.

### Phase 3: In-Context Learning Evaluation

#### Mistral ICL

**Zero-shot**:
```bash
python report_generation/scripts/evaluate_icl_mistral.py \
    --num_shots 0 \
    --output_dir report_generation/results
```

**Few-shot (3-shot)**:
```bash
python report_generation/scripts/evaluate_icl_mistral.py \
    --num_shots 3 \
    --selection random \
    --output_dir report_generation/results
```

**What happens**:
- Loads base Mistral model
- For zero-shot: No examples provided
- For few-shot: Selects k examples from training set (random or by task)
- Formats prompt in Mistral Instruct format: `<s>[INST] {prompt} [/INST]`
- Generates response and extracts prediction
- Calculates accuracy per task and overall

**Output**: `report_generation/results/icl_mistral_zero_shot.json` and `icl_mistral_3_shot.json`

#### Gemma ICL

**Zero-shot**:
```bash
python report_generation/scripts/evaluate_icl_gemma.py \
    --num_shots 0 \
    --output_dir report_generation/results
```

**Few-shot (3-shot)**:
```bash
python report_generation/scripts/evaluate_icl_gemma.py \
    --num_shots 3 \
    --selection random \
    --output_dir report_generation/results
```

**What happens**:
- Loads base Gemma model
- Formats prompt in Gemma-IT format: `<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`
- Rest is same as Mistral

**Output**: `report_generation/results/icl_gemma_zero_shot.json` and `icl_gemma_3_shot.json`

### Phase 4: Fine-tuned Model Evaluation

#### Mistral Fine-tuned

```bash
python report_generation/scripts/evaluate_finetuned_mistral.py \
    --adapter_path models/checkpoints/Polymarket-7B-LoRA \
    --output_dir report_generation/results
```

**What happens**:
- Loads base Mistral model
- Applies LoRA adapter from checkpoint
- Evaluates on test set
- Calculates accuracy, loss, and perplexity per task

**Output**: `report_generation/results/finetuned_mistral.json`

#### Gemma Fine-tuned

```bash
python report_generation/scripts/evaluate_finetuned_gemma.py \
    --adapter_path models/checkpoints/Polymarket-Gemma-7B-LoRA \
    --output_dir report_generation/results
```

**Output**: `report_generation/results/finetuned_gemma.json`

### Phase 5: Results Aggregation and Analysis

```bash
python report_generation/scripts/generate_comparison_tables.py \
    --results_dir report_generation/results \
    --output_dir report_generation/outputs
```

**What this generates**:
1. **Overall Comparison Table** (`overall_comparison.csv`): Accuracy and perplexity for all 4 methods
2. **Per-Task Comparison Table** (`per_task_comparison.csv`): Accuracy per task for each method
3. **Markdown Tables** (`comparison_tables.md`): Formatted tables for report
4. **Error Analysis** (`error_analysis.json`): Summary of errors with examples

## Evaluation Metrics

### Accuracy
- **Definition**: Exact match between predicted and ground truth labels
- **Calculation**: `correct_predictions / total_predictions`
- **Per-task**: Calculated separately for each task type
- **Overall**: Weighted average across all tasks

### Perplexity
- **Definition**: Exponential of average cross-entropy loss
- **Calculation**: `exp(avg_loss)`
- **Interpretation**: Lower is better (measures model confidence)

### Loss
- **Definition**: Cross-entropy loss on target tokens
- **Calculation**: Standard language modeling loss
- **Used for**: Perplexity calculation and training monitoring

## Prompt Formats

### Mistral Instruct Format

**Training/ICL**:
```
<s>[INST] {instruction}
{input} [/INST] {output} </s>
```

**Inference**:
```
<s>[INST] {instruction}
{input} [/INST]
```

### Gemma-IT Format

**Training/ICL**:
```
<start_of_turn>user
{instruction}
{input}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>
```

**Inference**:
```
<start_of_turn>user
{instruction}
{input}<end_of_turn>
<start_of_turn>model
```

## Prediction Extraction

### For Yes/No Tasks (Outcome Prediction, Manipulation Detection)

1. Search for "yes" or "no" in response (case-insensitive)
2. If found, return capitalized version
3. Otherwise, return first word of response

### For Classification Tasks (User Classification)

1. Search for "informed trader" or "noise trader" in response
2. Return exact match if found
3. Otherwise, return empty string (treated as error)

## Error Analysis Process

1. **Collect all incorrect predictions** from all methods
2. **Group by task type** to identify which tasks are hardest
3. **Compare across methods** to see if all fail on same examples
4. **Extract examples** for qualitative analysis
5. **Identify patterns**: 
   - Common failure modes
   - Task-specific challenges
   - Model-specific weaknesses

## Training/Validation Loss Curves

If using wandb:
1. Log in to wandb dashboard
2. Navigate to your project
3. Find the training run
4. Export loss curves (training vs validation)
5. Add to report

If not using wandb:
1. Check training logs in `wandb/run-*/files/output.log`
2. Extract loss values manually
3. Plot using matplotlib or similar

## Reproducibility

### Random Seeds
- **Data splitting**: seed=42
- **Example selection**: seed=42
- **Model initialization**: PyTorch default (no explicit seed for model weights)

### Environment
- **Python**: 3.8+
- **PyTorch**: 2.0.0+
- **Transformers**: 4.35.0+
- **CUDA**: (if using GPU) - document version
- **GPU**: (if using) - document model

### Hyperparameters
All hyperparameters are documented in:
- `report_generation/configs/finetuning_config_mistral.yaml`
- `report_generation/configs/finetuning_config_gemma.yaml`
- `report_generation/configs/icl_config.yaml`

## Common Issues and Solutions

### CUDA Out of Memory
- Use 4-bit quantization (default)
- Reduce batch size
- Reduce max_length
- Use gradient accumulation

### Model Not Found
- Ensure models are downloaded from HuggingFace
- Check internet connection
- Verify model names in configs

### Adapter Not Found
- Ensure fine-tuning completed successfully
- Check adapter path in evaluation scripts
- Verify checkpoint directory exists

### No Results Generated
- Check that dataset path is correct
- Verify test split is configured
- Ensure JSON result files are created

## Next Steps After Running Experiments

1. **Review Results**: Check `report_generation/results/` for JSON files
2. **Generate Tables**: Run comparison script
3. **Update Report**: Fill in `report.md` with results
4. **Add Visualizations**: Include loss curves, accuracy plots
5. **Error Analysis**: Document failure cases and patterns
6. **Conclusion**: Summarize findings and insights

## Additional Experiments (Optional)

### Retrieval-Augmented Generation (RAG)

The project includes `scripts/inference_with_news.py` for RAG experiments:
- Augments prompts with news search results
- Compares with/without news context
- Can be used as additional experiment section

### RAG Evaluation Script

A comprehensive RAG evaluation script has been created: `report_generation/scripts/evaluate_rag_integration.py`

**Features**:
- Evaluates both Mistral and Gemma fine-tuned models
- Compares baseline (no RAG) vs RAG performance
- Uses search/news retrieval for augmentation
- Generates responses with source citations and links
- Calculates all metrics (accuracy, loss, perplexity, per-task)
- Generates comparison tables (CSV and Markdown)

**Usage**:
```bash
# Evaluate both models with RAG (200 examples)
python report_generation/scripts/evaluate_rag_integration.py \
    --num_examples 200 \
    --search_provider duckduckgo \
    --num_search_results 5

# Evaluate only Mistral
python report_generation/scripts/evaluate_rag_integration.py \
    --models mistral \
    --num_examples 200

# Evaluate only Gemma
python report_generation/scripts/evaluate_rag_integration.py \
    --models gemma \
    --num_examples 200
```

**Output**:
- `rag_baseline_mistral.json` - Baseline Mistral results
- `rag_mistral.json` - RAG Mistral results (with source citations)
- `rag_baseline_gemma.json` - Baseline Gemma results
- `rag_gemma.json` - RAG Gemma results (with source citations)
- `rag_comparison_mistral.csv/md` - Mistral comparison tables
- `rag_comparison_gemma.csv/md` - Gemma comparison tables

**Key Implementation Details**:
- Uses `integration.prompt_augmenter.augment_prompt_with_search()` for RAG
- Extracts market questions from input text
- Formats responses with source citations: `[1] Title - Source (link)`
- Prioritizes outcome prediction examples (benefit most from external context)
- Caches search results to avoid redundant API calls
- Handles cases where no search results are found (falls back to baseline)

**For Report Section 4 (Additional Experiment)**:
- Use comparison tables from `rag_comparison_*.md` files
- Include examples of responses with source citations
- Compare baseline vs RAG accuracy improvements
- Analyze which tasks benefit most from RAG

## Report Sections Checklist

- [ ] Task & Dataset description
- [ ] Ethical considerations
- [ ] Training data formulation
- [ ] Evaluation method
- [ ] ICL prompt designs (both models)
- [ ] Fine-tuning setups (both models)
- [ ] ICL results (tables)
- [ ] Fine-tuning results (tables + loss curves)
- [ ] Error analysis (qualitative + quantitative)
- [ ] Best system selection
- [ ] Additional experiment (RAG evaluation)
  - [ ] Baseline vs RAG comparison tables
  - [ ] Accuracy improvements per task
  - [ ] Example responses with source citations
  - [ ] Analysis of which tasks benefit most from RAG
- [ ] Reproducibility details
- [ ] Conclusion

