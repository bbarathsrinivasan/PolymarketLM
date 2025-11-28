# Polymarket LLM

Project to preprocess Polymarket datasets and fine-tune an LLM using QLoRA for prediction market analysis.

## Overview

This project fine-tunes Mistral-7B-Instruct on Polymarket data to perform three tasks:
1. **Market Outcome Prediction**: Predict whether a market will resolve to "Yes" or "No"
2. **Manipulation Detection**: Identify if a market experienced manipulation
3. **User Classification**: Classify traders as "Noise Trader" or "Informed Trader"

## Structure

```
polymarket-llm/
├── data/
│   ├── raw/               # original CSV files (trades.csv, prices.csv, etc.)
│   ├── processed/         # intermediate data, e.g., merged by market_id
│   └── fine_tune.jsonl    # final training dataset in JSONL format
├── scripts/
│   ├── preprocess_data.py      # script to convert CSVs to JSONL
│   ├── finetune_qlora.py       # script to fine-tune the model with QLoRA (PEFT)
│   ├── test_inference.py       # script to test inference with fine-tuned model
│   └── create_sample_dataset.py # helper to create sample datasets for testing
├── models/
│   └── checkpoints/       # saved LoRA adapters or model checkpoints
├── requirements.txt       # Python dependencies
└── README.md              # this file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) with at least 16GB VRAM for full training
- For local testing, CPU or smaller GPU is acceptable

### Setup

1. **Clone the repository** (if applicable)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with `bitsandbytes`, you may need to install it separately:
```bash
pip install bitsandbytes
```

For CUDA support, ensure you have the appropriate CUDA toolkit installed.

3. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

4. **(Optional) Connect Weights & Biases for tracking**
   - Ensure `wandb` is installed (already in `requirements.txt`, but you can run `pip install wandb` if needed)
   - Log in once so Trainer can report metrics:

     ```bash
     wandb login
     ```

   - Configure logging by setting `report_to: wandb` (and optionally `run_name`) in `config/training_config.yaml`, or by passing `--report_to wandb --run_name <name>` via CLI. You can also set `WANDB_PROJECT` to control which project receives logs.

## Usage

### 1. Data Preprocessing

Convert raw CSV files into JSONL format for training:

```bash
python scripts/preprocess_data.py
```

This will:
- Load all market data from `data/raw/`
- Create training examples for all three tasks
- Output to `data/fine_tune.jsonl`

**Expected output**: ~1,954 examples (474 outcome prediction, 474 manipulation detection, ~1,006 user classification)

### 2. Local Testing (Dry-Run)

Before full training, test the pipeline with a small sample:

```bash
# Create a sample dataset (50 examples)
python scripts/create_sample_dataset.py --num_samples 50

# Run a quick training test
python scripts/finetune_qlora.py \
    --dataset_path data/fine_tune_sample.jsonl \
    --max_samples 50 \
    --num_epochs 1 \
    --save_steps 10 \
    --logging_steps 5
```

This will verify that:
- Model loads correctly
- Dataset is formatted properly
- Training loop works without errors

### 3. Full Fine-Tuning

Train the model on the full dataset:

```bash
python scripts/finetune_qlora.py \
    --dataset_path data/fine_tune.jsonl \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --output_dir models/checkpoints
```

> **Using YAML config**: The script automatically reads defaults from `config/training_config.yaml`.  
> Override values there or pass CLI arguments to supersede the config.

**Training parameters**:
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Per-device batch size (default: 4)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 4, effective batch size = 16)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--max_samples`: Limit dataset size for testing
- `--max_length`: Maximum sequence length (default: 512)

**Expected training time**: 
- Local GPU (RTX 3090/4090): ~2-4 hours for full dataset
- EC2 g5.2xlarge: ~3-5 hours
- CPU: Not recommended (will take days)

### 4. Test Inference

After training, test the model:

```bash
python scripts/test_inference.py \
    --adapter_path models/checkpoints/Polymarket-7B-LoRA
```

Or test with a custom prompt:

```bash
python scripts/test_inference.py \
    --adapter_path models/checkpoints/Polymarket-7B-LoRA \
    --custom_prompt "Predict the market outcome given the historical data.|Market ID: 12345\nQuestion: Will X happen?\nPrice History: 0.5, 0.6, 0.7"
```

## Resource Requirements

### Minimum (Local Testing)
- **GPU**: 8GB VRAM (with 4-bit quantization)
- **RAM**: 16GB
- **Storage**: 20GB (for model + dataset)

### Recommended (Full Training)
- **GPU**: 16GB+ VRAM (RTX 3090, A100, etc.)
- **RAM**: 32GB
- **Storage**: 50GB

### EC2 Instance Recommendations
- **g5.2xlarge**: 1x A10G (24GB VRAM) - Good for training
- **g5.4xlarge**: 1x A10G (24GB VRAM) - Better performance
- **p3.2xlarge**: 1x V100 (16GB VRAM) - Older but cheaper

See `EC2_SETUP.md` for detailed EC2 deployment instructions.

## Troubleshooting

### CUDA Out of Memory
If you encounter OOM errors:
- Reduce `--batch_size` (try 2 or 1)
- Increase `--gradient_accumulation_steps` to maintain effective batch size
- Reduce `--max_length` (try 256 or 384)
- Ensure 4-bit quantization is enabled (default)

### Model Loading Issues
- Verify you have internet connection (model downloads from Hugging Face)
- Check available disk space
- Ensure you have sufficient RAM/VRAM

### Training Too Slow
- Enable mixed precision training (fp16) - already enabled by default
- Use a GPU with more VRAM
- Reduce `--max_length` if sequences are shorter
- Consider using a smaller model for testing

## Output

After training, you'll find:
- **LoRA adapter**: `models/checkpoints/Polymarket-7B-LoRA/`
  - Contains only the fine-tuned adapter weights (~100-200MB)
  - Requires base model to load
- **Training checkpoints**: `models/checkpoints/checkpoint-*/`
  - Intermediate checkpoints saved during training
  - Can be used to resume training

## Next Steps

1. **Evaluate on held-out data**: Create a test set and measure accuracy
2. **Hyperparameter tuning**: Experiment with learning rates, LoRA rank, etc.
3. **Deploy to production**: See `EC2_SETUP.md` for deployment
4. **Integrate with Polymarket API**: Use the fine-tuned model for real-time predictions

## License

[Add your license here]

## Acknowledgments

- Mistral AI for the base model
- Hugging Face for Transformers, PEFT, and datasets libraries
- Polymarket for the prediction market data
