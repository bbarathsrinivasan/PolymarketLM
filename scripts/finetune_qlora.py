"""
Fine-tune an LLM with QLoRA (PEFT) using the prepared JSONL dataset.

Expected dataset: data/fine_tune.jsonl
Saves adapters/checkpoints under models/checkpoints/

This script:
- Loads Mistral-7B-Instruct in 4-bit mode using BitsAndBytes
- Configures LoRA adapters using PEFT
- Formats dataset into Mistral Instruct format
- Trains the model and saves LoRA adapters
"""

import transformers
import sys
print("SCRIPT Transformers version:", transformers.__version__)
print("SCRIPT Transformers path:", transformers.__file__)
print("SCRIPT Python:", sys.executable)
import os
import argparse
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import logging
import yaml
import math

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_mistral_prompt(example):
    """Format example into Mistral Instruct format."""
    user_prompt = example["instruction"]
    if example.get("input"):
        user_prompt += "\n" + example["input"]
    
    # Mistral instruct format: <s>[INST] prompt [/INST] response </s>
    formatted = f"<s>[INST] {user_prompt.strip()} [/INST] {example['output'].strip()} </s>"
    return {"text": formatted}


def load_and_prepare_dataset(dataset_path, max_samples=None, test_split=0.1):
    """Load JSONL dataset and format for training."""
    logger.info(f"Loading dataset from {dataset_path}")
    
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    if max_samples:
        logger.info(f"Limiting dataset to {max_samples} samples for testing")
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Format examples into Mistral prompt format
    dataset = dataset.map(format_mistral_prompt, remove_columns=["instruction", "input", "output"])
    
    # Split into train and validation if needed
    if test_split > 0:
        dataset = dataset.train_test_split(test_size=test_split, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
        return train_dataset, eval_dataset
    else:
        logger.info(f"Using all {len(dataset)} samples for training (no validation split)")
        return dataset, None


def load_model_and_tokenizer(model_name, use_4bit=True):
    """Load model in 4-bit mode and tokenizer."""
    logger.info(f"Loading model: {model_name}")
    
    if use_4bit:
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # Load in full precision (for testing without quantization)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def setup_peft_model(model, lora_r=32, lora_alpha=32, lora_dropout=0.1):
    """Configure and apply LoRA to the model."""
    logger.info("Setting up PEFT (LoRA) configuration")
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def create_data_collator(tokenizer, max_length=512):
    """Create data collator for training."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )


def tokenize_dataset(examples, tokenizer, max_length=512):
    """Tokenize dataset examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def load_config(config_path):
    """Load YAML configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found at {config_file}. Using defaults/CLI arguments.")
        return {}
    logger.info(f"Loading config from {config_file}")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral-7B with QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to YAML config file with training arguments"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/fine_tune.jsonl",
        help="Path to JSONL dataset file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/checkpoints",
        help="Output directory for checkpoints and adapters"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit dataset size for testing (e.g., 50 for quick test)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides num_epochs if set)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Validation split ratio (0 to disable)"
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization (for testing)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log every N steps"
    )
    
    args = parser.parse_args()

    # Load config defaults
    config_values = load_config(args.config)
    if config_values:
        for key, value in config_values.items():
            if not hasattr(args, key):
                continue
            current_value = getattr(args, key)
            default_value = parser.get_default(key)
            # If arg set to default (or None), override with config value
            if current_value == default_value or current_value is None:
                # Get the expected type from the argument parser
                for action in parser._actions:
                    if action.dest == key:
                        # Convert value to expected type
                        if action.type is not None and value is not None:
                            try:
                                if action.type == float:
                                    # Handle both numeric and string (including scientific notation)
                                    if isinstance(value, str):
                                        value = float(value)
                                    else:
                                        value = float(value)
                                elif action.type == int:
                                    value = int(value)
                                elif action.type == bool:
                                    if isinstance(value, str):
                                        value = value.lower() in ('true', '1', 'yes', 'on')
                                    else:
                                        value = bool(value)
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Could not convert {key}={value} (type: {type(value)}) to {action.type}: {e}, using as-is")
                        break
                setattr(args, key, value)
    
    # Validate paths
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Starting QLoRA Fine-Tuning")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Max samples: {args.max_samples or 'All'}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("=" * 60)
    
    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(
        str(dataset_path),
        max_samples=args.max_samples,
        test_split=args.test_split
    )
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model_name,
            use_4bit=not args.no_4bit
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Make sure you have sufficient GPU memory and the model is accessible")
        raise
    base_model_path = output_dir / "base_model"
    logger.info(f"Saving base model to {base_model_path}")
    base_model_path.mkdir(parents=True, exist_ok=True)

    # Save base model EXACTLY as loaded
    model.base_model.save_pretrained(str(base_model_path))
    tokenizer.save_pretrained(str(base_model_path))
    logger.info("Base model saved successfully.")
    
    # Setup PEFT
    model = setup_peft_model(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Enable gradient checkpointing to reduce memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "base_model") and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_dataset(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"]
    )
    
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda x: tokenize_dataset(x, tokenizer, args.max_length),
            batched=True,
            remove_columns=["text"]
        )
    
    # Create data collator
    data_collator = create_data_collator(tokenizer, args.max_length)

    from transformers import TrainingArguments
    import inspect
    print("TrainingArguments loaded from:", inspect.getfile(TrainingArguments))
    
    # Setup training arguments
    # If max_steps is set, use it instead of num_epochs
    training_args_dict = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size if eval_dataset else None,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "fp16": True,  # Mixed precision training
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.save_steps if eval_dataset else None,
        "save_total_limit": 2,
        "load_best_model_at_end": False,
        "metric_for_best_model": None,
        "report_to": "wandb",  # Change to "wandb" if using Weights & Biases
        "warmup_steps": 100,
        "lr_scheduler_type": "cosine",
    }
    
    # Use max_steps if provided, otherwise use num_epochs
    if args.max_steps is not None:
        training_args_dict["max_steps"] = args.max_steps
        logger.info(f"Using max_steps={args.max_steps} (overrides num_epochs)")
    else:
        training_args_dict["num_train_epochs"] = args.num_epochs
        logger.info(f"Using num_epochs={args.num_epochs}")
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
    
    # Train
    logger.info("Starting training...")
    try:
        trainer.train()
        # ---- Run evaluation after training (if we have a validation set) ----
        if eval_dataset is not None:
            logger.info("Running evaluation on validation set...")
            eval_metrics = trainer.evaluate()

            eval_loss = eval_metrics.get("eval_loss")
            if eval_loss is not None:
                try:
                    eval_metrics["perplexity"] = math.exp(eval_loss)
                except OverflowError:
                    eval_metrics["perplexity"] = float("inf")

            logger.info(f"Eval metrics: {eval_metrics}")
            # Log on wandb as well
            if trainer.args.report_to == "wandb":
                import wandb
                wandb.log(eval_metrics)

        else:
            logger.info("No eval dataset — skipping evaluation.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CUDA out of memory! Try:")
            logger.error("  - Reducing batch_size")
            logger.error("  - Increasing gradient_accumulation_steps")
            logger.error("  - Reducing max_length")
            logger.error("  - Using a smaller model")
        raise
    
    # Save final model
    final_model_path = output_dir / "Polymarket-7B-LoRA"
    logger.info(f"Saving final model to {final_model_path}")
    model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    from peft import PeftModel

    logger.info("Merging LoRA adapter into base model...")

    # Load base model again in full precision (must be FP16 or FP32)
    merged_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    lora_model = PeftModel.from_pretrained(merged_model, final_model_path)
    merged_model = lora_model.merge_and_unload()   # THIS merges LoRA → base

    merged_path = output_dir / "Polymarket-7B-MERGED"
    logger.info(f"Saving merged full model to: {merged_path}")

    merged_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))

    logger.info("Merged model saved successfully.")
    
    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
