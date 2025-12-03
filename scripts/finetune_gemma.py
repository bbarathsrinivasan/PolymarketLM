"""
Fine-tune an LLM with QLoRA (PEFT) using the prepared JSONL dataset — UPDATED FOR GEMMA 7B.

Expected dataset: data/fine_tune.jsonl
Saves adapters/checkpoints under models/checkpoints/

Changes from Llama version:
- Base model changed to google/gemma-7b-it
- Prompt format updated for Gemma dialogue format
- Everything else kept identical
"""

import transformers
import sys
print("SCRIPT Transformers version:", transformers.__version__)
print("SCRIPT Transformers path:", transformers.__file__)
print("SCRIPT Python:", sys.executable)

import os
# Fix tokenizers parallelism warning when using dataloader_num_workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import json
import math
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


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# ⭐ PROMPT FORMAT FOR GEMMA
# ---------------------------------------------------------

def format_gemma_prompt(example):
    """
    Gemma dialogue format:

    <bos><start_of_turn>user
    {instruction}
    <end_of_turn>
    <start_of_turn>model
    {response}
    <end_of_turn>
    """
    user_prompt = example["instruction"]
    if example.get("input"):
        user_prompt += "\n" + example["input"]

    formatted = (
        "<bos><start_of_turn>user\n"
        f"{user_prompt.strip()}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{example['output'].strip()}\n"
        "<end_of_turn>"
    )

    return {"text": formatted}


# ---------------------------------------------------------
# LOAD + PREPARE DATASET
# ---------------------------------------------------------

def load_and_prepare_dataset(dataset_path, max_samples=None, test_split=0.1):
    logger.info(f"Loading dataset from {dataset_path}")
    
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    if max_samples:
        logger.info(f"Limiting dataset to {max_samples} samples")
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # ⭐ Use Gemma formatting
    dataset = dataset.map(format_gemma_prompt, remove_columns=["instruction", "input", "output"])

    if test_split > 0:
        dataset = dataset.train_test_split(test_size=test_split, seed=42)
        logger.info(f"Train samples: {len(dataset['train'])}, Eval samples: {len(dataset['test'])}")
        return dataset["train"], dataset["test"]

    return dataset, None



# ---------------------------------------------------------
# LOAD MODEL + TOKENIZER
# ---------------------------------------------------------

def load_model_and_tokenizer(model_name, use_4bit=True):
    logger.info(f"Loading model: {model_name}")
    
    if use_4bit:
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
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ⭐ Gemma uses EOS as padding token
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer



# ---------------------------------------------------------
# PEFT / LORA CONFIG (same as Llama + Gemma)
# ---------------------------------------------------------

def setup_peft_model(model, lora_r=32, lora_alpha=32, lora_dropout=0.1):
    logger.info("Configuring PEFT (LoRA) for Gemma...")

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # ⭐ Gemma uses Llama-style projection names
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model



# ---------------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------------

def tokenize_dataset(examples, tokenizer, max_length=512):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False
    )



# ---------------------------------------------------------
# LOAD YAML
# ---------------------------------------------------------

def load_config(config_path):
    if not Path(config_path).exists():
        logger.warning("No config file found — using defaults.")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}



# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-7B with QLoRA")
    parser.add_argument("--config", type=str, default="config/training_config_llama.yaml")
    parser.add_argument("--dataset_path", type=str, default="data/fine_tune.jsonl")
    
    # ⭐ DEFAULT MODEL CHANGED
    parser.add_argument("--model_name", type=str, default="google/gemma-7b-it")

    parser.add_argument("--output_dir", type=str, default="models/checkpoints")

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of training steps (overrides num_epochs if set)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)

    args = parser.parse_args()

    cfg = load_config(args.config)
    for k, v in cfg.items():
        if hasattr(args, k):
            # Get the expected type from the argument parser
            for action in parser._actions:
                if action.dest == k:
                    # Convert value to expected type
                    if action.type is not None and v is not None:
                        try:
                            if action.type == float:
                                # Handle both numeric and string (including scientific notation)
                                if isinstance(v, str):
                                    v = float(v)
                                else:
                                    v = float(v)
                            elif action.type == int:
                                v = int(v)
                            elif action.type == bool:
                                if isinstance(v, str):
                                    v = v.lower() in ('true', '1', 'yes', 'on')
                                else:
                                    v = bool(v)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not convert {k}={v} (type: {type(v)}) to {action.type}: {e}, using as-is")
                    break
            setattr(args, k, v)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error("Dataset not found.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_path, args.max_samples, args.test_split
    )

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        use_4bit=not args.no_4bit
    )

    model = setup_peft_model(
        model, args.lora_r, args.lora_alpha, args.lora_dropout
    )

    model.config.use_cache = False
    
    # Enable gradient checkpointing to reduce memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "base_model") and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()

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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Use max_steps if provided, otherwise use num_epochs
    training_args_dict = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.save_steps if eval_dataset else None,
        "fp16": True,
        "warmup_steps": 100,
        "lr_scheduler_type": "cosine",
        "report_to": "wandb",
        "load_best_model_at_end": False,
        "greater_is_better": False,
        "gradient_checkpointing": True,  # Reduce memory usage
        "optim": "adamw_torch",  # Use torch optimizer for better memory efficiency
        "dataloader_num_workers": 4,  # Speed up data loading
        "dataloader_pin_memory": True,  # Speed up data transfer to GPU
        "remove_unused_columns": True,  # Reduce memory overhead
    }
    
    if args.max_steps is not None:
        training_args_dict["max_steps"] = args.max_steps
        logger.info(f"Using max_steps={args.max_steps} (overrides num_epochs)")
    else:
        training_args_dict["num_train_epochs"] = args.num_epochs
        logger.info(f"Using num_epochs={args.num_epochs}")
    
    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # Clear GPU cache before training
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")
    
    logger.info("Starting training...")
    trainer.train()

    # Save adapter
    final_path = output_dir / "Polymarket-Gemma-7B-LoRA"
    logger.info(f"Saving LoRA adapter to: {final_path}")
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

