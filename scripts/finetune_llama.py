"""
Fine-tune an LLM with QLoRA (PEFT) using the prepared JSONL dataset — UPDATED FOR LLAMA 7B.

Expected dataset: data/fine_tune.jsonl
Saves adapters/checkpoints under models/checkpoints/

Changes from Mistral version:
- Base model changed to Llama-2-7b-chat-hf
- Prompt format updated for Llama
- LoRA target modules updated
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

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# PROMPT FORMAT FOR LLAMA-2-CHAT
# ---------------------------------------------------------

def format_llama_prompt(example):
    """
    Llama 2 Chat format (system prompt optional):

    <s>[INST] <<SYS>>
    system prompt here
    <</SYS>>

    {user_message}
    [/INST] {assistant_response} </s>
    """

    user_prompt = example["instruction"]
    if example.get("input"):
        user_prompt += "\n" + example["input"]

    SYSTEM_PROMPT = "You are a helpful AI assistant."

    formatted = (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_prompt.strip()} [/INST] {example['output'].strip()} </s>"
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

    dataset = dataset.map(format_llama_prompt, remove_columns=["instruction", "input", "output"])

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
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------
# PEFT / LORA CONFIG FOR LLAMA-2
# ---------------------------------------------------------

def setup_peft_model(model, lora_r=32, lora_alpha=32, lora_dropout=0.1):
    logger.info("Configuring PEFT (LoRA) for Llama...")

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Llama supports these LoRA modules:
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
# LOAD YAML CONFIG
# ---------------------------------------------------------

def load_config(config_path):
    if not Path(config_path).exists():
        logger.warning("No config file found — using defaults.")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------
# MAIN TRAINING LOGIC
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-2-7B with QLoRA")
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    parser.add_argument("--dataset_path", type=str, default="data/fine_tune.jsonl")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")
    
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=3)
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

    # Apply config override
    cfg = load_config(args.config)
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)

    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error("Dataset file not found, exiting.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_path, args.max_samples, args.test_split
    )

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        use_4bit=not args.no_4bit
    )

    # Setup LoRA
    model = setup_peft_model(
        model, args.lora_r, args.lora_alpha, args.lora_dropout
    )

    model.config.use_cache = False

    # Tokenization
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=True,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        report_to="wandb",
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # Train
    trainer.train()

    # Save LoRA adapter
    final_path = output_dir / "Llama-7B-LoRA"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    logger.info(f"Finished! LoRA adapter saved to: {final_path}")


if __name__ == "__main__":
    main()

