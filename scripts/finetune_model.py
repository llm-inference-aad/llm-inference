#!/usr/bin/env python3
"""
Finetune a code generation model for successful mutations.

Usage:
    python scripts/finetune_model.py --model deepseek-ai/deepseek-coder-6.7b-instruct
    python scripts/finetune_model.py --model google/gemma-2-9b-it --use-lora
"""

import sys
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse


def load_dataset(jsonl_file):
    """Load training data from JSONL file"""
    examples = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    formatted = []
    for ex in examples:
        text = f"{ex['prompt']}\n\n{ex['completion']}" # Format as instruction-response pairs
        formatted.append({"text": text})
    
    return Dataset.from_list(formatted)


def prepare_model_and_tokenizer(model_name, use_lora=True):
    """Load model and tokenizer"""
    print(f"\n[INFO] Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 8-bit for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_8bit=True if use_lora else False,
        trust_remote_code=True
    )
    
    # Apply LoRA -- this is pretty nice to have for efficient finetuning
    if use_lora:
        print("[INFO] Applying LoRA configuration")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # r = LoRA rank
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Adjust based on model
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def tokenize_dataset(dataset, tokenizer, max_length=2048):
    """Tokenize the dataset"""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized


def finetune(
    model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
    dataset_file="finetuning_dataset.jsonl",
    output_dir="finetuned_model",
    use_lora=True,
    epochs=3,
    batch_size=4,
    learning_rate=2e-5
):
    """Main finetuning function"""
    
    # Load dataset
    print(f"[INFO] Loading dataset from {dataset_file}")
    dataset = load_dataset(dataset_file)
    print(f"[INFO] Dataset size: {len(dataset)} examples")
    
    # Split train/val
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Load model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(model_name, use_lora)
    
    # Tokenize
    print("[INFO] Tokenizing dataset...")
    tokenized_train = tokenize_dataset(dataset["train"], tokenizer)
    tokenized_val = tokenize_dataset(dataset["test"], tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    # Train
    print("\n[INFO] Starting training...")
    trainer.train()
    
    # Save final model
    print(f"\n[SUCCESS] Training complete!")
    print(f"Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n[NEXT STEPS]")
    print(f"1. Update MODEL_PATH in .env to point to: {Path(output_dir).absolute()}")
    print(f"2. Set LLM_MODEL='local_server' in src/cfg/constants.py")
    print(f"3. Start server: sbatch server.sh")
    print(f"4. Run experiments with your finetuned model!")


def main():
    parser = argparse.ArgumentParser(description="Finetune code generation model")
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-6.7b-instruct",
                       help="Base model to finetune")
    parser.add_argument("--dataset", default="finetuning_dataset.jsonl",
                       help="Training dataset (JSONL)")
    parser.add_argument("--output", default="finetuned_model",
                       help="Output directory")
    parser.add_argument("--use-lora", action="store_true", default=True,
                       help="Use LoRA for efficient finetuning")
    parser.add_argument("--full-finetune", action="store_true",
                       help="Full model finetuning (not LoRA)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    
    args = parser.parse_args()
    
    use_lora = not args.full_finetune
    
    print("=" * 80)
    print("MODEL FINETUNING")
    print("=" * 80)
    print(f"Base model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Method: {'LoRA (efficient)' if use_lora else 'Full finetuning'}")
    print(f"Epochs: {args.epochs}")
    print("=" * 80)
    
    finetune(
        model_name=args.model,
        dataset_file=args.dataset,
        output_dir=args.output,
        use_lora=use_lora,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()

