import os
import json
import argparse
import torch

# Set HuggingFace cache to scratch directory (avoid home quota issues)
os.environ["HF_HOME"] = "/home/hice1/rmanimaran8/scratch/.cache/huggingface"
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

def load_and_format_data(dataset_path):
    """
    Loads dataset.json and formats it for Conditional SFT.
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Convert to HuggingFace Dataset
    # Format:
    # ### Instruction:
    # {prompt}
    #
    # ### Target Fitness:
    # {fitness}
    #
    # ### Response:
    # {generated_text}
    
    formatted_data = []
    for entry in data:
        prompt = entry.get('prompt', '')
        fitness = entry.get('fitness', '')
        response = entry.get('generated_text', '')
        
        text = f"### Instruction:\n{prompt}\n\n### Target Fitness:\n{fitness}\n\n### Response:\n{response}"
        formatted_data.append({"text": text})
        
    return Dataset.from_list(formatted_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Verify setup without loading model")
    parser.add_argument("--output-dir", type=str, default=os.path.expanduser("~/deepseek-mutation-finetune"), help="Output directory for checkpoints")
    args = parser.parse_args()

    # Configuration - Using GPT-2 for testing (124M params - reliable small model)
    model_path = "gpt2"
    dataset_path = "dataset.json"
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_and_format_data(dataset_path)
        print(f"Loaded {len(dataset)} examples.")
        if len(dataset) > 0:
            print("Sample entry:\n", dataset[0]['text'][:500], "...")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.dry_run:
        print("Dry run complete. Dataset loaded successfully.")
        print(f"Model would be saved to: {output_dir}")
        return

    # Skip quantization for tiny model - not needed for 70M params
    print(f"Loading model from {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # LoRA Config - target modules for GPT-2 architecture
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc"] 
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Training Arguments using SFTConfig (new TRL API)
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        optim="adamw_torch",  # Standard optimizer for small model
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        max_length=512,  # Shorter for GPT-2
        dataset_text_field="text",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()
