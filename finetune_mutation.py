import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

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

    # Configuration
    model_path = "/storage/ice-shared/vip-vvk/llm_storage/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
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

    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    print(f"Loading model from {model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Enable gradient checkpointing and k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1, # Low batch size for 32B model
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        optim="paged_adamw_32bit", # Memory efficient optimizer
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
    )

    # Formatting function for SFTTrainer
    def formatting_func(examples):
        return examples["text"]
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=2048, # Adjust based on prompt length
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()
