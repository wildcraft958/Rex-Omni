import os
import torch
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from qwen_vl_utils import process_vision_info
import argparse

def train():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with LoRA")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL data file")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--model_id", type=str, default="IDEA-Research/Rex-Omni", help="Model ID")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1) # Small batch size for VRAM efficiency
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--use_qlora", action="store_true", help="Use 4-bit quantization")
    args = parser.parse_args()

    # 1. Load Processor
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=256*28*28, max_pixels=1280*28*28)

    # 2. Load Dataset
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    # 3. Data Formatting Function
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True) for example in examples]
        image_inputs, video_inputs = process_vision_info(examples)
        
        # Tokenize
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Create labels (same as input_ids, but masked for padding)
        # Note: This is a simplified labeling. For strict instruction tuning, you might want to mask user prompts.
        # But for general SFT, this often works or the Trainer handles it if using DataCollatorForCompletionOnlyLM (from trl)
        # Here we use standard causal LM training where labels = input_ids
        
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100 # Ignore padding in loss
        
        # Optional: Mask image tokens if needed, but usually fine to train on them or they are handled by model
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        labels[labels == image_token_id] = -100  # Usually we don't want to predict image tokens

        inputs["labels"] = labels
        return inputs

    # 4. Load Model
    bnb_config = None
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # 5. LoRA Config
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05,
        bias="none",
        modules_to_save=[] # Add "embed_tokens", "lm_head" if you want to tune embeddings
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        remove_unused_columns=False, # Important for custom collator with VLM
        gradient_checkpointing=True,
        dataloader_pin_memory=False # Sometimes helps with VLM data loading issues
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    # 8. Train
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    train()
