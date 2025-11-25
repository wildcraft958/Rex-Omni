import os
import modal
from pathlib import Path

# 1. Define the Modal Image
# We need a GPU environment with PyTorch, Transformers, PEFT, and Qwen-VL utils.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.46.0",
        "peft>=0.13.0",
        "bitsandbytes>=0.44.0",
        "accelerate>=1.0.0",
        "qwen-vl-utils",
        "datasets",
        "wandb",
        "scipy",
        "hf_transfer" # Faster downloads
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# 2. Define the App and Volumes
app = modal.App("rex-omni-finetune")
volume = modal.Volume.from_name("rex-omni-data", create_if_missing=True)
model_volume = modal.Volume.from_name("rex-omni-models", create_if_missing=True)

# 3. Training Function
@app.function(
    image=image,
    gpu="A100", # Use A100 for best performance, or "T4" / "L4" for lower cost
    volumes={"/data": volume, "/models": model_volume},
    timeout=86400, # 24 hours
)
def train_rex_omni(
    data_filename: str,
    num_epochs: int = 3,
    batch_size: int = 1,
    lora_rank: int = 64,
    use_qlora: bool = True
):
    import torch
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        TrainingArguments,
        Trainer
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from datasets import load_dataset
    from qwen_vl_utils import process_vision_info

    print("Starting training setup...")
    
    # Paths
    data_path = f"/data/{data_filename}"
    output_dir = f"/models/rex-omni-lora-{modal.current_function_call_id}"
    model_id = "IDEA-Research/Rex-Omni"

    # Load Processor
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=1280*28*28)

    # Load Dataset
    print(f"Loading data from {data_path}...")
    dataset = load_dataset("json", data_files=data_path, split="train")

    # Data Collator
    def collate_fn(examples):
        texts = [processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True) for example in examples]
        image_inputs, video_inputs = process_vision_info(examples)
        
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        labels[labels == image_token_id] = -100
        inputs["labels"] = labels
        return inputs

    # Load Model
    print("Loading model...")
    bnb_config = None
    if use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    # LoRA Config
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05,
        bias="none",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training Args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    # Train
    print("Starting training loop...")
    trainer.train()
    
    # Save
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    # Commit volume to ensure persistence
    model_volume.commit()
    
    return output_dir

@app.local_entrypoint()
def main():
    # Example usage for local testing
    # In practice, you'd trigger this from the notebook or CLI
    print("Run this app using 'modal run modal_train.py' or import it in a notebook.")
