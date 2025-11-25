# LoRA Fine-Tuning for Rex-Omni (Qwen2.5-VL)

This directory contains scripts to fine-tune the Rex-Omni model using LoRA/QLoRA on your custom remote sensing data.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

1.  **Format**: Your data needs to be in a JSONL format where each line is a conversation.
2.  **Conversion Tool**: Use `convert_to_jsonl.py` to convert a folder of images and a metadata JSON file.
    ```bash
    python convert_to_jsonl.py --image_folder /path/to/images --metadata_file /path/to/metadata.json --output_file train_data.jsonl
    ```
    *Note: The metadata file should be a JSON list of objects with `image` (filename) and `text` (caption/ground truth).*

3.  **Sample Format**: See `sample_data.json` for the expected structure of a single entry.

## Training

Run the training script:

```bash
python train_lora.py \
    --data_path train_data.jsonl \
    --output_dir output_lora \
    --num_epochs 3 \
    --batch_size 1 \
    --use_qlora
```

### Key Arguments:
- `--use_qlora`: Enables 4-bit quantization (saves VRAM).
- `--lora_rank`: LoRA rank (default 64).
- `--grad_accum`: Gradient accumulation steps (increase if batch size is small).
