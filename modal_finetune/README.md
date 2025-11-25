# Modal Fine-tuning for Rex-Omni

This directory contains the setup to fine-tune Rex-Omni on the cloud using [Modal](https://modal.com).

## Feasibility
Yes, it is absolutely feasible to fine-tune Rex-Omni using LoRA. Rex-Omni is built on the Qwen2-VL architecture, which is fully supported by the Hugging Face ecosystem (`transformers`, `peft`). The link you found (`eduardolsmj/Rex-Omni-HFendpoints`) appears to be a repository for inference deployment, not training. We will use the official `IDEA-Research/Rex-Omni` model as the base for fine-tuning.

## Files
- **`modal_train.py`**: The core script. It defines the Modal App, the GPU environment (Docker image), and the training function.
- **`finetune.ipynb`**: A Jupyter notebook to guide you through uploading data, running the job, and downloading results.

## How to Use

1.  **Install Modal**:
    ```bash
    pip install modal
    modal setup
    ```

2.  **Prepare Data**:
    Use the tools in `../lora_finetuning/` to create your `train_data.jsonl`.

3.  **Run the Notebook**:
    Open `finetune.ipynb` and follow the steps.
    - It will upload your data to a Modal Volume.
    - It will launch a training job on an A100 GPU (or other specified GPU).
    - It will save the trained LoRA adapters to a Modal Volume.

4.  **Download Results**:
    Use the command provided in the notebook to download your fine-tuned model.

## Cost Note
Modal charges for GPU usage. An A100 costs around $4-5/hr. Fine-tuning on 10-20k samples might take a few hours depending on the number of epochs. You can switch to `gpu="A10G"` or `gpu="L4"` in `modal_train.py` to save costs, though training will be slower.
