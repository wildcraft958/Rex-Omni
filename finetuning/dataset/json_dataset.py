import json
import copy
import os
from collections import Counter
from typing import Dict, List, Any, Sequence
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from engine.registry import BUILDER
from utils.box_utils import xywh2xwxy
from utils.constants import IGNORE_INDEX
from utils.rope2d import get_rope_index_2, get_rope_index_25
from .tsv_dataset import preprocess_qwen_2_visual

class GroundingJsonDataset(Dataset):
    """
    Detection Dataset that reads from a JSON file containing image paths and annotations.
    Replaces GroundingTSVDataset.
    
    Expected JSON format (List[Dict]):
    [
        {
            "image_path": "/path/to/image.jpg",
            "boxes": [[x0, y0, x1, y1], ...],
            "labels": ["description1", "description2", ...]
        },
        ...
    ]
    """

    def __init__(
        self,
        json_file: str,
        tokenizer,
        data_args,
        image_min_pixels=224 * 224,
        image_max_pixels=1024 * 1024,
        max_num_samples=None,
        task_fn=None,
        system_message="You are a helpful assistant.",
        ori_box_format="xyxy",
        dataset_name=None,
        max_length=4096,
    ):
        super().__init__()
        
        # Load data
        print(f"Loading JSON dataset from {json_file}...")
        with open(json_file, 'r') as f:
            self.data = json.load(f)
            
        if max_num_samples is not None:
            self.data = self.data[:max_num_samples]
            
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.system_message = system_message
        self.ori_box_format = ori_box_format
        self.dataset_name = dataset_name or "grounding_json"
        self.max_length = max_length
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.max_retries = min(len(self.data), 32) if self.data else 1
        self.skip_reasons = Counter()
        self.sample_health = Counter()
        self.grid_threshold = 256
        
        # Setup model specific helpers
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2
        processor_for_merge = getattr(self.data_args.image_processor, "image_processor", self.data_args.image_processor)
        self.merge_size = getattr(
            processor_for_merge,
            "merge_size",
            getattr(processor_for_merge, "spatial_merge_size", 2),
        )
        
        self._configure_image_processor(image_min_pixels, image_max_pixels)
            
        # Setup task function
        if task_fn is not None:
            self.task_fn = BUILDER.build(task_fn)
        else:
            self.task_fn = None
            
        self.id2name = None

    def _configure_image_processor(self, min_pixels: int, max_pixels: int):
        processor = self.data_args.image_processor
        for attr, value in [
            ("min_pixels", min_pixels),
            ("max_pixels", max_pixels),
            ("image_min_pixels", min_pixels),
            ("image_max_pixels", max_pixels),
        ]:
            if hasattr(processor, attr):
                setattr(processor, attr, value)
        if hasattr(processor, "size") and isinstance(processor.size, dict):
            processor.size.setdefault("shortest_edge", int(np.sqrt(min_pixels)))
            processor.size.setdefault("longest_edge", int(np.sqrt(max_pixels)))
        self.merge_size = getattr(processor, "merge_size", getattr(processor, "spatial_merge_size", self.merge_size))

    def __len__(self):
        return len(self.data)

    def process_image_unified(self, image):
        processor = copy.deepcopy(self.data_args.image_processor)
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def _log_skip(self, reason: str):
        self.skip_reasons[reason] += 1
        self.sample_health["skipped"] += 1

    def _validate_grid_alignment(self, image_tensor, grid_thw, idx: int) -> bool:
        if not isinstance(image_tensor, torch.Tensor):
            return True
        if not isinstance(grid_thw, list) or len(grid_thw) == 0:
            return True
        first_grid = grid_thw[0]
        if not isinstance(first_grid, torch.Tensor):
            return True

        try:
            num_patches_img = int(image_tensor.shape[0])
            num_patches_grid = int(first_grid[0].item())
        except Exception:
            return True

        if num_patches_img != num_patches_grid:
            print(
                f"CRITICAL WARNING: Image patches {num_patches_img} != Grid T {num_patches_grid} "
                f"for image {idx}"
            )
            self._log_skip("grid_mismatch")
            return False

        if (first_grid > self.grid_threshold).any():
            print(f"WARNING: Large grid values detected for image {idx}: {first_grid.tolist()}")
            self._log_skip("grid_too_large")
            return False
        return True

    def _prepare_sample(self, idx: int) -> Dict[str, torch.Tensor] | None:
        item = self.data[idx]

        image_path = item.get("image_path")
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            self._log_skip("missing_image")
            return None
            
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            self._log_skip("image_load_failed")
            return None

        w, h = image_pil.size
        if w < 28 or h < 28:
            print(f"Image too small: {w}x{h}, skipping")
            self._log_skip("image_too_small")
            return None

        boxes = item.get("boxes", [])
        labels = item.get("labels", [])

        if len(boxes) == 0:
            print(f"No boxes for image {idx}, skipping")
            self._log_skip("no_boxes")
            return None

        data_dict = dict(
            boxes=np.array(boxes),
            labels=labels,
            size=(h, w),  # Height, Width
        )

        image, grid_thw = self.process_image_unified(image_pil)

        merge_size = getattr(self, "merge_size", getattr(self.data_args.image_processor, "merge_size", 2))
        grid_thw_merged = copy.deepcopy(grid_thw)
        
        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]
        
        grid_thw_merged = [
            merged_thw.prod() // merge_size**2
            for merged_thw in grid_thw_merged
        ]

        if not self._validate_grid_alignment(image, grid_thw, idx):
            return None

        if self.task_fn is not None:
            data_dict = self.task_fn(
                {"id2name": self.id2name, "annotations": data_dict},
                w,
                h,
            )

        sources = copy.deepcopy([e["conversations"] for e in [data_dict]])
        data_dict = preprocess_qwen_2_visual(
            sources,
            self.tokenizer,
            grid_thw=grid_thw_merged,
            visual_type="image",
            system_message=self.system_message,
        )

        position_ids, _ = self.get_rope_index(
            merge_size,
            data_dict["input_ids"],
            torch.stack(grid_thw, dim=0),
        )

        final_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            position_ids=position_ids,
            pixel_values=image,
            image_grid_thw=grid_thw
        )

        if final_dict["input_ids"].size(0) > self.max_length:
            print(f"Input too long ({final_dict['input_ids'].size(0)}), skipping")
            self._log_skip("input_too_long")
            return None

        self.sample_health["ok"] += 1
        return final_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if len(self.data) == 0:
            raise RuntimeError("The JSON dataset is empty. Please regenerate the JSON file.")

        for attempt in range(self.max_retries):
            candidate_idx = (i + attempt) % len(self.data)
            sample = self._prepare_sample(candidate_idx)
            if sample is not None:
                return sample

        raise RuntimeError(
            f"Unable to fetch a valid sample after {self.max_retries} attempts. "
            "Check the dataset export step for corrupt entries."
        )

    def get_health_report(self) -> Dict[str, Any]:
        return {
            "ok": int(self.sample_health.get("ok", 0)),
            "skipped": int(self.sample_health.get("skipped", 0)),
            "reasons": dict(self.skip_reasons),
            "total_records": len(self.data),
        }

    def reset_health_report(self):
        self.skip_reasons = Counter()
        self.sample_health = Counter()

