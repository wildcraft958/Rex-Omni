import json
import copy
import os
from typing import Dict, List, Any
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
        
        # Setup model specific helpers
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2
            
        # Setup task function
        if task_fn is not None:
            self.task_fn = BUILDER.build(task_fn)
        else:
            self.task_fn = None
            
        self.id2name = None

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

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.data[i]
        
        # Load image
        image_path = item.get("image_path")
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return self.__getitem__((i + 1) % len(self.data))
            
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((i + 1) % len(self.data))

        w, h = image_pil.size
        if w < 28 or h < 28:
            print(f"Image too small: {w}x{h}, skipping")
            return self.__getitem__((i + 1) % len(self.data))

        # Parse annotations
        boxes = item.get("boxes", [])
        labels = item.get("labels", [])
        
        if len(boxes) == 0:
            print(f"No boxes for image {i}, skipping")
            return self.__getitem__((i + 1) % len(self.data))

        # Prepare data dict for task_fn
        data_dict = dict(
            boxes=np.array(boxes),
            labels=labels,
            size=(h, w), # Height, Width
        )

        # Process image
        image, grid_thw = self.process_image_unified(image_pil)
        
        # Handle grid merge size
        merge_size = getattr(self.data_args.image_processor, 'merge_size', 2)
        grid_thw_merged = copy.deepcopy(grid_thw)
        
        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]
            
        grid_thw_merged = [
            merged_thw.prod() // merge_size**2
            for merged_thw in grid_thw_merged
        ]

        # Apply task function (formatting prompt)
        if self.task_fn is not None:
            data_dict = self.task_fn(
                {"id2name": self.id2name, "annotations": data_dict},
                w, h,
            )

        # Preprocess conversation for Qwen2.5-VL
        sources = copy.deepcopy([e["conversations"] for e in [data_dict]])
        data_dict = preprocess_qwen_2_visual(
            sources,
            self.tokenizer,
            grid_thw=grid_thw_merged,
            visual_type="image",
            system_message=self.system_message,
        )
        
        # Get RoPE index
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
            return self.__getitem__((i + 1) % len(self.data))

        return final_dict

