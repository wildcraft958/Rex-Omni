#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rex Vision Service - SAM + Spacy service
Responsibilities: Image processing, SAM segmentation, phrase extraction
Calls Rex LLM Service for object detection
"""

import io
import os
import base64
from typing import List, Dict, Any, Optional, Union

from modal import Image, App, method, fastapi_endpoint, enter
from fastapi import Body
import modal

# We'll lookup the LLM service at runtime, not at import time
# This avoids circular dependencies during image build


def download_sam_checkpoint():
    """Download SAM checkpoint during image build"""
    import subprocess
    subprocess.run([
        "wget", "-O", "/root/sam_vit_h_4b8939.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    ])


def build_vision_image():
    """Build image for SAM + Spacy vision processing"""
    return (
        Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
        .apt_install("git", "libgl1", "wget")
        .run_commands(
            # Core stack
            "pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124",
            
            # Vision dependencies (no vLLM!)
            "pip install git+https://github.com/facebookresearch/segment-anything.git",
            "pip install spacy opencv-python pillow numpy shapely pycocotools matplotlib fastapi requests",
            
            # Download spacy model
            "python -m spacy download en_core_web_sm",
        )
        .run_function(download_sam_checkpoint)
    )


image = build_vision_image()
app = App("rex-vision-service", image=image)


@app.cls(
    gpu="A100-40GB",  # SAM needs GPU but not as beefy as vLLM
    memory=65536,     # 64 GB
    cpu=16,
    scaledown_window=300,
    timeout=600
)
class VisionService:
    @enter()
    def initialize(self):
        import sys
        import traceback
        sys.path.append("/root")

        from segment_anything import sam_model_registry, SamPredictor
        import spacy
        import torch

        print(">>> ENTER VisionService.initialize()")
        
        # Initialize SAM
        print("Initializing SAM...")
        try:
            ckpt = "/root/sam_vit_h_4b8939.pth"
            self.sam = sam_model_registry["vit_h"](checkpoint=ckpt)
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"SAM using device: {device}")
            self.sam.to(device=device)
            self.sam_predictor = SamPredictor(self.sam)
            print(">>> SAM initialized successfully")
        except Exception as e:
            print(f"ERROR initializing SAM: {e}")
            traceback.print_exc()
            raise

        # Initialize Spacy
        print("Initializing Spacy...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print(">>> Spacy initialized successfully")
        except Exception as e:
            print(f"ERROR initializing Spacy: {e}")
            traceback.print_exc()
            raise

        print(">>> sam_predictor set?", hasattr(self, "sam_predictor"))
        print(">>> nlp set?", hasattr(self, "nlp"))
        print(">>> VisionService initialization complete")

    def _call_rex_llm(self, image_data: str, task: str, categories: List[str]) -> Dict[str, Any]:
        """Call Rex LLM service via Modal lookup, local import, or HTTP fallback"""
        payload = {
            "image": image_data,
            "task": task,
            "categories": categories,
        }

        # 1. Try Modal function lookup if supported in this runtime
        lookup_target = getattr(getattr(modal, "Function", None), "lookup", None)
        if callable(lookup_target):
            try:
                print(">>> Calling Rex LLM via modal.Function.lookup('rex-llm-service', 'rex_inference')")
                rex_func = lookup_target("rex-llm-service", "rex_inference")
                return rex_func.call(item=payload)
            except Exception as exc:
                print(f"WARNING: Modal function lookup failed: {exc}. Falling back...")

        # 2. Optional local import (only works if rex_llm_app is bundled in image)
        try:
            import importlib

            rex_module = importlib.import_module("rex_llm_app")
            if hasattr(rex_module, "rex_inference"):
                print(">>> Calling rex_llm_app.rex_inference directly (local import)")
                return rex_module.rex_inference(item=payload)
        except ModuleNotFoundError:
            print("INFO: rex_llm_app module not found inside vision image. Skipping local import.")
        except Exception as exc:
            print(f"WARNING: Local rex_llm_app import failed: {exc}. Falling back...")

        # 3. HTTP fallback as a last resort
        import requests

        url = os.getenv(
            "REX_LLM_HTTP_URL",
            "https://animeshraj958--rex-llm-service-rex_inference.modal.run",
        )

        try:
            print(f">>> Calling Rex LLM service over HTTP at {url}")
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"ERROR calling Rex LLM service over HTTP: {e}")
            return {"success": False, "error": str(e)}

    def _decode_image(self, image_data: Union[str, bytes]):
        from PIL import Image as PILImage
        import numpy as np

        try:
            if isinstance(image_data, str):
                # Handle data URI format (data:image/jpeg;base64,...)
                if "," in image_data and image_data.startswith("data:"):
                    image_data = image_data.split(",", 1)[1]
                
                # Remove whitespace
                image_data = image_data.strip()
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data

            img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            return img, np.array(img)
        except Exception as e:
            print(f"ERROR decoding image: {e}")
            if isinstance(image_data, str):
                print(f"Image data length: {len(image_data)}")
                print(f"First 100 chars: {image_data[:100]}")
            raise ValueError(f"Failed to decode image: {e}")

    @method()
    def sam_inference(self, image_data: str, categories: List[str]) -> Dict[str, Any]:
        """
        SAM + Rex pipeline:
        1. Call Rex LLM service for detection
        2. Use predicted boxes as SAM prompts
        3. Generate segmentation masks
        4. Return polygons + metadata
        """
        import numpy as np
        import torch
        import cv2

        print(">>> VisionService.sam_inference() called")
        
        if not hasattr(self, "sam_predictor"):
            raise RuntimeError("SAM not initialized. Check container logs.")

        # 1. Decode image
        pil_img, image_np = self._decode_image(image_data)

        # 2. Call Rex LLM service for detection
        print(">>> Calling Rex LLM service...")
        rex_results = self._call_rex_llm(
            image_data=image_data,
            task="detection",
            categories=categories
        )

        if not rex_results.get("success", False):
            return {
                "success": False,
                "error": rex_results.get("error", "Rex detection failed")
            }

        predictions = rex_results["extracted_predictions"]

        # Collect boxes
        input_boxes = []
        labels = []
        for cat, items in predictions.items():
            for item in items:
                if item.get("type") == "box":
                    input_boxes.append(item["coords"])
                    labels.append(cat)

        if not input_boxes:
            return {
                "success": True,
                "predictions": predictions,
                "sam_results": [],
                "message": "No objects detected by Rex-Omni",
            }

        # 3. Run SAM on those boxes
        print(f">>> Running SAM on {len(input_boxes)} boxes...")
        self.sam_predictor.set_image(image_np)

        boxes_t = torch.tensor(input_boxes, device=self.sam_predictor.device)
        transformed = self.sam_predictor.transform.apply_boxes_torch(
            boxes_t, image_np.shape[:2]
        )

        masks, scores, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )

        masks_np = masks.cpu().numpy()
        scores_np = scores.cpu().numpy()

        # 4. Convert masks to polygons
        sam_results = []
        for mask, score, box, label in zip(masks_np, scores_np, input_boxes, labels):
            mask_uint8 = (mask[0] > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            polys = []
            for c in contours:
                if cv2.contourArea(c) > 10:
                    polys.append(c.flatten().tolist())

            sam_results.append({
                "category": label,
                "box": box,
                "score": float(score[0]),
                "polygons": polys,
            })

        return {
            "success": True,
            "predictions": predictions,
            "sam_results": sam_results,
        }

    @method()
    def grounding_inference(self, image_data: str, caption: str) -> Dict[str, Any]:
        """
        Grounding pipeline:
        1. Extract noun phrases with Spacy
        2. Call Rex LLM service for grounding
        3. Map phrases to boxes
        """
        print(">>> VisionService.grounding_inference() called")
        
        if not hasattr(self, "nlp"):
            raise RuntimeError("Spacy not initialized. Check container logs.")

        pil_img, _ = self._decode_image(image_data)
        doc = self.nlp(caption)

        # Extract noun phrases
        phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 2:
                phrases.append({
                    "text": chunk.text,
                    "start": chunk.start_char,
                    "end": chunk.end_char,
                })

        if not phrases:
            return {
                "success": True,
                "message": "No phrases found in caption",
                "annotations": []
            }

        unique_phrases = list(set(p["text"] for p in phrases))

        # Call Rex LLM for phrase grounding
        print(f">>> Calling Rex LLM service for grounding {len(unique_phrases)} phrases...")
        rex_results = self._call_rex_llm(
            image_data=image_data,
            task="detection",
            categories=unique_phrases
        )

        if not rex_results.get("success", False):
            return {
                "success": False,
                "error": rex_results.get("error", "Rex detection failed")
            }

        predictions = rex_results["extracted_predictions"]

        # Map phrases to boxes
        annotations = []
        for phrase in phrases:
            text = phrase["text"]
            if text in predictions:
                boxes = [
                    item["coords"]
                    for item in predictions[text]
                    if item.get("type") == "box"
                ]
                if boxes:
                    annotations.append({
                        "phrase": text,
                        "start_char": phrase["start"],
                        "end_char": phrase["end"],
                        "boxes": boxes,
                    })

        return {
            "success": True,
            "image_size": pil_img.size,
            "caption": caption,
            "annotations": annotations,
        }


# External HTTP endpoints (public API surface)
@app.function()
@fastapi_endpoint(method="POST")
def api_sam(item: Dict = Body(...)):
    """
    SAM segmentation endpoint
    
    Input: {
        "image": base64_str,
        "categories": [str]
    }
    
    Output: {
        "success": bool,
        "predictions": {...},
        "sam_results": [{category, box, score, polygons}]
    }
    """
    return VisionService().sam_inference.remote(
        image_data=item["image"],
        categories=item["categories"],
    )


@app.function()
@fastapi_endpoint(method="POST")
def api_grounding(item: Dict = Body(...)):
    """
    Phrase grounding endpoint
    
    Input: {
        "image": base64_str,
        "caption": str
    }
    
    Output: {
        "success": bool,
        "image_size": [w, h],
        "caption": str,
        "annotations": [{phrase, start_char, end_char, boxes}]
    }
    """
    return VisionService().grounding_inference.remote(
        image_data=item["image"],
        caption=item["caption"],
    )
