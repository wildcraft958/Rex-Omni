#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rex LLM Service - Pure vLLM inference service
Responsibilities: Run Rex-Omni with vLLM backend, return structured predictions
No SAM, no Spacy, no OpenCV - just LLM inference
"""

import io
import base64
from typing import List, Dict, Any, Optional

from modal import Image, App, method, fastapi_endpoint, enter
from fastapi import Body


def build_llm_image():
    """Build isolated image for Rex-Omni vLLM inference"""
    return (
        Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
        .apt_install("git", "wget")
        .run_commands(
            # 1. Core stack
            "pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124",
            "pip install ninja packaging psutil setuptools wheel",
            
            # 2. NumPy + Numba (critical for vLLM)
            "pip install 'numpy<2.1' 'numba<0.61'",
            
            # 3. vLLM + flash-attn
            "pip install vllm==0.8.2",
            "pip install flash-attn==2.7.4.post1 --no-build-isolation",
            
            # 4. Minimal dependencies for Rex-Omni
            "pip install pillow transformers qwen_vl_utils",
            
            # 5. Force-reinstall to prevent version drift
            "pip install --force-reinstall --no-deps 'numpy<2.1' 'numba<0.61'",
            
            # 6. Verify
            "python -c \"import numpy, numba; print('NUMPY', numpy.__version__, 'NUMBA', numba.__version__)\""
        )
        .add_local_dir("/home/bakasur/Desktop/Rex-Omni/rex_omni", remote_path="/root/rex_omni")
    )


image = build_llm_image()
app = App("rex-llm-service", image=image)


@app.cls(
    gpu="A100-80GB",      # Production GPU
    memory=131072,        # 128 GB RAM
    cpu=32,               # 32 vCPUs
    scaledown_window=300,
    timeout=600
)
class RexLLMService:
    @enter()
    def initialize(self):
        import sys
        import traceback
        sys.path.append("/root")
        
        from rex_omni import RexOmniWrapper

        print(">>> ENTER RexLLMService.initialize()")
        try:
            self.rex_model = RexOmniWrapper(
                model_path="IDEA-Research/Rex-Omni-AWQ",
                backend="vllm",
                quantization="awq",
                max_tokens=2048,
                temperature=0.0,
                top_p=0.05,
                top_k=1,
                repetition_penalty=1.05,
                gpu_memory_utilization=0.9,  # Can use more since SAM is elsewhere
            )
            print(">>> Rex-Omni LLM initialized successfully")
            print(">>> rex_model set?", hasattr(self, "rex_model"))
        except Exception as e:
            print(f"CRITICAL: Failed to initialize Rex-Omni: {e}")
            traceback.print_exc()
            raise

    def _decode_image(self, image_data: str):
        from PIL import Image as PILImage
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        return PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

    @method()
    def inference(
        self,
        image_data: str,
        task: str,
        categories: Optional[List[str]] = None,
        keypoint_type: Optional[str] = None,
        visual_prompt_boxes: Optional[List[List[float]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        print(">>> RexLLMService.inference() called")
        if not hasattr(self, "rex_model"):
            raise RuntimeError("Rex-Omni LLM not initialized. Check container logs.")

        image = self._decode_image(image_data)

        if categories and isinstance(categories, str):
            categories = [c.strip() for c in categories.split(",")]

        results = self.rex_model.inference(
            images=[image],
            task=task,
            categories=categories,
            keypoint_type=keypoint_type,
            visual_prompt_boxes=visual_prompt_boxes,
            **kwargs,
        )
        return results[0]


# External HTTP endpoint for direct testing
@app.function()
@fastapi_endpoint(method="POST")
def rex_inference(item: Dict = Body(...)):
    """
    Rex-Omni LLM inference endpoint
    
    Input: {
        "image": base64_str,
        "task": "detection" | "grounding" | ...,
        "categories": [str],
        "keypoint_type": str (optional),
        "visual_prompt_boxes": [[x1,y1,x2,y2]] (optional),
        "kwargs": {} (optional)
    }
    
    Output: Single Rex-Omni result dict
    """
    return RexLLMService().inference.remote(
        image_data=item["image"],
        task=item.get("task", "detection"),
        categories=item.get("categories"),
        keypoint_type=item.get("keypoint_type"),
        visual_prompt_boxes=item.get("visual_prompt_boxes"),
        **item.get("kwargs", {})
    )
