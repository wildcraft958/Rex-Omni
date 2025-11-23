import os
import io
import json
import base64
from typing import List, Dict, Any, Optional, Union
from modal import Image, App, method, gpu, fastapi_endpoint
from fastapi import Body, Request

# Define the image with necessary dependencies
def download_models():
    # Download Spacy model
    import spacy
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    
    # Download SAM checkpoint
    subprocess.run(["wget", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"])

image = (
    Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .run_commands("pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124")
    .run_commands("pip install vllm==0.8.2")
    .run_commands("pip install flash-attn==2.7.4.post1")
    .run_commands("pip install segment-anything spacy opencv-python pillow numpy transformers accelerate qwen_vl_utils shapely pycocotools gradio_image_prompter")
    .run_function(download_models)
    .add_local_dir("/home/bakasur/Desktop/Rex-Omni/rex_omni", remote_path="/root/rex_omni")
)

app = App("rex-omni-service", image=image)

@app.cls(
    gpu="A100-40GB",
    scaledown_window=300,
    timeout=600
)
class RexOmniService:
    def __enter__(self):
        import sys
        sys.path.append("/root") # Ensure rex_omni is importable
        
        from rex_omni import RexOmniWrapper
        from segment_anything import sam_model_registry, SamPredictor
        import spacy
        import torch

        print("Initializing Rex-Omni Model...")
        self.rex_model = RexOmniWrapper(
            model_path="IDEA-Research/Rex-Omni-AWQ",
            backend="vllm",
            quantization="awq",
            max_tokens=2048,
            temperature=0.0,
            top_p=0.05,
            top_k=1,
            repetition_penalty=1.05,
            gpu_memory_utilization=0.7, # Leave some memory for SAM
        )

        print("Initializing SAM Model...")
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.sam = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint)
        self.sam.to(device="cuda")
        self.sam_predictor = SamPredictor(self.sam)

        print("Initializing Spacy...")
        self.nlp = spacy.load("en_core_web_sm")
        
        print("Initialization Complete.")

    def _decode_image(self, image_data: Union[str, bytes]) -> Any:
        from PIL import Image as PILImage
        import numpy as np
        
        if isinstance(image_data, str):
            # Check if it's a base64 string
            if "," in image_data:
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
            
        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        return image

    @method()
    def inference(
        self, 
        image_data: str, 
        task: str, 
        categories: Optional[List[str]] = None,
        keypoint_type: Optional[str] = None,
        visual_prompt_boxes: Optional[List[List[float]]] = None,
        **kwargs
    ):
        image = self._decode_image(image_data)
        
        # Ensure categories is properly formatted
        if categories and isinstance(categories, str):
            categories = [c.strip() for c in categories.split(",")]
            
        results = self.rex_model.inference(
            images=[image],
            task=task,
            categories=categories,
            keypoint_type=keypoint_type,
            visual_prompt_boxes=visual_prompt_boxes,
            **kwargs
        )
        return results[0]

    @method()
    def sam_inference(self, image_data: str, categories: List[str]):
        import numpy as np
        import torch
        
        image = self._decode_image(image_data)
        image_np = np.array(image)
        
        # 1. Rex-Omni Detection
        rex_results = self.rex_model.inference(
            images=[image],
            task="detection",
            categories=categories
        )[0]
        
        if not rex_results.get("success", False):
            return {"success": False, "error": rex_results.get("error", "Rex-Omni detection failed")}
            
        predictions = rex_results["extracted_predictions"]
        
        # Collect all boxes
        input_boxes = []
        box_labels = [] # To keep track of which box belongs to which category
        
        for cat, items in predictions.items():
            for item in items:
                if item.get("type") == "box":
                    input_boxes.append(item["coords"])
                    box_labels.append(cat)
        
        if not input_boxes:
            return {
                "success": True, 
                "predictions": predictions, 
                "masks": [], 
                "message": "No objects detected by Rex-Omni"
            }

        # 2. SAM Segmentation
        self.sam_predictor.set_image(image_np)
        
        input_boxes_tensor = torch.tensor(input_boxes, device=self.sam_predictor.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            input_boxes_tensor, image_np.shape[:2]
        )
        
        masks, scores, logits = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # Process results
        # masks shape: [N, 1, H, W]
        masks_np = masks.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        sam_results = []
        for i, (mask, score, box, label) in enumerate(zip(masks_np, scores_np, input_boxes, box_labels)):
            # Encode mask to RLE or just return polygon? 
            # For simplicity in this API, let's return bounding box and a simplified polygon or RLE.
            # Returning full binary mask in JSON is too heavy.
            # Let's return the box and score for now, and maybe a simplified polygon if needed.
            # For this implementation, we'll just return the metadata.
            # If the user needs the mask, we can encode it. 
            # Let's use a simple RLE encoding or just return the fact that we have it.
            # Actually, let's return the polygon using opencv.
            
            import cv2
            mask_uint8 = (mask[0] > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            polygons = []
            for contour in contours:
                if cv2.contourArea(contour) > 10: # Filter small noise
                    polygons.append(contour.flatten().tolist())
            
            sam_results.append({
                "category": label,
                "box": box,
                "score": float(score[0]),
                "polygons": polygons
            })
            
        return {
            "success": True,
            "predictions": predictions,
            "sam_results": sam_results
        }

    @method()
    def grounding_inference(self, image_data: str, caption: str):
        image = self._decode_image(image_data)
        
        # 1. Extract phrases with Spacy
        doc = self.nlp(caption)
        phrases = []
        for chunk in doc.noun_chunks:
            # Filter out stop words or very short phrases if needed
            if len(chunk.text) > 2:
                phrases.append({
                    "text": chunk.text,
                    "start": chunk.start_char,
                    "end": chunk.end_char
                })
        
        if not phrases:
             return {"success": True, "message": "No phrases found in caption", "annotations": []}
             
        unique_phrases = list(set(p["text"] for p in phrases))
        
        # 2. Ground with Rex-Omni
        rex_results = self.rex_model.inference(
            images=[image],
            task="detection",
            categories=unique_phrases
        )[0]
        
        if not rex_results.get("success", False):
            return {"success": False, "error": rex_results.get("error", "Rex-Omni detection failed")}
            
        predictions = rex_results["extracted_predictions"]
        
        # 3. Match back to spans
        annotations = []
        for phrase_info in phrases:
            text = phrase_info["text"]
            if text in predictions:
                # Get boxes for this phrase
                boxes = [item["coords"] for item in predictions[text] if item.get("type") == "box"]
                if boxes:
                    annotations.append({
                        "phrase": text,
                        "start_char": phrase_info["start"],
                        "end_char": phrase_info["end"],
                        "boxes": boxes
                    })
                    
        return {
            "success": True,
            "image_size": image.size,
            "caption": caption,
            "annotations": annotations
        }

@app.function()
@fastapi_endpoint(method="POST")
def api_inference(item: Dict = Body(...)):
    service = RexOmniService()
    return service.inference.remote(
        image_data=item["image"],
        task=item.get("task", "detection"),
        categories=item.get("categories"),
        keypoint_type=item.get("keypoint_type"),
        visual_prompt_boxes=item.get("visual_prompt_boxes"),
        **item.get("kwargs", {})
    )

@app.function()
@fastapi_endpoint(method="POST")
def api_sam(item: Dict = Body(...)):
    service = RexOmniService()
    return service.sam_inference.remote(
        image_data=item["image"],
        categories=item["categories"]
    )

@app.function()
@fastapi_endpoint(method="POST")
def api_grounding(item: Dict = Body(...)):
    service = RexOmniService()
    return service.grounding_inference.remote(
        image_data=item["image"],
        caption=item["caption"]
    )

@app.function()
@fastapi_endpoint(method="GET")
def health():
    try:
        import rex_omni
        import torch
        return {
            "status": "ok", 
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
