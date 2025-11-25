import json
import os
import argparse
from pathlib import Path
from PIL import Image

def normalize_bbox(bbox, width, height):
    """
    Normalizes bbox (x0, y0, x1, y1) to 0-999 range and returns token string.
    """
    x0, y0, x1, y1 = bbox
    
    # Normalize to 0-1
    x0_n = max(0.0, min(1.0, x0 / width))
    y0_n = max(0.0, min(1.0, y0 / height))
    x1_n = max(0.0, min(1.0, x1 / width))
    y1_n = max(0.0, min(1.0, y1 / height))
    
    # Scale to 0-999
    x0_bin = int(x0_n * 999)
    y0_bin = int(y0_n * 999)
    x1_bin = int(x1_n * 999)
    y1_bin = int(y1_n * 999)
    
    return f"<{x0_bin}><{y0_bin}><{x1_bin}><{y1_bin}>"

def convert(image_folder, metadata_file, output_file):
    output_path = Path(output_file)
    image_folder_path = Path(image_folder)
    
    with open(metadata_file, 'r') as f:
        data = json.load(f)
        
    with open(output_path, 'w') as f_out:
        for item in data:
            image_filename = item.get('image')
            if not image_filename:
                continue
                
            image_path = image_folder_path / image_filename
            if not image_path.exists():
                print(f"Warning: Image {image_path} not found. Skipping.")
                continue
            
            # Get image dimensions for normalization
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                continue

            # Construct the conversation
            # Expected metadata format for grounding:
            # "objects": [{"category": "ship", "bbox": [x0, y0, x1, y1]}, ...]
            
            objects = item.get('objects', [])
            if not objects:
                continue

            # Group boxes by category
            cat_to_boxes = {}
            for obj in objects:
                cat = obj['category']
                bbox = obj['bbox']
                if cat not in cat_to_boxes:
                    cat_to_boxes[cat] = []
                cat_to_boxes[cat].append(bbox)
            
            # Format answer
            # <|object_ref_start|>cat<|object_ref_end|><|box_start|><x><y><x><y>,<x><y><x><y><|box_end|>
            answer_parts = []
            for cat, boxes in cat_to_boxes.items():
                box_tokens = [normalize_bbox(b, width, height) for b in boxes]
                box_str = ",".join(box_tokens)
                part = f"<|object_ref_start|>{cat}<|object_ref_end|><|box_start|>{box_str}<|box_end|>"
                answer_parts.append(part)
            
            answer_text = ", ".join(answer_parts)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path.absolute())},
                        {"type": "text", "text": "<image>\nDetect the objects in this image."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_text}]
                }
            ]
            
            entry = {"messages": messages}
            f_out.write(json.dumps(entry) + '\n')
            
    print(f"Converted {len(data)} items to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", required=True, help="Folder containing images")
    parser.add_argument("--metadata_file", required=True, help="JSON file with metadata")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    args = parser.parse_args()
    
    convert(args.image_folder, args.metadata_file, args.output_file)
