# Rex-Omni Modal Microservices Deployment

## Architecture

Two separate Modal services:

1. **rex-llm-service** (`rex_llm_app.py`)
   - Pure Rex-Omni vLLM inference
   - GPU: A100-80GB, CPU: 32, RAM: 128GB
   - No SAM, no Spacy, no OpenCV

2. **rex-vision-service** (`rex_vision_app.py`)
   - SAM segmentation + Spacy phrase extraction
   - GPU: A100-40GB, CPU: 16, RAM: 64GB
   - Calls rex-llm-service internally

## Deployment

### 1. Deploy Rex LLM Service (FIRST)

```bash
modal deploy rex_llm_app.py
```

This creates the LLM inference backend. Wait for deployment to complete.

### 2. Deploy Rex Vision Service

```bash
modal deploy rex_vision_app.py
```

This creates the public API endpoints that use the LLM service.

## Testing

### Test LLM Service Directly

```bash
curl -X POST https://animeshraj958--rex-llm-service-rex-inference.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_image>",
    "task": "detection",
    "categories": ["person", "cup", "laptop"]
  }'
```

### Test Vision Service (Public API)

Update `test_modal.py` to use `rex-vision-service`:

```python
# SAM endpoint
url = "https://animeshraj958--rex-vision-service-api-sam.modal.run"

# Grounding endpoint  
url = "https://animeshraj958--rex-vision-service-api-grounding.modal.run"
```

## Benefits

1. **Fault Isolation**: vLLM crashes don't kill SAM
2. **Independent Scaling**: Scale LLM and vision separately
3. **Cleaner Debugging**: Hit each service independently
4. **Resource Optimization**: Each service has appropriate resources

## Migration Notes

- Old `modal_app.py` can be deprecated
- Update frontend/clients to hit `rex-vision-service` endpoints
- Monitor both services separately in Modal dashboard
