# MedGemma 1.5 Backend

Backend API for running Google's MedGemma 1.5 model locally and on Modal serverless platform.

## Features

- 🏥 **Medical AI**: MedGemma 1.5 4B instruction-tuned model
- 🖼️ **Multimodal**: Supports both text-only and image+text inputs
- 💻 **Local Development**: Run on your GPU for development
- ☁️ **Serverless Deployment**: Deploy to Modal with auto-scaling
- ⚡ **Optional Quantization**: 8-bit quantization support to reduce VRAM usage

## Prerequisites

### For Local Development
- Python 3.10+
- NVIDIA GPU with 12GB+ VRAM (or 6GB+ with quantization)
- CUDA toolkit installed

### For Modal Deployment
- Modal account ([sign up here](https://modal.com))
- Modal CLI installed

### Hugging Face Access
1. Go to https://huggingface.co/google/medgemma-1.5-4b-it
2. Accept the Health AI Developer Foundations terms
3. Create a Hugging Face token: https://huggingface.co/settings/tokens
4. Save your token - you'll need it for both local and Modal deployment

## Local Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note**: For PyTorch with CUDA support, you may need to install from the PyTorch website:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Configure Environment

Copy the example environment file:
```bash
copy .env.example .env
```

Edit `.env` and add your Hugging Face token:
```
HUGGINGFACE_TOKEN=your_actual_token_here
```

### 3. Run the Server

```bash
python local_app.py
```

The API will be available at `http://localhost:8000`

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Modal Deployment

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal token set
```

### 3. Create Hugging Face Secret

Store your Hugging Face token as a Modal secret:

```bash
modal secret create huggingface-secret HUGGINGFACE_TOKEN=your_token_here
```

### 4. Deploy to Modal

```bash
modal deploy modal_app.py
```

Modal will output your deployment URLs. You'll get:
- Inference endpoint: `https://your-app--infer-endpoint.modal.run`
- Health check: `https://your-app--health.modal.run`

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Text-Only Inference

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What are the symptoms of pneumonia?",
    "max_new_tokens": 512,
    "temperature": 0.7
  }'
```

### Multimodal Inference (Text + Image)

```bash
# First, convert your image to base64
# Linux/Mac:
base64 -i your_image.jpg > image.b64

# Windows PowerShell:
[Convert]::ToBase64String([IO.File]::ReadAllBytes("your_image.jpg")) | Out-File image.b64

# Then make the request:
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"What do you see in this medical image?\",
    \"image_base64\": \"$(cat image.b64)\",
    \"max_new_tokens\": 512
  }"
```

### Python Example

```python
import requests
import base64

# Text-only request
response = requests.post(
    "http://localhost:8000/infer",
    json={
        "text": "What are the common treatments for Type 2 diabetes?",
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9
    }
)
print(response.json()["response"])

# With image
with open("medical_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/infer",
    json={
        "text": "Describe this X-ray image",
        "image_base64": image_base64,
        "max_new_tokens": 512
    }
)
print(response.json()["response"])
```

## Configuration Options

Edit `.env` or set environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `google/medgemma-1.5-4b-it` | Hugging Face model ID |
| `USE_QUANTIZATION` | `false` | Enable 8-bit quantization (reduces VRAM) |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `MAX_NEW_TOKENS` | `512` | Default max tokens to generate |
| `TEMPERATURE` | `0.7` | Default sampling temperature |
| `TOP_P` | `0.9` | Default nucleus sampling parameter |
| `DEVICE` | `cuda` | Device to use (`cuda` or `cpu`) |

### Enabling Quantization

To reduce VRAM usage (from ~12GB to ~4-6GB), set in `.env`:
```
USE_QUANTIZATION=true
```

## Project Structure

```
backend/
├── config.py           # Configuration management
├── model.py            # MedGemma model handler
├── main.py             # FastAPI application
├── local_app.py        # Local development server
├── modal_app.py        # Modal serverless deployment
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment variables
└── README.md           # This file
```

## GPU Requirements

| Configuration | VRAM Required | Speed |
|---------------|---------------|-------|
| Full precision (fp16) | 12-16GB | Fast |
| 8-bit quantization | 4-6GB | Moderate |
| CPU only | 0GB (uses RAM) | Slow |

## Troubleshooting

### CUDA Out of Memory
- Enable quantization: `USE_QUANTIZATION=true`
- Reduce batch size if processing multiple requests
- Use a GPU with more VRAM

### Model Download Issues
- Ensure your Hugging Face token is valid
- Check you've accepted the model terms
- Verify internet connection for initial download

### Modal Deployment Issues
- Ensure Modal secret is created: `modal secret list`
- Check logs: `modal app logs medgemma-api`
- Verify your Modal account has GPU access

## License

This project uses MedGemma 1.5, which is subject to Google's Health AI Developer Foundations terms. Please review the license at:
https://huggingface.co/google/medgemma-1.5-4b-it

## Support

For issues with:
- **MedGemma model**: Check [Hugging Face model page](https://huggingface.co/google/medgemma-1.5-4b-it)
- **Modal platform**: Check [Modal documentation](https://modal.com/docs)
- **This codebase**: Open an issue in your repository

## Disclaimer

⚠️ This model is intended for research and development purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
