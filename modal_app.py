import modal
import os
from typing import Optional

# Create Modal app
app = modal.App("medgemma-api")

# Define container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.50.0",
        "accelerate",
        "bitsandbytes",
        "fastapi",
        "pillow",
        "pydantic-settings",
    )
)

# Create a volume for model caching
model_cache = modal.Volume.from_name("medgemma-cache", create_if_missing=True)

# GPU configuration - using A10G for cost-effectiveness
GPU_CONFIG = modal.gpu.A10G()


@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],  # Store HF token as Modal secret
    container_idle_timeout=300,  # Keep container alive for 5 minutes
    allow_concurrent_inputs=10,  # Handle up to 10 concurrent requests
)
class MedGemmaInference:
    """MedGemma inference class for Modal deployment."""
    
    @modal.enter()
    def load_model(self):
        """Load model when container starts."""
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
        import logging
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        model_id = "google/medgemma-1.5-4b-it"
        cache_dir = "/cache"
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        
        self.logger.info(f"Loading model: {model_id}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir=cache_dir
        )
        
        # Load model with fp16 for A10G
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model.eval()
        self.logger.info("Model loaded successfully")
    
    @modal.method()
    def generate(
        self,
        text: str,
        image_base64: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate response from MedGemma.
        
        Args:
            text: Input text/question
            image_base64: Optional base64 encoded image
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text response
        """
        import torch
        from PIL import Image
        import io
        import base64
        
        # Prepare image if provided
        pil_image = None
        if image_base64:
            # Remove data URL prefix if present
            if image_base64.startswith('data:'):
                image_base64 = image_base64.split(',', 1)[1]
            # Decode base64
            image_bytes = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Prepare inputs
        if pil_image is not None:
            inputs = self.processor(text=text, images=pil_image, return_tensors="pt")
        else:
            inputs = self.processor(text=text, return_tensors="pt")
        
        # Move to GPU
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )
        
        # Decode
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if generated_text.startswith(text):
            generated_text = generated_text[len(text):].strip()
        
        return generated_text


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_endpoint(method="POST")
def infer_endpoint(request: dict):
    """
    Web endpoint for inference.
    
    Example request:
    {
        "text": "What are the symptoms of pneumonia?",
        "image_base64": null,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9
    }
    """
    inference = MedGemmaInference()
    
    text = request.get("text")
    if not text:
        return {"error": "Missing 'text' field in request"}, 400
    
    try:
        response = inference.generate.remote(
            text=text,
            image_base64=request.get("image_base64"),
            max_new_tokens=request.get("max_new_tokens", 512),
            temperature=request.get("temperature", 0.3),
            top_p=request.get("top_p", 0.9),
        )
        
        return {
            "response": response,
            "model_id": "google/medgemma-1.5-4b-it"
        }
    except Exception as e:
        return {"error": str(e)}, 500


@app.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_id": "google/medgemma-1.5-4b-it",
        "gpu": "A10G"
    }
