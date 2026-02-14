import uvicorn
from config import settings

if __name__ == "__main__":
    print(f"Starting MedGemma server on {settings.api_host}:{settings.api_port}")
    print(f"Model: {settings.model_id}")
    print(f"Quantization: {settings.use_quantization}")
    print(f"Device: {settings.device}")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,  # Disabled to prevent constant reloading from model cache changes
        log_level="info"
    )
