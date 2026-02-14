from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging

from model import model_instance
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MedGemma 1.5 API",
    description="Medical AI inference API using MedGemma 1.5",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ViewportState(BaseModel):
    """Current state of the medical image viewer."""
    window_width: float
    window_center: float
    current_slice: int
    total_slices: int


class InferenceRequest(BaseModel):
    """Request model for inference endpoint."""
    text: str = Field(..., description="Input text or question")
    image_base64: Optional[str] = Field(None, description="Optional base64 encoded image")
    max_new_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    current_state: Optional[ViewportState] = Field(None, description="Current viewer state for agentic control")
    previous_action: Optional[dict] = Field(None, description="Action taken in the previous step (for memory)")
    previous_thought: Optional[str] = Field(None, description="Thought from the previous step (for memory)")
    skip_actions: bool = Field(False, description="If True, skip navigation and force explanation immediately")


class InferenceResponse(BaseModel):
    """Response model for inference endpoint."""
    response: str
    model_id: str
    action: Optional[dict] = Field(None, description="Optional agentic action to execute")
    thought: Optional[str] = Field(None, description="Chain-of-thought reasoning")


class AttentionInferenceRequest(InferenceRequest):
    """Request model for attention-enabled inference."""
    return_attention: bool = Field(True, description="Whether to return attention heatmaps")


class AttentionHeatmap(BaseModel):
    """Attention heatmap for a single token."""
    token: str
    heatmap: str  # Base64 encoded PNG


class AttentionInferenceResponse(BaseModel):
    """Response model for attention-enabled inference."""
    response: str
    model_id: str
    attention_maps: Optional[list[AttentionHeatmap]] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting MedGemma API server...")
    try:
        model_instance.load_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_id": settings.model_id,
        "quantization_enabled": settings.use_quantization,
        "device": settings.device
    }

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Run inference on MedGemma model.
    
    Supports both text-only and multimodal (text + image) inputs.
    """
    try:
        # Pass current_state to generate method
        current_state_dict = request.current_state.dict() if request.current_state else None
        
        response_text, action, thought = model_instance.generate(
            text=request.text,
            image=request.image_base64,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            current_state=current_state_dict,
            previous_action=request.previous_action,
            previous_thought=request.previous_thought,
            skip_actions=request.skip_actions
        )
        
        return InferenceResponse(
            response=response_text,
            model_id=settings.model_id,
            action=action,
            thought=thought
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer_with_attention", response_model=AttentionInferenceResponse)
async def infer_with_attention(request: AttentionInferenceRequest):
    """
    Run inference with attention heatmap visualization.
    
    Returns cross-attention heatmaps showing which image regions
    the model focused on for each generated token.
    
    Requires an image input.
    """
    try:
        if not request.image_base64:
            raise HTTPException(status_code=400, detail="Image is required for attention extraction")
        
        # Generate with attention
        response, attention_data = model_instance.generate_with_attention(
            text=request.text,
            image=request.image_base64,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        
        # Process attention if requested
        attention_maps = None
        if request.return_attention:
            from utils.attention import process_input_token_attentions
            
            # Process input token attentions (where model looked when reading the question)
            try:
                heatmap_data = process_input_token_attentions(
                    attentions=attention_data['attentions'],
                    input_text=request.text,
                    image_size=attention_data['image_size'],
                    num_image_tokens=256,
                    layer_idx=-1  # Use last layer
                )
                
                attention_maps = [
                    AttentionHeatmap(token=item['token'], heatmap=item['heatmap'])
                    for item in heatmap_data
                ]
            except Exception as e:
                logger.warning(f"Failed to process attention: {e}")
                # Continue without attention maps
        
        return AttentionInferenceResponse(
            response=response,
            model_id=settings.model_id,
            attention_maps=attention_maps
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Attention inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MedGemma 1.5 API",
        "version": "1.0.0",
        "model": settings.model_id,
        "endpoints": {
            "health": "/health",
            "inference": "/infer (POST)",
            "attention_inference": "/infer_with_attention (POST)",
            "docs": "/docs"
        }
    }
