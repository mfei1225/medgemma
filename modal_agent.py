
import modal
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json

# Import from modules
from common import app, image
from perception import MedGemmaPerception
from reasoning import GemmaReasoning
from segmentation import SegmentationAgent

# --- FastAPI App for Frontend Orchestration ---

web_app = FastAPI()

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    print(f"Response status: {response.status_code}")
    return response

@web_app.post("/segment_dicom")
async def segment_dicom_endpoint(request: Request):
    """
    Accepts list of DICOM URLs and a structure name.
    Downloads, stacks, and segments.
    """
    data = await request.json()
    dicom_urls = data.get("dicom_urls", [])
    structure = data.get("structure")
    modality = data.get("modality") # Optional: "CT", "MR", or None
    
    if not dicom_urls or not structure:
        return {"error": "Missing dicom_urls or structure"}
        
    model = SegmentationAgent()
    return model.get_centroid_from_dicom_urls.remote(dicom_urls, structure, modality)

@web_app.post("/check-window")
async def check_window_endpoint(request: Request):
    data = await request.json()
    model = GemmaReasoning()
    # Now returns object with { "thought": ..., "action": ... }
    result = model.suggest_window.remote(data.get("text"))
    return result

@web_app.post("/identify_structure")
async def identify_structure_endpoint(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text: return {"structure": None}
    
    model = GemmaReasoning()
    return model.identify_structure.remote(text)

@web_app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    model = GemmaReasoning()
    # Expects { "messages": [ {role, content}, ... ] }
    response = model.chat.remote(data.get("messages", []))
    return {"response": response}

@web_app.post("/agent-route")
async def agent_route_endpoint(request: Request):
    """
    Classifies user message into one of 6 agent actions.
    Returns: { "action": "adjust_window"|"explain_finding"|"detect_modality"|"show_organ"|"generate_share"|"chat", "params": {...} }
    """
    data = await request.json()
    message = data.get("message", "")
    history = data.get("history", [])
    if not message:
        return {"action": "chat", "params": {"query": ""}}
    model = GemmaReasoning()
    return model.route_intent.remote(message=message, history=history)


@web_app.post("/detect-modality")
async def detect_modality_endpoint(request: Request):
    data = await request.json()
    dicom_urls = data.get("dicom_urls", [])
    if not dicom_urls:
        return {"error": "Missing dicom_urls"}
    model = MedGemmaPerception()
    return model.detect_modality.remote(dicom_urls=dicom_urls)

@web_app.post("/summary-multi")
async def summary_multi_endpoint(request: Request):
    data = await request.json()
    model = MedGemmaPerception()
    imgs = data.get("images_base64")
    text = data.get("text")
    dicom_urls = data.get("dicom_urls") # Optional
    
    if not imgs and not dicom_urls:
        return {"error": "Missing images_base64 or dicom_urls"}
        
    summary = model.summarize.remote(images_base64=imgs, task_context=text, dicom_urls=dicom_urls)
    return {"summary": summary}


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=600)
@modal.asgi_app()
def api():
    return web_app
