
import modal
import json
import os
import jwt
from supabase import create_client, Client
from fastapi import FastAPI, Request, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

# Import from modules
from common import app, image
from perception import MedGemmaPerception
from reasoning import GemmaReasoning
from segmentation import SegmentationAgent
from normal_atlas import get_or_compute_atlas_entry, list_available_structures

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

# --- Auth Dependency ---
security = HTTPBearer()

def verify_supabase_jwt(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    secret = os.environ.get("SUPABASE_JWT_SECRET")
    if not secret:
        # Fail open locally if secret isn't mounted, but fail closed in prod
        print("WARNING: SUPABASE_JWT_SECRET not found in environment.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: missing JWT secret."
        )
    
    try:
        # Supabase uses HS256 to sign its JWTs
        payload = jwt.decode(
            token, 
            secret, 
            algorithms=["HS256"], 
            options={"verify_aud": False}
        )
        user_id = payload.get("sub")
        
        # --- Credit Verification and Decrement ---
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            print("WARNING: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY missing. Skipping credit check.")
            return payload
            
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Fetch current credits
        response = supabase.table("chat_credits").select("credits_remaining").eq("user_id", user_id).maybe_single().execute()
        
        if response.data:
            current_credits = response.data.get("credits_remaining", 0)
        else:
            # First time user? They should have been initialized on frontend, but just in case:
            current_credits = 15
            supabase.table("chat_credits").insert({"user_id": user_id, "credits_remaining": current_credits}).execute()
            
        if current_credits <= 0:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Out of AI credits"
            )
            
        # Decrement credits
        supabase.table("chat_credits").update({"credits_remaining": current_credits - 1}).eq("user_id", user_id).execute()
        
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        print(f"JWT Verification Failed: {repr(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {repr(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

@web_app.post("/segment_dicom")
async def segment_dicom_endpoint(request: Request, user: dict = Security(verify_supabase_jwt)):
    """
    Accepts list of DICOM URLs and one or more structure names.
    Downloads, stacks, and segments all in one TotalSegmentator pass.
    """
    data = await request.json()
    dicom_urls = data.get("dicom_urls", [])
    structures = data.get("structures") or []
    if not structures and data.get("structure"):
        structures = [data["structure"]]
    modality = data.get("modality")
    orientation = data.get("orientation")

    if not dicom_urls or not structures:
        return {"error": "Missing dicom_urls or structures"}

    model = SegmentationAgent()
    return model.get_centroid_from_dicom_urls.remote(
        dicom_urls, structures, modality, orientation=orientation
    )

@web_app.post("/check-window")
async def check_window_endpoint(request: Request, user: dict = Security(verify_supabase_jwt)):
    data = await request.json()
    model = GemmaReasoning()
    # Now returns object with { "thought": ..., "action": ... }
    result = model.suggest_window.remote(data.get("text"))
    return result

@web_app.post("/identify_structure")
async def identify_structure_endpoint(request: Request, user: dict = Security(verify_supabase_jwt)):
    data = await request.json()
    text = data.get("text")
    if not text: return {"structure": None}
    
    model = GemmaReasoning()
    return model.identify_structure.remote(text)

@web_app.post("/chat")
async def chat_endpoint(request: Request, user: dict = Security(verify_supabase_jwt)):
    data = await request.json()
    model = GemmaReasoning()
    # Expects { "messages": [ {role, content}, ... ] }
    response = model.chat.remote(data.get("messages", []))
    return {"response": response}

@web_app.post("/agent-route")
async def agent_route_endpoint(request: Request, user: dict = Security(verify_supabase_jwt)):
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
async def detect_modality_endpoint(request: Request, user: dict = Security(verify_supabase_jwt)):
    data = await request.json()
    dicom_urls = data.get("dicom_urls", [])
    if not dicom_urls:
        return {"error": "Missing dicom_urls"}
    model = MedGemmaPerception()
    return model.detect_modality.remote(dicom_urls=dicom_urls)

@web_app.post("/summary-multi")
async def summary_multi_endpoint(request: Request, user: dict = Security(verify_supabase_jwt)):
    data = await request.json()
    model = MedGemmaPerception()
    text = data.get("text")
    dicom_urls = data.get("dicom_urls") # Optional
    
    if not dicom_urls:
        return {"error": "Missing dicom_urls"}
        
    summary = model.summarize.remote(task_context=text, dicom_urls=dicom_urls)
    return {"summary": summary}


@web_app.get("/normal-atlas/{structure}")
async def normal_atlas_endpoint(structure: str, orientation: str = "axial", user: dict = Security(verify_supabase_jwt)):
    """
    Return pre-segmented normal CT data for a given structure.
    If not cached, segments the normal CT on-demand (first call is slower).
    """
    return get_or_compute_atlas_entry(structure, orientation)


@web_app.get("/normal-atlas")
async def normal_atlas_list_endpoint(orientation: str = "axial", user: dict = Security(verify_supabase_jwt)):
    """List all structures that have cached atlas data."""
    return {"structures": list_available_structures(orientation)}


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("supabase-secret")], timeout=600)
@modal.asgi_app()
def api():
    return web_app
