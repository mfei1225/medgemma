import modal
import os
from typing import Optional, Dict, Any
import json
import torch
import re
import base64

# Create Modal app
app = modal.App("medgemma-dual-agent-v11")

# Define container image with dependencies for both models
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "accelerate",
        "bitsandbytes",  # For 4-bit quantization of 27B model
        "pillow",
        "fastapi",
        "pydantic",
    )
    # Add environment variable for better memory allocation if needed
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

# Shared volume for model caching
model_cache = modal.Volume.from_name("medgemma-cache", create_if_missing=True)
def extract_first_json(text: str):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except:
                    return None
    return None
# GPU Configs
# Perception (4B) fits easily on A10G
PERCEPTION_GPU = "A10G"
# Reasoning (27B) needs 80GB A100 to avoid loading OOM (failed on 40GB).
REASONING_GPU = "A100-80GB"

def expected_regions_for_target(task_context: str):
    t = (task_context or "").lower()
    # very small ruleset—add more over time
    if "pericard" in t or "heart" in t or "cardiac" in t or "effusion" in t:
        return {"Chest"}
    if "brain" in t or "intracran" in t or "subdural" in t or "ich" in t:
        return {"Brain"}
    if "pelvis" in t or "bladder" in t or "prostate" in t or "uterus" in t:
        return {"Pelvis"}
    return None

@app.cls(
    image=image,
    gpu=PERCEPTION_GPU,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    cpu=4,
    memory=16384,
)

class MedGemmaPerception:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model_id = "google/medgemma-1.5-4b-it"
        cache_dir = "/cache"
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")

        print(f"Loading Perception Model: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=hf_token, cache_dir=cache_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            token=hf_token,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

    def _generate(self, images_base64: Any, prompt: str) -> str:
        import torch
        from PIL import Image
        import io, base64

        # Normalize input to list
        if isinstance(images_base64, str):
            images_base64 = [images_base64]

        pil_images = []
        for img_b64 in images_base64:
            if img_b64.startswith("data:"):
                img_b64 = img_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(img_b64)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil_images.append(pil_image)

        # Build content with multiple images
        content = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{
            "role": "user",
            "content": content
        }]

        device = next(self.model.parameters()).device

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)

        # IMPORTANT: deterministic output helps JSON compliance a lot
        with torch.no_grad():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=512, # Increased for multi-slice
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_text = self.processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()

        print(f"--- PERCEPTION OUTPUT ---\n{generated_text}\n-------------------------")
        return generated_text


    @modal.method()
    def analyze(self, image_base64: str, task_context: str) -> str:
        prompt = f"""
You are a radiology perception model analyzing ONE CT/MRI slice.

User Target: "{task_context}"

Return ONLY valid JSON. No extra text. No markdown.

Schema:
{{
  "plane": "Axial|Coronal|Sagittal|Unknown",
  "region_guess": "Brain|Neck|Chest|Upper Abdomen|Mid Abdomen|Pelvis|Lower Extremity|Unknown",
  "visible_structures": ["... up to 10 short items ..."],
  "target_visible": true|false,
  "explanation": "1-2 sentences describing ONLY what is visible",
  "confidence": 0.0
}}

4) explanation MUST be 1–2 sentences:
   - What you see that supports plane/region (or why Unknown),
   - and why target_visible is true/false.
   - If uncertain, explicitly say you are uncertain.
5) confidence is your confidence in plane+region+target_visible as a whole:
   - 0.0–0.4 if uncertain
   - 0.5–0.7 if moderate
   - 0.8–1.0 only if clear

Rules:
- Describe ONLY what is strictly visible in THIS slice.
- visible_structures: MAX 6 items. High-confidence anatomy only.
- target_visible: false if the region is wrong (e.g., Heart is NOT visible in Mid Abdomen).
- Do not hallucinate the target simply because the user asked for it.
- JSON ONLY.
"""
        raw = self._generate(image_base64, prompt)
        parsed = extract_first_json(raw)
        if parsed is None:
            return {"ok": False, "raw": raw}
        return {"ok": True, "perception": parsed}


    @modal.method()
    def summarize(self, images_base64: list, task_context: str) -> str:
        # Normalize to list if string passed
        if isinstance(images_base64, str):
            images_base64 = [images_base64]
            
        prompt = f"""
You are a compassionate medical communicator explaining imaging findings to a patient.

IMPORTANT:
- Base your answer ONLY on what is visible in these {len(images_base64)} CONSECUTIVE image slices.
- Do NOT invent details that are not visible.
- If something cannot be determined from these slices, say so clearly.
- Use plain language (8th-grade level).
- Be calm and non-alarming.

USER’S TARGET / CONCERN:
"{task_context}"

Write a patient-friendly explanation with these labeled sections:

1) What is seen (the imaging finding)
- Name/describe the abnormality in neutral terms.

2) Where it is
- Describe the body region and nearby structures.

3) What it looks like
- Describe appearance and progression across slices.

4) What it could mean (possibilities)
- Give 2–4 common, non-technical possibilities.
- Include at least one benign possibility.

5) What we cannot tell from these images
- Mention key limitations.

6) Key takeaway
- 1–2 sentences summarizing the most important point.

Length: 120–200 words. No bullet explosions; keep it readable.
"""
        return self._generate(images_base64, prompt)



@app.cls(
    image=image,
    gpu=REASONING_GPU,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    cpu=8,
    memory=32768, # Increased to 32GB for 27B model loading
)
class GemmaReasoning:
    """
    Stage 2: Reasoning
    """
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        self.model_id = "google/medgemma-27b-it"
        cache_dir = "/cache"
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        
        print(f"Loading Reasoning Model: {self.model_id}")
        
        # 4-bit Quantization Config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=hf_token, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=hf_token,
            cache_dir=cache_dir,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.eval()

    @modal.method()
    def decide(self, perception_text: str, user_request: str, current_state: Dict, history: str) -> Dict[str, Any]:        
        # System Prompt for the Reasoner
        system_prompt = f"""
You are a Medical CT/MRI Slice Navigation Agent.

Your ONLY job is to move through slices to locate and clearly visualize the requested target.

You are NOT responsible for windowing.
You MUST NEVER suggest window/level changes.
You may ONLY scroll or stop.

============================================================
USER TARGET:
"{user_request}"

============================================================

============================================================
CURRENT PERCEPTION (JSON):
{perception_text}
- Current slice: {current_state.get("current_slice")}
- Total slices: {current_state.get("total_slices")}

============================================================
HISTORY:
{history}

============================================================
BODY REGION ORDER (Cranial → Caudal):

Brain
Neck
Chest
Upper Abdomen
Mid Abdomen
Pelvis
Lower Extremity

============================================================
NAVIGATION RULES

1. STOP CONDITION
If:
- target_visible == true
AND
- confidence >= 0.75
Then:
- action = null

You may NOT stop otherwise.

------------------------------------------------------------

2. REGION MATCH RULE
If the anatomical region does NOT match the region where the target organ should be located,
you MUST scroll toward the correct region.

You may NOT diagnose pathology outside the correct region.

------------------------------------------------------------

============================================================
NAVIGATION RULES

1. SEMANTIC MOVEMENT:
   - If target is above (cranial to) current region, use "scroll_cranial".
   - If target is below (caudal to) current region, use "scroll_caudal".
   - Do NOT guess the sign. The frontend handles the mapping.

2. STEP SIZE STRATEGY:

2. AVOID OSCILLATION:
   - Do NOT flip-flop between +30 and -30.
   - If you moved -30 and got closer (e.g., Abdomen -> Chest), your next step MUST be negative (e.g., -10 or -5) to fine-tune.
   - Do NOT go back +30.
orientation direction (index mapping) is uncertain:
→ Perform small probe scroll of ±3 slices.

------------------------------------------------------------

4. STEP SIZE STRATEGY
- Large distance (far region) → step 30
- Moderate distance → step 10
- Close to expected region → step 3–5
- If near boundaries → reduce step size

------------------------------------------------------------

5. ANTI-OSCILLATION & BOUNDARY RULE (CRITICAL)
- If history says "Hit volume boundary":
  You MUST reverse direction immediately.
  (e.g., if you tried to scroll down, now scroll up).
- If last action did not change region_guess:
  Reduce step size.
- If repeated back-and-forth detected:
  Reduce magnitude and maintain direction.

------------------------------------------------------------

CRITICAL CONSTRAINTS
- You MUST output JSON only.
- You MUST NEVER output windowing actions.
- Allowed action types: scroll_delta or null.
- No explanations outside JSON.

============================================================

OUTPUT FORMAT:

{{
  "thought": "brief reasoning",
  "action": 
      {{ "type": "scroll_delta", "step": <int> }}
   OR null
}}
"""

        
        messages = [
            {"role": "user", "content": system_prompt}
        ]
        
        # Get device mapping
        device = next(self.model.parameters()).device

        # apply_chat_template with return_tensors="pt" returns a dictionary-like object (BatchEncoding)
        # or a Tensor depending on tokenizer. Safest way is to access input_ids if it's a dict, or use as is if tensor.
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        
        # Determine input_ids
        if isinstance(encodeds, dict) or hasattr(encodeds, 'keys'):
             input_ids = encodeds["input_ids"]
        else:
             input_ids = encodeds
             
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=2000, # Increased to avoid truncation
                do_sample=False,
                # temperature removed because do_sample=False
            )
            
        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"--- REASONING OUTPUT ---\n{generated_text}\n------------------------")
        
        # Parse JSON with robustness for missing wrappers
        try:
            parsed = extract_first_json(generated_text)
            if parsed and "thought" in parsed and "action" in parsed:
                return parsed
            if parsed and "type" in parsed:
                return {"thought": generated_text, "action": parsed}
            return {"thought": generated_text, "action": None}
        except Exception as e:
            return {"thought": f"Error parsing: {e} - Raw: {generated_text}", "action": None}

    @modal.method()
    def chat(self, messages: list) -> str:
        """
        General chat with the 27B model.
        messages: list of {"role": "user"|"assistant", "content": "..."}
        """
        import torch
        
        # Ensure system prompt is present or added
        if not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {
                "role": "system", 
                "content": "You are MedGemma, a helpful medical AI assistant. Answer healthcare questions accurately and cautiously."
            })
            
        # Gemma requires conversation to start with user (after system)
        # Filter out leading assistant messages (e.g. initial greeting)
        while len(messages) > 1 and messages[1].get("role") == "assistant":
            messages.pop(1)

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        
        if isinstance(encodeds, dict) or hasattr(encodeds, 'keys'):
             input_ids = encodeds["input_ids"]
        else:
             input_ids = encodeds
             
        input_ids = input_ids.to(next(self.model.parameters()).device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
            )
            
        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text

    @modal.method()
    def suggest_window(self, user_request: str) -> Dict[str, Any]:
        """
        Uses medical knowledge to suggest optimal Window/Level purely from text.
        Returns: { "thought": str, "action": { "type": "set_window_level", ... } or null }
        """
        import torch
        import re
        import json
        
        system_prompt = f"""
You are a Medical Imaging VOI (Window/Level) Agent for CT.

Your job: choose a single window width (WW) and window center/level (WC) in Hounsfield Units (HU)
that best visualizes the PRIMARY target in the user request.

You MUST return an ACTION object that can be executed by a DICOM viewer.
You are NOT allowed to scroll slices. Only set window/level or do nothing.

============================================================
USER REQUEST:
"{user_request}"

============================================================
RULES (STRICT):
1) Output JSON ONLY. No prose, no markdown, no extra keys.
2) WW and WC MUST be integers in HU.
3) Choose ONE best preset based on the PRIMARY target.
4) If multiple targets exist, prefer:
   a) most clinically urgent (hemorrhage > stroke > fracture > soft tissue),
   b) otherwise the first explicit target mentioned.
5) If the request is generic/unclear, default to Soft Tissue (400, 50).
6) If the request is clearly NOT CT (e.g., MRI-only terms like "T1", "FLAIR") OR the target is not window-dependent,
   return action = null.

============================================================
PRESET OPTIONS (WW, WC in HU):
- Brain: 80, 40
- Subdural / Hemorrhage: 200, 80
- Stroke (early ischemia): 40, 40
- Temporal Bone: 2800, 600
- Soft Tissue (Abdomen/Chest): 400, 50
- Lung: 1500, -600
- Liver: 150, 30
- Bone: 2000, 500
- Spine: 1800, 400

============================================================
TERM MAPPING (examples):
- "pericardial effusion", "heart", "mediastinum" -> Soft Tissue
- "lung", "pulmonary", "pleura", "pneumothorax" -> Lung
- "fracture", "osseous", "rib", "bone" -> Bone
- "spine", "vertebra", "canal" -> Spine
- "liver", "hepatic" -> Liver
- "brain", "head CT" -> Brain
- "subdural", "ICH", "hemorrhage" -> Subdural
- "stroke", "acute infarct" -> Stroke

============================================================
OUTPUT FORMAT (JSON ONLY):
{{
  "thought": "<brief>",
  "action": {{ "type": "set_window_level", "window": <int>, "level": <int>, "name": "<preset name>" }}
  OR
  "action": null
}}
"""
        messages = [{"role": "user", "content": system_prompt}]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        
        if isinstance(encodeds, dict) or hasattr(encodeds, 'keys'):
             input_ids = encodeds["input_ids"]
        else:
             input_ids = encodeds
             
        input_ids = input_ids.to(next(self.model.parameters()).device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=500, 
                do_sample=False,
            )
            
        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"--- WINDOW AGENT OUTPUT ---\n{generated_text}\n---------------------------")
        
        # Parse JSON and force the {thought, action} wrapper
        try:
            match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                # If the model output the action object directly, wrap it
                if "type" in parsed and "window" in parsed:
                    return {"thought": generated_text.split('{')[0].strip(), "action": parsed}
                # If it's already wrapped, return as is
                if "action" in parsed:
                    return parsed
                return {"thought": generated_text, "action": None}
            else:
                return {"thought": "No JSON found in model output.", "action": None}
        except Exception as e:
             return {"thought": f"Error parsing window suggestion: {e}", "action": None}

# --- Separate Endpoints for Frontend Orchestration ---

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

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

@web_app.post("/check-window")
async def check_window_endpoint(request: Request):
    data = await request.json()
    model = GemmaReasoning()
    # Now returns object with { "thought": ..., "action": ... }
    result = model.suggest_window.remote(data.get("text"))
    return result

@web_app.post("/perception")
async def perception_endpoint(request: Request):
    data = await request.json()
    model = MedGemmaPerception()
    analysis = model.analyze.remote(data.get("image_base64"), data.get("text"))
    return analysis


@web_app.post("/reasoning")
async def reasoning_endpoint(request: Request):
    data = await request.json()
    model = GemmaReasoning()

    perception = data.get("perception_output")
    if isinstance(perception, dict):
        perception = json.dumps(perception, ensure_ascii=False)

    history = f"""
Last Action: {data.get("previous_action")}
Last Thought: {data.get("previous_thought")}
Last Execution Result: {data.get("previous_exec_reason")}
"""

    result = model.decide.remote(
        perception_text=perception,
        user_request=data.get("text"),
        current_state=data.get("current_state", {}),
        history=history
    )
    return result

@web_app.post("/summarize")
async def summarize_endpoint(request: Request):
    data = await request.json()
    model = MedGemmaPerception()
    # Support both key names
    imgs = data.get("images_base64") or data.get("image_base64")
    text = data.get("text")
    analysis = model.summarize.remote(imgs, data.get("text"))
    return {"summary": analysis}

@web_app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    model = GemmaReasoning()
    # Expects { "messages": [ {role, content}, ... ] }
    response = model.chat.remote(data.get("messages", []))
    return {"response": response}

@web_app.post("/summary-multi")
async def summary_multi_endpoint(request: Request):
    """Alias for summarize to support explicit multi-slice calls."""
    return await summarize_endpoint(request)

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=600)
@modal.asgi_app()
def api():
    return web_app

