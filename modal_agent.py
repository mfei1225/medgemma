import modal
import os
from typing import Optional, Dict, Any
import json

# Create Modal app
# Create Modal app
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

# GPU Configs
# Perception (4B) fits easily on A10G
PERCEPTION_GPU = "A10G"
# Reasoning (27B) needs 80GB A100 to avoid loading OOM (failed on 40GB).
REASONING_GPU = modal.gpu.A100(size="80GB")

@app.cls(
    image=image,
    gpu=PERCEPTION_GPU,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
)
class MedGemmaPerception:
    """
    Stage 1: Perception & Windowing
    Uses MedGemma 1.5 4B to "see" the image.
    """
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

    def _generate(self, image_base64: str, prompt: str) -> str:
        import torch
        from PIL import Image
        import io
        import base64

        # Decode image
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',', 1)[1]
        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=300, 
                do_sample=True,
                temperature=0.2
            )
            
        return self.processor.decode(generation[0], skip_special_tokens=True)

    @modal.method()
    def analyze(self, image_base64: str, task_context: str) -> str:
        """Standard perception analysis."""
        prompt = (
            "Describe this medical image in detail. "
            "1. Identify the body part and anatomical plane (Axial/Coronal/Sagittal). "
            "   - Look for Pelvis landmarks (Bladder, Femoral Heads, Rectum) vs Abdomen (Liver, Kidneys, Spleen). "
            "2. List visible organs and structures. "
            "3. Is the pathology or the region of interest provided by user visible? "
            f"User Input: {task_context}"
        )
        return self._generate(image_base64, prompt)

@app.cls(
    image=image,
    gpu=REASONING_GPU,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
)
class GemmaReasoning:
    """
    Stage 2: Reasoning
    Uses Gemma 2 27B (Quantized) to decide ACTIONS based on findings.
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
    def decide(self, perception_text: str, window_analysis: str, user_request: str, current_state: Dict, history: str) -> Dict[str, Any]:
        import torch
        import re
        import json
        
        # System Prompt for the Reasoner
        system_prompt = f"""You are an expert Medical AI Navigation Agent.
Your goal is to navigate a CT/MRI scan to find specific pathology or anatomy requested by the user.

**Inputs**:
1.  **User Request**: "{user_request}"
2.  **Window Analysis**: "{window_analysis}"
3.  **Visual Findings**: "{perception_text}"
4.  **Current State**: Slice {current_state.get('current_slice')}/{current_state.get('total_slices')}.
5.  **History**: {history}

**Dynamic Reasoning Logic**:
1.  **STOP CONDITION**: If *Visual Findings* confirm target is visible -> STOP.
2.  **Window Check**: If *Window Analysis* says image is poor -> Adjust Window.
3.  **Orientation**: Deduce direction from *History*. If unknown, TEST by scrolling small amount.
4.  **Distance**: Adjust step size (Far=30+, Close=1-5).

**Anatomy Reference**:
Head <-> Neck <-> Chest (Lungs/Heart) <-> Abdomen (Liver/Spleen/Stomach -> Kidneys -> Intestines) <-> Pelvis (Bladder/Reproductive) <-> Legs

**Task**:
Decide the next action.
1.  **Check Window/Level FIRST**: Is the image contrast appropriate for the target?
2.  **Check for Target**: Is it visible? If yes, STOP.
3.  Identify Current Location vs Target Location.
4.  Determine Direction (Cranial or Caudal).
5.  Map Direction to Index Change (Increase or Decrease) based on Orientation Logic.
6.  Determine Step Size based on Distance.

**Output Format**:
Return specific JSON only.
{{
  "thought": "Brief reasoning...",
  "action": {{ "type": "scroll_delta", "step": 30 }} OR {{ "type": "set_window_level", "window": 400, "level": 50 }} OR null
}}
"""
        
        messages = [
            {"role": "user", "content": system_prompt}
        ]
        
        # apply_chat_template with return_tensors="pt" returns a dictionary-like object (BatchEncoding)
        # or a Tensor depending on tokenizer. Safest way is to access input_ids if it's a dict, or use as is if tensor.
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        
        # Determine input_ids
        if isinstance(encodeds, dict) or hasattr(encodeds, 'keys'):
             input_ids = encodeds["input_ids"]
        else:
             input_ids = encodeds
             
        input_ids = input_ids.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=2000, # Increased to avoid truncation
                do_sample=False,
                # temperature removed because do_sample=False
            )
            
        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON with robustness for missing wrappers
        try:
            # 1. Try finding complete JSON block
            match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                parsed = json.loads(json_str)
                # Check if it has 'thought' and 'action'
                if "thought" in parsed and "action" in parsed:
                    return parsed
                # If it's just the action object (e.g. {'type': 'scroll...'})
                if "type" in parsed:
                     return {"thought": generated_text, "action": parsed}
                     
            # 2. Fallback: Model output might be just the thinking text if it failed to output JSON
            return {"thought": generated_text, "action": None}
            
        except Exception as e:
            return {"thought": f"Error parsing: {e} - Raw: {generated_text}", "action": None}

    @modal.method()
    def suggest_window(self, user_request: str) -> Dict[str, Any]:
        """
        Uses medical knowledge to suggest optimal Window/Level purely from text.
        Returns: { "window": int, "level": int, "name": str }
        """
        import torch
        import re
        import json
        
        system_prompt = f"""You are an expert radiologist.
Task: Suggest the optimal DICOM Window Width (WW) and Window Center (WC) for the following user request.

**User Request**: "{user_request}"

**Output Format**:
Return specific JSON only.
{{
  "window": <int>,
  "level": <int>,
  "name": "<string (e.g. Lung Window, Bone Window)>"
}}

**Common Values**:
- Brain: 80, 40
- Subdural: 200, 80
- Stroke: 40, 40
- Temporal Bone: 2800, 600
- Soft Tissue (Abdomen/Chest): 400, 50
- Lung: 1500, -600
- Liver: 150, 30
- Bone: 2000, 500
- Spine: 1800, 400

If the request is generic or unknown, default to Soft Tissue (400, 50).
"""
        messages = [{"role": "user", "content": system_prompt}]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        
        if isinstance(encodeds, dict) or hasattr(encodeds, 'keys'):
             input_ids = encodeds["input_ids"]
        else:
             input_ids = encodeds
             
        input_ids = input_ids.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=100, # Short output
                do_sample=False,
            )
            
        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON
        try:
            match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                return {"window": 400, "level": 50, "name": "Default Soft Tissue"}
        except:
             return {"window": 400, "level": 50, "name": "Default Soft Tissue"}

# --- Separate Endpoints for Frontend Orchestration ---

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=600)
@modal.fastapi_endpoint(method="POST")
def check_window_endpoint(request: dict):
    # Use REASONING model for knowledge-based suggestion
    model = GemmaReasoning()
    analysis = model.suggest_window.remote(request.get("text"))
    return {"window_analysis": analysis}

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=600)
@modal.fastapi_endpoint(method="POST")
def perception_endpoint(request: dict):
    model = MedGemmaPerception()
    analysis = model.analyze.remote(request.get("image_base64"), request.get("text"))
    return {"perception_output": analysis}

@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=600)
@modal.fastapi_endpoint(method="POST")
def reasoning_endpoint(request: dict):
    model = GemmaReasoning()
    result = model.decide.remote(
        perception_text=request.get("perception_output"),
        window_analysis=request.get("window_analysis", "Not checked"),
        user_request=request.get("text"),
        current_state=request.get("current_state", {}),
        history=f"Last Action: {request.get('previous_action')}, Last Thought: {request.get('previous_thought')}"
    )
    return result
