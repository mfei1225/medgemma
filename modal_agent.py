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
        "TotalSegmentator",
        "nibabel",
        "scipy",
        "pydicom",
        "requests",
    )
    # Add environment variable for better memory allocation if needed
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .run_commands("pip install scipy nibabel TotalSegmentator pydicom requests")
)

# Shared volume for model caching
model_cache = modal.Volume.from_name("medgemma-cache", create_if_missing=True)

# List of valid TotalSegmentator structures (subset for efficiency/prompting)
VALID_STRUCTURES = [
    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas",
    "adrenal_gland_right", "adrenal_gland_left", "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right", "esophagus",
    "trachea", "thyroid_gland", "small_bowel", "duodenum", "colon", "urinary_bladder",
    "prostate", "kidney_cyst_left", "kidney_cyst_right", "sacrum", "vertebrae_L5", "vertebrae_L4",
    "vertebrae_L3", "vertebrae_L2", "vertebrae_L1", "vertebrae_T12", "vertebrae_T11", "vertebrae_T10",
    "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4",
    "vertebrae_T3", "vertebrae_T2", "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5",
    "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1", "heart", "aorta", "pulmonary_vein",
    "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left", "common_carotid_artery_right",
    "common_carotid_artery_left", "brachiocephalic_vein_left", "brachiocephalic_vein_right", "atrial_appendage_left",
    "superior_vena_cava", "inferior_vena_cava", "portal_vein_and_splenic_vein", "iliac_artery_left",
    "iliac_artery_right", "iliac_vena_left", "iliac_vena_right", "humerus_left", "humerus_right",
    "scapula_left", "scapula_right", "clavicula_left", "clavicula_right", "femur_left", "femur_right",
    "hip_left", "hip_right", "spinal_cord", "gluteus_maximus_left", "gluteus_maximus_right",
    "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left", "gluteus_minimus_right",
    "autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right", "brain", "skull"
]
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

        # DEBUG: Log image stats
        import numpy as np
        if pil_images:
            first_img = np.array(pil_images[0])
            print(f"Received {len(pil_images)} images. Size: {pil_images[0].size}")
            print(f"Img[0] stats: Mean={first_img.mean():.2f}, Std={first_img.std():.2f}, Min={first_img.min()}, Max={first_img.max()}")
            if first_img.mean() < 5:
                print("WARNING: Image seems very dark/black!")

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
    def analyze(self, images_base64: Any, task_context: str) -> Dict:
        # Normalize to list
        if isinstance(images_base64, str):
            images_base64 = [images_base64]

        num_slices = len(images_base64)
        slice_desc = "ONE CT/MRI slice" if num_slices == 1 else f"a stack of {num_slices} CONSECUTIVE CT/MRI slices"

        prompt = f"""
You are a radiology PERCEPTION model analyzing {slice_desc}.
Use ONLY what is visible in these images. Do NOT guess from the user text.

USER TARGET TEXT:
\"\"\"{task_context}\"\"\"

Goal:
1) Identify plane + coarse anatomic region.
2) Decide if the target (structure/finding/ROI) is visible in these slices.
3) If visible, give minimal location cues.

Return ONLY valid JSON. No extra text. No markdown.

Schema:
{{
  "plane": "Axial|Coronal|Sagittal|Unknown",
  "region_guess": "Brain|Neck|Chest|Upper Abdomen|Mid Abdomen|Pelvis|Lower Extremity|Unknown",

  "target_keywords": ["... up to 3 short phrases copied from the user text ..."],

  "target_visible": true|false,

  "target_location": {{
    "laterality": "Left|Right|Midline|Bilateral|Unknown",
    "relative_position": "Anterior|Posterior|Central|Peripheral|Unknown"
  }},

  "visible_structures": ["... up to 5 high-confidence anatomy items ..."],

  "evidence": ["... 1-2 concrete visible cues OR 1-2 concrete reasons not visible ..."],

  "confidence": 0.0
}}

STRICT RULES:
- target_keywords: pick 1–3 key phrases from the user text. If vague, use [].
- PERCEPTION ONLY: describe only what you can directly see in these images.
- EVIDENCE GATE (CRITICAL):
  - You may set target_visible=true ONLY if evidence includes at least ONE concrete visible cue.
  - If you cannot name a concrete visible cue, target_visible MUST be false.
- What counts as a “concrete visible cue” (choose the closest that fits what you see):
  - "focal mass/lesion", "nodule", "abnormal fluid collection/effusion", "free air/gas", "fracture/disruption",
    "hyperdense blood", "edema/swelling", "fat stranding/inflammation", "dilated bowel/obstruction pattern",
    "stone/calcification", "enlarged organ", "vascular dilation/aneurysm", "device/tube/line", "abnormal opacity in lung".
- If region_guess is clearly incompatible with the target_keywords (e.g., brain target but pelvis is shown), set target_visible=false.
- evidence must NEVER be "none". If not visible, say why (e.g., "target organ not in these slices", "no clear abnormality matching keywords").
- visible_structures: max 5 items, anatomy only, high confidence.
- confidence reflects plane+region+target_visible together:
  0.0–0.4 uncertain, 0.5–0.7 moderate, 0.8–1.0 only if clear.

JSON ONLY.
"""


        raw = self._generate(images_base64, prompt)
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

Your ONLY job is to move through slices to locate and clearly visualize the requested USER TARGET.

You are NOT responsible for windowing.
You MUST NEVER suggest window/level changes.
You may ONLY scroll slices or stop.

============================================================
USER TARGET:
"{user_request}"

============================================================
CURRENT PERCEPTION (JSON):
{perception_text}

CURRENT SLICE STATE:
- Current slice index: {current_state.get("current_slice")}
- Total slices: {current_state.get("total_slices")}

============================================================
HISTORY (most recent last):
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
NAVIGATION RULES (STRICT)

1) STOP CONDITION
If:
- target_visible == true
AND
- confidence >= 0.75
Then:
- action = null

You may NOT stop otherwise.

------------------------------------------------------------

2) REGION MATCH RULE
If the anatomical region does NOT match where the USER TARGET should be located,
you MUST scroll toward the correct region.

You may NOT diagnose pathology outside the correct region.

------------------------------------------------------------

3) SLICE ORIENTATION RULE (CRITICAL)
Slice direction is DEFINED as:

- Cranial (toward Brain)  = NEGATIVE step
- Caudal  (toward Pelvis) = POSITIVE step

So:
- To move cranial: step must be < 0
- To move caudal:  step must be > 0

You MUST follow this convention.

------------------------------------------------------------

4) STEP SIZE STRATEGY
Choose step magnitude based on distance:

- Far away (2+ regions away): step = 30
- Moderate distance (1 region away): step = 10
- Close (in correct region but target not visible): step = 3 to 5
- Fine-tuning: step = 1 to 3

------------------------------------------------------------

5) BOUNDARY RULE (CRITICAL)
If history says "Hit volume boundary":
- You MUST reverse direction immediately.
- Reduce step size to 5 or less.

------------------------------------------------------------

CRITICAL CONSTRAINTS
- Output JSON ONLY.
- You MUST NEVER output windowing actions.
- Allowed action types: `scroll_delta` or `segment_structure` or `null`.
- No explanations outside JSON.
- No extra keys beyond "thought" and "action".

============================================================
OUTPUT FORMAT (JSON ONLY)

{{
  "thought": "brief reasoning (1 sentence max)",
  "action": {{ "type": "scroll_delta", "step": <signed int> }}
  OR
  "action": {{ "type": "segment_structure", "structure": "<valid_structure_name>" }}
  OR
  "action": null
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

    @modal.method()
    def identify_structure(self, user_text: str) -> Dict[str, Any]:
        """
        Maps user text (pathology/finding) to a valid TotalSegmentator structure.
        Returns: { "structure": "name" or None, "thought": "reasoning" }
        """
        import torch
        import json
        import re

        system_prompt = f"""
You are a Medical Query Router.
Your job is to map a user's request (pathology, organ, or finding) to a SINGLE anatomical structure from the valid list below.

VALID STRUCTURES:
[""" + ", ".join(VALID_STRUCTURES) + """]

RULES:
1. If the user mentions a specific organ (e.g. "liver", "spleen"), return that structure.
2. If the user mentions a pathology (e.g. "kidney cyst", "tumor in lung"), map it to the specific structure if available (e.g. "kidney_cyst_left") OR the container organ (e.g. "lung_upper_lobe_left").
3. If specific laterality (left/right) is NOT mentioned but required, return the base organ or both if possible? No, pick the most likely or generic one.
   - Actually, if "kidney cyst" (unspecified), return "kidney_right" (as a proxy) OR "kidney_cyst_right" (random guess) OR try to find a generic.
   - Better: Return "kidney_right" and if that fails, the agent can try left?
   - Instruction: If laterality is ambiguous, pick RIGHT side by default or the most common variant.
4. If the entity is NOT in the list (e.g. "appendicitis" but "appendix" is not listed), map to the closest landmark (e.g. "cecum" or "colon").
5. If NO relevant structure is found, return null.

OUTPUT JSON ONLY:
{{
  "thought": "brief reasoning",
  "structure": "valid_structure_name" OR null
}}
"""
        messages = [
            {"role": "user", "content": f"User Request: \"{user_text}\"\n\n{system_prompt}"}
        ]

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        if isinstance(encodeds, dict) or hasattr(encodeds, 'keys'):
             input_ids = encodeds["input_ids"]
        else:
             input_ids = encodeds
        input_ids = input_ids.to(next(self.model.parameters()).device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=200,
                do_sample=False,
            )
        
        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        try:
            parsed = extract_first_json(generated_text)
            if parsed and "structure" in parsed:
                return parsed
            return {"thought": generated_text, "structure": None}
        except:
            return {"thought": "Failed to parse", "structure": None}


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
    cpu=4,
    memory=16384,
)
class SegmentationAgent:
    @modal.method()
    def get_centroid(self, contrast_nifti_path: str, structure_name: str) -> Dict[str, Any]:
        """
        Runs TotalSegmentator on the provided NIfTI file for the specific structure
        and returns the centroid (x, y, z).
        Note: contrast_nifti_path should be a path in the container (e.g. /tmp/...).
        However, since we pass data via Modal, we might need to handle bytes or mount a volume.
        For simplicity, I'll assume we pass the image data as base64 or similar, BUT
        TotalSegmentator works on files. Best to write to a temp file.
        """
        import os
        import nibabel as nib
        import numpy as np
        from totalsegmentator.python_api import totalsegmentator
        from scipy.ndimage import center_of_mass
        import tempfile

        if structure_name not in VALID_STRUCTURES:
            return {"error": f"Structure '{structure_name}' not supported."}

        # For now, let's assume the input is a path to a file already in the volume or similar.
        # But wait, the frontend sends slices. Reconstructing a 3D volume from slices in base64
        # is heavy. 
        # Ideally, `contrast_nifti_path` refers to a file that we can access.
        # IF the frontend sends a stack of images, we need to stack them.
        
        # simplified for this step: assumes we can just import logic
        # Implementation details will depend on how we get the 3D volume.
        
        pass

    @modal.method()
    def get_centroid_from_bytes(self, nifti_bytes: bytes, structure_name: str) -> Dict[str, Any]:
        import os
        import nibabel as nib
        import numpy as np
        from totalsegmentator.python_api import totalsegmentator
        from scipy.ndimage import center_of_mass
        import tempfile

        if structure_name not in VALID_STRUCTURES:
             return {"error": f"Structure '{structure_name}' not supported."}

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_in:
            tmp_in.write(nifti_bytes)
            input_path = tmp_in.name
            
        output_path = input_path.replace(".nii.gz", "_seg.nii.gz")
        
        try:
            # Run Segmentation (Fast -> Normal retry)
            success = self._run_segmentation(input_path, output_path, structure_name)
            
            if not success:
                 return {"found": False, "centroid": None, "message": "Structure not found (segmentation failed or empty)."}

            img = nib.load(output_path)
            data = img.get_fdata()
            com_voxel = center_of_mass(data)
            
            return {
                "found": True,
                "centroid_voxel": com_voxel, 
                "structure": structure_name
            }
            
        except Exception as e:
            return {"error": str(e)}
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)

    @modal.method()
    def get_centroid_from_raw(self, volume_bytes: bytes, metadata: Dict, structure_name: str) -> Dict[str, Any]:
        """
        Creates a NIfTI from raw bytes and metadata, then segments.
        metadata: {rows, columns, slices, spacing: [row_spacing, col_spacing, slice_thickness]}
        """
        import os
        import nibabel as nib
        import numpy as np
        from totalsegmentator.python_api import totalsegmentator
        from scipy.ndimage import center_of_mass
        import tempfile

        if structure_name not in VALID_STRUCTURES:
             return {"error": f"Structure '{structure_name}' not supported."}

        # 1. Reconstruct Numpy Array
        # Assuming Int16 (common for CT)
        # Note: JavaScript sent it as a flat sequence of Int16s
        # We need to reshape it.
        try:
            arr = np.frombuffer(volume_bytes, dtype=np.int16)
            
            rows = metadata.get("rows")
            cols = metadata.get("columns")
            slices = metadata.get("slices")
            
            if len(arr) != rows * cols * slices:
                 # Try Float32?
                 return {"error": f"Data size mismatch. Expected {rows*cols*slices}, got {len(arr)}"}
            
            # Reshape to (rows, cols, slices) -> (x, y, z)
            # DICOM pixel data is usually (rows, cols). Stack is (rows, cols, slices).
            # BUT: Nibabel expects (x, y, z).
            # If we just reshape to (rows, cols, slices), that's usually correct for NIfTI if we set affine right.
            # But let's assume standard orientation for now.
            volume_data = arr.reshape((rows, cols, slices), order='F') # 'F' for column-major? JS is row-major usually?
            # JS flat array: [row0, row1...] -> this is 'C' order (row-major).
            volume_data = arr.reshape((rows, cols, slices), order='C')
            
            # However, prompt says "volumeData.set(pixelData, i * rows * columns)".
            # pixelData is usually row-major.
            # So the flat buffer is slice0_row0, slice0_row1... slice1_row0...
            # This is (slices, rows, cols) in C order if we look at it that way?
            # Wait, JS loop: loadedImages.forEach((img, i) => ... set at i*rows*cols)
            # So the outer dimension is slices.
            # So the flat array is [slice0, slice1, slice2...]
            # Inside a slice: [row0, row1...]
            # So it is (slices, rows, cols).
            
            volume_data = arr.reshape((slices, rows, cols), order='C')
            
            # Nibabel expects (x, y, z) usually (rows, cols, slices) equivalent?
            # It depends on affine.
            # Let's transpose to (rows, cols, slices) which is (y, x, z)? Or (x, y, z).
            # Let's try (cols, rows, slices) = (x, y, z).
            # Usually DICOM images are (rows, cols).
            # Let's swap axes to get (rows, cols, slices).
            volume_data = np.transpose(volume_data, (1, 2, 0))
            
            # 2. Create NIfTI image
            # Simple identity affine with scaling
            spacing = metadata.get("spacing", [1.0, 1.0, 1.0])
            affine = np.diag(spacing + [1.0])
            
            nifti_img = nib.Nifti1Image(volume_data, affine)
            
        except Exception as e:
            return {"error": f"Failed to reconstruct NIfTI: {e}"}

        # 3. Save to temp
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_in:
            nib.save(nifti_img, tmp_in.name)
            input_path = tmp_in.name
            
        output_path = input_path.replace(".nii.gz", "_seg.nii.gz")
        
        try:
            # Run Segmentation (Fast -> Normal retry)
            success = self._run_segmentation(input_path, output_path, structure_name)
            
            if not success:
                 return {"found": False, "centroid": None, "message": "Structure not found (segmentation failed or empty)."}

            img = nib.load(output_path)
            data = img.get_fdata()
            com_voxel = center_of_mass(data)
            
            # Coordinate Transform: Mask Voxel -> World -> Input Voxel
            # 1. Mask Voxel to World
            com_world = nib.affines.apply_affine(img.affine, com_voxel)
            
            # 2. World to Input Voxel
            com_input_voxel = nib.affines.apply_affine(np.linalg.inv(nifti_img.affine), com_world)
            
            return {
                "found": True,
                "centroid_voxel": com_input_voxel.tolist(), 
                "structure": structure_name
            }
            
        except Exception as e:
            return {"error": str(e)}
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)

    @modal.method()
    def get_centroid_from_dicom_urls(self, dicom_urls: list[str], structure_name: str) -> Dict[str, Any]:
        """
        Downloads DICOMs from URLs in parallel, stacks them, converts to NIfTI, and segments.
        Fastest method for cloud-to-cloud transfers.
        """
        import os
        import nibabel as nib
        import numpy as np
        import pydicom
        import requests
        import io
        import concurrent.futures
        import tempfile
        from scipy.ndimage import center_of_mass

        if structure_name not in VALID_STRUCTURES:
             return {"error": f"Structure '{structure_name}' not supported."}

        print(f"Downloading {len(dicom_urls)} DICOMs in parallel...")

        def download_dicom(url, index):
            try:
                # Handle potentially missing protocol or dicomweb: prefix
                clean_url = url
                if clean_url.startswith("dicomweb:"):
                    clean_url = clean_url.replace("dicomweb:", "")
                
                resp = requests.get(clean_url, timeout=10)
                if resp.status_code != 200:
                    print(f"Failed to download {url}: {resp.status_code}")
                    return None
                
                ds = pydicom.dcmread(io.BytesIO(resp.content))
                return (index, ds)
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                return None

        # Download in parallel
        slices = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # Pass index to preserve order if needed, though we should likely sort by InstanceNumber/ImagePosition
            futures = [executor.submit(download_dicom, url, i) for i, url in enumerate(dicom_urls)]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    slices.append(res)
        
        if not slices:
            return {"error": "Failed to download any DICOM files."}

        # Sort by index (URL order) - assumption: URLs are sorted
        # Or better: sort by InstanceNumber if available
        # Let's try to sort by InstanceNumber first, falling back to index
        try:
            slices.sort(key=lambda x: int(x[1].InstanceNumber))
        except:
            print("InstanceNumber sorting failed, using URL order.")
            slices.sort(key=lambda x: x[0])

        sorted_ds = [s[1] for s in slices]
        
        # Stack into 3D volume
        # Expecting (rows, cols) pixel data
        try:
            # Check PixelData type
            # pydicom .pixel_array handles types
            first_ds = sorted_ds[0]
            rows = first_ds.Rows
            cols = first_ds.Columns
            
            # Stack: (slices, rows, cols)
            volume = np.stack([ds.pixel_array for ds in sorted_ds])
            
            # Convert to (rows, cols, slices) => (y, x, z) or (x, y, z)?
            # Nibabel usually wants (x, y, z). 
            # Dicom pixel_array is (y, x) [rows, cols].
            # So volume is (z, y, x).
            # We want (x, y, z).
            # Transpose (2, 1, 0) -> (x, y, z)
            volume_nifti_orient = np.transpose(volume, (2, 1, 0))
            
            # Create Affine
            # We need PixelSpacing and SliceThickness/ImagePosition
            ps = first_ds.PixelSpacing # [row, col] -> [y, x]
            # Spacing vector for affine: [x, y, z]
            # x_spacing = ps[1]
            # y_spacing = ps[0]
            # z_spacing: Calc from ImagePosition (z-diff)
            
            try:
                z_spacing = abs(sorted_ds[1].ImagePositionPatient[2] - sorted_ds[0].ImagePositionPatient[2])
            except:
                z_spacing = first_ds.SliceThickness if hasattr(first_ds, 'SliceThickness') else 1.0
                
            affine = np.diag([ps[1], ps[0], z_spacing, 1.0])
            
            nifti_img = nib.Nifti1Image(volume_nifti_orient, affine)
            
        except Exception as e:
            return {"error": f"Failed to stack/convert DICOMs: {e}"}

        # Save and Segment
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_in:
             nib.save(nifti_img, tmp_in.name)
             input_path = tmp_in.name
             
        output_path = input_path.replace(".nii.gz", "_seg.nii.gz")
        
        try:
            success = self._run_segmentation(input_path, output_path, structure_name)
            
            if not success:
                 return {"found": False, "centroid": None, "message": "Structure not found."}

            img = nib.load(output_path)
            data = img.get_fdata()
            com_voxel = center_of_mass(data)
            
            # Coordinate Transform: Mask Voxel -> World -> Input Voxel
            # This ensures correctness even if mask is lower resolution (fast mode)
            # 1. Mask Voxel to World
            com_world = nib.affines.apply_affine(img.affine, com_voxel)
            
            # 2. World to Input Voxel
            # We use nifti_img.affine (from input) to map world back to input voxel space
            com_input_voxel = nib.affines.apply_affine(np.linalg.inv(nifti_img.affine), com_world)
            
            # RLE Encode Mask for Frontend Overlay
            mask_binary = (data > 0).astype(np.uint8)
            f = mask_binary.flatten()
            f_padded = np.concatenate([[0], f, [0]])
            runs = np.where(f_padded[1:] != f_padded[:-1])[0] + 1
            runs[1::2] -= runs[::2] # Lengths
            

            return {
                "found": True,
                "centroid_voxel": com_input_voxel.tolist(), 
                "structure": structure_name,
                "mask_rle": runs.tolist(),
                "shape": mask_binary.shape,
                "affine": img.affine.tolist(),
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            if os.path.exists(input_path): os.remove(input_path)
            if os.path.exists(output_path): os.remove(output_path)

    def _run_segmentation(self, input_path, output_path, structure_name):
        from totalsegmentator.python_api import totalsegmentator
        import nibabel as nib
        import numpy as np
        
        # Try Fast mode first
        print(f"Attempting FAST segmentation for {structure_name}...")
        try:
            totalsegmentator(input_path, output_path, roi_subset=[structure_name], fast=True, ml=True)
            if os.path.exists(output_path):
                img = nib.load(output_path)
                if np.sum(img.get_fdata()) > 0:
                    print("FAST segmentation successful.")
                    return True
                # If existing but empty, we might want to retry normal mode?
                # TotalSegmentator fast mode is usually adequate, but if it misses, normal might find it.
                print("FAST segmentation produced empty mask. Retrying in NORMAL mode...")
        except Exception as e:
            print(f"FAST segmentation failed: {e}. Retrying in NORMAL mode...")
            
        # Retry Normal mode
        try:
            if os.path.exists(output_path): os.remove(output_path)
            totalsegmentator(input_path, output_path, roi_subset=[structure_name], fast=False, ml=True)
            if os.path.exists(output_path):
                img = nib.load(output_path)
                if np.sum(img.get_fdata()) > 0:
                    print("NORMAL segmentation successful.")
                    return True
        except Exception as e:
             print(f"NORMAL segmentation failed: {e}")
             
        return False




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

@web_app.post("/segment_dicom")
async def segment_dicom_endpoint(request: Request):
    """
    Accepts list of DICOM URLs and a structure name.
    Downloads, stacks, and segments.
    """
    data = await request.json()
    dicom_urls = data.get("dicom_urls", [])
    structure = data.get("structure")
    
    if not dicom_urls or not structure:
        return {"error": "Missing dicom_urls or structure"}
        
    model = SegmentationAgent()
    return model.get_centroid_from_dicom_urls.remote(dicom_urls, structure)

@web_app.post("/segment_dicom")
async def segment_dicom_endpoint(request: Request):
    """
    Accepts list of DICOM URLs and a structure name.
    Downloads, stacks, segments, and determines windowing via LLM.
    """
    import asyncio # Local import
    
    data = await request.json()
    dicom_urls = data.get("dicom_urls", [])
    structure = data.get("structure")
    
    if not dicom_urls or not structure:
        return {"error": "Missing dicom_urls or structure"}
        
    seg_model = SegmentationAgent()
    reasoning_model = GemmaReasoning()
    
    # Parallel execution of Segmentation and Window Reasoning
    try:
        prompt = f"Show me the {structure}"
        results = await asyncio.gather(
            seg_model.get_centroid_from_dicom_urls.remote.aio(dicom_urls, structure),
            reasoning_model.suggest_window.remote.aio(prompt),
            return_exceptions=True
        )
        
        seg_result = results[0]
        window_result = results[1]
        
        # Check segmentation success
        if isinstance(seg_result, Exception):
            print(f"Segmentation failed: {seg_result}")
            return {"error": str(seg_result)}
            
        if seg_result.get("found"):
            # Try to apply LLM windowing
            if (not isinstance(window_result, Exception) and 
                window_result and 
                window_result.get("action") and 
                window_result["action"].get("window") is not None):
                
                print(f"Applying LLM Windowing: {window_result['action']}")
                seg_result["window_level"] = window_result["action"]
                seg_result["window_thought"] = window_result.get("thought")
            else:
                print(f"LLM Windowing failed or yielded no action: {window_result}")
                # Fallback to hardcoded window_level if it exists in seg_result
        
        return seg_result
        
    except Exception as e:
        print(f"Endpoint error: {e}")
        return {"error": f"Endpoint error: {e}"}

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

@web_app.post("/perception")
async def perception_endpoint(request: Request):
    data = await request.json()
    model = MedGemmaPerception()
    # Support both key names: 'images_base64' (list) or 'image_base64' (single/list)
    imgs = data.get("images_base64") or data.get("image_base64")
    analysis = model.analyze.remote(imgs, data.get("text"))
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

@web_app.post("/segment_centroid")
async def segment_centroid_endpoint(request: Request):
    """
    Endpoint to get the centroid of a structure.
    Expects JSON: { "nifti_bytes_base64": "...", "structure": "liver" }
    """
    import base64
    data = await request.json()
    model = SegmentationAgent()
    
    b64_data = data.get("nifti_bytes_base64")
    if not b64_data:
        return {"error": "Missing nifti_bytes_base64"}
        
    structure = data.get("structure")
    
    # Decode base64
    try:
        if "," in b64_data:
            b64_data = b64_data.split(",")[1]
        file_bytes = base64.b64decode(b64_data)
    except Exception as e:
        return {"error": f"Invalid base64: {str(e)}"}
        
    return model.get_centroid_from_bytes.remote(file_bytes, structure)

@web_app.post("/segment_raw")
async def segment_raw_endpoint(request: Request):
    """
    Endpoint to get centroid from raw volume data.
    Expects JSON: { "volume_base64": "...", "metadata": {...}, "structure": "..." }
    """
    import base64
    data = await request.json()
    model = SegmentationAgent()
    
    b64_data = data.get("volume_base64")
    if not b64_data:
        return {"error": "Missing volume_base64"}
        
    metadata = data.get("metadata")
    structure = data.get("structure")
    
    # Decode
    try:
        # It's raw binary data of Int16s
        # JS btoa might have encoding issues if we just did binary string?
        # But we did: new Uint8Array(buffer).reduce... String.fromCharCode
        # That creates a binary string. btoa works on that.
        # Python base64.b64decode should work.
        volume_bytes = base64.b64decode(b64_data)
    except Exception as e:
        return {"error": f"Invalid base64: {str(e)}"}
        
    return model.get_centroid_from_raw.remote(volume_bytes, metadata, structure)



@app.function(image=image, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=600)
@modal.asgi_app()
def api():
    return web_app

