
import modal
import os
from typing import Optional, Dict, Any, List
from common import app, image, model_cache, extract_first_json, VALID_STRUCTURES, get_modality, get_valid_structures_for_modality

# Reasoning (27B) needs 80GB A100 to avoid loading OOM (failed on 40GB).
REASONING_GPU = "A100-80GB"
#REASONING_GPU = "H100" 
@app.cls(
    image=image,
    gpu=REASONING_GPU,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    # keep_warm=1,          # UNCOMMENT FOR DEMOS: Always keep 1 container warm to avoid cold starts
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
            cache_dir=cache_dir,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.eval()

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
        Maps user text (pathology/finding) to one or more valid TotalSegmentator structures.
        Returns: { "structures": ["name", ...], "thought": "reasoning" }
        """
        import torch
        import json
        import re

        system_prompt = f"""
You are a Medical Structure Identifier.
Map the user's request to one or more SPECIFIC anatomical structures from the valid list below.

VALID STRUCTURES:
[""" + ", ".join(VALID_STRUCTURES) + """]

RULES:
1. Be as SPECIFIC as possible. Use individual lobes, vessels, or segments when available.
2. If the user mentions a GENERAL region that has sub-structures in the list, return ALL relevant sub-structures.
   Examples:
   - "right lung" → ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
   - "left lung"  → ["lung_upper_lobe_left", "lung_lower_lobe_left"]
   - "lungs" or "bilateral lungs" → ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right", "lung_upper_lobe_left", "lung_lower_lobe_left"]
   - "kidneys" → ["kidney_right", "kidney_left"]
3. If the user mentions a SPECIFIC organ that is a single entry (e.g. "liver", "spleen", "heart"), return just that one.
4. If the user mentions a pathology, map to the relevant anatomical structure(s).
   Example: "right lung opacity" → ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
   Example: "liver lesion" → ["liver"]
5. If laterality is unspecified and required, default to RIGHT side.
6. If NO relevant structure matches, return an empty array.
7. Maximum 5 structures per request.
8. Every structure in the output MUST be from the valid list above. Do not invent names.

OUTPUT JSON ONLY:
{{
  "thought": "brief reasoning",
  "structures": ["structure_1", "structure_2"]
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
                max_new_tokens=300,
                do_sample=False,
            )

        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        try:
            parsed = extract_first_json(generated_text)
            if parsed:
                # Normalize: accept both "structure" (legacy) and "structures"
                structures = parsed.get("structures") or []
                if not structures and parsed.get("structure"):
                    structures = [parsed["structure"]]
                # Filter to only valid names
                structures = [s for s in structures if s in VALID_STRUCTURES]
                return {"thought": parsed.get("thought", ""), "structures": structures}
            return {"thought": generated_text, "structures": []}
        except:
            return {"thought": "Failed to parse", "structures": []}

    @modal.method()
    def route_intent(self, message: str, history: list = []) -> Dict[str, Any]:
        """
        Classifies a user message into one of 7 agent actions.
        Returns: { "action": str, "params": dict }

        Actions:
          adjust_window   - user wants to change CT window/level
          explain_finding - user wants to analyze a radiology finding/impression
          detect_modality - user wants to know what imaging modality is loaded
          show_organ      - user wants to see/segment an organ or structure
          compare_normal  - user wants to compare current scan with a normal reference
          generate_share  - user wants to share or generate a link
          chat            - general medical question, answer directly
        """
        import torch
        import json
        import re

        # Format last few turns for context (keep it short)
        history_text = ""
        if history:
            recent = history[-4:]  # last 2 exchanges max
            for turn in recent:
                role = turn.get("role", "user")
                content = str(turn.get("content", ""))[:200]  # truncate long content
                history_text += f"{role.upper()}: {content}\n"

        system_prompt = f"""You are a medical imaging AI agent router.
Classify the user message into EXACTLY ONE action from this list:

- adjust_window: user explicitly wants to adjust/change/optimize the window, level, contrast, or brightness on a CT scan
- explain_finding: user wants to analyze, explain, or run a diagnostic pipeline on a radiology finding, impression, or report — whether they paste the text inline OR simply ask about it (e.g. "explain my report", "analyze this finding", "what does this impression mean", "run the pipeline on my CT report")
- detect_modality: user wants to know what type of imaging scan is loaded (CT, MRI, X-Ray, etc.)
- show_organ: user wants to see, highlight, segment, or locate an organ or anatomical structure
- compare_normal: user wants to compare the current scan with a normal/healthy/reference CT, or see what normal looks like, or open a side-by-side comparison (e.g. "compare with normal", "show me what normal looks like", "compare this to a healthy scan", "side by side with normal")
- generate_share: user wants to share, copy a link, or send the current session to someone
- chat: general medical question with NO radiology report/finding context, or a follow-up to a prior AI answer

RULES (apply in order, stop at first match):
1. FOLLOW-UP: If the history already has an ASSISTANT response AND the new message is a brief follow-up question using pronouns ("this", "these", "it", "that") OR asks "why", "what causes", "tell me more" about the prior answer → CHAT.
2. FINDING: If the message mentions "radiology report", "impression", "finding", "opacity", "nodule", "lesion", "effusion", "mass", "infiltrate", or asks to "explain/analyze/run" something related to a scan report → explain_finding.
3. COMPARE: If the message asks to compare with normal, healthy, or reference, or asks for a side-by-side comparison → compare_normal.
4. ORGAN: If the message asks to show/find/segment a specific body part → show_organ.
5. DEFAULT: Anything else → chat.

RECENT CONVERSATION:
{history_text if history_text else "(no prior conversation)"}

OUTPUT JSON ONLY. No prose. No markdown.
{{
  "action": "<one of the 7 actions>",
  "params": {{
    "finding": "<copy the finding/impression text from the message if present, else null>",
    "structure": "<organ/structure name if show_organ or compare_normal, else null>",
    "query": "<the user's original message>"
  }}
}}

User message: "{message}"
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
                max_new_tokens=150,
                do_sample=False,
            )

        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"--- ROUTE INTENT OUTPUT ---\n{generated_text}\n---------------------------")

        try:
            parsed = extract_first_json(generated_text)
            if parsed and "action" in parsed:
                valid_actions = {"adjust_window", "explain_finding", "detect_modality", "show_organ", "compare_normal", "generate_share", "chat"}
                if parsed["action"] not in valid_actions:
                    parsed["action"] = "chat"
                if "params" not in parsed:
                    parsed["params"] = {"query": message}
                return parsed
            return {"action": "chat", "params": {"query": message}}
        except Exception as e:
            print(f"route_intent parse error: {e}")
            return {"action": "chat", "params": {"query": message}}
