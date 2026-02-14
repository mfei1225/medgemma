import torch
from transformers import pipeline
from PIL import Image
import io
import base64
from typing import Optional, Union, Tuple
import logging
import json
import re

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedGemmaModel:
    """MedGemma 1.5 model handler for inference."""
    
    def __init__(self):
        self.pipe = None
        self.device = settings.device
        
    def load_model(self):
        """Load the MedGemma model using pipeline API."""
        logger.info(f"Loading MedGemma model: {settings.model_id}")
        logger.info(f"Using quantization: {settings.use_quantization}")
        
        # Determine dtype
        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        # Create pipeline
        # Using official settings provided by user
        self.pipe = pipeline(
            "image-text-to-text",
            model=settings.model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device=self.device,
            token=settings.huggingface_token,
            model_kwargs={
                "cache_dir": settings.cache_dir,
                "attn_implementation": "eager"
            }
        )
        # Note: We don't load self.processor manually to avoid tokenizer/processor mismatches
        # The pipeline handles preprocessing internally.
        
        logger.info("Model loaded successfully")
        
    def prepare_image(self, image_data: Union[str, bytes]) -> Image.Image:
        """
        Convert base64 string or bytes to PIL Image.
        
        Args:
            image_data: Base64 encoded image string or raw bytes
            
        Returns:
            PIL Image object
        """
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if image_data.startswith('data:'):
                image_data = image_data.split(',', 1)[1]
            # Decode base64
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
            
        return Image.open(io.BytesIO(image_bytes))
    

    def _plan_windowing(self, text: str, current_state: dict, image: Optional[Image.Image] = None) -> Tuple[str, Optional[dict]]:
        """
        Stage 1A: WINDOWING AGENT
        Determines if the window/level needs adjustment based on the target tissue.
        """
        system_prompt = f"""You are the Windowing Logic Module. Based on User input, determine if the window/level needs adjustment to see presented pathology.
Current State: Window={current_state.get('window_width', 'N/A')} HU, Level={current_state.get('window_center', 'N/A')} HU.

Common Presets (Window/Level in HU):
- Soft Tissue: 400/50 (Default for abdomen/appendicitis)
- Lung: 1500/-600
- Bone: 1800/400
- Brain: 80/40
- Liver: 150/30
- Vascular/Angio: 600/200

Task:
1. **Identify Target**: Preset for pathology (e.g. "appendicitis" -> Soft Tissue).
2. **Calculate Difference**: Compare Current vs Target.
3. **DECISION**:
   - IF (Delta Window > 30 OR Delta Level > 30): GENERATE "set_window_level" action.
   - ELSE: Return empty action {{}}.

Output format (JSON ONLY):
Example:
{{
  "thought": "Target Soft Tissue (400/50). Current is 329/-172. Delta > 30. MUST adjust.",
  "action": {{ "type": "set_window_level", "window": 400, "level": 50 }}
}}

**CRITICAL**: Return ONLY a valid JSON object. End with '}}'.
User input: {text}"""
        return self._run_agent_prompt(system_prompt, image, temp=0.1)

    def _plan_scrolling(self, text: str, current_state: dict, image: Optional[Image.Image] = None, previous_action: Optional[dict] = None, previous_thought: str = "") -> Tuple[str, Optional[dict]]:
        """
        Stage 1B: SCROLLING AGENT
        Determines navigation based on visible anatomy vs target.
        """
        prev_context = ""
        if previous_action:
            prev_context = f"\nPrevious Action Taken: {previous_action}\nIMPORTANT: Use this to determine direction. Did the last scroll move you closer to or further from the target?"
        
        if previous_thought:
            prev_context += f"\nPrevious Thought: \"{previous_thought}\"\nReflect on this. Was your previous reasoning correct?"

        system_prompt = f"""You are the Scrolling Logic Module. Based on User input, determine if the scroll position needs adjustment to see presented pathology.
Current Slice: {current_state.get('current_slice', 'N/A')}/{current_state.get('total_slices', 'N/A')}.
Anatomical Hierarchy: Neck → Thoracic inlet/upper mediastinum → Aortic arch → Heart → Diaphragm → Liver → Stomach/Spleen → Pancreas → Kidneys/Adrenals → Bowel/Colon → Pelvis
{prev_context}

Task:
1. **Analyze Image**: Identify visible anatomy.
2. **Determine Target**: Anatomical goal (e.g., Gallbladder).
3. **Anatomical Comparison**: Is Target *Cephalad* (Head) or *Caudal* (Feet) relative to current?
   - *Example*: Gallbladder is *Caudal* to Heart, but *Cephalad* to Pelvis.
4. **Orientation Deduction**:
   - IF no history: Scroll +30 to "probe" orientation.
   - IF history exists: Use `previous_action` and `Previous Thought` to see if your last move worked.
   - **Deduction Rule**: 
     - If you moved +30 and saw more *Caudal* anatomy (e.g. went from Heart to Liver), then `+` = Caudal.
     - If you saw more *Cephalad* anatomy (e.g. went from Heart to Neck), then `+` = Cephalad.
5. **Calculate NEW_INDEX**:
   - `NEW_INDEX = Current +/- Offset` (e.g. +/- 30).
   - **CRITICAL**: Calculate final integer. Do NOT put Math (e.g. 60+30) in "action".

Output format (JSON ONLY):
Example:
{{
  "thought": "I see Heart (Slice 30). Target Gallbladder is Caudal. Previous move (+20) showed Liver, which is Caudal to Heart. So Increasing (+) = Caudal. NEW_INDEX = 30 + 40 = 70.",
  "action": {{ "type": "scroll_to_slice", "index": 70 }}
}}

**CRITICAL**: Return ONLY a valid JSON object. End with '}}'.
User input: {text}"""
        thought, action = self._run_agent_prompt(system_prompt, image, temp=0.2)
        
        # Safety Clamp for Scrolling Index
        if action and action.get("type") == "scroll_to_slice":
            total = current_state.get('total_slices', 1000) # Default high if missing
            idx = action.get("index", 0)
            # Clamp between 0 and total-1
            safe_idx = max(0, min(idx, total - 1))
            
            if safe_idx != idx:
                logger.info(f"Clamping scroll index from {idx} to {safe_idx}")
                action["index"] = safe_idx
                # Inject awareness into the thought for the NEXT turn
                thought += f" [SYSTEM NOTE: I attempted to scroll to slice {idx}, but the image only has {total} slices. My action was clamped to {safe_idx}. I may have reached the end of the volume.]"
                
        return thought, action
        
    def _run_agent_prompt(self, system_prompt: str, image: Optional[Image.Image], temp: float) -> Tuple[str, Optional[dict]]:
        """Helper to run a JSON-forcing agent prompt. Returns (thought, action)."""
        logger.info("Running _run_agent_prompt with ROBUST JSON PARSING v3")
        messages = [{"role": "user", "content": []}]
        if image:
             messages[0]["content"].append({"type": "image", "image": image})
        messages[0]["content"].append({"type": "text", "text": system_prompt})
        
        # Pre-fill JSON start with opening quote for thought to force structure
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "{\n  \"thought\": \""}]})
        
        try:
            outputs = self.pipe(messages, max_new_tokens=1024, generate_kwargs={"temperature": temp, "repetition_penalty": 1.2}) 
            print(outputs)
            gen_text_obj = outputs[0]["generated_text"]
            if isinstance(gen_text_obj, list):
                # Pipeline returns full conversation, but we only want the last message's content
                # The last message is the assistant's response which includes our pre-fill + generation
                last_msg_content = gen_text_obj[-1]["content"]
            else:
                last_msg_content = gen_text_obj

            # Extract text from blocks if it's a list (multimodal output format)
            generated_suffix = ""
            if isinstance(last_msg_content, list):
                for block in last_msg_content:
                     if isinstance(block, dict) and block.get("type") == "text":
                         generated_suffix += block.get("text", "")
            else:
                generated_suffix = str(last_msg_content)

            full_json_str = generated_suffix.strip()
            
            # If it doesn't start with the opening brace, it likely means the pipeline returned only the *new* tokens
            # (which would be just the text of the thought, without the pre-fill)
            # OR the pipeline returned result without pre-fill.
            if not full_json_str.startswith('{'):
                full_json_str = '{\n  "thought": "' + full_json_str
            
            # Try to parse the reconstructed JSON
            # Use stack-based extraction to find the first valid JSON object
            stack = []
            json_start = full_json_str.find('{')
            json_end = -1
            
            if json_start != -1:
                for i, char in enumerate(full_json_str[json_start:], start=json_start):
                    if char == '{':
                        stack.append(char)
                    elif char == '}':
                        if stack:
                            stack.pop()
                            if not stack:
                                json_end = i + 1
                                break
            
            if json_end != -1:
                json_str = full_json_str[json_start:json_end]
                try:
                    data = json.loads(json_str)
                    return data.get("thought", ""), data.get("action")
                except json.JSONDecodeError:
                    # Repair attempt: 
                    # If "thought" contains unescaped quotes, it usually breaks between "thought": " and ", "action":
                    # We can try to extract action directly with regex if JSON fails
                    logger.info("Standard JSON parse failed, attempting regex recovery...")
                    
                    thought = ""
                    # Try to extract thought manually with regex
                    thought_match = re.search(r'"thought":\s*"(.*?)"', json_str, re.DOTALL)
                    if thought_match:
                         thought = thought_match.group(1)

                    # Look for the action dictionary specifically
                    action_match = re.search(r'"action":\s*(\{.*?\})', json_str, re.DOTALL)
                    if action_match:
                        try:
                            action_json = action_match.group(1)
                            # Patch attempt for math like "index": 60+30
                            # Replace unquoted arithmetic with its result (very basic)
                            def patch_math(m):
                                try:
                                    # Look for patterns like : 60+30 or : 60 - 30
                                    expr = m.group(1)
                                    if any(op in expr for op in '+-*/'):
                                        # Very basic santization
                                        clean_expr = re.sub(r'[^0-9+\-*/().\s]', '', expr)
                                        return f': {eval(clean_expr)}'
                                except:
                                    pass
                                return m.group(0)
                            
                            action_json = re.sub(r':\s*([0-9+\-*/\s.]{3,})', patch_math, action_json)

                            # Ensure it's valid JSON (or fix it if it's just empty braces)
                            if action_json == '{}':
                                 return thought, {}
                            return thought, json.loads(action_json)
                        except:
                            pass
                    
                    logger.error(f"JSON Recovery failed. Text: {json_str[:200]}...")
            else:
                 logger.warning("No JSON block found in plan output.")
        except Exception as e:
            logger.error(f"Agent prompt failed: {e}")
        return "", None

    # ... (Keeping _generate_explanation unchanged) ...
    # Refactoring generate to orchestrate:
    
    def generate(
        self,
        text: str,
        image: Optional[Union[str, bytes, Image.Image]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        current_state: Optional[dict] = None,
        previous_action: Optional[dict] = None,
        previous_thought: str = "",
        skip_actions: bool = False,
    ) -> Tuple[str, Optional[dict], str]:
        if self.pipe is None: raise RuntimeError("Model not loaded")
        
        pil_image = None
        if image is not None:
            pil_image = image if isinstance(image, Image.Image) else self.prepare_image(image)

        # Stage 1: Navigation Logic
        action = None
        thought = ""
        
        # Only run navigation if state is present AND we are not skipping actions (e.g. forced summary)
        if current_state and not skip_actions:
            # Priority 1: Windowing
            logger.info("Stage 1A: Checking Windowing...")
            thought, action = self._plan_windowing(text, current_state, pil_image)
            print(action)
            # Priority 2: Scrolling (only if no window action)
            if not action:
                logger.info("Stage 1B: Checking Scrolling...")
                thought, action = self._plan_scrolling(text, current_state, pil_image, previous_action, previous_thought)
                
            if action:
                logger.info(f"Action determined: {action}")
                # Use a specific "Moving..." response so user knows agent is working
                return f"I am adjusting the view ({action['type']})...", action, thought
        
        # Stage 2: Explanation (Only if no navigation action OR skipping actions)
        logger.info("Stage 2: Generating Explanation...")
        response_text = self._generate_explanation(text, pil_image, action)
        # For explanation, the thought is effectively the response text itself
        return response_text, None, response_text

    def _generate_explanation(self, text: str, image: Optional[Image.Image], action_taken: Optional[dict]) -> str:
        """
        Stage 2: EXPLAIN
        Generates patient-friendly explanation based on the image and the action performed.
        """
        context = ""
        if action_taken:
            context = f"Context: I have just adjusted the view: {action_taken}. "
            
        system_prompt = f"""You are a helpful medical assistant explaining radiology findings to a patient.
{context}
User Query: {text}

Provide a clear, simple explanation of the findings in the image. Use simple language.
Format visually with Markdown (bolding, lists)."""

        messages = [{"role": "user", "content": []}]
        if image:
            messages[0]["content"].append({"type": "image", "image": image})
        messages[0]["content"].append({"type": "text", "text": system_prompt})
        
        try:
            outputs = self.pipe(messages, max_new_tokens=1024, generate_kwargs={"temperature": 0.3})
            return outputs[0]["generated_text"][-1]["content"]
        except Exception as e:
            return f"Error analyzing image: {e}"




    def generate_with_attention(
        self,
        text: str,
        image: Union[str, bytes, Image.Image],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[str, dict]:
        """
        Generate response with attention weights for visualization.
        
        Args:
            text: Input text/question
            image: Image (base64 string, bytes, or PIL Image) - REQUIRED for attention
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Tuple of (generated_text, attention_data)
            attention_data contains: {
                'attentions': attention tensors,
                'tokens': list of generated tokens,
                'image_size': (height, width)
            }
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if image is None:
            raise ValueError("Image is required for attention extraction")
        
        # Use defaults from settings if not provided
        max_new_tokens = max_new_tokens or settings.max_new_tokens
        temperature = temperature or settings.temperature
        top_p = top_p or settings.top_p
        
        # Prepare image
        if not isinstance(image, Image.Image):
            pil_image = self.prepare_image(image)
        else:
            pil_image = image
        
        # Get image size
        image_size = pil_image.size  # (width, height)
        image_size = (image_size[1], image_size[0])  # Convert to (height, width)
        
        # Build message in chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": text}
                ]
            }
        ]
        
        # Try to use pipeline with output_attentions
        try:
            # Attempt with pipeline (may not support output_attentions)
            output = self.pipe(
                messages, 
                max_new_tokens=max_new_tokens,
                output_attentions=True,
                return_dict_in_generate=True
            )
            
            # Extract basic info
            if isinstance(output, list) and len(output) > 0:
                generated_text = output[0].get("generated_text", "")
                # If generated_text is a list (from messages format), get the last content
                if isinstance(generated_text, list) and len(generated_text) > 0:
                    generated_text = generated_text[-1].get("content", "")
                attentions = output[0].get('attentions', None)
            else:
                attentions = getattr(output, 'attentions', None)
                generated_text = getattr(output, 'generated_text', "")
            
            if attentions is None:
                logger.warning("Pipeline did not return attentions, using fallback")
                raise AttributeError("No attentions in pipeline output")
                
        except (TypeError, AttributeError, KeyError) as e:
            logger.info(f"Pipeline doesn't support attention extraction: {e}")
            logger.info("Falling back to direct model access")
            
            # Fallback: Use direct model access (requires accessing pipe internals)
            # This is a workaround if pipeline doesn't support output_attentions
            model = self.pipe.model
            # For PaliGemma, the pipeline.tokenizer often behaves as the processor
            # but we'll be careful here
            processor = getattr(self.pipe, 'processor', self.pipe.tokenizer)
            
            # Apply chat template
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = inputs.to(self.device)
            input_len = inputs["input_ids"].shape[-1]

            # Determine sampling parameters
            do_sample = temperature is not None and temperature > 0
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "output_attentions": True,
                "return_dict_in_generate": True
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
                
            # Generate with attentions
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    **gen_kwargs
                )
            
            # Extract generated tokens
            generation = outputs.sequences[0][input_len:]
            generated_text = processor.decode(generation, skip_special_tokens=True)
            
            # Get attentions - they're in a specific format from generate
            # outputs.attentions is a tuple of tuples: (decoder_attentions_for_each_generated_token)
            attentions = outputs.attentions
            
            # Debug logging
            logger.info(f"Attention structure: {type(attentions)}, length: {len(attentions) if attentions else 0}")
            if attentions and len(attentions) > 0:
                logger.info(f"First attention element: {type(attentions[0])}, length: {len(attentions[0]) if hasattr(attentions[0], '__len__') else 'N/A'}")
                if hasattr(attentions[0], '__len__') and len(attentions[0]) > 0:
                    first_layer = attentions[0][0] if isinstance(attentions[0], tuple) else attentions[0]
                    logger.info(f"First layer shape: {first_layer.shape if hasattr(first_layer, 'shape') else 'N/A'}")
        
        # Prepare attention data
        attention_data = {
            'attentions': attentions,
            'image_size': image_size,
            'generated_text': generated_text
        }
        
        return generated_text, attention_data


# Global model instance
model_instance = MedGemmaModel()


