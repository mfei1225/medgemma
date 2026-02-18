
import modal
import os
from typing import Optional, Dict, Any, List
from common import app, image, model_cache, extract_first_json

# GPU Configs
# Perception (4B) fits easily on A10G
PERCEPTION_GPU = "A10G"

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

        if not images_base64:
            return "No image data provided for analysis."

        # Normalize input to list
        if isinstance(images_base64, str):
            images_base64 = [images_base64]

        pil_images = []
        for img_input in images_base64:
            if isinstance(img_input, str):
                if img_input.startswith("data:"):
                    img_input = img_input.split(",", 1)[1]
                image_bytes = base64.b64decode(img_input)
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                pil_images.append(pil_image)
            else:
                # Assume it's already a PIL Image object
                pil_images.append(img_input.convert("RGB"))

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

    def _download_dicoms(self, urls: List[str]) -> List[Any]:
        import requests
        import pydicom
        import io
        import gzip
        import concurrent.futures

        def fetch(url):
            try:
                clean = url.replace("dicomweb:", "")
                r = requests.get(clean, timeout=10)
                if r.status_code == 200:
                    data = r.content
                    # Try gzip decompress if magic bytes match
                    if data[:2] == b'\x1f\x8b':
                        try:
                            data = gzip.decompress(data)
                        except Exception:
                            pass  # Not actually gzip, use raw
                    return pydicom.dcmread(io.BytesIO(data))
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
            return None

        datasets = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
            futures = [exe.submit(fetch, u) for u in urls]
            for f in concurrent.futures.as_completed(futures):
                ds = f.result()
                if ds: datasets.append(ds)

        datasets.sort(key=lambda x: int(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
        return datasets

    def _preprocess_ct_slice(self, ds, target_size=(896, 896)):
        import numpy as np
        from PIL import Image

        # 1. Get Rescaled HU
        # pydicom.pixel_data_handlers.util.apply_modality_lut handles rescale slope/intercept
        arr = ds.pixel_array.astype(np.float32)
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        hu = arr * slope + intercept

        # 2. Three Windows (Paper Recipe)
        # Channel 0: Bone/Lung (W:2250, L:-100) -> range [-1225, 1025]
        # Channel 1: Soft Tissue (W:350, L:40) -> range [-135, 215]
        # Channel 2: Brain (W:80, L:40) -> range [0, 80]
        
        def apply_window(data, w, l):
            lower = l - w / 2
            upper = l + w / 2
            return np.clip((data - lower) / (upper - lower), 0, 1)

        c0 = apply_window(hu, 2250, -100)
        c1 = apply_window(hu, 350, 40)
        c2 = apply_window(hu, 80, 40)

        # Stack to RGB
        stacked = np.stack([c0, c1, c2], axis=-1) # (H, W, 3) relative 0-1

        # Resize
        img = Image.fromarray((stacked * 255).astype(np.uint8))
        img = img.resize(target_size, Image.Resampling.BICUBIC)
        
        return img

    def _preprocess_mri_slice(self, ds, target_size=(896, 896)):
        import numpy as np
        from PIL import Image

        arr = ds.pixel_array.astype(np.float32)

        # 1. Percentile Clipping (Robust Scaling)
        p01, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
        if p99 > p01:
            arr = np.clip(arr, p01, p99)
            arr = (arr - p01) / (p99 - p01) # 0-1
        else:
            arr = np.zeros_like(arr) # Flat image handling

        # 2. Replicate to 3 channels
        stacked = np.stack([arr, arr, arr], axis=-1)

        # Resize
        img = Image.fromarray((stacked * 255).astype(np.uint8))
        img = img.resize(target_size, Image.Resampling.BICUBIC)

        return img

    def _preprocess_generic(self, ds, target_size=(896, 896)):
        import numpy as np
        from PIL import Image
        
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        stacked = np.stack([arr, arr, arr], axis=-1)
        img = Image.fromarray((stacked * 255).astype(np.uint8))
        img = img.resize(target_size, Image.Resampling.BICUBIC)
        return img

    @modal.method()
    def detect_modality(self, dicom_urls: List[str]) -> Dict:
        """
        Detects modality from DICOM header tags.
        dicom_urls: list of wadouri/dicomweb URLs — first one is used for header inspection.
        """
        if not dicom_urls:
            return {"modality": "Unknown", "confidence": 0.0, "error": "No dicom_urls provided"}
        try:
            import requests, pydicom, io
            url = dicom_urls[0].replace("dicomweb:", "")
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                ds = pydicom.dcmread(io.BytesIO(r.content))
                mod = str(ds.get("Modality", "Unknown"))
                if "MR" in mod:  return {"modality": "MRI",       "confidence": 1.0, "reasoning": "DICOM Header (0008,0060)"}
                if "CT" in mod:  return {"modality": "CT",        "confidence": 1.0, "reasoning": "DICOM Header (0008,0060)"}
                if "US" in mod:  return {"modality": "Ultrasound", "confidence": 1.0, "reasoning": "DICOM Header (0008,0060)"}
                if any(x in mod for x in ("XR", "CR", "DX")): return {"modality": "X-Ray", "confidence": 1.0, "reasoning": "DICOM Header (0008,0060)"}
                return {"modality": mod, "confidence": 1.0, "reasoning": "DICOM Header (0008,0060)"}
        except Exception as e:
            print(f"DICOM header detection failed: {e}")
        return {"modality": "Unknown", "confidence": 0.0, "error": "Header read failed"}

    @modal.method()
    def summarize(self, images_base64: Optional[List[str]] = None, task_context: str = "", dicom_urls: Optional[List[str]] = None) -> str:
        # Normalize to list if string passed
        if isinstance(images_base64, str):
            images_base64 = [images_base64]

        # Determine Modality & Load Images (Reuse logic?)
        # For brevity, duplicating the DICOM loading logic since it's short
        final_inputs = []
        datasets = self._download_dicoms(dicom_urls)
        if datasets:
            mod_tag = datasets[0].get("Modality", "CT")
            if "MR" in mod_tag: mode = "MRI"
            elif "CT" in mod_tag: mode = "CT"
            else: mode = "Other"
            
            for ds in datasets:
                if mode == "CT": final_inputs.append(self._preprocess_ct_slice(ds))
                elif mode == "MRI": final_inputs.append(self._preprocess_mri_slice(ds))
                else: final_inputs.append(self._preprocess_generic(ds))

            
        prompt = f"""
        You are a compassionate medical communicator. You are looking at {len(final_inputs)} consecutive medical image slices.

        RULES:
        - Describe ONLY what you can directly see in these images.
        - If you cannot see something clearly, say so briefly.
        - Plain language, 8th-grade level. Calm, non-alarming tone.
        - No definitive diagnoses unless unmistakably obvious.

        CLINICAL CONTEXT: "{task_context}"

        Look carefully at the images and describe:

        1) **What type of scan is this?** (CT/MRI/X-ray/Ultrasound — what you can see in the images)

        2) **What area of the body is shown?** (organ, region, and what it normally does)

        3) **What do you see?** (describe the finding visible across the slices — shape, brightness, location, size if measurable, how it changes slice to slice)

        4) **What could this mean?** (2-3 plain-language possibilities, include at least one benign option)

        5) **What can't we tell from these images alone?** (key limitations — what tests or context would help)

        6) End with a warm, conversational sentence inviting the user to ask more, then naturally suggest 2-3 follow-up questions inline — NOT as bullet points.
   - Write it like: "If you'd like to understand more, you could ask me things like question 1, question 2, or question 3."
   - Questions should focus on basic anatomy, physiology, or biology related to what is visible — things Claude can answer from general knowledge.
   - Keep it feeling like a conversation, not a list.

        Keep the entire response concise. Let the images guide your answer — if something isn't visible, skip it rather than speculate.
        """
        return self._generate(final_inputs, prompt)

