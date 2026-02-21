import modal
import os
from typing import Optional, Dict, Any, List
from common import app, image, model_cache, extract_first_json, get_modality, clean_url

PERCEPTION_GPU = "A10G"
#PERCEPTION_GPU = "A100-40GB" 
@app.cls(
    image=image,
    gpu=PERCEPTION_GPU,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    # keep_warm=1,
    cpu=4,
    memory=16384,
)
class MedGemmaPerception:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model_id = "google/medgemma-1.5-4b-it"
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")

        print(f"Loading Perception Model: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=hf_token, cache_dir="/cache")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            token=hf_token,
            cache_dir="/cache",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def _generate(self, images: List[Any], prompt: str) -> str:
        import torch
        from PIL import Image
        import numpy as np

        if not images:
            return "No image data provided for analysis."

        pil_images = [img.convert("RGB") for img in images]

        # Debug logging
        first_img = np.array(pil_images[0])
        print(f"Received {len(pil_images)} images. Size: {pil_images[0].size}")
        print(f"Img[0] stats: Mean={first_img.mean():.2f}, Std={first_img.std():.2f}, Min={first_img.min()}, Max={first_img.max()}")
        if first_img.mean() < 5:
            print("WARNING: Image seems very dark/black!")

        content = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": prompt})

        device = next(self.model.parameters()).device
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=512,
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
                clean = clean_url(url)
                r = requests.get(clean, timeout=10)
                if r.status_code == 200:
                    data = r.content
                    if data[:2] == b'\x1f\x8b':
                        try:
                            data = gzip.decompress(data)
                        except Exception:
                            pass
                    return pydicom.dcmread(io.BytesIO(data))
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
            return None

        datasets = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
            futures = [exe.submit(fetch, u) for u in urls]
            for f in concurrent.futures.as_completed(futures):
                ds = f.result()
                if ds:
                    datasets.append(ds)

        datasets.sort(key=lambda x: float(
            getattr(x, 'SliceLocation', None) or getattr(x, 'InstanceNumber', 0)
        ))
        return datasets

    def _preprocess_ct_slice(self, ds, target_size=(448, 448)):
        import numpy as np
        from PIL import Image

        arr = ds.pixel_array.astype(np.float32)
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        hu = arr * slope + intercept

        def apply_window(data, w, l):
            lower, upper = l - w / 2, l + w / 2
            return np.clip((data - lower) / (upper - lower), 0, 1)

        c0 = apply_window(hu, 2250, -100)  # Bone/Lung
        c1 = apply_window(hu, 350, 40)     # Soft Tissue
        c2 = apply_window(hu, 80, 40)      # Brain

        img = Image.fromarray((np.stack([c0, c1, c2], axis=-1) * 255).astype(np.uint8))
        return img.resize(target_size, Image.Resampling.BICUBIC)

    def _preprocess_mri_slice(self, ds, target_size=(448, 448)):
        import numpy as np
        from PIL import Image

        arr = ds.pixel_array.astype(np.float32)
        p01, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
        if p99 > p01:
            arr = np.clip(arr, p01, p99)
            arr = (arr - p01) / (p99 - p01)
        else:
            arr = np.zeros_like(arr)

        img = Image.fromarray((np.stack([arr, arr, arr], axis=-1) * 255).astype(np.uint8))
        return img.resize(target_size, Image.Resampling.BICUBIC)

    def _preprocess_generic(self, ds, target_size=(448, 448)):
        import numpy as np
        from PIL import Image

        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        img = Image.fromarray((np.stack([arr, arr, arr], axis=-1) * 255).astype(np.uint8))
        return img.resize(target_size, Image.Resampling.BICUBIC)

    @modal.method()
    def detect_modality(self, dicom_urls: List[str]) -> Dict:
        """Detects modality and orientation from DICOM header tags."""
        if not dicom_urls:
            return {"modality": "Unknown", "orientation": "axial", "confidence": 0.0, "error": "No dicom_urls provided"}
        try:
            import requests, pydicom, io
            import numpy as np
            url = clean_url(dicom_urls[0])
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                ds = pydicom.dcmread(io.BytesIO(r.content))

                mod = str(ds.get("Modality", "Unknown"))
                if "MR" in mod:                             result = {"modality": "MRI",        "confidence": 1.0, "reasoning": "DICOM Header (0008,0060)"}
                elif "CT" in mod:                           result = {"modality": "CT",         "confidence": 1.0, "reasoning": "DICOM Header (0008,0060)"}
                elif "US" in mod:                           result = {"modality": "Ultrasound", "confidence": 1.0, "reasoning": "DICOM Header (0008,0060)"}
                elif any(x in mod for x in ("XR","CR","DX")): result = {"modality": "X-Ray", "confidence": 1.0, "reasoning": "DICOM Header (0008,0060)"}
                else:                                       result = {"modality": mod, "confidence": 1.0, "reasoning": "DICOM Header (0008,0060)"}

                orientation = "axial"
                try:
                    iop = ds.get("ImageOrientationPatient")
                    if iop and len(iop) == 6:
                        row_dir = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
                        col_dir = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
                        normal = np.cross(row_dir, col_dir)
                        normal = normal / (np.linalg.norm(normal) + 1e-10)
                        abs_normal = np.abs(normal)
                        max_idx = int(np.argmax(abs_normal))
                        if max_idx == 1:
                            orientation = "coronal"
                        elif max_idx == 0:
                            orientation = "sagittal"
                except Exception:
                    pass

                result["orientation"] = orientation
                return result
        except Exception as e:
            print(f"DICOM header detection failed: {e}")
        return {"modality": "Unknown", "orientation": "axial", "confidence": 0.0, "error": "Header read failed"}

    @modal.method()
    def summarize(self, dicom_urls: List[str], task_context: str = "") -> str:
        """Summarize a DICOM series for a patient-facing audience."""
        datasets = self._download_dicoms(dicom_urls)
        if not datasets:
            return "Could not load images."

        mode = get_modality(datasets[0])
        preprocess = {
            "CT":  self._preprocess_ct_slice,
            "MRI": self._preprocess_mri_slice,
        }.get(mode, self._preprocess_generic)

        images = [preprocess(ds) for ds in datasets]

        prompt = f"""
You are a compassionate medical communicator. You are looking at {len(images)} consecutive medical image slices.

RULES:
- Describe ONLY what you can directly see in these images.
- If you cannot see something clearly, say so briefly.
- Plain language, 8th-grade level. Calm, non-alarming tone.
- No definitive diagnoses unless unmistakably obvious.

CLINICAL CONTEXT: "{task_context}"

Look carefully at the images and describe:

1) **What area of the body is shown?** (organ, region, and what it normally does)

2) **What do you see?** (describe the finding visible across the slices — shape, brightness, location, how it changes slice to slice)

3) **What could this mean?** (2-3 plain-language possibilities, include at least one benign option)

4) **What can't we tell from these images alone?** (key limitations — what tests or context would help)

5) End with a warm, conversational sentence inviting the user to ask more, then naturally suggest 2-3 follow-up questions inline — NOT as bullet points.
   Write it like: "If you'd like to understand more, you could ask me things like question 1, question 2, or question 3."
   Questions should focus on basic anatomy, physiology, or biology related to what is visible.
   Keep it feeling like a conversation, not a list.

Keep the entire response concise. Let the images guide your answer — if something isn't visible, skip it rather than speculate.
"""
        return self._generate(images, prompt)