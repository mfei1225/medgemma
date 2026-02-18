
import modal
import os
import json
import re

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
        "dicom2nifti",
    )
    # Add environment variable for better memory allocation if needed
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .run_commands("pip install scipy nibabel TotalSegmentator pydicom requests dicom2nifti")
    # Mount the local directory to /root so modules can be imported
    .add_local_dir(".", remote_path="/root")
)

# Shared volume for model caching
model_cache = modal.Volume.from_name("medgemma-cache", create_if_missing=True)

# List of valid TotalSegmentator structures (subset for efficiency/prompting)
# --- CT Classes (117) ---
CT_STRUCTURES = [
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
    "autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right", "brain", "skull",
    "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5", "rib_left_6",
    "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
    "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6",
    "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12",
    "sternum", "costal_cartilages"
]

# --- MR Specific Classes ---
MR_STRUCTURES = [
    # Common shared
    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas",
    "adrenal_gland_right", "adrenal_gland_left", "esophagus", "trachea", "thyroid_gland", 
    "small_bowel", "duodenum", "colon", "urinary_bladder", "prostate", "sacrum", 
    "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1", 
    "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", 
    "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", 
    "vertebrae_T2", "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", 
    "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1", 
    "heart", "aorta", "pulmonary_vein", "inferior_vena_cava", "portal_vein_and_splenic_vein",
    "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right",
    "humerus_left", "humerus_right", "scapula_left", "scapula_right", "clavicula_left", "clavicula_right",
    "femur_left", "femur_right", "hip_left", "hip_right", "spinal_cord", 
    "gluteus_maximus_left", "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right",
    "gluteus_minimus_left", "gluteus_minimus_right", "autochthon_left", "autochthon_right",
    "iliopsoas_left", "iliopsoas_right", "brain", 

    # MR Specific extras from list
    "intervertebral_discs", "patella", "tibia", "fibula", "tarsal", "metatarsal", "phalanges_feet", 
    "ulna", "radius", "face", "subcutaneous_fat", "skeletal_muscle", "torso_fat", 
    "quadriceps_femoris_left", "quadriceps_femoris_right", 
    "thigh_medial_compartment_left", "thigh_medial_compartment_right",
    "thigh_posterior_compartment_left", "thigh_posterior_compartment_right",
    "sartorius_left", "sartorius_right", "deltoid", "supraspinatus", "infraspinatus", 
    "subscapularis", "coracobrachial", "trapezius", "pectoralis_minor", "serratus_anterior",
    "teres_major", "triceps_brachii"
]

# Combined list for validation
VALID_STRUCTURES = list(set(CT_STRUCTURES + MR_STRUCTURES))

def extract_first_json(text: str):
    import json
    import re
    
    # 1. Try to find markdown code block first
    code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except:
            pass

    # 2. Try to find the first '{' and balance braces
    start = text.find("{")
    if start == -1:
        return None
        
    depth = 0
    in_string = False
    escape = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if in_string:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    
                    # 3. Try standard parse
                    try:
                        return json.loads(candidate)
                    except:
                        # 4. Handle double braces {{ ... }} common in some LLM outputs
                        # If candidate starts with {{ and ends with }}, try stripping
                        if candidate.startswith("{{") and candidate.endswith("}}"):
                            try:
                                return json.loads(candidate[1:-1])
                            except:
                                pass
                        
                        # 5. Handle trailing text/garbage inside the block? 
                        # Usually regex handles this better if we just look for valid JSON.
                        # But simple balancing is usually robust enough for pure JSON.
                        pass
                        
    # Last ditch: try finding any valid JSON object using a regex that matches balanced braces (non-nested)
    # This is hard with regex. The iterative approach above is better.
    
    return None
