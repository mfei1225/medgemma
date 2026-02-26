import json
import os
import sys
sys.path.append(os.path.dirname(__file__))

from common import app, CT_STRUCTURES
from segmentation import SegmentationAgent
from normal_atlas import (
    HEAD_CT_DICOM_URLS,
    HEAD_CT_DICOMWEB_URLS,
    HEAD_CT_WW,
    HEAD_CT_WC,
)
from totalseg_tasks import get_task_for_structure

ATLAS_OUTPUT_FILE = "normal_atlas_data.json"

@app.local_entrypoint()
def main():
    agent = SegmentationAgent()
    
    # Load existing atlas to merge into
    atlas = {}
    if os.path.exists(ATLAS_OUTPUT_FILE):
        with open(ATLAS_OUTPUT_FILE, "r") as f:
            atlas = json.load(f)

    # Filter CT_STRUCTURES to only head-related tasks to save time and prevent timeout
    head_tasks = {
        "head_glands_cavities", "head_muscles", "headneck_bones_vessels", 
        "headneck_muscles", "craniofacial_structures", "oculomotor_muscles", 
        "brain_structures", "cerebral_bleed"
    }
    
    head_structures = [
        "brain", "skull", "face"
    ]
    
    for s in CT_STRUCTURES:
        task = get_task_for_structure(s, "CT")
        if task in head_tasks and s not in head_structures:
            head_structures.append(s)

    orientations = [
        ("axial", HEAD_CT_DICOM_URLS, HEAD_CT_DICOMWEB_URLS, HEAD_CT_WW, HEAD_CT_WC),
    ]

    for orn, dicom_urls, dicomweb_urls, ww, wc in orientations:
        print(f"\n======================================")
        print(f"Normal CT (HEAD {orn}): {len(dicom_urls)} slices")
        print(f"Structures to segment: {len(head_structures)}")
        print(f"Window: WW={ww}, WC={wc}")
        print(f"======================================\n")

        print("Starting FINE segmentation (this may take a few minutes)...")
        # Run only the HEAD CT structures
        result = agent.get_centroid_from_dicom_urls.remote(
            dicom_urls,
            head_structures,
            "CT",
            fast_mode=False,
            fallback_to_normal=False,
        )

        if "error" in result:
            print(f"Segmentation failed for {orn}: {result['error']}")
            continue

        detected_orientation = result.get("orientation", orn)
        
        if detected_orientation not in atlas:
            atlas[detected_orientation] = {}

        found = []
        missing = []

        import copy
        # We must NOT overwrite existing good body CT data with 'missing' from head CT!
        for res in result.get("results", []):
            name = res["structure"]
            if res.get("found"):
                # Always safely override if found in head CT
                atlas[detected_orientation][name] = {
                    "dicom_urls": dicomweb_urls,
                    "ww": ww,
                    "wc": wc,
                    "mask_rle": res["mask_rle"],
                    "shape": res["shape"],
                    "centroid_voxel": res["centroid_voxel"],
                    "orientation": detected_orientation,
                }
                found.append(name)
            else:
                missing.append(name)
                
        print(f"[{orn}] Found: {len(found)} structures")
        print(f"[{orn}] Missing: {len(missing)} structures")
        if missing:
            print(f"  Not found: {', '.join(missing[:20])}{'...' if len(missing) > 20 else ''}")

    # Write the combined atlas out once at the very end
    with open(ATLAS_OUTPUT_FILE, "w") as f:
        json.dump(atlas, f)

    size_mb = len(json.dumps(atlas)) / (1024 * 1024)
    print()
    print(f"Saved {ATLAS_OUTPUT_FILE} ({size_mb:.1f} MB)")
