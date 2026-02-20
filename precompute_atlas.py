"""
One-time script to pre-segment the normal CT reference and save results as JSON.

Usage:
    cd medgemma
    modal run precompute_atlas.py

This runs TotalSegmentator in FINE (non-fast) mode on the normal CT with all
CT structures, then saves the result as normal_atlas_data.json (used by the
atlas endpoint).
"""

import json
from common import app, CT_STRUCTURES
from segmentation import SegmentationAgent
from normal_atlas import (
    NORMAL_CT_DICOM_URLS,
    NORMAL_CT_DICOMWEB_URLS,
    NORMAL_CT_WW,
    NORMAL_CT_WC,
    CORONAL_CT_DICOM_URLS,
    CORONAL_CT_DICOMWEB_URLS,
    CORONAL_CT_WW,
    CORONAL_CT_WC,
)

ATLAS_OUTPUT_FILE = "normal_atlas_data.json"


@app.local_entrypoint()
def main():
    agent = SegmentationAgent()
    atlas = {}

    orientations = [
        ("axial", NORMAL_CT_DICOM_URLS, NORMAL_CT_DICOMWEB_URLS, NORMAL_CT_WW, NORMAL_CT_WC),
        ("coronal", CORONAL_CT_DICOM_URLS, CORONAL_CT_DICOMWEB_URLS, CORONAL_CT_WW, CORONAL_CT_WC),
    ]

    for orn, dicom_urls, dicomweb_urls, ww, wc in orientations:
        print(f"\n======================================")
        print(f"Normal CT ({orn}): {len(dicom_urls)} slices")
        print(f"Structures to segment: {len(CT_STRUCTURES)}")
        print(f"Window: WW={ww}, WC={wc}")
        print(f"======================================\n")

        print("Starting FINE segmentation (this may take 10+ minutes)...")
        result = agent.get_centroid_from_dicom_urls.remote(
            dicom_urls,
            CT_STRUCTURES,
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

        for res in result.get("results", []):
            name = res["structure"]
            if res.get("found"):
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
