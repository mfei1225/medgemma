"""
Normal Atlas — pre-segmented reference CT data for "Compare with Normal".

The atlas is stored as `normal_atlas_data.json` alongside this file,
generated once by running:  modal run precompute_atlas.py

Each entry is keyed by structure name and contains:
  - dicom_urls       list of dicomweb: URLs for the normal CT
  - ww, wc           window width / center
  - mask_rle         RLE-encoded 3D binary mask
  - shape            [W, H, D]
  - centroid_voxel   [x, y, z] voxel centroid
  - orientation      'axial' | 'coronal' | 'sagittal'
"""

import json
import os
from typing import Optional, Dict, Any, List

# ── Normal CT reference configuration ─────────────────────────────────────────

NORMAL_CT_PREFIX = "https://images.pacsbin.com/dicom/production/-yK_x5lnHY_1.2.840.113711.999999.27013.1665069946.2/"
NORMAL_CT_SUFFIX = ".dcm.gz"
NORMAL_CT_START = 1
NORMAL_CT_END = 220
NORMAL_CT_WW = 350
NORMAL_CT_WC = 1074

CORONAL_CT_PREFIX = "https://images.pacsbin.com/dicom/production/-yK_x5lnHY_1.2.840.113711.999999.27013.1665069946.6/"
CORONAL_CT_SUFFIX = ".dcm.gz"
CORONAL_CT_START = 1
CORONAL_CT_END = 165
CORONAL_CT_WW = 350
CORONAL_CT_WC = 1074

HEAD_CT_PREFIX = "https://images.pacsbin.com/dicom/production/Zk7qRUbE5O_1.2.840.113619.2.437.3.2299157841.358.1663827198.554/"
HEAD_CT_SUFFIX = ".dcm.gz"
HEAD_CT_START = 1
HEAD_CT_END = 269
HEAD_CT_WW = 150
HEAD_CT_WC = 1059


def _build_urls(prefix: str, suffix: str, start: int, end: int) -> List[str]:
    return [
        f"{prefix}{i}{suffix}"
        for i in range(start, end + 1)
    ]


def _build_dicomweb_urls(urls: List[str]) -> List[str]:
    return [f"dicomweb:{url}" for url in urls]


NORMAL_CT_DICOM_URLS = _build_urls(NORMAL_CT_PREFIX, NORMAL_CT_SUFFIX, NORMAL_CT_START, NORMAL_CT_END)
NORMAL_CT_DICOMWEB_URLS = _build_dicomweb_urls(NORMAL_CT_DICOM_URLS)

CORONAL_CT_DICOM_URLS = _build_urls(CORONAL_CT_PREFIX, CORONAL_CT_SUFFIX, CORONAL_CT_START, CORONAL_CT_END)
CORONAL_CT_DICOMWEB_URLS = _build_dicomweb_urls(CORONAL_CT_DICOM_URLS)

HEAD_CT_DICOM_URLS = _build_urls(HEAD_CT_PREFIX, HEAD_CT_SUFFIX, HEAD_CT_START, HEAD_CT_END)
HEAD_CT_DICOMWEB_URLS = _build_dicomweb_urls(HEAD_CT_DICOM_URLS)

# ── Atlas store (loaded from JSON on import) ──────────────────────────────────

_ATLAS_FILE = os.path.join(os.path.dirname(__file__), "normal_atlas_data.json")
_ATLAS: Dict[str, Dict[str, Any]] = {}


def _load_persisted():
    global _ATLAS
    if os.path.exists(_ATLAS_FILE):
        try:
            with open(_ATLAS_FILE, "r") as f:
                _ATLAS = json.load(f)
            print(f"[normal_atlas] Loaded {len(_ATLAS)} structures from {_ATLAS_FILE}")
        except Exception as e:
            print(f"[normal_atlas] Failed to load {_ATLAS_FILE}: {e}")


_load_persisted()


def get_atlas_entry(structure: str, orientation: str = "axial") -> Optional[Dict[str, Any]]:
    """Look up pre-segmented normal data for a given structure across axes."""
    if orientation not in _ATLAS:
        return _ATLAS.get(structure) # Fallback to old flat struct if un-migrated
    return _ATLAS[orientation].get(structure)


def list_available_structures(orientation: str = "axial") -> List[str]:
    """Return structure names that have atlas data."""
    if orientation not in _ATLAS:
        return list(_ATLAS.keys())
    return list(_ATLAS[orientation].keys())


def get_or_compute_atlas_entry(structure: str, orientation: str = "axial") -> Dict[str, Any]:
    """
    Return cached atlas entry. If not cached, attempt on-demand segmentation.
    Prefer running `modal run precompute_atlas.py` once instead.
    """
    cached = get_atlas_entry(structure, orientation)
    if cached is not None:
        return cached

    # On-demand fallback — slow (~1 min), segments only the requested structure
    try:
        from segmentation import SegmentationAgent

        print(f"[normal_atlas] On-demand segmentation for: {structure} ({orientation})")
        agent = SegmentationAgent()
        
        # Determine if structure is primarily a head structure
        head_structures = {"brain", "skull", "face", "eye_left", "eye_right"} # Plus any others that might be queried standalone
        
        if structure in head_structures or "brain" in structure or "skull" in structure:
            dicom_urls = HEAD_CT_DICOM_URLS
            dicomweb_urls = HEAD_CT_DICOMWEB_URLS
            ww = HEAD_CT_WW
            wc = HEAD_CT_WC
        elif orientation == "coronal":
            dicom_urls = CORONAL_CT_DICOM_URLS
            dicomweb_urls = CORONAL_CT_DICOMWEB_URLS
            ww = CORONAL_CT_WW
            wc = CORONAL_CT_WC
        else:
            dicom_urls = NORMAL_CT_DICOM_URLS
            dicomweb_urls = NORMAL_CT_DICOMWEB_URLS
            ww = NORMAL_CT_WW
            wc = NORMAL_CT_WC

        result = agent.get_centroid_from_dicom_urls.remote(
            dicom_urls, [structure]
        )

        detected_orientation = result.get("orientation", orientation)
        for res in result.get("results", []):
            if res.get("found"):
                entry = {
                    "dicom_urls": dicomweb_urls,
                    "ww": ww,
                    "wc": wc,
                    "mask_rle": res["mask_rle"],
                    "shape": res["shape"],
                    "centroid_voxel": res["centroid_voxel"],
                    "orientation": detected_orientation,
                }
                
                if detected_orientation not in _ATLAS:
                    _ATLAS[detected_orientation] = {}
                _ATLAS[detected_orientation][structure] = entry
                
                # Persist so next restart has it
                try:
                    with open(_ATLAS_FILE, "w") as f:
                        json.dump(_ATLAS, f)
                except Exception:
                    pass
                return entry

        return {"error": f"Structure '{structure}' not found in the {orientation} normal CT."}

    except Exception as e:
        return {
            "error": f"Atlas not pre-computed for '{structure}'. Run: modal run precompute_atlas.py",
            "available_structures": list_available_structures(),
        }
