
import modal
import os
from typing import Optional, Dict, Any, List
from common import app, image, model_cache, VALID_STRUCTURES

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
    def get_centroid_from_dicom_urls(self, dicom_urls: list[str], structure_name: str, modality: Optional[str] = None) -> Dict[str, Any]:
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
        import shutil
        import dicom2nifti
        from scipy.ndimage import center_of_mass
        
        # If modality is passed as "MRI", normalize to "MR" for consistency
        if modality and modality.upper() in ["MRI", "MR"]:
            modality = "MR"
        elif modality and modality.upper() == "CT":
            modality = "CT"
        else:
            modality = None # Let detection handle it if invalid/unknown

        if structure_name not in VALID_STRUCTURES:
             return {"error": f"Structure '{structure_name}' not supported."}

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
        datasets_with_idx = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
            futures = [exe.submit(download_dicom, u, i) for i, u in enumerate(dicom_urls)]
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                if res: datasets_with_idx.append(res)
        
        if not datasets_with_idx:
             return {"error": "Failed to download any DICOMs."}

        # Sort by index (preserves original order)
        datasets_with_idx.sort(key=lambda x: x[0])
        sorted_ds = [x[1] for x in datasets_with_idx]

        # Use temporary directory for dicom2nifti
        tmp_dir = tempfile.mkdtemp()
        dcm_dir = os.path.join(tmp_dir, "dcms")
        os.makedirs(dcm_dir)
        
        for i, ds in enumerate(sorted_ds):
            ds.save_as(os.path.join(dcm_dir, f"slice_{i:03d}.dcm"))
            
        nifti_tmp = os.path.join(tmp_dir, "input.nii.gz")
        output_path = os.path.join(tmp_dir, "seg.nii.gz")

        # Convert to NIfTI
        try:
            dicom2nifti.dicom_series_to_nifti(dcm_dir, nifti_tmp, reorient_nifti=True)
        except Exception as e:
            shutil.rmtree(tmp_dir)
            return {"error": f"DICOM to NIfTI conversion failed: {e}"}

        input_path = nifti_tmp
        
        # Load the NIfTI to get the affine for coordinate transforms later
        try:
            nifti_img = nib.load(input_path)
        except Exception as e:
            shutil.rmtree(tmp_dir)
            return {"error": f"Failed to load generated NIfTI: {e}"}
             
        # Detect Modality from the first DICOM (if available) OR use provided override
        scan_modality = "CT" # Default
        
        # 1. Use override if provided
        if modality:
            scan_modality = modality
            print(f"Using explicit modality override: {scan_modality}")
        
        # 2. Fallback to detection if no override
        elif sorted_ds:
            try:
                mod = sorted_ds[0].get("Modality", "CT")
                if "MR" in mod: 
                    scan_modality = "MR"
            except:
                pass
            print(f"Detected Modality from DICOM: {scan_modality}")
        
        try:
            success = self._run_segmentation(input_path, output_path, structure_name, scan_modality)
            
            if not success:
                 return {"found": False, "centroid": None, "message": "Structure not found."}

            img = nib.load(output_path)
            data = img.get_fdata()
            com_voxel = center_of_mass(data)
            
            # Coordinate Transform: Mask Voxel -> World -> Input Voxel
            # 1. Mask Voxel to World
            com_world = nib.affines.apply_affine(img.affine, com_voxel)
            
            # 2. World to Input Voxel
            com_input_voxel = nib.affines.apply_affine(np.linalg.inv(nifti_img.affine), com_world)
            
            # RLE Encode Mask
            mask_binary = (data > 0).astype(np.uint8)
            f = mask_binary.flatten()
            f_padded = np.concatenate([[0], f, [0]])
            runs = np.where(f_padded[1:] != f_padded[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            
            return {
                "found": True,
                "centroid_voxel": com_input_voxel.tolist(), 
                "structure": structure_name,
                "mask_rle": runs.tolist(),
                "shape": list(mask_binary.shape),
                "affine": img.affine.tolist(),
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

    def _run_segmentation(self, input_path, output_path, structure_name, modality="CT"):
        from totalsegmentator.python_api import totalsegmentator
        import nibabel as nib
        import numpy as np
        import os
        
        # Select task based on modality
        task_name = "total"
        if modality == "MR":
            task_name = "total_mr"
            
        print(f"Running TotalSegmentator task='{task_name}' for structure='{structure_name}'...")
        
        # Try Fast mode first
        print(f"Attempting FAST segmentation for {structure_name}...")
        try:
            totalsegmentator(input_path, output_path, roi_subset=[structure_name], fast=True, ml=True, task=task_name)
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
            # Note: fast=False is default
            totalsegmentator(input_path, output_path, roi_subset=[structure_name], fast=False, ml=True, task=task_name)
            if os.path.exists(output_path):
                img = nib.load(output_path)
                if np.sum(img.get_fdata()) > 0:
                    print("NORMAL segmentation successful.")
                    return True
        except Exception as e:
             print(f"NORMAL segmentation failed: {e}")
             
        return False
