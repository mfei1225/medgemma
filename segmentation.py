import modal
import os
from typing import Optional, Dict, Any, List
from common import app, image, model_cache, get_modality, get_valid_structures_for_modality, clean_url

@app.cls(
    image=image,
    gpu="H100",
    #gpu="A10G",
    volumes={"/cache": model_cache},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("totalsegmentator-secret")
    ],
    timeout=3600,
    scaledown_window=300,
    #keep_warm =1,
    cpu=4,
    memory=16384,
)

#A100-40GB

class SegmentationAgent:

    @modal.enter()
    def preload_models(self):
        """Pre-warm TotalSegmentator models on container start to avoid cold-start latency."""
        try:
            from totalsegmentator.libs import download_pretrained_weights
            download_pretrained_weights(task="total")
            download_pretrained_weights(task="total_mr")
            print("TotalSegmentator models pre-loaded.")
        except Exception as e:
            print(f"Model pre-load warning (non-fatal): {e}")

    @modal.method()
    def get_centroid_from_dicom_urls(
        self,
        dicom_urls: list[str],
        structure_names: list[str],
        modality: Optional[str] = None,
        orientation: Optional[str] = None,
        fast_mode: bool = True,
        fallback_to_normal: bool = True,
    ) -> Dict[str, Any]:
        """
        Downloads DICOMs, converts to NIfTI, segments one or more structures
        in a single TotalSegmentator pass, and returns per-structure RLE masks.
        """
        import asyncio
        import io
        import tempfile
        import shutil

        import httpx
        import nibabel as nib
        import numpy as np
        import pydicom
        import dicom2nifti.convert_dicom as convert_dicom
        from scipy.ndimage import center_of_mass

        if not structure_names:
            return {"error": "No structure names provided."}

        # Normalize modality
        if modality:
            upper = modality.upper()
            if upper in ("MRI", "MR"):
                modality = "MR"
            elif upper == "CT":
                modality = "CT"
            else:
                modality = None

        # ── 1. Async download ──────────────────────────────────────────────────
        async def fetch_all(urls):
            async with httpx.AsyncClient(timeout=15) as client:
                async def fetch(url, index):
                    clean = clean_url(url)
                    try:
                        resp = await client.get(clean)
                        if resp.status_code != 200:
                            print(f"Failed {url}: {resp.status_code}")
                            return None
                        ds = pydicom.dcmread(io.BytesIO(resp.content))
                        return (index, ds)
                    except Exception as e:
                        print(f"Error downloading {url}: {e}")
                        return None

                results = await asyncio.gather(*[fetch(u, i) for i, u in enumerate(urls)])
                return [r for r in results if r is not None]

        datasets_with_idx = asyncio.run(fetch_all(dicom_urls))

        if not datasets_with_idx:
            return {"error": "Failed to download any DICOMs."}

        datasets_with_idx.sort(key=lambda x: x[0])
        sorted_ds = [x[1] for x in datasets_with_idx]

        # ── 2. Detect modality ─────────────────────────────────────────────────
        if modality:
            scan_modality = modality
            print(f"Using explicit modality override: {scan_modality}")
        else:
            scan_modality = "CT"
            try:
                mod = sorted_ds[0].get("Modality", "CT")
                if "MR" in mod:
                    scan_modality = "MR"
            except Exception:
                pass
            print(f"Detected modality: {scan_modality}")

        valid = get_valid_structures_for_modality(scan_modality)
        structure_names = [s for s in structure_names if s in valid]
        if not structure_names:
            return {"error": "None of the requested structures are supported for this modality."}

        # ── 3. Detect orientation if not provided ──────────────────────────────
        scan_orientation = orientation if orientation else "axial"
        if not orientation:
            try:
                ds = sorted_ds[0]
                iop = ds.get("ImageOrientationPatient")
                if iop and len(iop) == 6:
                    row_dir = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
                    col_dir = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
                    normal = np.cross(row_dir, col_dir)
                    normal = normal / (np.linalg.norm(normal) + 1e-10)
                    abs_normal = np.abs(normal)
                    max_idx = np.argmax(abs_normal)
                    if max_idx == 1:
                        scan_orientation = "coronal"
                    elif max_idx == 0:
                        scan_orientation = "sagittal"
            except Exception as orient_err:
                print(f"Orientation detection failed: {orient_err}, defaulting to axial")
        print(f"Orientation: {scan_orientation}")

        # ── 4. In-memory DICOM → NIfTI ────────────────────────────────────────
        tmp_dir = tempfile.mkdtemp()
        nifti_tmp = os.path.join(tmp_dir, "input.nii.gz")
        output_dir = os.path.join(tmp_dir, "seg_output")

        try:
            result = convert_dicom.dicom_array_to_nifti(sorted_ds, nifti_tmp, reorient_nifti=True)
            nifti_img = result["NII"]
        except Exception as e:
            shutil.rmtree(tmp_dir)
            return {"error": f"DICOM to NIfTI conversion failed: {e}"}

        # ── 5. Segment all structures organized by tasks ──────────────────────
        try:
            import sys
            sys.path.append(os.path.dirname(__file__))
            from totalseg_tasks import group_structures_by_task
            
            task_groups = group_structures_by_task(structure_names, scan_modality)
            
            for task_name, items in task_groups.items():
                self._run_segmentation(
                    nifti_tmp, output_dir, items, task_name=task_name,
                    fast_mode=fast_mode, fallback_to_normal=fallback_to_normal,
                )

            # ── 6. Process each structure's binary mask ───────────────────────
            inv_affine = np.linalg.inv(nifti_img.affine)
            per_structure = []

            for sname in structure_names:
                seg_path = os.path.join(output_dir, f"{sname}.nii.gz")
                if not os.path.exists(seg_path):
                    per_structure.append({"structure": sname, "found": False})
                    continue

                img = nib.load(seg_path)
                data = img.get_fdata()
                if np.sum(data) == 0:
                    per_structure.append({"structure": sname, "found": False})
                    continue

                com_voxel = center_of_mass(data)
                com_world = nib.affines.apply_affine(img.affine, com_voxel)
                com_input = nib.affines.apply_affine(inv_affine, com_world)

                mask_binary = (data > 0).astype(np.uint8)

                # Transpose so the slice axis is always last
                if scan_orientation == "coronal":
                    mask_binary = np.transpose(mask_binary, (0, 2, 1))
                    com_input = np.array([com_input[0], com_input[2], com_input[1]])
                elif scan_orientation == "sagittal":
                    mask_binary = np.transpose(mask_binary, (1, 2, 0))
                    com_input = np.array([com_input[1], com_input[2], com_input[0]])

                f = mask_binary.flatten()
                f_padded = np.concatenate([[0], f, [0]])
                runs = np.where(f_padded[1:] != f_padded[:-1])[0] + 1
                runs[1::2] -= runs[::2]

                per_structure.append({
                    "structure": sname,
                    "found": True,
                    "centroid_voxel": com_input.tolist(),
                    "mask_rle": runs.tolist(),
                    "shape": list(mask_binary.shape),
                })

            return {
                "results": per_structure,
                "orientation": scan_orientation,
                "affine": nifti_img.affine.tolist(),
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _run_segmentation(
        self,
        input_path: str,
        output_dir: str,
        structure_names: list[str],
        task_name: str,
        fast_mode: bool = True,
        fallback_to_normal: bool = True,
    ) -> bool:
        """Runs TotalSegmentator with ml=False so each structure gets its own binary NIfTI."""
        from totalsegmentator.python_api import totalsegmentator
        import nibabel as nib
        import numpy as np
        import shutil

        print(f"Running TotalSegmentator task='{task_name}' structures={structure_names} fast={fast_mode}")

        def attempt(fast: bool) -> bool:
            label = "FAST" if fast else "NORMAL"
            
            # Optionally setup license from loaded secret
            import os
            import subprocess
            license_key = os.environ.get("TOTALSEG_LICENSE")
            if license_key:
                try:
                    subprocess.run(["totalseg_set_license", "-l", license_key], check=True, capture_output=True)
                except Exception as e:
                    print(f"Warning: Failed to set Totalsegmentator license: {e}")
            
            try:
                os.makedirs(output_dir, exist_ok=True)
                
                # Clear ONLY the targeted structures so we don't wipe other tasks' outputs during retries
                for sname in structure_names:
                    p = os.path.join(output_dir, f"{sname}.nii.gz")
                    if os.path.exists(p):
                        os.remove(p)

                # TotalSegmentator only allows roi_subset for the base tasks
                use_roi = structure_names if task_name in ["total", "total_mr"] else None
                
                totalsegmentator(
                    input_path, output_dir,
                    roi_subset=use_roi,
                    fast=fast,
                    ml=False,
                    task=task_name,
                )

                for sname in structure_names:
                    seg_path = os.path.join(output_dir, f"{sname}.nii.gz")
                    if os.path.exists(seg_path) and np.sum(nib.load(seg_path).get_fdata()) > 0:
                        print(f"{label} segmentation found at least: {sname}")
                        return True

                print(f"{label} segmentation produced no non-empty masks.")
            except Exception as e:
                print(f"{label} segmentation failed: {e}")
            return False

        if fast_mode:
            if attempt(fast=True):
                return True
            if fallback_to_normal:
                print("Retrying in NORMAL (fine) mode...")
                return attempt(fast=False)
            return False

        return attempt(fast=False)