# Project Deep Dive: Clinical Knee Reconstruction Pipeline
**Version**: 1.1 (Hardware-Aware Revision)  
**Objective**: Fully automated, clinical-grade 3D reconstruction of the femur and tibia from orthopedic CT scans, featuring advanced hardware rejection and anatomical validation.

---

## 1. Data Ingestion & Clinical Preprocessing
The pipeline begins by converting raw DICOM series into standardized NIfTI volumes.

*   **HU Windowing**: The raw CT values (Hounsfield Units) are windowed to optimize bone contrast.
*   **Denoising (Anisotropic Diffusion)**: We apply edge-preserving smoothing to the intensity data. This reduces noise while sharpening the cortical bone boundaries.
*   **Resampling**: All patient scans are resampled to a high-resolution grid (typically **0.5mm isotropic**). This ensures that every sub-millimeter anatomical feature is preserved for the mesh generation phase.

---

## 2. Phase 1: AI Core (Segmentation)
The pipeline uses a modified **TotalSegmentator v2** (AI) model running on PyTorch to identify primary bone structures.

*   **Anatomical Label Mapping**:
    *   **Femur**: Label 75  
    *   **Tibia**: Label 77  
*   **The "Quality Gate"**: The script automatically validates the AI's output. If the labels are swapped (e.g., AI thinks the pelvis is the tibia due to a large rod) or if the bones are merged, the "Anatomical Verification" logic triggers a **Surgical Fallback**.

---

## 3. Phase 2: Surgical Fallback (Advanced Anatomy Logic)
When the AI fails (common in patients with large metal implants or unusual posture), the pipeline switches to an **Intensity + Spatial Logic** system.

### A. Anatomical Z-Axis Anchoring
We use the **successfully identified femur** as a spatial anchor.
*   **Logic**: The tibia *must* be inferior to the distal femur.
*   **Constraint**: The pipeline automatically calculates the femur's distal end and creates a "Spatial Cut-off". Everything above this point (Pelvis, Rod, Femur) is strictly ignored for the tibia extraction.

### B. Hardware-Aware Thresholding (The 1800 HU ceiling)
Metal implants (IM nails, rods) are much denser than bone.
*   **Cortical Bone Range**: ~400 – 1200 HU.
*   **Metal Hardware Range**: > 2000 HU.
*   **Implementation**: During surface extraction, the pipeline applies a hard **1800 HU ceiling**. This effectively "hollowed out" the metal rod from the interior of the bone.

### C. Multi-Part Connectivity Cleaning (85-Fragment Rejection)
After subtracting the rod, the computer often finds "debris" or shell fragments from the hardware.
*   **Logic**: The script performs a **Binary Connected Component Analysis**. 
*   **Refinement**: It calculates the volume of every single disconnected object. It identifies the largest anatomical body (the Tibia) and **automatically discards all other fragments** (e.g., discarding 85+ hardware-related particles).

---

## 4. Phase 3: Clinical Mesh Refinement
Raw 3D meshes (Marching Cubes) are mathematically "noisy" due to voxel artifacts (staircasing).

*   **Taubin Smoothing (Non-Shrinking)**: Unlike standard smoothing (Gaussian) which "shrinks" the bone and rounds off joints, we use **Taubin Smoothing**. This preserves the exact volume and anatomical dimensions while removing surface jitter.
*   **Quadratic Edge Collapse (QEM)**: We decimate the high-resolution meshes (often 1M+ triangles) down to **150,000 faces**. This provides a clinical balance between high-fidelity detail and smooth performance in surgery planning software.

---

## 5. Phase 4: Validation & "Ground Truth" Metrics
To prove our mesh is better than other software (like Cuvis), we use **Image-to-Surface Agreement**.

*   **Sampling**: We take every point on the 3D surface and sample the original CT scan HU intensity at that exact XYZ location.
*   **The Truth Score**: 
    *   **Agreement > 50%**: Proves the mesh is perfectly tracking the hard cortical bone edge.
    *   **Agreement ~0%**: Proves the mesh is displaced from the anatomy (common in other software).
*   **Geometrical Registration**: We use **ICP (Iterative Closest Point)** to align external meshes and measure the precise Root-Mean-Square (RMS) error between models.

---

## 6. Project Directory Structure
*   `data/NIfTI`: Preprocessed medical volumes.
*   `data/meshes`: Final Clinical STL files (ready for Slicer/Printing).
*   `scripts/phase1`: Core AI and Extraction logic.
*   `scripts/validation`: Comparative accuracy metrics.
*   `PROJECT_DEEP_DIVE.md`: This technical documentation.

---
**Conclusion**: This pipeline is designed to prioritize **Anatomical Truth** over aesthetic smoothness. By combining AI speed with Surgical Fallback reliability, it produces sub-millimeter accurate bone models even in the most complex orthopedic cases.
