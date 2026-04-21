# Implementation Plan — Restoring Anatomical Joint Geometry

## Goal
Remove the "Flat/Plane" look from the knee joint (Femur distal and Tibia proximal) by replacing the hard Z-slice crop with **Morphological Separation**. This will preserve the natural curved "Condyle" and "Plateau" shapes.

## User Review Required

> [!IMPORTANT]
> I am removing the `mask_ds[:, :, Z:] = 0` hard cut entirely. Instead, I will use a **3D Morphological Erosion** pass to break any contact points between the femur and tibia, then select the largest bone component. This respects the true density-based shape of your bones.

## Proposed Changes

### 1. Mesh Extraction Logic (`scripts/phase1/03_extract_and_mesh.py`)

#### [MODIFY] [03_extract_and_mesh.py](file:///d:/knee-implant-pipeline/knee-implant-pipeline/scripts/phase1/03_extract_and_mesh.py)

- **Remove Hard Crops:** Delete the lines that zero out slices above/below a specific Z-coordinate.
- **Implement `anatomical_separation(mask)`:**
    - Use a **3D Distance Transform** or **Morphological Erosion** (radius 2-3) to separate the Femur and Tibia if they are touching.
    - Identify the Tibia by finding the largest component in the *lower half* of the volume after erosion.
    - Use **Morphological Dilation** (matching the erosion radius) to restore the original outer surface of the bone, while keeping the joint space separate.
- **Result:** The bones will have their natural rounded ends (Condyles) instead of flat "planed" ends.

### 2. Verification Plan

#### Automated Verification
- Rerun the segmentation.
- Check the mesh topology—the "top" of the Tibia mesh should now have varied Z-coordinates (rounded) rather than a single constant Z-value.

#### Manual Verification
- Visualization in **3D Slicer** to confirm the "Rounded/Natural" look of the knee joint.
