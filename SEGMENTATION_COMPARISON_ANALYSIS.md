# 🦴 Segmentation Comparison Analysis & Improvement Plan

## Patient: AB 72Y Male — Left Knee
**Comparison:** Knee Segmentation pipeline (KS) vs Cuvis Software (medical-grade) vs Ground Truth

---

## 1. Overall Verdict

| Metric | Cuvis Score | KneeSeg Score | Winner |
|--------|------------|---------------|--------|
| Mean Surface Distance | 1/5 | 1/5 | Tie |
| Hausdorff Distance | 1/5 | 1/5 | Tie |
| Dice Similarity | 1/5 | 1/5 | Tie |
| Volume Accuracy | 2/5 | **3/5** | **KS** |
| Mesh Quality | **5/5** | 3/5 | **Cuvis** |
| **Total** | **10/25** | **9/25** | Cuvis (barely) |

> [!IMPORTANT]
> Both software systems score **very poorly** against the Ground Truth (all 1/5 on the key accuracy metrics). This is **NOT** because both are terrible — it's because **the Ground Truth STL appears to be a combined femur+tibia mesh** (601 cm³, Euler -4262, NOT watertight) in a **different coordinate system**. The evaluation's vs-GT metrics are dominated by **coordinate misalignment**, not segmentation quality. The Cuvis-vs-KS comparison is far more meaningful.

---

## 2. Key Findings — Where KS Falls Behind Cuvis

### 🔴 Problem #1: Mesh Topology (Biggest Gap — KS: 3/5 vs Cuvis: 5/5)

| Property | Cuvis Femur | KS Femur | Cuvis Tibia | KS Tibia |
|----------|-------------|----------|-------------|----------|
| Watertight | ✔ | ✘ | ✔ | ✘ |
| Euler Number | 2 (perfect) | 8 | 2 (perfect) | **-434** |
| Faces | 93,296 | 150,000 | 68,050 | **820,803** |

**Diagnosis:** 
- KS meshes are **not watertight** — they have holes and self-intersections
- The KS Tibia has an **extremely negative Euler number (-434)** indicating hundreds of topological defects (handles/holes)
- The KS Tibia has **820K faces** — 12× more than Cuvis (68K). This signals insufficient decimation and/or noisy surface extraction

**Root Cause in Pipeline:**
- `meshing_close_holes(maxholesize=100)` in [03_extract_and_mesh.py](file:///d:/knee-implant-pipeline/knee-implant-pipeline/scripts/phase1/03_extract_and_mesh.py#L54) is wrapped in a try/except that **silently ignores failures**
- Taubin smoothing (`stepsmoothnum=10`) is **too mild** — the surface retains noise from marching cubes
- The pipeline caps at `MAX_TRIANGLES = 150,000` but the tibia mesh has **820K faces**, meaning decimation failed/was skipped for the tibia in this case

---

### 🟡 Problem #2: Volume Discrepancy (Over-segmentation)

| Bone | Cuvis Volume | KS Volume | Difference |
|------|-------------|-----------|------------|
| Femur | 400.18 cm³ | 429.85 cm³ | **+7.4% over** |
| Tibia | 262.60 cm³ | 137.62 cm³ | **-47.6% under** |

**Diagnosis:**
- The **Femur is over-segmented by ~30 cm³** — likely including soft tissue or joint capsule
- The **Tibia is catastrophically under-segmented** — capturing only ~52% of the expected volume
- Looking at the tibia mesh overlay, the KS tibia (green) extends well beyond the Cuvis tibia (gray) in some areas but **misses large anatomical portions in others**

**Root Cause in Pipeline:**
- HU thresholds in [config.py](file:///d:/knee-implant-pipeline/knee-implant-pipeline/config.py): `HU_CORTICAL_MIN = 700` is very aggressive — this captures mainly cortical bone but **misses cancellous/spongy bone** (typically 200–600 HU)
- The fallback `floor_hu = 350` for tibia still misses significant trabecular bone
- The `HU_CANAL_MAX = 400` range suggests the pipeline is aware of medullary canal but doesn't properly handle it during surface extraction
- Aggressive metal filtering (`sub_vol < 1800` for tibia) may remove legitimate high-density cortical bone near metal implants

---

### 🟡 Problem #3: Surface Accuracy (Cuvis vs KS)

| Metric | Femur (C vs KS) | Tibia (C vs KS) |
|--------|-----------------|-----------------|
| Mean Surface Distance | 1.23 mm | 3.74 mm |
| Hausdorff Distance | 138.98 mm | 26.18 mm |
| 95th Percentile | 2.18 mm | 19.38 mm |
| Dice Score | 0.947 | 0.329 |

**Diagnosis:**
- **Femur** is reasonably close — Dice 0.947 (near-excellent), MSD 1.23 mm (acceptable). The Hausdorff 138.98 mm is driven by a few **outlier points** (visible in heatmaps as scattered yellow points)
- **Tibia** has severe problems — Dice 0.329 (terrible), MSD 3.74 mm. This confirms the tibia segmentation is fundamentally wrong in shape/extent
- The tibia heatmap (KS vs Cuvis) shows **high-distance regions concentrated on the distal shaft** — the KS tibia is either capturing too much shaft material or a different region entirely

**Root Cause:**
- The tibia fallback code uses a 2× downsampled volume for component detection, then upsamples — this introduces **spatial imprecision**
- The connectivity-based "largest component" selection may select the wrong anatomical structure when metal hardware creates bridges between bones
- The `aspect_ratio > 7.0` penalty for rod detection may not be aggressive enough for this patient's hardware

---

### 🟡 Problem #4: Surface Area Anomaly

| Bone | Cuvis SA | KS SA | Ratio |
|------|----------|-------|-------|
| Femur | 548.52 cm² | 725.34 cm² | 1.32× |
| Tibia | 397.87 cm² | 734.42 cm² | **1.85×** |

**Diagnosis:**
- KS surface areas are **1.3–1.8× larger** than Cuvis despite (for tibia) having **less volume**
- This is a classic sign of **rough/noisy surfaces** — marching cubes creates jagged surfaces that have high surface area relative to enclosed volume
- The Cuvis meshes are properly smoothed and decimated, producing clean surfaces

---

## 3. Prioritized Improvement Recommendations

### Priority 1: Fix Mesh Topology (Highest Impact)

> [!TIP]
> This is the single most impactful fix — achieving watertight, Euler=2 meshes dramatically improves clinical usability.

```diff
# In 03_extract_and_mesh.py — replace the current refinement pipeline:

- ms.apply_coord_taubin_smoothing(stepsmoothnum=SMOOTH_ITERS, lambda_=0.5, mu=-0.53)
+ # Stage 2a: Aggressive hole closing (increase max hole size)
+ ms.apply_filter("meshing_close_holes", maxholesize=500)
+ 
+ # Stage 2b: Smooth MORE aggressively (increase from 10 to 30+ iterations)
+ ms.apply_coord_taubin_smoothing(stepsmoothnum=30, lambda_=0.5, mu=-0.53)
+ 
+ # Stage 2c: Post-smooth duplicate/degenerate cleanup
+ ms.apply_filter("meshing_remove_duplicate_vertices")
+ ms.apply_filter("meshing_remove_duplicate_faces")
+ ms.apply_filter("meshing_repair_non_manifold_edges")
+ ms.apply_filter("meshing_repair_non_manifold_vertices")
+ 
+ # Stage 2d: Final hole closing pass after smoothing
+ ms.apply_filter("meshing_close_holes", maxholesize=500)
```

**Also in config.py:**
```diff
- SMOOTH_ITERS     = 10
+ SMOOTH_ITERS     = 30

- MAX_TRIANGLES    = 150_000
+ MAX_TRIANGLES    = 100_000  # Cuvis uses ~93K for femur, ~68K for tibia
```

---

### Priority 2: Improve Tibia Segmentation Volume

> [!WARNING]
> The tibia is only capturing 52% of the expected volume — this is the most critical accuracy problem.

**Recommended Changes:**

1. **Lower the cortical bone threshold for the tibia**:
```diff
# In 03_extract_and_mesh.py fallback_to_hu():
- floor_hu = 350 if bone_name == "tibia" else 250
+ floor_hu = 200 if bone_name == "tibia" else 200  # Capture trabecular bone (200-400 HU range)
```

2. **Use adaptive thresholding** instead of a fixed HU floor:
   - Analyze the HU histogram of the ROI
   - Use Otsu's method or a two-class threshold to separate bone from soft tissue
   - This adapts to each patient's density characteristics

3. **Morphological closing** after thresholding to fill internal gaps:
```python
from scipy import ndimage
# After bone thresholding, close small gaps
bone_mask = ndimage.binary_closing(bone_mask, structure=np.ones((5,5,5)), iterations=2).astype(np.uint8)
```

4. **Reconsider the 1800 HU metal ceiling for tibia**:
   - Dense cortical bone near implants can reach 1500–2000 HU
   - Use a more nuanced approach: instead of a hard ceiling, apply **morphological erosion** to remove only the metal rod's cross-section

---

### Priority 3: Reduce Surface Roughness

**Recommended Changes:**

1. **Pre-smooth the binary mask** before marching cubes (Gaussian blur):
```python
from scipy.ndimage import gaussian_filter
# Before marching cubes
mask_smooth = gaussian_filter(mask.astype(np.float32), sigma=1.0)
verts, faces, _, _ = measure.marching_cubes(mask_smooth, level=0.5)
```

2. **Use higher-resolution marching cubes** with `step_size=1` (default), not subsampled

3. **Increase Taubin smoothing iterations** from 10 to 30-50 (as noted above)

4. **Add Laplacian smoothing** as a second pass after Taubin for ultra-smooth surfaces

---

### Priority 4: Improve Coordinate System Alignment

> [!NOTE]
> The enormous vs-GT distances (MSD 24-164 mm, Hausdorff 445-514 mm) strongly suggest a **coordinate system mismatch** between your meshes and the ground truth. Both Cuvis and KS show nearly identical poor scores vs GT, confirming this is a systematic alignment issue, not a segmentation quality issue.

**Recommended Changes:**

1. **Verify the affine transform**: Ensure the NIfTI-to-mesh coordinate transform matches what the ground truth software uses (LPS vs RAS?)
2. **Apply ICP alignment** as a post-processing step before evaluation, to remove rigid-body misalignment artifacts
3. **Export meshes in the same coordinate convention** as Cuvis (check if Cuvis uses patient coordinate space, voxel space, or scanner space)

---

### Priority 5: AI Segmentation Label Quality

**Recommended Changes:**

1. **Use the latest TotalSegmentator v2 model** if not already — v2 has significantly improved knee segmentation with dedicated left/right femur and tibia labels (75-78)
2. **Post-process the AI segmentation mask**:
   - Apply morphological operations (closing → opening) to fill holes and remove noise
   - Smooth label boundaries with a majority-vote filter
3. **Consider fine-tuning** on a small set of knee-specific cases if possible

---

## 4. Summary — Quick Wins vs Long-Term

| Improvement | Effort | Impact | Priority |
|-------------|--------|--------|----------|
| Increase smoothing iterations (10→30+) | ⚡ Low | 🟢 High | Do Now |
| Add non-manifold repair filters | ⚡ Low | 🟢 High | Do Now |
| Pre-smooth mask before marching cubes | ⚡ Low | 🟡 Medium | Do Now |
| Lower tibia HU threshold (350→200) | ⚡ Low | 🟢 High | Do Now |
| Add morphological closing after threshold | ⚡ Low | 🟡 Medium | Do Now |
| Fix mesh decimation for tibia | ⚡ Low | 🟡 Medium | Do Now |
| Coordinate system alignment fix | 🟡 Medium | 🟢 High | This Week |
| Adaptive/Otsu thresholding | 🟡 Medium | 🟡 Medium | This Week |
| Better rod separation (erosion-based) | 🟡 Medium | 🟡 Medium | This Week |
| TotalSegmentator v2 fine-tuning | 🔴 High | 🟢 High | Long-Term |

> [!IMPORTANT]
> **The bottom line**: Your KS pipeline is **surprisingly close to Cuvis on femur segmentation** (Dice 0.947 is near-excellent!). The main battles to fight are:
> 1. **Fix tibia segmentation** — it's capturing only half the volume
> 2. **Fix mesh topology** — achieve watertight, Euler=2 meshes like Cuvis
> 3. **Fix coordinate alignment** — the vs-GT scores are artificially terrible due to coord system mismatch
> 
> Addressing these three areas will make your pipeline **competitive with or better than Cuvis**.
