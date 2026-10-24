# GuessMyText V3 — Paragraph Recognition Implementation Documentation

**Status:** Implemented, Unified & Verified ✅  
**Scope:** V3 paragraph recognition  
**Compatibility:** V1 and V2 remain functional and behaviorally unchanged.

---

## 1. Overview & Canvas Technology Alignment

V3 implements full paragraph recognition for multi-line handwriting on an expandable, 2D scrollable canvas. Lines of text are detected from the foreground pixel distribution, sorted top-to-bottom, segmented left-to-right into characters, classified using the ensemble of 3 CNN models (32×32, 64×64, 128×128), and reconstructed with preserved line breaks (`\n`) and estimated word spaces (` `).

### Canvas Technology & Source of Truth
- **Source of Truth:** Directly reuses and extends CaptureMyText's original canvas technology (`CaptureMyText_Copy/frontend/src/features/capture/DrawingCanvas.tsx`).
- **Unified Component:** `GuessMyText/frontend/src/components/DrawingCanvas.tsx` now powers all 3 versions of GuessMyText:
  - **Mode 1 (V1 Single Character):** Fixed square aspect ratio (`scrollable=false`).
  - **Mode 2 (V2 Line Recognition):** Horizontal expansion + draggable scrollbar (`scrollable="horizontal"` or `scrollable={true}`).
  - **Mode 3 (V3 Paragraph Recognition):** 4-way expansion (Left, Right, Top, Bottom) + 2D scrolling viewport (`scrollable="2d"`).
- **Core Technology:**
  - Native HTML5 `<canvas>` 2D context (`ctx.lineCap = 'round'`, `ctx.lineJoin = 'round'`, stroke `#FFFFFF`, background `#000000`).
  - Pointer capture API (`onPointerDown`, `onPointerMove`, `onPointerUp`, `onPointerCancel`).
  - Pointer mapping accounting for container scroll offsets and logical origin shifts:
    `x = (e.clientX - rect.left) * (canvas.width / rect.width)`
    `y = (e.clientY - rect.top) * (canvas.height / rect.height)`
  - Snapshot-based stroke preservation: Before changing canvas dimensions, the bitmap is blitted to an offscreen `tempCanvas`, the main canvas dimensions are updated, filled with black, and the snapshot is drawn back at `(dx, dy)`.
  - Full-canvas export via `canvas.toBlob(..., 'image/png')` capturing off-screen content.

All processing is performed strictly in memory without persisting user drawings, crops, or predictions.

---

## 2. Actual Files & Routes

### Backend
- **`GuessMyText/backend/paragraph_segmentation.py`**:
  - `LineRegion`: Data class storing line index, vertical bounds (`y_start`, `y_end`), and bounding boxes.
  - `detect_line_regions`: Groups connected components into vertical line clusters via centroid proximity and vertical overlap; merges intra-line split marks (dots, bars, accents); sorts lines top-to-bottom.
  - `segment_paragraph_lines`: Integrates CCA extraction, detached mark merging, vertical line clustering, touching-character valley splitting, and left-to-right sorting.
  - `recognize_paragraph`: Top-level pipeline verifying canvas dimensions, decoding PNG bytes, running independent per-line word boundary estimation, executing 3-model CNN majority voting, assembling lines with newlines (`\n`), and compiling actual inference intermediates for the Segmentation Preview.
- **`GuessMyText/backend/app.py`**:
  - `POST /api/predict-paragraph`: Dedicated, isolated V3 endpoint accepting multipart form with `image` (PNG). Validates payload size, checks all 3 CNN models are loaded (returning HTTP 500 on technical failure without partial voting), and returns JSON response with status, reconstructed text, line structures, and preview metadata.
- **`GuessMyText/backend/test_paragraph_recognition.py`**:
  - 9 automated unit/integration tests validating empty canvas, line sorting, newline preservation, word gaps, dimension limits, model failure handling, preview intermediates, Flask endpoint, and V1/V2 regression.

### Frontend
- **`GuessMyText/frontend/src/components/DrawingCanvas.tsx`**:
  - Unified component implementing the shared CaptureMyText/V1/V2 canvas technology.
  - For V3, set `scrollable="2d"`, `internalWidth={1200}`, `internalHeight={600}`, and `brushSizePx={18}`.
  - Implements 4-way expansion (`expandLeft`, `expandRight`, `expandTop`, `expandBottom`), dimension resets, and scroll helpers.
- **`GuessMyText/frontend/src/pages/ParagraphRecognitionPage.tsx`**:
  - Mounted at route `/v3`.
  - Reuses `DrawingCanvas` from `../components/DrawingCanvas`.
  - Dismissible notice banner with exact specified wording (persisted in `localStorage`).
  - Four external `+` expansion controls (+600px Above, +600 Left, +600 Right, +600px Below) placed outside the drawing surface.
  - Canvas toolbar with logical dimensions badge, "Reset to 1200 × 600", and "Scroll to Top-Left".
  - Three-section status card with Reference link, recognition feedback, and Ready indicator.
  - Reconstructed paragraph display with line badges (`L1`, `L2`, ...), `[?]` badges for uncertain positions, and plain text copyable `textarea`.
  - Segmentation Preview section grouped by detected line, displaying crop image, model-input tensor image, dimensions, and resolution toggle (32×32, 64×64, 128×128) using nearest-neighbor display (`imageRendering: 'pixelated'`).
- **`GuessMyText/frontend/src/App.tsx`**:
  - Routes `/v3` to `<ParagraphRecognitionPage />`.

---

## 3. Dimensions & Expansion Rules

- **Initial Logical Dimensions:** `1200 × 600 px` (sensible starting size accommodating 3–4 lines of handwriting with height ~120–150px each).
- **Fixed Viewport Display:** `w-full h-[480px]` with standard 2D scrolling (`overflow-auto`).
- **Expansion Increment:** Exactly **600 logical pixels** per click:
  - **Left `+`:** Width increases by 600 px; all existing strokes shifted right by 600 logical px (`dx = 600, dy = 0`). Container scrolls to left (`0`).
  - **Right `+`:** Width increases by 600 px; existing stroke coordinates unchanged (`dx = 0, dy = 0`). Container scrolls to right.
  - **Top `+`:** Height increases by 600 px; all existing strokes shifted down by 600 logical px (`dx = 0, dy = 600`). Container scrolls to top (`0`).
  - **Bottom `+`:** Height increases by 600 px; existing stroke coordinates unchanged (`dx = 0, dy = 0`). Container scrolls to bottom.
- **Clear Behavior Decision:** `clear()` erases all strokes on the logical canvas, resets results and segmentation preview, and smoothly scrolls to the top-left origin `(0, 0)`. The canvas explicitly **retains its expanded dimensions** upon Clear so the user does not lose their customized writing canvas layout if they choose to re-write their paragraph. An explicit "Reset to 1200 × 600" control is provided in the canvas toolbar if the user wishes to restore initial dimensions.
- **Maximum Canvas Limit:**
  - `MAX_CANVAS_DIM = 6000 px`
  - `MAX_CANVAS_PIXELS = 25_000_000 px`
  - Payloads exceeding these limits are rejected with HTTP 400 before expensive processing.

---

## 4. Segmentation & Preprocessing Details

### Preprocessing & Label Ordering Verification
- Verified against `New/generate_dataset.py`, `New/datasets/label_mapping.json`, and CNN training checkpoints:
  - Input: Grayscale image decoded via OpenCV.
  - Binarization: `(resized > 100).astype(np.uint8)` with foreground = 1 (white) and background = 0 (black).
  - Interpolation: `cv2.INTER_AREA` when downsampling to model resolution.
  - Padding & Framing: Character bounding box is extracted, framed into a square with 20% padding while preserving aspect ratio, centered by mass/dimensions, and resized to (32×32), (64×64), or (128×128).
  - Tensor: Converted to `torch.float32` tensor of shape `(1, 1, H, W)`.
  - Classes: Exactly 67 classes (digits 0–9, uppercase A–Z, lowercase a–z, math operators `+`, `-`, `*`, `/`, `=`, `%`, `÷`).

### Diagnosed Segmentation Issues & Causes
1. **Premature Cross-Line Stroke Merging:** `merge_detached_strokes` was initially invoked globally across the entire canvas before lines were clustered. When characters on Line 1 were positioned above characters on Line 2 (e.g. 'H' above 'T'), they were erroneously merged into single bounding boxes spanning across lines, collapsing paragraphs into single lines and yielding multi-character vertical crops.
2. **Fragile Component-Based Line Clustering:** Line grouping previously used dynamic centroid bounds that progressively expanded until adjacent lines merged. Furthermore, an adjacent merge check (`v_gap <= 5 px`) merged distinct lines whenever a descender anywhere in Line 1 came within 5px of an ascender anywhere in Line 2.
3. **Over-Aggressive Detached Stroke Merging:** The heuristic condition in `merge_detached_strokes` treated any horizontal overlap (`overlap_ratio >= 0.35`) with `v_gap = 0` as vertically stacked, causing adjacent separate characters with natural kerning (e.g. 'To', 'AV') to merge.
4. **Touching-Character Valley Splitting Without Tight Bounds:** Split character halves inherited the full unrefined bounding box height and y-origin of the parent component, leaving empty space that skewed square centering and caused CNN recognition errors.
5. **Fixed Word Gap Floor:** A hardcoded `24.0 px` floor in `estimate_word_boundaries` prevented space detection for compact or smaller handwriting scales.

### Implemented Segmentation Pipeline & Algorithms
1. **Hybrid Line Detection (`detect_line_regions`):**
   - **Horizontal Projection Profiling (HPP):** Computes row pixel density $P(y) = \sum_x \text{binary}[y, x] > 0$ to identify continuous ink row runs and empty inter-line rows.
   - **Intra-Line Band Bridging:** Vertically bridges small gaps ($\le 22\text{ px}$ or $\le 40\text{ px}$ for small dots/bars $\le 30\text{ px}$) to keep characters with detached strokes (e.g. 'i', 'j', '=') within their proper line.
   - **Overlapping Line Valley Splitting:** When an ink band is unusually tall ($> 1.85 \times \text{med\_h}$ and $\ge 135\text{ px}$) due to descenders touching ascenders, 1D convolution smoothing identifies the projection valley ($\le 0.45 \times \text{peak}$) to split the joined lines.
   - **Component Assignment:** Connected components are assigned to line bands according to their vertical centroid ($y + h/2$).
   - **Top-to-Bottom Line Ordering:** Lines are sorted strictly top-to-bottom by median component vertical center.
2. **Per-Line Intra-Line Segmentation (`segment_paragraph_lines`):**
   - Detached stroke merging (`merge_detached_strokes`) runs **strictly within each detected line**, completely preventing cross-line stroke bleeding.
   - Detached stroke merging checks whether both components are full-sized characters ($h \ge 0.50 \times \text{line\_height}$) with separated horizontal centers ($\Delta x_c > 0.35 \times \max(w_1, w_2)$) to ensure adjacent characters (e.g. 'To', 'AV') are never merged.
   - Touching-character valley splitting (`split_touching_characters`) computes tight bounding boxes via `cv2.findNonZero` on each slice, eliminating padding distortion.
   - Character bounding boxes are sorted strictly left-to-right (`box.x`).
3. **Scale-Adaptive Word Spacing (`estimate_word_boundaries`):**
   - Horizontal gaps $g_i = x_{i+1} - x_{i,2}$ are measured.
   - Adaptive space threshold: $\text{space\_threshold} = \max(0.45 \times \text{med\_width}, 1.85 \times \text{med\_gap}, \min(20.0, 0.55 \times \text{med\_width}))$.
4. **Full Canvas Overview & Diagnostic Preview:**
   - Backend renders an annotated visualization (`annotated_canvas`) with emerald green line bounding boxes (`L1`, `L2`, ...) and cyan character bounding boxes.
   - Includes structured diagnostic metadata (`diagnostics`) with line region coordinates, height spans, and threshold configurations.
   - Frontend features a toggleable **Diagnostics** view for inspection without cluttering user-facing production results.

---

## 5. Test Suite & Verification Results

| Test Suite | Commands Run | Tests Count | Status | Notes |
|:---|:---|:---:|:---:|:---|
| **V3 Paragraph Tests** | `python GuessMyText/backend/test_paragraph_recognition.py` | 16 | **PASSED** | Validates single-line, multi-line varying spacing, wide/narrow characters, disconnected strokes intra-line, overlapping side-by-side characters, expanded canvas outside viewport, preview intermediates, annotated canvas & diagnostics, Flask endpoint, model failure HTTP 500, and V1/V2 regression. |
| **V2 Line Tests** | `python GuessMyText/backend/test_line_recognition.py` | 10 | **PASSED** | CCA, stroke merging, touching split, word gaps, majority voting, `/api/predict-line` endpoint. |
| **V1 Backend Tests** | `python GuessMyText/backend/test_backend.py` | 4 | **PASSED** | Label mapping, model checkpoints on CUDA, preprocessing shapes, and inference accuracy. |
| **Voting Logic** | `python GuessMyText/backend/test_prediction_logic.py` | 8 | **PASSED** | 3-agree, 2-agree, 0-agree, CUDA error simulation, technical failure abort. |
| **Frontend Production Build** | `npm.cmd run build` | 1 | **PASSED** | `tsc -b && vite build` built in 681ms with 0 errors. |
| **End-to-End Browser** | Browser Subagent | Complete workflow | **PASSED** | Verified page load, normal mouse pointer (no crosshairs), multi-line drawing, prediction, line separation (`L1`, `L2`), Canvas Segmentation Overview with bounding boxes, toggleable Diagnostics view, and resolution switching (32/64/128). |

---

## 6. Known Limitations

1. **Character-Level Ensemble:** The underlying CNN models classify individual characters without word-level language modeling or dictionary spelling correction.
2. **Heavily Connected Cursive:** Highly connected or cursive writing where multiple letters flow together without vertical projection valleys may require manual character spacing for optimal segmentation.
3. **Severe Slanted Lines:** Paragraph segmentation assumes approximately horizontal line orientation; handwriting written diagonally (> 35° tilt) across multiple line paths may intersect adjacent line bands.
