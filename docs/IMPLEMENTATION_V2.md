# GuessMyText V2 — Line Recognition Implementation Plan & Report

**Status:** Completed  
**Version:** V2  
**Parent project:** GuessMyText  
**Related feature:** V1 — Single Character Recognition  
**Completion Date:** October 24, 2026  

---

## 1. Overview and Purpose

GuessMyText V2 implements handwritten **line recognition**, enabling users to draw multi-character lines and multi-word phrases. The application segments the drawing into individual character regions, processes and frames each candidate to match the training distribution of the existing CNN models, classifies each character using a 3-model majority-voting ensemble, estimates inter-character word boundaries, and reconstructs the full line of text while clearly preserving and highlighting any uncertain character positions.

The feature is built on the existing GuessMyText architecture, preserves full backward compatibility with V1 Single Character Recognition, reuses the visual language and components of CaptureMyText, and strictly avoids modifying `CaptureMyText_Copy` or retraining the existing CNN models.

---

## 2. Architecture & Component Reuse

### Frontend Architecture
- **Framework & Styling**: React 19, TypeScript, Tailwind CSS v4, DaisyUI v5, Phosphor Icons (`@phosphor-icons/react`), and React Hot Toast.
- **Theming**: Centralized primary accent color `--primary-accent: #89CFF0` defined in `GuessMyText/frontend/src/index.css` on `:root`.
- **Navigation**: Preserves `Sidebar.tsx` and `AppLayout.tsx`. Route `/v2` maps to the new `LineRecognitionPage.tsx`. V1 route `/v1` remains active and functional.
- **Reusable Drawing Canvas**: Adapted `DrawingCanvas.tsx` with backward-compatible support for custom dimensions (`internalWidth`, `internalHeight`), height-proportional brush scaling (`brushSizeRatio`), and responsive container styling (`className`).

### Backend Architecture
- **Framework & Server**: Flask API (`GuessMyText/backend/app.py`) running with evaluation-mode PyTorch models loaded once on startup (`cuda` if available, otherwise `cpu`).
- **Endpoint**: `POST /api/predict-line` accepting PNG image as multipart form-data or JSON base64.
- **Line Segmentation Engine**: Dedicated modular pipeline in `GuessMyText/backend/line_segmentation.py` handling image decoding, binarization, empty canvas detection, Connected Component Analysis (CCA), detached stroke association (e.g., dot of 'i'/'j', '=', '÷'), touching character projection splitting, dynamic gap analysis for word boundaries, square training-compatible character framing, multi-scale tensor creation, and 3-model majority voting.

---

## 3. Canvas Dimensions, Expandable System & Brush Proportions

### Dimensions & Expandable Architecture
- **Initial Logical Canvas Dimensions**: `1200 × 300` pixels (width × height).
- **Displayed Height**: Fixed at `300` CSS pixels (`h-[300px]`), preventing page stretching and avoiding unwanted vertical scrolling.
- **Expansion Increment**: Exactly `600` logical canvas pixels per click (`+600px`).
- **Scroll Container**: The drawing surface is housed in an outer fixed-height container (`h-[300px]`, `overflow-x-auto overflow-y-hidden no-scrollbar`) that prevents horizontal page expansion while allowing smooth horizontal traversal.
- **Custom Draggable Scrollbar & Visual Cue**: Positioned directly beneath the canvas drawing viewport:
  - **Visual Cue Header**: Displays the exact active visible region (e.g. `Visible View: 659px – 1544px`), percentage of canvas visible (`49% visible`), and total canvas length (`1800px total length`).
  - **Draggable Thumb**: A tactile pill-shaped thumb styled with `--primary-accent` (`#89CFF0`), a subtle border, drop shadow, and 3 vertical grip bars. Dragging the thumb moves the canvas in real time with pointer capture.
  - **Track Jumping**: Clicking anywhere along the track smoothly jumps/scrolls directly to that relative coordinate on the canvas.
  - **Bidirectional Sync**: The thumb automatically tracks and mirrors canvas scrolling regardless of input method (trackpad gestures, mouse dragging, click-to-jump, Left/Right expansion buttons, or window resize via `ResizeObserver`).
- **Canvas Element**: Rendered at `width={currentWidth}` and `height={300}` with CSS `minWidth: `${currentWidth}px`` and `width: `${currentWidth}px``. This ensures that on viewports narrower than the canvas, the container scrolls horizontally without squeezing or distorting the canvas bitmap.

### Left & Right Expansion Controls
Two vertical expansion buttons flank the canvas container, placed entirely outside the drawing surface so they never appear in exported prediction images:
1. **Right Expansion (`+600 Right`)**:
   - Increases logical canvas width by exactly 600px ($W_{\text{new}} = W_{\text{old}} + 600$).
   - Existing strokes remain at their original $(x, y)$ coordinates.
   - Canvas height and brush thickness remain strictly unchanged.
   - Creates 600px of clean black drawing space on the right $[W_{\text{old}}, W_{\text{new}}]$.
   - Automatically smooth-scrolls the container to `scrollWidth` so the user can continue writing seamlessly.
2. **Left Expansion (`+600 Left`)**:
   - Increases logical canvas width by exactly 600px ($W_{\text{new}} = W_{\text{old}} + 600$).
   - Adds 600px of clean black empty space before the existing drawing $[0, 600]$.
   - Shifts the existing drawing by exactly $+600$ logical pixels to the right.
   - Preserves stroke appearance, relative spacing, thickness, and color without re-rasterization.
   - Automatically smooth-scrolls the container to `scrollLeft = 0` toward the newly created left-hand space.

### Stroke Preservation Mechanism (Avoiding Canvas Context Wipe)
In HTML5, updating `canvas.width` resets the 2D context and wipes the canvas bitmap to blank. To prevent data loss:
1. An offscreen snapshot canvas is created: `tempCanvas = document.createElement('canvas')`.
2. Existing bitmap is copied to `tempCanvas` via `tempCtx.drawImage(canvas, 0, 0)`.
3. In React, `useLayoutEffect` synchronizes with the updated DOM width, fills the entire canvas with `#000000`, and restores the snapshot:
   - For Right expansion: `ctx.drawImage(tempCanvas, 0, 0)`.
   - For Left expansion: `ctx.drawImage(tempCanvas, 600, 0)`.
4. Drawing context settings (`lineWidth = 18`, `strokeStyle = '#FFFFFF'`, `lineCap = 'round'`, `lineJoin = 'round'`) are immediately reconfigured.
5. The container scroll is updated via `requestAnimationFrame`.

### Brush-Scaling & Coordinate Invariance
- **Brush Thickness Invariance**:
  $$\text{Brush Size (px)} = \text{Internal Height} \times (\text{brushSizeRatio} \lor 0.06) = 300 \times 0.06 = 18\text{px}$$
  Because canvas height remains fixed at 300px across all expansion states (1200×300, 1800×300, 2400×300, etc.), brush thickness remains invariant at exactly 18px.
- **Coordinate Transformation**:
  $$\text{scaleX} = \frac{\text{canvas.width}}{\text{rect.width}} = 1.0, \quad \text{scaleY} = \frac{\text{canvas.height}}{\text{rect.height}} = 1.0$$
  $$\text{logical } x = (e.\text{clientX} - \text{rect.left}) \times \text{scaleX}$$
  When the container is scrolled by $\text{scrollLeft}$, `rect.left` shifts by $-\text{scrollLeft}$ relative to the viewport. Therefore, $(e.\text{clientX} - \text{rect.left})$ automatically compensates for horizontal scroll offset with zero distortion.

### Full-Canvas Export Behavior
- `DrawingCanvas.exportToBlob()` invokes native `canvas.toBlob()`, which encodes the entire logical bitmap ($W \times 300$, e.g. 1800×300 or 2400×300) into a single PNG blob.
- The export is never clipped to the visible container viewport.
- The backend CCA and segmentation engine processes all characters across the full width, sorting them from left to right, and populating the line prediction and Segmentation Preview.

### Clear & Reset Behavior
- **Clear Button**: Wipes handwriting across the entire current expanded width (`ctx.fillRect(0, 0, canvas.width, canvas.height)`), scrolls back to `scrollLeft = 0`, and preserves the expanded canvas width.
- **Reset to 1200px Button**: Displayed in the canvas toolbar when expanded ($W > 1200$), allowing users to explicitly reset canvas width back to the initial 1200px.

---

## 4. Line Recognition Pipeline & Preprocessing

### 1. Binarization & Empty Canvas Detection
- Background is black (`#000000`), strokes are white (`#FFFFFF`).
- The image is converted to grayscale and thresholded at 100.
- If foreground pixel count is $< 15$ pixels, the canvas is immediately recognized as `empty` without triggering model inference.

### 2. Connected Component Analysis (CCA) & Region Extraction
- `cv2.connectedComponentsWithStats` identifies all 8-connected stroke regions.
- Noise filtering removes components with area $< 8$ pixels or bounding box dimension $< 2$ pixels.

### 3. Detached Stroke Association
- Handles characters with disconnected marks (lowercase `i`, `j`, punctuation `!`, `?`, `=`, `÷`, `%`, etc.).
- Compares small upper components with taller base components underneath whose horizontal span overlaps by $\ge 30\%$ and vertical distance is within $1.5 \times$ the base height.
- Merges the stroke bounding boxes into a unified character region.

### 4. Overlapping / Touching Character Splitting
- Detects wide components where aspect ratio $\frac{W}{H} > 1.35$.
- Computes vertical projection histogram (sum of white pixels per column).
- Identifies significant local minimum valleys in the central $30\%–70\%$ region with low pixel density ($< 35\%$ of peak column density).
- Splits touching strokes at the optimal valley point.

### 5. Training-Compatible Character Framing
- Inspected `New/generate_dataset.py` and `New/Images/`: Each character in the training set was drawn in a square canvas with ~15–25% margins, resized directly via `cv2.INTER_AREA` to resolution, and thresholded at 100.
- Squeezing rectangular crops directly into a square tensor distorts aspect ratio and degrades accuracy.
- Solution: Each character crop is centered and placed inside a square bounding box with $20\%$ uniform padding margins before resizing to $32\times32$, $64\times64$, and $128\times128$.
- Tensors are created with shape `(1, 1, res, res)`, normalized to float range `[0.0, 1.0]`, and binarized at $> 0.392$ (`100/255`).

### 6. Word-Boundary Estimation
- Evaluates horizontal gaps $g_i = x_{i+1} - (x_i + w_i)$ between neighboring characters sorted left-to-right.
- Computes character median width $W_{\text{med}}$ and median gap $G_{\text{med}}$.
- Dynamic threshold:
  $$T_{\text{space}} = \max\left(14, \max\left(0.55 \times W_{\text{med}}, 1.7 \times G_{\text{med}}\right)\right)$$
- If $g_i \ge T_{\text{space}}$, a space character `' '` is marked as a word boundary between character $i$ and character $i+1$.

---

## 5. Model Loading, Voting & Uncertainty

### Loaded Models
Loaded on server startup, set to `.eval()`, with inference run inside `torch.inference_mode()` on `cuda` (or `cpu` fallback):
1. **CNN 32×32**: `cnn_32_8x8_30epochs.pth` (Test acc: 96.04%)
2. **CNN 64×64**: `cnn_64_8x8_30epochs.pth` (Test acc: 95.82%)
3. **CNN 128×128**: `cnn_128_8x8_30epochs.pth` (Test acc: 95.15%)
4. **Class Mapping**: `New/datasets/label_mapping.json` (67 classes).

### Voting Rules
For each character segment:
- **Case A (All 3 agree)**: Unanimous consensus $\rightarrow$ return predicted character.
- **Case B (Exactly 2 agree)**: Majority consensus $\rightarrow$ return majority character.
- **Case C (All 3 differ)**: No consensus $\rightarrow$ character marked as uncertain (`is_uncertain = True`, character = `"[?]"`).
- **Technical Failure**: If any model fails to run or is missing, a technical error `500` is returned. Predictions are never silently made with partial models.

### Line Reconstruction
- Combines character predictions in left-to-right reading order.
- Inserts spaces at detected word boundaries.
- Reconstructed string preserves recognized characters (e.g., `"Hello [?]orld"`).
- In the UI, uncertain characters are distinctly highlighted with a warning badge `[?]` explaining that the 3 CNN models lacked consensus at that position.

---

## 6. API Endpoint Specification

### `POST /api/predict-line`

**Request Format**:
- Multipart form-data with `image: File` (PNG), or
- JSON body `{"image": "data:image/png;base64,..."}`.

**Response Schema**:
```json
{
  "status": "success",
  "text": "7 + 1",
  "num_characters": 3,
  "num_words": 3,
  "has_uncertain": false,
  "message": "Line recognized successfully.",
  "segments": [
    {
      "index": 0,
      "character": "7",
      "is_uncertain": false,
      "bbox": { "x": 120, "y": 60, "width": 45, "height": 90 },
      "crop_dimensions": { "width": 45, "height": 90 },
      "has_space_after": true,
      "crop_image": "data:image/png;base64,...",
      "model_inputs": {
        "32": { "resolution": 32, "dimensions": "32×32", "image": "data:image/png;base64,..." },
        "64": { "resolution": 64, "dimensions": "64×64", "image": "data:image/png;base64,..." },
        "128": { "resolution": 128, "dimensions": "128×128", "image": "data:image/png;base64,..." }
      }
    }
  ]
}
```

**Status Values**:
- `success`: One or more characters detected and classified.
- `empty`: Canvas has no drawing ($< 15$ pixels).
- `no_characters`: Noise-only pixels or no valid character contours found.
- `error`: Technical failure (HTTP 400 for bad request, HTTP 500 for missing model or internal crash).

---

## 7. Segmentation Preview (Visual Debugging Feature)

### Purpose & Architecture
To assist in verifying segmentation accuracy and identifying model failure modes (such as touching characters, detached marks, excessive whitespace, or aspect ratio distortion), V2 includes an interactive **Segmentation Preview** section beneath the prediction results.

### Features
1. **Side-by-Side Comparison per Character**:
   - **Segmented Crop**: The raw unpadded character bounding box extracted directly from the line drawing.
   - **Model Input**: The exact preprocessed, square-padded (20% margin), centered, resized, and binarized uint8 image supplied to the CNNs.
2. **Multi-Resolution Switching**:
   - Interactive toggle allows switching the Model Input display between `32×32`, `64×64`, and `128×128` in real-time.
3. **Debugging Metrics**:
   - Displays character position (`Character 1`, `Character 2`, ...).
   - Prediction badge (e.g. `'7'` or `[?] Uncertain`).
   - Original crop dimensions vs model input resolution.
   - Bounding box coordinates `(x, y)` and word boundary indicator (`␣ Space after`).
4. **Guaranteed Consistency**:
   - Debug previews are extracted directly inside `prepare_character_tensor_with_preview()` at the exact moment the PyTorch inference tensor is produced. Previews cannot diverge from model evaluation.
   - Images are rendered with CSS `image-rendering: pixelated` for razor-sharp pixel inspection.
   - Processed purely in-memory as compact base64 PNG data URLs with zero permanent disk writes.
5. **Resolution Scaling & Display Investigation**:
   - Programmatic measurements confirmed that the PyTorch tensors for 32×32, 64×64, and 128×128 models occupy the exact same proportion (~70%) of the tensor array dimensions.
   - The visual difference observed in initial previews was caused by frontend CSS: `max-w-full max-h-full` left the 32×32 PNG at its intrinsic size of 32 CSS pixels inside the 80px container, while 128×128 filled the 80px box.
   - Fixed by applying `w-full h-full object-contain` with `image-rendering: pixelated`. All three model input resolutions now scale to fill the preview tile uniformly, clearly displaying nearest-neighbor pixel blocks for 32×32 while preserving exact handwriting proportions.

---

## 8. Verification & Tests Executed

### Automated Unit & Integration Tests (`GuessMyText/backend/test_line_recognition.py`)
Ran `python test_line_recognition.py`:
1. `test_empty_canvas`: Successfully returns `status: empty`.
2. `test_bounding_box_and_ordering`: Correct left-to-right ordering of characters.
3. `test_detached_stroke_merging`: Dot above lowercase `i` correctly merged into single character component.
4. `test_touching_character_split`: Wide touching component correctly split via projection valley.
5. `test_word_boundary_estimation`: Small within-word gaps have no space; larger between-word gaps insert space.
6. `test_multiscale_tensors`: Tensors for 32x32, 64x64, 128x128 generated with correct shape and normalization.
7. `test_voting_rules`: Verified unanimous agreement, majority agreement, and 3-way disagreement (`is_uncertain = True`).
8. `test_model_failure_behavior`: Verified that missing a required model causes an error (no silent partial voting).
9. `test_api_endpoint`: Flask client integration test with real rendered line drawing returned HTTP 200 and valid JSON schema.
10. `test_segmentation_preview_data`: Verified that `crop_image`, `crop_dimensions`, and `model_inputs` (32, 64, 128) are generated with valid data URLs and correct image dimensions upon decoding.
**Result**: **10 / 10 TESTS PASSED**.

### Regression Tests (`GuessMyText/backend/test_prediction_logic.py`)
Ran `python test_prediction_logic.py`:
- All V1 voting scenarios (3-agree, 2-agree, all-differ) tested against `New/models/CNN/`.
**Result**: **ALL TESTS PASSED**.

### Frontend Compilation
Ran `cmd /c "npm run build"` in `GuessMyText/frontend`:
- TypeScript typecheck (`tsc -b`) and Vite production bundle succeeded with 0 errors.

### Browser End-to-End Testing (via `browser_subagent`)
Executed on live frontend (`http://localhost:5173`) and backend (`http://localhost:5000`):
1. **Initial Canvas Dimensions**: Loaded V2 page with canvas badge showing `1200 × 300 px`, flanked by left and right `+` expansion buttons outside the scroll container.
2. **Pre-Expansion Drawing**: Drew initial character strokes on the 1200px canvas.
3. **Right Expansion (+600px)**:
   - Clicked Right `+` button. Badge updated immediately to `1800 × 300 px`.
   - Existing strokes remained perfectly preserved in their original coordinates.
   - Container smooth-scrolled toward the newly added right space.
   - Drew additional strokes in the new space without distortion.
4. **Left Expansion (+600px)**:
   - Clicked Left `+` button. Badge updated immediately to `2400 × 300 px`.
   - Existing strokes shifted exactly $+600$px right and remained fully intact.
   - Container smooth-scrolled toward `scrollLeft = 0` (newly added left space).
   - Drew additional strokes in the new left-hand space.
5. **Full-Canvas Export & Line Recognition**:
   - Clicked Predict with drawing spanning across the 2400px canvas.
   - Complete 2400×300 PNG exported to `/api/predict-line`.
   - All characters across the line were segmented in correct left-to-right order.
   - Result displayed recognized text.
   - Segmentation Preview rendered individual cards showing Segmented Crops and Model Input tensors for characters across all canvas regions.
6. **Clear Behavior**:
   - Clicked Clear. Handwriting across the entire 2400px canvas was erased to black.
   - Canvas width remained at 2400px (preserved).
   - Container scroll reset to leftmost (`0`).
   - Results and segmentation preview reset cleanly.
7. **Reset Canvas Width**:
   - Clicked `Reset to 1200px` toolbar button. Canvas returned to initial 1200×300 dimensions.
8. **V1 Regression Test**:
   - Navigated to `/v1` (Single Character Recognition).
   - Verified 400×400 square canvas and consensus inference remain fully functional.

---

## 8. Known Limitations

1. **Arbitrary Cursive Handwriting**: Connected cursive writing with continuous baseline ligatures (e.g., Spencerian or continuous cursive script) is challenging for simple projection valley splitting and may be recognized as a single merged block or uncertain character. The dismissible guidance banner explicitly informs users that separated characters yield the highest accuracy.
2. **Horizontal Mathematical Fraction Bars / Multi-Line Expressions**: V2 is scoped to single-line horizontal recognition. Stacked numerator/denominator formulas belong to V4 Math Solver.
3. **Ambiguous Detached Marks**: Punctuation marks positioned very far away from their parent character may be treated as independent characters or filtered if below the noise threshold.

---

## 9. Boundary Confirmations

- **`CaptureMyText_Copy`**: Strictly unmodified (verified: 0 files altered).
- **CNN Checkpoints**: Unmodified and reused (`cnn_32_8x8_30epochs.pth`, `cnn_64_8x8_30epochs.pth`, `cnn_128_8x8_30epochs.pth`).
- **Database / Storage**: No database introduced. Ordinary predictions are processed purely in memory with zero permanent file writes.
- **Server Shutdown**: Background Flask and Vite servers cleanly terminated after verification.
