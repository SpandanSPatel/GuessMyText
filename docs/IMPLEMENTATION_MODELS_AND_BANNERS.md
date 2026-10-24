# GuessMyText — Specialized Model Integration and Banner Updates

## 1. Objective

Integrate the newly trained specialized CNN models into GuessMyText V2, V3, and V4. Update each version's handwriting-guidance banner to accurately describe its supported character classes.

Preserve V1 and ensure each version uses the correct models, label mapping, preprocessing, and inference pipeline.

## 2. Existing Model Structure

The trained models are stored under `New/models/CNN/`.

### V1 — Existing 67-Class Models

Location:
`New/models/CNN/`

Models:

* `cnn_32_8x8_30epochs.pth`
* `cnn_64_8x8_30epochs.pth`
* `cnn_128_8x8_30epochs.pth`

Supported classes:

* Digits: `0–9`
* Uppercase letters: `A–Z`
* Lowercase letters: `a–z`
* Operators: `÷`, `-`, `*`, `%`, `+`

V1 must continue using these original models and the original `label_mapping.json`.

### V2 and V3 — Uppercase Letters and Digits

Location:
`New/models/CNN/uppercase_digits/`

Models:

* `cnn_32_uppercase_digits_8x8_30epochs.pth`
* `cnn_64_uppercase_digits_8x8_30epochs.pth`
* `cnn_128_uppercase_digits_8x8_30epochs.pth`

Mapping:

* `uppercase_digits_label_mapping.json`

Supported classes:

* Uppercase letters: `A–Z`
* Digits: `0–9`

Total: 36 classes.

V2 and V3 must use these specialized models instead of the original 67-class models.

### V4 — Mathematical Expression Solver

Location:
`New/models/CNN/math_symbols/`

Models:

* `cnn_32_math_symbols_8x8_30epochs.pth`
* `cnn_64_math_symbols_8x8_30epochs.pth`
* `cnn_128_math_symbols_8x8_30epochs.pth`

Mapping:

* `math_symbols_label_mapping.json`

Supported classes:

* Digits: `0–9`
* Operators: `+`, `-`, `*`, `÷`, `%`

Total: 15 classes.

V4 must use these specialized models instead of the original 67-class models.

## 3. Repository Inspection

Before modifying files, inspect the actual GuessMyText repository.

Identify:

* Frontend framework and routing.
* Existing V1, V2, V3, and V4 pages.
* Flask API endpoints and model-loading code.
* Model architecture and checkpoint-loading conventions.
* Image preprocessing and character segmentation.
* Label mapping and output decoding.
* Existing majority-voting and uncertainty-handling logic.
* Banner components, styles, and version-specific guidance.
* Existing documentation and tests.

Do not assume filenames or routes. Follow the actual repository structure.

Treat `CaptureMyText_Copy/` as read-only.

## 4. Model Integration

### 4.1 Preserve the Existing Architecture

Reuse the existing CNN architecture and inference utilities wherever possible.

Do not retrain, modify, or overwrite the trained models.

Load each required model once at backend startup where compatible with the existing architecture. Put models in evaluation mode and use inference without gradient calculation.

Use CUDA when available and otherwise fall back to CPU, following existing project conventions.

### 4.2 Version-Specific Configuration

Keep model configuration separate for each version.

| Version | Model directory                    | Class count |
| ------- | ---------------------------------- | ----------: |
| V1      | `New/models/CNN/`                  |          67 |
| V2      | `New/models/CNN/uppercase_digits/` |          36 |
| V3      | `New/models/CNN/uppercase_digits/` |          36 |
| V4      | `New/models/CNN/math_symbols/`     |          15 |

Use the appropriate label mapping for each model family.

Do not reuse the original 67-class mapping for the specialized models.

### 4.3 Checkpoint Compatibility

Inspect the actual checkpoint format before implementing model loading.

Some checkpoints may contain a dictionary with metadata and a `state_dict`, rather than a raw state dictionary.

The loader must correctly handle the format actually saved by the training scripts.

Verify:

* The model architecture matches the checkpoint.
* The output layer has the correct number of classes.
* The class mapping matches the output indices.
* The expected input dimensions are correct.
* All three models load successfully.

### 4.4 Preprocessing Compatibility

Verify preprocessing against the training and dataset-generation code.

Preserve the correct:

* Image polarity.
* Thresholding and binarization.
* Resizing and padding.
* Input normalization.
* Tensor dimensions.
* Character-crop preparation.

Do not assume that the same preprocessing works for every model without checking.

### 4.5 Majority Voting

Preserve the existing voting behavior where applicable.

For V2 and V3:

* Use the three uppercase-and-digit models.
* Apply the existing voting rules.
* Preserve the existing uncertainty handling.

For V4:

* Use the three math-symbol models.
* Apply the same voting principles.
* Do not evaluate an expression containing an uncertain character.

If a required model fails, return an appropriate technical error rather than silently using an incomplete vote.

## 5. Banner Updates

Update each banner without redesigning the rest of the interface.

Preserve the existing component structure, visual style, spacing, typography, dismissal behavior, and theme.

### 5.1 V1 Banner

Do not modify the V1 banner or V1 behavior as part of this task.

### 5.2 V2 — Line Recognition

Suggested banner:

**Tips for better recognition**

GuessMyText V2 recognizes uppercase letters (A–Z) and digits (0–9).

Write characters clearly and leave sufficient space between words. Connected or overlapping handwriting can be harder to segment and may produce inaccurate results.

Lowercase letters and mathematical operators are not supported in this version.

### 5.3 V3 — Paragraph Recognition

Suggested banner:

**Tips for better recognition**

GuessMyText V3 recognizes uppercase letters (A–Z) and digits (0–9) across multiple lines.

Write characters clearly, leave sufficient space between words, and leave clear vertical space between lines. Connected or overlapping handwriting and touching lines can be harder to segment.

Lowercase letters and mathematical operators are not supported in this version.

### 5.4 V4 — Math Solver

Suggested banner:

**Tips for better recognition**

GuessMyText V4 recognizes handwritten digits (0–9) and the operators `+`, `-`, `*`, `÷`, and `%`.

Write each digit and operator clearly, leaving enough space to distinguish neighboring characters.

The `%` symbol represents modulo/remainder, not percentage. Parentheses are not supported in this version.

## 6. UI Requirements

* Keep each banner associated with the correct version.
* Preserve existing banner dismissal behavior.
* Ensure banners do not block drawing, prediction, or solving.
* Use the project's existing theme and styling conventions.
* Avoid unrelated visual redesigns.
* Do not change V2's Predict button into Solve.
* V4 must retain its Solve action and expression-evaluation behavior.

## 7. Testing and Validation

### Model Loading

* Confirm all three V1 models still load.
* Confirm all three V2/V3 specialized models load.
* Confirm all three V4 specialized models load.
* Confirm the correct mapping is selected for each version.
* Confirm model output indices decode to the intended characters.

### Inference

* Test representative uppercase letters and digits in V2 and V3.
* Test representative digits and operators in V4.
* Verify preprocessing and segmentation remain compatible.
* Test majority voting, uncertainty, and model-failure handling.

### Version Isolation

* Confirm V1 still uses its original 67-class models.
* Confirm V2 and V3 use the 36-class models.
* Confirm V4 uses the 15-class models.
* Confirm V2, V3, and V4 do not accidentally load another version's mapping.

### UI

* Confirm the correct banner appears on each page.
* Confirm banners can be dismissed.
* Confirm V1 remains unchanged.
* Confirm V2 still has Predict and V4 still has Solve.
* Confirm the rest of the application remains functional.

Report only tests that were actually executed.

## 8. Constraints

* Do not retrain or modify model checkpoints.
* Do not modify the original dataset or label mappings.
* Do not modify `CaptureMyText_Copy/`.
* Do not rewrite the application unnecessarily.
* Do not introduce unrelated dependencies or redesigns.
* Do not claim successful testing without executing the tests.

## 9. Implementation Summary and Completion Report

### 9.1 Files Created and Modified

1. **`GuessMyText/backend/app.py`**
   - Configured separate model directories and mapping paths for V1 (67 classes), V2/V3 (36 classes), and V4 (15 classes).
   - Implemented `load_mapping_file()` supporting both flat and structured JSON mapping schemas with bijective integrity validation.
   - Implemented `load_checkpoint_family()` to load, verify shapes and class alignments, and set `eval()` mode on `DEVICE` (`cuda` or `cpu`).
   - Updated endpoints:
     - `POST /api/predict`: Uses V1 67-class models and mapping.
     - `POST /api/predict-line`: Uses V2/V3 36-class specialized models (`models_v2_v3`, `id_to_label_v2_v3`).
     - `POST /api/predict-paragraph`: Uses V2/V3 36-class specialized models (`models_v2_v3`, `id_to_label_v2_v3`).
     - `POST /api/solve-math`: Uses V4 15-class specialized models (`models_v4`, `id_to_label_v4`).
     - `GET /api/health`: Reports loaded status for all 9 models and version breakdown.
   - Preserved backward compatibility aliases (`models = models_v1`, `id_to_label = id_to_label_v1`) and legacy 2-parameter mock compatibility.

2. **`GuessMyText/frontend/src/pages/LineRecognitionPage.tsx`**
   - Updated guidance banner title to `"Tips for better recognition"`.
   - Updated copy to specify that GuessMyText V2 recognizes uppercase letters (A–Z) and digits (0–9), with warning that lowercase letters and operators are not supported.
   - Maintained `"Predict"` button, canvas interactions, and dismissal persistence under `gmt_v2_dismiss_guidance_v2`.

3. **`GuessMyText/frontend/src/pages/ParagraphRecognitionPage.tsx`**
   - Updated guidance banner title to `"Tips for better recognition"`.
   - Updated copy to specify that GuessMyText V3 recognizes uppercase letters (A–Z) and digits (0–9) across multiple lines, and lowercase letters and operators are not supported.
   - Maintained multi-line 2D canvas navigation and dismissal persistence under `gmt_v3_dismiss_guidance_v2`.

4. **`GuessMyText/frontend/src/pages/MathSolverPage.tsx`**
   - Updated guidance banner title to `"Tips for better recognition"`.
   - Updated copy to state that GuessMyText V4 recognizes handwritten digits (0–9) and operators `+`, `-`, `*`, `÷`, and `%` (modulo/remainder).
   - Maintained `"Solve"` action, equation evaluation, and dismissal persistence under `gmt_v4_dismiss_guidance_v2`.

5. **`GuessMyText/backend/test_specialized_models.py`** (New Test Suite)
   - 5 comprehensive tests validating all 9 model checkpoints, label mappings, real handwritten character inferences from `New/Images/`, endpoint version isolation, and `/api/health`.

### 9.2 Version-to-Model Configuration

| Version | Endpoint | Model Family Directory | Checkpoints | Classes | Mapping File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | `/api/predict` | `New/models/CNN/` | `cnn_32_8x8_30epochs.pth`<br>`cnn_64_8x8_30epochs.pth`<br>`cnn_128_8x8_30epochs.pth` | 67 (0–9, A–Z, a–z, +, -, *, ÷, %) | `New/datasets/label_mapping.json` |
| **V2** | `/api/predict-line` | `New/models/CNN/uppercase_digits/` | `cnn_32_uppercase_digits_8x8_30epochs.pth`<br>`cnn_64_uppercase_digits_8x8_30epochs.pth`<br>`cnn_128_uppercase_digits_8x8_30epochs.pth` | 36 (A–Z, 0–9) | `uppercase_digits_label_mapping.json` |
| **V3** | `/api/predict-paragraph` | `New/models/CNN/uppercase_digits/` | `cnn_32_uppercase_digits_8x8_30epochs.pth`<br>`cnn_64_uppercase_digits_8x8_30epochs.pth`<br>`cnn_128_uppercase_digits_8x8_30epochs.pth` | 36 (A–Z, 0–9) | `uppercase_digits_label_mapping.json` |
| **V4** | `/api/solve-math` | `New/models/CNN/math_symbols/` | `cnn_32_math_symbols_8x8_30epochs.pth`<br>`cnn_64_math_symbols_8x8_30epochs.pth`<br>`cnn_128_math_symbols_8x8_30epochs.pth` | 15 (0–9, +, -, *, ÷, %) | `math_symbols_label_mapping.json` |

### 9.3 Checkpoint Format & Preprocessing Compatibility

- **Checkpoint Structure:** All 9 checkpoints contain a top-level dictionary with `state_dict`, `classes`, and `resolution`.
- **Output Layer:** Linear output feature dimension matches class count exactly: 67 for V1, 36 for V2/V3, 15 for V4.
- **Preprocessing:** All models were trained with binary 0/1 float32 tensors resized using `cv2.INTER_AREA` and thresholded at 100. The existing pipeline in `preprocess_image()` and `prepare_character_tensor_with_preview()` is 100% compatible with all three model families.

### 9.4 Verification & Test Results

All tests executed with actual results recorded:

1. **`test_backend.py`**: PASSED (4/4 tests passed).
2. **`test_prediction_logic.py`**: PASSED (8/8 tests passed).
3. **`test_line_recognition.py`**: PASSED (10/10 tests passed).
4. **`test_paragraph_recognition.py`**: PASSED (16/16 tests passed).
5. **`test_math_solver.py`**: PASSED (13/13 tests passed).
6. **`test_specialized_models.py`**: PASSED (5/5 tests passed).
7. **Frontend Production Build (`npm run build`)**: PASSED (`tsc -b && vite build` built in 689ms with 0 errors).
8. **Browser UI & Banner Verification**:
   - V2 Line Recognition: Banner displayed `"Tips for better recognition"`, copy verified, `"Predict"` button verified, banner dismissal verified.
   - V3 Paragraph Recognition: Banner displayed `"Tips for better recognition"`, copy verified, banner dismissal verified.
   - V4 Math Solver: Banner displayed `"Tips for better recognition"`, copy verified, `"Solve"` button verified, banner dismissal verified.
   - V1 Single Character: No banner displayed, `"Predict"` button verified.
9. **Server Cleanup**: All background processes on ports 5000 and 5173 terminated after testing.

### 9.5 Remaining Issues

None. All specialized models and updated guidance banners are fully integrated and verified across V1, V2, V3, and V4.
