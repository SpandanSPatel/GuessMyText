# GuessMyText — V1 Implementation Plan

**Version:** 1.0
**Scope:** Single-character handwriting recognition
**Status:** Planned

---

## 1. Project Objective

Build a temporary, functional web application called GuessMyText using the existing CaptureMyText project as the source for its frontend design and reusable components.

V1 will allow a user to draw a single character or mathematical symbol on a canvas and receive predictions from three existing, trained CNN models.

The application will compare the predictions from the 32×32, 64×64, and 128×128 models.

The primary objective is to validate the complete image-to-prediction pipeline using the original drawing interface and training-compatible image preprocessing.

This is an initial working version, not the final polished website.

### V1 requirements

* Reuse the CaptureMyText design and relevant frontend components.
* Replace the purple accent color with light blue `#89CFF0`.
* Preserve the existing drawing canvas behavior.
* Load the existing CNN checkpoints without retraining.
* Process one drawing through all three models.
* Display predictions and debugging information.
* Avoid permanent image storage and database operations.
* Include navigation for V1–V4, with V2–V4 displaying Coming Soon pages.

## 2. Existing Project Structure

The current project root contains:

```text
Project/
├── CaptureMyText_Copy/
│   ├── Assets/
│   ├── backend/
│   ├── Docs/
│   └── frontend/
│
├── New/
│   ├── datasets/
│   ├── Images/
│   ├── models/
│   ├── notebooks/
│   ├── results/
│   ├── generate_dataset.py
│   └── ...
│
└── Old/
```

The new GuessMyText project must be created inside the existing project root.

The intended structure is:

```text
Project/
├── CaptureMyText_Copy/
├── New/
├── Old/
└── GuessMyText/
    ├── docs/
    │   ├── IMPLEMENTATION_PLAN_V1.md
    │   └── AGENT_PROMPT_V1.md
    ├── frontend/
    ├── backend/
    └── README.md
```

The exact frontend and backend structure must be decided after inspecting the existing projects.

Do not create unnecessary directories or move the existing ML project.

### Source of truth

* `CaptureMyText_Copy/`: Existing website implementation, frontend design, canvas, navigation, and reusable components.
* `New/`: Existing datasets, preprocessing implementation, training notebooks, class mapping, and model checkpoints.
* `GuessMyText/`: New application, isolated from the original website and existing ML project.

## 3. Phase 1 — Inspect Existing Projects

Before implementation, inspect the existing code.

### 3.1 CaptureMyText inspection

Inspect the relevant files, including:

* Frontend package configuration.
* Existing application routing.
* Navbar implementation.
* Drawing canvas component.
* Canvas image-export functionality.
* Shared styles and theme configuration.
* Existing page layout and reusable UI components.
* Existing backend configuration, if relevant.

The previously inspected frontend includes components such as:

* `CapturePage.tsx`
* `DrawingCanvas.tsx`
* `ClassSelector.tsx`
* `ProgressIndicator.tsx`

Confirm their actual locations before using them.

### Requirements

1. Reuse the existing drawing canvas wherever practical.
2. Reuse the navbar and relevant layout components.
3. Preserve the existing interaction behavior and visual design.
4. Do not rebuild working components unnecessarily.
5. Do not modify the original CaptureMyText project.

### 3.2 ML project inspection

Inspect:

* `New/generate_dataset.py`
* The CNN training notebook.
* The CNN architecture.
* The checkpoint-saving implementation.
* The dataset-generation pipeline.
* `datasets/label_mapping.json`.
* Any metadata saved in the checkpoints.

Confirm the actual filenames and locations.

Do not assume that the existing checkpoints contain complete model objects. Determine whether they contain state dictionaries or other saved structures.

Document the findings that affect inference before implementing preprocessing.

## 4. Phase 2 — Create the GuessMyText Project

Create a separate GuessMyText application.

The frontend should use the same framework, dependencies, styling conventions, and component patterns as CaptureMyText wherever practical.

The backend should use Python and Flask for model inference.

Do not migrate or replace the existing frontend stack without a technical reason.

### Design requirements

* Preserve CaptureMyText's existing design and layout.
* Reuse the existing navbar, drawing canvas, and relevant shared components.
* Change the purple accent to `#89CFF0`.
* Retain the existing dark theme, white elements, spacing, typography, borders, and component styling wherever practical.
* Do not use the Python desktop tester as a design reference.
* Avoid unnecessary redesigns, animations, and new UI frameworks.

The visual result should look like a lightly rebranded CaptureMyText application.

## 5. Phase 3 — Add Navigation and Page Routing

The navbar must provide four navigation options.

| Version | Page                         | Status      |
| ------- | ---------------------------- | ----------- |
| V1      | Single Character Recognition | Functional  |
| V2      | Line Recognition             | Coming Soon |
| V3      | Paragraph Recognition        | Coming Soon |
| V4      | Math Solver                  | Coming Soon |

### V1

The V1 page contains the drawing canvas and model prediction interface.

### V2–V4

These pages should be simple Coming Soon placeholders that follow the existing website's visual style.

Do not implement their future recognition functionality.

Do not implement character segmentation, line segmentation, paragraph processing, or mathematical expression evaluation during V1.

## 6. Phase 4 — Implement the ML Inference Pipeline

This is the most important phase.

### 6.1 Existing models

Use the existing checkpoints:

```text
New/models/CNN/cnn_32_8x8_30epochs.pth
New/models/CNN/cnn_64_8x8_30epochs.pth
New/models/CNN/cnn_128_8x8_30epochs.pth
```

Resolve the paths relative to the project configuration or a stable absolute project-root path. Do not depend on the current terminal working directory.

Load the checkpoints directly from `New/models/CNN/`. Do not duplicate the model files.

### 6.2 Model architecture

Inspect the training notebook and reproduce the exact architecture used for each checkpoint.

The previously documented CNN uses convolutional layers, pooling, adaptive average pooling, and a fully connected classifier. Verify the exact layer definitions before implementation.

Confirm:

* Convolutional layer parameters.
* Pooling operations.
* Adaptive pooling output dimensions.
* Fully connected layer dimensions.
* Dropout configuration where relevant.
* Number of output classes.

Load the saved weights correctly and verify compatibility.

Do not retrain, alter, or overwrite the existing models.

### 6.3 Model initialization

At backend startup:

1. Resolve model paths.
2. Load the label mapping.
3. Instantiate the correct architecture.
4. Load each checkpoint.
5. Verify output dimensions and class ordering.
6. Set each model to evaluation mode.
7. Move models to the selected inference device.

Load the models once rather than reloading them for each prediction request.

Use CUDA if available and appropriate; otherwise, use CPU inference.

If a checkpoint cannot be loaded, report a clear error. Do not silently substitute another model or fabricated result.

### 6.4 Class mapping

Use the existing `New/datasets/label_mapping.json`.

The model recognizes 67 classes:

* Digits `0–9`.
* Uppercase letters `A–Z`.
* Lowercase letters `a–z`.
* Operators `÷`, `-`, `*`, `%`, and `+`.

Verify the direction of the mapping and the exact relationship between class IDs and model output positions.

Do not assume the output classes are ordered alphabetically.

If checkpoint metadata contains class-order information, inspect it. Otherwise, reconstruct the correct ordering from the training implementation.

The same mapping must be used consistently across all three models.

### 6.5 Input image

Reuse the original CaptureMyText drawing canvas.

Preserve its existing:

* Internal canvas dimensions.
* Coordinate scaling.
* Stroke rendering.
* Brush behavior.
* Background and stroke colors.
* PNG-export behavior.

Do not replace it with a new canvas implementation unless the existing component cannot be reused.

The canvas image should be exported and sent to Flask for inference.

The image should be processed in memory and must not be permanently stored.

### 6.6 Training-compatible preprocessing

Before writing preprocessing code, inspect the dataset generator and training notebook.

Determine:

* How source images were decoded.
* Whether grayscale conversion was performed.
* Whether images were cropped, padded, centered, or otherwise transformed.
* Which resize method and interpolation settings were used.
* Whether pixel values were thresholded or binarized.
* Whether CSV pixel values represent binary integers, normalized floats, or another representation.
* The expected input tensor shape and dtype.
* Whether any normalization was applied during training.

Implement the inference preprocessing to match the actual training pipeline.

Do not introduce additional transformations without evidence that they were part of training.

In particular, do not automatically introduce:

* ImageNet normalization.
* Automatic cropping.
* Automatic centering.
* Padding.
* Additional thresholding.
* New aspect-ratio adjustments.

These may be investigated later if baseline results indicate a problem.

### 6.7 Generate three model inputs

Start from the same decoded canvas image.

Create the three inputs using the verified training-compatible pipeline:

```text
Original canvas image
          │
          ▼
Training-compatible preprocessing
          │
          ├──► 32×32 tensor ──► CNN 32
          │
          ├──► 64×64 tensor ──► CNN 64
          │
          └──► 128×128 tensor ► CNN 128
```

Each model must receive the correct input resolution, tensor dimensions, data type, pixel range, and channel ordering.

Do not use a single resized image for all three models.

### 6.8 Prediction

Run inference for all three models using the same original drawing.

Use evaluation mode and disable gradient calculation.

For each model:

1. Obtain the model's output logits.
2. Apply softmax to obtain class probabilities.
3. Identify the highest-probability class.
4. Identify the top three classes.
5. Convert class IDs to characters using the verified mapping.
6. Record the inference duration.

The displayed confidence should be the predicted class's softmax probability expressed as a percentage.

Confidence is not a guarantee of correctness and should not be presented as measured model accuracy.

### 6.9 Inference timing

Measure inference time independently for each model.

Do not include model loading or application startup time.

When using CUDA, synchronize appropriately before and after timed inference so asynchronous GPU execution does not distort the measurements.

Treat the timings as approximate per-request measurements, not benchmark results.

## 7. Phase 5 — Implement the Flask Prediction API

Create a minimal Flask backend.

The backend should be responsible only for:

1. Loading the existing models.
2. Receiving the drawing image.
3. Decoding the image in memory.
4. Applying the verified preprocessing pipeline.
5. Running inference through the three models.
6. Returning the predictions and diagnostic information.

Use an image-upload mechanism appropriate for the existing frontend and canvas-export implementation.

### API behavior

Provide a prediction endpoint that accepts one image and returns the results from all three models.

The response should include:

* Model identifier.
* Predicted character.
* Confidence.
* Top-three predictions and their probabilities.
* Inference time.
* Processed input preview data or another suitable representation for the frontend.

Include meaningful error responses for invalid input, empty images, missing checkpoints, and inference failures.

Do not add database connections, authentication, cloud storage, or image-collection functionality.

## 8. Phase 6 — Implement the V1 Prediction Interface

Reuse the existing CaptureMyText page layout and canvas.

Replace the original capture workflow with a Predict workflow.

### Required controls

* Drawing canvas.
* Clear button.
* Predict button.

### Required results

Display three clearly separated model result sections.

Each section should include:

* Model resolution.
* Predicted character.
* Confidence percentage.
* Top-three predictions.
* Inference time.
* Processed input preview.

Display a clear agreement indicator when all three models predict the same character.

If the models disagree, display their independent results without forcing a consensus prediction.

The frontend should handle loading states, API errors, and invalid input.

Keep the drawing visible after prediction so the user can inspect the result.

Clear should reset the canvas and prediction results.

Prevent duplicate requests while a prediction is running.

Do not include the original class selector, progress indicators, capture quotas, or image-storage workflow unless a component is explicitly required for another purpose.

## 9. Phase 7 — Verification and Testing

The agent must test the application rather than only generating code.

### 9.1 Model tests

Verify that:

* All three checkpoints load successfully.
* All three models use the expected architecture.
* Output dimensions match the class mapping.
* The correct character is associated with each output ID.
* All models can perform inference.

### 9.2 Preprocessing tests

Verify that:

* The image is decoded correctly.
* The background and stroke colors are preserved.
* The preprocessing reproduces the training pipeline.
* Each model receives its correct resolution.
* Tensor shapes, dtype, and pixel values match the training requirements.
* The generated previews match the actual model inputs.

### 9.3 Frontend tests

Verify that:

* The reused canvas works.
* Drawing and clearing work.
* Predict sends the correct image.
* Results are displayed correctly.
* All three models are compared.
* V2–V4 navigation opens Coming Soon pages.
* The design retains CaptureMyText's structure and styling, with the blue accent.

### 9.4 End-to-end test

Draw several characters and symbols, including uppercase letters, lowercase letters, digits, and supported operators.

Test confusing pairs such as:

* `O` and `0`.
* `l` and `1`.
* `S` and `5`.

Compare the displayed predictions and processed inputs.

Do not expect every prediction to be correct. Record the observed behavior and distinguish actual model errors from preprocessing or integration problems.

### 9.5 Storage verification

Confirm that ordinary prediction requests do not create permanent image files, dataset records, or database entries.

## 10. Out of Scope

Do not implement:

* Model retraining.
* New CNN architectures.
* Dataset generation.
* Dataset collection.
* Character segmentation.
* Line recognition.
* Paragraph recognition.
* Mathematical expression evaluation.
* Database integration.
* User authentication.
* Cloud image storage.
* Production deployment.
* Unnecessary UI redesign.

These may be considered in later versions.

## 11. Completion Criteria

V1 is complete when:

1. The GuessMyText project runs independently of CaptureMyText.
2. The existing canvas and navbar have been reused.
3. The primary accent color is `#89CFF0`.
4. The three existing CNN checkpoints load successfully.
5. Preprocessing matches the verified training pipeline.
6. One drawing produces predictions from all three models.
7. The results include character, confidence, top-three predictions, and inference time.
8. Processed input previews are available for debugging.
9. V2–V4 display Coming Soon pages.
10. Prediction images are not permanently stored.
11. The main workflow has been tested successfully.

The final report must list the files created, installation requirements, startup commands, tests performed, and any unresolved issues.
