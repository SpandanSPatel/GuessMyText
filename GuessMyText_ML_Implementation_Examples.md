# GuessMyText: Core Machine Learning & Image Segmentation Implementation

This educational document extracts and simplifies the core machine learning and computer vision algorithms from **GuessMyText**. It demonstrates how the system detects, segments, and classifies handwritten characters from images for academic presentation in Machine Learning Essentials.

---

## Academic & Architectural Context

In the project's experimental evaluation matrix (documented in `docs/TECHNICAL_INVESTIGATION_REPORT.md`), multiple machine learning approaches were evaluated across the 67-class handwritten dataset:

| Model Architecture | Evaluated Input Resolution | Recorded Test Accuracy (67 Classes) | Inference Latency | Academic Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **SVM (RBF Kernel)** | $32\times32$, $64\times64$, $128\times128$ | 83.88% ($32\times32$) | High (O(N) support vectors) | Struggles with high-dimensional raw pixel vectors. |
| **Random Forest (100 Trees)** | $32\times32$, $64\times64$, $128\times128$ | 84.03% ($32\times32$) | Moderate | Plateaus at ~84%; lacks spatial translation invariance. |
| **MLP (Linear-ReLU-Dropout)** | $32\times32$, $64\times64$, $128\times128$ | 82.49% ($32\times32$) | Low (<1.0 ms) | Cannot exploit 2D spatial locality; lags behind CNNs. |
| **Deep CNN (Selected Model)** | $32\times32$ | **96.04%** | Fast (<1.5 ms) | **Selected primary model:** Highest accuracy, robust local spatial feature hierarchies, translation invariance. |

For this academic demonstration, **all single-character inference examples exclusively use the single $32\times32$ CNN model** (`cnn_32_8x8_30epochs.pth`, test accuracy **96.04%** on 67 classes).

---

## A. Single-Character Prediction Using CNN

### Brief Explanation
The single-character prediction pipeline takes a handwritten character image (from a canvas or segmented crop), prepares it according to the exact training protocol, and classifies it using the trained $32\times32$ PyTorch CNN.

1. **Model Architecture**: A 2-stage convolutional feature extractor with $3\times3$ kernels, ReLU activations, $2\times2$ max-pooling, and an `AdaptiveAvgPool2d((8, 8))` layer producing a fixed 4,096-dimensional representation ($64 \times 8 \times 8$), followed by a fully-connected classifier with Dropout ($p=0.2$) projecting onto the 67 output classes.
2. **Preprocessing**:
   - Convert image to grayscale.
   - Resize to $32\times32$ pixels using area interpolation (`cv2.INTER_AREA`), matching training downsampling.
   - Threshold at pixel intensity 100 to produce a binary image ($0$ or $1$).
   - Wrap into a 4D float32 tensor of shape `(1, 1, 32, 32)` representing `(batch_size, channels, height, width)`.
3. **Inference**: A single forward pass through the $32\times32$ CNN in inference mode (`torch.inference_mode()`), followed by Softmax normalization to obtain the predicted class and confidence percentage.
4. **Single-Model Guarantee**: This code loads and runs **only the single $32\times32$ CNN model**. It contains no model ensembles, no runtime resolution switching, and no majority voting.

### Simplified Code

```python
"""
GuessMyText — Educational Example 1: Single-Character Prediction (32×32 CNN Only)
Source files: backend/model.py, backend/app.py
"""

import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. CNN Architecture Definition (matches backend/model.py)
# ---------------------------------------------------------------------------
class CNN(nn.Module):
    """
    2-stage Convolutional Neural Network with Adaptive Average Pooling.
    Shared architecture across GuessMyText CNN models.
    """
    def __init__(self, num_classes: int = 67):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# 2. Model Loading (Single 32×32 Checkpoint Only)
# ---------------------------------------------------------------------------
def load_32x32_model_and_mapping(
    checkpoint_path: str,
    label_mapping_path: str,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, dict[int, str]]:
    """
    Loads ONLY the single 32×32 CNN checkpoint (96.04% test accuracy)
    and the 67-class label mapping.
    """
    # Load 67-class label mapping (char -> id) and invert to (id -> char)
    with open(label_mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    id_to_char = {int(v): str(k) for k, v in mapping_data.items()}

    # Load 32×32 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Verify checkpoint resolution and class count
    assert checkpoint["resolution"] == 32, "Checkpoint is not 32×32"
    assert len(checkpoint["classes"]) == 67, "Expected 67 classes"

    # Instantiate single CNN and load state dict
    model = CNN(num_classes=67)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, id_to_char


# ---------------------------------------------------------------------------
# 3. Preprocessing Pipeline (matches backend/app.py:preprocess_image)
# ---------------------------------------------------------------------------
def preprocess_character_image(image_bgr_or_gray: np.ndarray) -> torch.Tensor:
    """
    Prepares an input character image for the 32×32 CNN:
      1. Converts to grayscale.
      2. Resizes to (32, 32) using area interpolation (cv2.INTER_AREA).
      3. Thresholds at pixel intensity 100 to form binary 0/1 pixels.
      4. Converts to float32 tensor of shape (1, 1, 32, 32).
    """
    # Step 1: Ensure grayscale
    if image_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_gray

    # Step 2: Resize to 32×32 (matches CNN resolution)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

    # Step 3: Threshold at 100 -> binary 0.0 or 1.0 (matches training dataset)
    binary = (resized > 100).astype(np.float32)

    # Step 4: Add batch and channel dimensions -> shape (1, 1, 32, 32)
    tensor = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)
    return tensor


# ---------------------------------------------------------------------------
# 4. Single-Character Inference
# ---------------------------------------------------------------------------
def predict_character(
    model: nn.Module,
    tensor: torch.Tensor,
    id_to_char: dict[int, str],
    device: torch.device = torch.device("cpu"),
) -> tuple[str, float]:
    """
    Executes inference using the single 32×32 CNN model.
    Returns: (predicted_character, confidence_percentage)
    """
    tensor_device = tensor.to(device)

    with torch.inference_mode():
        logits = model(tensor_device)                     # Shape: (1, 67)
        probabilities = F.softmax(logits, dim=1).squeeze(0)  # Shape: (67,)

    predicted_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_id].item() * 100.0)
    predicted_char = id_to_char[predicted_id]

    return predicted_char, confidence


# ---------------------------------------------------------------------------
# Example Usage Walkthrough
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "backend/models/CNN/original/cnn_32_8x8_30epochs.pth"
    mapping_path = "backend/models/CNN/original/label_mapping.json"

    # 1. Load model and mapping
    model, id_to_char = load_32x32_model_and_mapping(ckpt_path, mapping_path, device)

    # 2. Simulate or load an input character image (e.g., white letter on black canvas)
    sample_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.putText(sample_img, "A", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 6)

    # 3. Preprocess to 32×32 binary tensor
    tensor_32 = preprocess_character_image(sample_img)

    # 4. Predict using the single 32×32 CNN
    char, conf = predict_character(model, tensor_32, id_to_char, device)
    print(f"Prediction: '{char}' with confidence {conf:.2f}%")
```

### Explanation of Important Lines

* **`self.features = nn.Sequential(...)`** (lines 20–28): Two convolutional layers with $3\times3$ kernels and padding=1 maintain spatial dimensions before $2\times2$ max-pooling downsamples the feature maps. `AdaptiveAvgPool2d((8, 8))` produces a consistent $64 \times 8 \times 8$ tensor regardless of earlier layer spatial variations.
* **`self.classifier = nn.Sequential(...)`** (lines 30–37): Flattens the 4,096 feature values into a dense hidden layer with 128 units, followed by ReLU, Dropout ($p=0.2$) for regularization, and the final linear layer with 67 logits.
* **`assert checkpoint["resolution"] == 32`** (line 61): Confirms that this pipeline exclusively loads the $32\times32$ checkpoint (`cnn_32_8x8_30epochs.pth`).
* **`cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)`** (line 88): Downsamples handwritten ink using area-based pixel averaging, avoiding aliasing artifacts on thin strokes.
* **`binary = (resized > 100).astype(np.float32)`** (line 91): Converts pixel intensities into binary values ($0.0$ for background, $1.0$ for handwriting). The model was trained directly on these thresholded values without mean/variance normalization.
* **`torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)`** (line 94): Constructs the 4-dimensional tensor `(N=1, C=1, H=32, W=32)` expected by PyTorch 2D convolutional layers.
* **`with torch.inference_mode():`** (line 110): Disables autograd tracking and intermediate activation storage, minimizing memory and accelerating inference.
* **`probabilities = F.softmax(logits, dim=1)`** (line 112): Converts unnormalized network logits into a valid probability distribution over the 67 classes.

---

## B. Character Segmentation for Line Recognition

### Brief Explanation
Handwritten line recognition requires isolating individual characters from a continuous horizontal image of text before classification. The actual GuessMyText implementation in `backend/line_segmentation.py` uses a multi-stage computer vision algorithm:

1. **Binarization**: Converts the canvas image to a single-channel binary image ($0 = \text{black background}$, $255 = \text{white strokes}$) thresholded at intensity 100.
2. **Connected-Component Analysis (CCA)**: Uses OpenCV's 8-connectivity component labeling (`cv2.connectedComponentsWithStats`) to detect contiguous ink strokes and extract preliminary bounding boxes. Small noise artifacts ($<10$ pixels) are filtered out.
3. **Detached Stroke Merging (`merge_detached_strokes`)**: Solves a fundamental challenge in handwriting OCR: multi-stroke characters (such as the dot above 'i' and 'j', the two parallel bars of '=', or the division symbol '÷') naturally appear as disconnected components. The algorithm checks horizontal overlap and vertical proximity:
   - If a small component is positioned vertically above a character stem with significant horizontal alignment, it merges them into a single bounding box.
   - Side-by-side full-height characters are strictly prevented from merging.
4. **Touching Character Splitting (`split_touching_characters`)**: In natural cursive or cramped handwriting, adjacent characters frequently touch. The algorithm identifies abnormally wide bounding boxes ($w \ge 1.35 \times \text{median character width}$) and analyzes their **Vertical Projection Profile** ($V(x) = \sum_y \text{pixel}(x, y)$). If a clear valley exists in the central region ($25\%\text{--}75\%$) where stroke density drops below $45\%$ of the mean stroke height, the component is sliced at the valley index.
5. **Left-to-Right Ordering**: Bounding boxes are sorted by their horizontal coordinate (`boxes.sort(key=lambda b: b.x)`).
6. **Square Framing & Aspect-Ratio Preservation (`prepare_character_tensor`)**: Instead of stretching a narrow character (like '1' or 'l') directly into a $32\times32$ square (which causes severe horizontal distortion), the crop is centered within a square canvas with a 20% margin, resized via `cv2.INTER_AREA` to $32\times32$, thresholded, and passed directly into the **single $32\times32$ CNN**.

### Simplified Code

```python
"""
GuessMyText — Educational Example 2: Horizontal Line Character Segmentation
Source file: backend/line_segmentation.py
"""

from dataclasses import dataclass
from typing import List
import cv2
import numpy as np
import torch


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    def union(self, other: "BoundingBox") -> "BoundingBox":
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return BoundingBox(x=x, y=y, w=x2 - x, h=y2 - y)


# ---------------------------------------------------------------------------
# 1. Connected Component Extraction & Stroke Merging
# ---------------------------------------------------------------------------
def extract_components(binary: np.ndarray, min_area: int = 10) -> List[BoundingBox]:
    """Finds all 8-connected components with area >= min_area."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    boxes = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            boxes.append(BoundingBox(
                x=int(stats[i, cv2.CC_STAT_LEFT]),
                y=int(stats[i, cv2.CC_STAT_TOP]),
                w=int(stats[i, cv2.CC_STAT_WIDTH]),
                h=int(stats[i, cv2.CC_STAT_HEIGHT]),
            ))
    return boxes


def merge_detached_strokes(boxes: List[BoundingBox]) -> List[BoundingBox]:
    """
    Merges multi-stroke components belonging to the same character
    (e.g., dot of 'i' or 'j', two bars of '=', or parts of '÷').
    """
    if not boxes:
        return []

    line_height = max(40.0, float(max(b.h for b in boxes)))
    merged = list(boxes)
    changed = True

    while changed:
        changed = False
        new_merged = []
        skip_indices = set()

        for i in range(len(merged)):
            if i in skip_indices:
                continue
            curr = merged[i]

            for j in range(i + 1, len(merged)):
                if j in skip_indices:
                    continue
                cand = merged[j]

                # Calculate horizontal overlap and center distance
                overlap_x = max(0, min(curr.x2, cand.x2) - max(curr.x, cand.x))
                min_w = min(curr.w, cand.w)
                max_w = max(curr.w, cand.w)
                overlap_ratio = (overlap_x / min_w) if min_w > 0 else 0.0

                curr_mid_x = curr.x + curr.w / 2.0
                cand_mid_x = cand.x + cand.w / 2.0
                dx_center = abs(curr_mid_x - cand_mid_x)

                # Vertical gap between boxes
                v_gap = max(0, max(curr.y, cand.y) - min(curr.y2, cand.y2))

                # Never merge two full-height side-by-side letters
                if curr.h >= 0.50 * line_height and cand.h >= 0.50 * line_height:
                    if dx_center > 0.35 * max_w:
                        continue

                # Rule: Small dot near a larger stem (dot of 'i' / 'j')
                is_dot = (
                    min(curr.h, cand.h) <= 0.45 * line_height
                    and (overlap_ratio >= 0.15 or dx_center <= max_w * 0.60)
                    and v_gap <= line_height * 0.70
                )

                # Rule: Vertically stacked strokes ('=', '÷', ':')
                is_stacked = (
                    dx_center <= max_w * 0.45
                    and overlap_ratio >= 0.30
                    and v_gap <= line_height * 0.60
                )

                if is_dot or is_stacked:
                    curr = curr.union(cand)
                    skip_indices.add(j)
                    changed = True

            new_merged.append(curr)
        merged = new_merged

    return merged


# ---------------------------------------------------------------------------
# 2. Touching Character Splitting via Vertical Projection Profile (VPP)
# ---------------------------------------------------------------------------
def split_touching_characters(binary: np.ndarray, boxes: List[BoundingBox]) -> List[BoundingBox]:
    """
    Identifies unusually wide character components and splits them at
    vertical projection valleys if a clear valley exists.
    """
    if not boxes:
        return []

    widths = [b.w for b in boxes if b.w >= 10]
    med_width = float(np.median(widths)) if widths else 40.0

    result = []
    for box in boxes:
        # Detect candidate touching characters by relative width
        is_wide = (box.w >= 1.35 * med_width and box.w >= 30) or (box.w >= 1.05 * box.h and box.w >= 35)

        if is_wide:
            crop = binary[box.y : box.y2, box.x : box.x2]
            v_proj = np.sum(crop > 0, axis=0)  # Sum ink pixels down each column

            # Search central 25% to 75% region for a separation valley
            start_col = int(box.w * 0.25)
            end_col = int(box.w * 0.75)
            mid_proj = v_proj[start_col:end_col]

            if len(mid_proj) > 0:
                min_idx = int(np.argmin(mid_proj)) + start_col
                min_val = v_proj[min_idx]
                mean_val = float(np.mean(v_proj))

                # Split if the valley is significantly lower than average stroke density
                if min_val <= 0.45 * mean_val and min_val < box.h * 0.40:
                    box1 = BoundingBox(x=box.x, y=box.y, w=min_idx, h=box.h)
                    box2 = BoundingBox(x=box.x + min_idx, y=box.y, w=box.w - min_idx, h=box.h)
                    result.extend([box1, box2])
                    continue

        result.append(box)

    return result


# ---------------------------------------------------------------------------
# 3. Complete Line Segmentation & Square Framing for 32×32 CNN
# ---------------------------------------------------------------------------
def segment_line_characters(binary_line: np.ndarray) -> List[BoundingBox]:
    """
    Segments all characters in a line image, ordered strictly left-to-right.
    """
    raw_boxes = extract_components(binary_line)
    merged_boxes = merge_detached_strokes(raw_boxes)
    split_boxes = split_touching_characters(binary_line, merged_boxes)

    # Sort characters strictly left-to-right
    split_boxes.sort(key=lambda b: b.x)
    return split_boxes


def prepare_character_crop_for_32x32_cnn(
    binary: np.ndarray, bbox: BoundingBox
) -> torch.Tensor:
    """
    Crops character, frames it in a square canvas with 20% margin to preserve
    aspect ratio, and resizes to a (1, 1, 32, 32) tensor for CNN inference.
    """
    crop = binary[bbox.y : bbox.y2, bbox.x : bbox.x2]
    h, w = crop.shape
    if h == 0 or w == 0:
        crop = np.zeros((10, 10), dtype=np.uint8)
        h, w = 10, 10

    # Square framing with 20% margin
    max_side = max(h, w)
    pad = max(4, int(max_side * 0.20))
    square_size = max_side + 2 * pad
    square = np.zeros((square_size, square_size), dtype=np.uint8)

    y_offset = (square_size - h) // 2
    x_offset = (square_size - w) // 2
    square[y_offset : y_offset + h, x_offset : x_offset + w] = crop

    # Resize to 32×32 and threshold at 100
    resized = cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)
    bin_float = (resized > 100).astype(np.float32)

    return torch.from_numpy(bin_float).unsqueeze(0).unsqueeze(0)
```

### Explanation of Important Lines

* **`cv2.connectedComponentsWithStats(..., connectivity=8)`** (lines 45–47): Analyzes 8-neighbor connectivity to locate every separate ink cluster in a single pass, outputting exact pixel bounding boxes `[LEFT, TOP, WIDTH, HEIGHT, AREA]`.
* **`merge_detached_strokes(...)`** (lines 62–136): Iteratively resolves the multi-stroke disconnectedness problem. Lines 111–123 distinguish between genuine side-by-side characters (which must not be merged) and vertical stroke pairs (like the dot on 'i' or the two horizontal strokes of an '=' sign) based on vertical gap and center alignment.
* **`v_proj = np.sum(crop > 0, axis=0)`** (line 157): Calculates the vertical projection profile by summing foreground pixels along each column of the candidate crop.
* **`min_idx = int(np.argmin(mid_proj)) + start_col`** (line 164): Searches for the local minimum (the stroke "valley") between characters in cursive handwriting, splitting the connected glyphs at the point of minimal stroke thickness.
* **`split_boxes.sort(key=lambda b: b.x)`** (line 192): Ensures that the segmented bounding boxes follow natural left-to-right reading order.
* **`square_size = max_side + 2 * pad`** (lines 212–217): Creates an isotropic square canvas and centers the character inside it. This critical step ensures that narrow characters (e.g., '1', 'I') and wide characters (e.g., 'W', 'M') preserve their natural proportions when resized to the $32\times32$ CNN input grid.

---

## C. Line Segmentation for Paragraph Recognition

### Brief Explanation
When recognizing multi-line handwritten text (a paragraph), the vertical structure must be analyzed to separate individual lines of text before character segmentation can proceed. The actual GuessMyText implementation in `backend/paragraph_segmentation.py` uses **Horizontal Projection Profiling (HPP)** combined with adaptive smoothing and connected-component vertical centroid clustering:

1. **Horizontal Projection Profile (HPP)**: Sums foreground ink pixels horizontally across each row of the canvas ($H(y) = \sum_x \text{pixel}(x, y)$). High values indicate rows densely populated with text strokes; rows with $H(y) = 0$ represent clear whitespace between lines.
2. **Ink Run Detection**: Groups consecutive rows with $H(y) > 0$ into initial candidate horizontal bands.
3. **Intra-Line Gap Bridging**: In handwritten text, floating dots (on 'i'/'j'), accents, and detached strokes often create small vertical gaps within a single line. The algorithm bridges adjacent bands separated by small vertical gaps ($\le 22$ pixels, or up to 40 pixels for small dots) so a single line is not fragmented.
4. **Overlapping Line Splitting (Valley Detection)**: When an author writes compactly, descenders from the upper line (e.g., 'g', 'p', 'y') often touch or overlap with ascenders from the lower line (e.g., 'b', 'd', 'h'), producing an abnormally tall merged band ($> 1.85 \times \text{median character height}$). The algorithm applies a 1D moving average filter to smooth the HPP and searches the central $25\%\text{--}75\%$ region for a deep valley ($min\_val \le 0.45 \times max\_val$). If found, the merged band is split at the valley row.
5. **Component Assignment to Lines**: Initial connected components from the entire canvas are assigned to line bands based on their vertical center coordinate ($y_{mid} = y + h/2$).
6. **Top-to-Bottom Line Ordering**: Lines are sorted strictly by their vertical position, preserving natural document reading order.
7. **Passing to Line Pipeline**: Each line's assigned components are passed directly to the horizontal character segmentation stage (Example B) and classified individually by the **$32\times32$ CNN** (Example A).

### Simplified Code

```python
"""
GuessMyText — Educational Example 3: Vertical Paragraph Segmentation
Source file: backend/paragraph_segmentation.py
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np

# Reusing BoundingBox and extract_components from Example B
from line_segmentation_simplified import BoundingBox, extract_components


@dataclass
class LineRegion:
    line_index: int
    y_start: int
    y_end: int
    boxes: List[BoundingBox]

    @property
    def height(self) -> int:
        return self.y_end - self.y_start


# ---------------------------------------------------------------------------
# 1. Line Detection via Horizontal Projection Profile (HPP)
# ---------------------------------------------------------------------------
def detect_line_bands(binary: np.ndarray, components: List[BoundingBox]) -> List[Tuple[int, int]]:
    """
    Finds vertical start and end boundaries (y_start, y_end) for each text line:
      1. Computes Horizontal Projection Profile (HPP) across all rows.
      2. Groups continuous ink rows into runs.
      3. Bridges small intra-line gaps (dots on 'i', accents).
      4. Splits abnormally tall joined bands (overlapping descenders/ascenders).
    """
    h_img, w_img = binary.shape

    # Step 1: Compute Horizontal Projection Profile (row-wise ink sum)
    hpp = np.sum(binary > 0, axis=1)

    # Step 2: Find contiguous rows where ink is present (hpp > 0)
    runs = []
    in_run = False
    start = 0
    for y, val in enumerate(hpp):
        if val > 0 and not in_run:
            in_run = True
            start = y
        elif val == 0 and in_run:
            in_run = False
            runs.append((start, y))
    if in_run:
        runs.append((start, len(hpp)))

    if not runs:
        return []

    # Estimate median character height from individual components
    valid_heights = [c.h for c in components if c.h >= 20]
    med_h = float(np.median(valid_heights)) if valid_heights else 50.0

    # Step 3: Bridge small intra-line gaps within the same text line
    merged_bands = []
    for r in runs:
        if not merged_bands:
            merged_bands.append(r)
            continue
        prev_start, prev_end = merged_bands[-1]
        curr_start, curr_end = r
        gap = curr_start - prev_end
        prev_h = prev_end - prev_start
        curr_h = curr_end - curr_start

        # Bridge gap if <= 22px, or up to 40px if one component is a small dot/accent
        if gap <= 22 or ((prev_h <= 30 or curr_h <= 30) and gap <= 40):
            merged_bands[-1] = (prev_start, curr_end)
        else:
            merged_bands.append(r)

    # Step 4: Split abnormally tall merged bands (where descenders meet ascenders)
    final_bands = []
    for b_start, b_end in merged_bands:
        band_h = b_end - b_start
        if band_h > 1.85 * med_h and band_h >= 135:
            # Check smoothed HPP for a separation valley
            sub_hpp = hpp[b_start:b_end].astype(float)
            kernel_size = max(5, int(med_h * 0.25))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(sub_hpp, kernel, mode="same")

            s_idx = int(band_h * 0.25)
            e_idx = int(band_h * 0.75)
            min_rel_idx = int(np.argmin(smoothed[s_idx:e_idx])) + s_idx
            min_val = smoothed[min_rel_idx]
            max_val = np.max(smoothed)

            # Split at valley if significantly lower than the band peak
            if min_val <= 0.45 * max_val:
                split_row = b_start + min_rel_idx
                final_bands.append((b_start, split_row))
                final_bands.append((split_row, b_end))
                continue

        final_bands.append((b_start, b_end))

    return final_bands


# ---------------------------------------------------------------------------
# 2. Assign Components to Lines & Sort Top-to-Bottom
# ---------------------------------------------------------------------------
def segment_paragraph_into_lines(
    binary: np.ndarray,
) -> List[LineRegion]:
    """
    Segments a paragraph canvas into ordered LineRegion objects from top to bottom.
    """
    initial_boxes = extract_components(binary)
    if not initial_boxes:
        return []

    line_bands = detect_line_bands(binary, initial_boxes)

    # Assign each component to a line based on its vertical center (y + h/2)
    lines_components: List[List[BoundingBox]] = [[] for _ in line_bands]
    for comp in initial_boxes:
        comp_mid_y = comp.y + comp.h / 2.0
        best_line = -1
        min_dist = float("inf")

        for idx, (b_start, b_end) in enumerate(line_bands):
            if b_start <= comp_mid_y <= b_end:
                best_line = idx
                break
            # Find closest band if stroke extends slightly outside
            dist = min(abs(b_start - comp_mid_y), abs(comp_mid_y - b_end))
            if dist < min_dist:
                min_dist = dist
                best_line = idx

        if best_line >= 0:
            lines_components[best_line].append(comp)

    # Filter empty lines and sort strictly top-to-bottom by vertical median
    non_empty = [grp for grp in lines_components if grp]
    non_empty.sort(key=lambda grp: np.median([b.y + b.h / 2.0 for b in grp]))

    h_img = binary.shape[0]
    line_regions = []
    for idx, grp in enumerate(non_empty):
        y_start = max(0, min(b.y for b in grp) - 4)
        y_end = min(h_img, max(b.y2 for b in grp) + 4)
        line_regions.append(LineRegion(line_index=idx, y_start=y_start, y_end=y_end, boxes=grp))

    return line_regions
```

### Explanation of Important Lines

* **`hpp = np.sum(binary > 0, axis=1)`** (line 40): Projects the 2D binary paragraph onto a 1D vertical vector where each value represents total ink pixels in row $y$.
* **`gap <= 22 or ((prev_h <= 30 or curr_h <= 30) and gap <= 40)`** (lines 76–77): Bridges intra-line white space caused by detached dots and accents without merging distinct text lines.
* **`smoothed = np.convolve(sub_hpp, kernel, mode="same")`** (line 92): Applies moving average smoothing to eliminate high-frequency stroke noise, ensuring the valley finder targets true line gaps rather than intra-stroke loops (e.g., inside 'e' or 'o').
* **`min_val <= 0.45 * max_val`** (line 101): Validates that a valley is deep enough to represent a genuine separation between two vertically overlapping lines.
* **`comp_mid_y = comp.y + comp.h / 2.0`** (line 128): Uses the vertical centroid rather than bounding box edges to assign characters to lines. This correctly preserves characters with long ascenders or descenders in their parent text line.
* **`non_empty.sort(key=lambda grp: np.median([b.y + b.h / 2.0 for b in grp]))`** (line 147): Sorts all detected lines strictly from top to bottom based on the median vertical position of their characters, guaranteeing correct reading order.

---

## D. How the Three Components Work Together

### End-to-End Hierarchical Architecture

GuessMyText decomposes document recognition into a hierarchical, three-level pipeline:

1. **Document / Paragraph Level**: Coarse spatial layout analysis isolates horizontal line strips.
2. **Line Level**: Fine spatial segmentation isolates individual character glyphs and detects word spaces.
3. **Character Level**: Deep convolutional neural network ($32\times32$ CNN) performs pattern recognition.

```
+-----------------------------------------------------------------------------------+
|                           INPUT: MULTI-LINE PARAGRAPH                             |
|                           (Handwritten canvas image)                              |
+-----------------------------------------------------------------------------------+
                                          |
                                          v  [COMPONENT C: PARAGRAPH SEGMENTATION]
+-----------------------------------------------------------------------------------+
|  1. Horizontal Projection Profile (HPP): Calculate row-wise ink sum               |
|  2. Gap Bridging & Overlapping Valley Splitting: Separate lines                   |
|  3. Connected Component Vertical Centroid Clustering                              |
|  4. Top-to-Bottom Line Ordering                                                   |
+-----------------------------------------------------------------------------------+
                                          |
                +-------------------------+-------------------------+
                | Line 1 (Top)            | Line 2                  | Line N (Bottom)
                v                         v                         v
+-----------------------------------------------------------------------------------+
|                         [COMPONENT B: LINE SEGMENTATION]                          |
|  1. Connected Component Analysis (cv2.connectedComponentsWithStats)               |
|  2. Detached Stroke Merging (merge dots on 'i'/'j', parallel bars '=', '÷')       |
|  3. Touching Character Splitting (Vertical Projection Profile Valley Detection)   |
|  4. Strict Left-to-Right Sorting by X-Coordinate                                   |
|  5. Horizontal Gap Analysis -> Detect Word Boundaries / Spaces                    |
+-----------------------------------------------------------------------------------+
                                          |
        +---------------------------------+---------------------------------+
        | Char 1                          | Char 2                          | Char M
        v                                 v                                 v
+-----------------------------------------------------------------------------------+
|                     [COMPONENT A: 32×32 CNN PREDICTION]                          |
|  1. Square Framing: Center in square canvas with 20% margin (preserve ratio)      |
|  2. Downsample: Resize to 32×32 using cv2.INTER_AREA                              |
|  3. Binarize: Threshold at intensity 100 -> Float32 Tensor (1, 1, 32, 32)        |
|  4. Inference: Single 32×32 CNN forward pass (96.04% test accuracy)               |
|  5. Softmax -> Argmax -> 67-class Label Mapping                                   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        RECONSTRUCTED DOCUMENT TEXT                                |
|  - Characters concatenated into words using detected space thresholds             |
|  - Lines concatenated preserving newline breaks (\n)                              |
+-----------------------------------------------------------------------------------+
```

### Combined Integration Script

The following script ties all three components together in a clean, self-contained educational demonstration:

```python
"""
GuessMyText — Educational Example 4: End-to-End Paragraph Recognition Pipeline
Demonstrates how Line Segmentation, Character Segmentation, and the 32×32 CNN collaborate.
"""

import numpy as np
import torch
import cv2

# Import from simplified components above
from cnn_single_char_simplified import CNN, load_32x32_model_and_mapping, predict_character
from line_segmentation_simplified import segment_line_characters, prepare_character_crop_for_32x32_cnn
from paragraph_segmentation_simplified import segment_paragraph_into_lines


def recognize_handwritten_paragraph(
    canvas_image_bgr: np.ndarray,
    model: torch.nn.Module,
    id_to_char: dict[int, str],
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Executes the full hierarchical recognition pipeline:
      1. Binarizes the canvas.
      2. Segments into top-to-bottom lines.
      3. For each line, segments into left-to-right characters.
      4. For each character, prepares a 32×32 tensor and classifies using the 32×32 CNN.
      5. Reconstructs text preserving spaces and newlines (\n).
    """
    # Step 1: Binarization (threshold 100)
    gray = cv2.cvtColor(canvas_image_bgr, cv2.COLOR_BGR2GRAY) if canvas_image_bgr.ndim == 3 else canvas_image_bgr
    binary = np.where(gray > 100, 255, 0).astype(np.uint8)

    # Step 2: Vertical paragraph segmentation (top-to-bottom lines)
    line_regions = segment_paragraph_into_lines(binary)
    if not line_regions:
        return ""

    reconstructed_lines = []

    for line_region in line_regions:
        # Step 3: Horizontal line segmentation (left-to-right characters)
        # Extract the binary crop of this line band
        line_crop = binary[line_region.y_start : line_region.y_end, :]
        char_boxes = segment_line_characters(line_crop)

        if not char_boxes:
            continue

        # Estimate word spaces based on horizontal gaps between characters
        spaces_after = []
        if len(char_boxes) > 1:
            gaps = [char_boxes[i + 1].x - char_boxes[i].x2 for i in range(len(char_boxes) - 1)]
            widths = [b.w for b in char_boxes if b.w >= 8]
            med_w = float(np.median(widths)) if widths else 40.0
            med_gap = float(np.median([g for g in gaps if g > 2])) if any(g > 2 for g in gaps) else 10.0
            space_threshold = max(0.45 * med_w, 1.85 * med_gap, min(20.0, 0.55 * med_w))
            spaces_after = [g >= space_threshold for g in gaps] + [False]
        else:
            spaces_after = [False]

        # Step 4: Classify each character with the single 32×32 CNN
        line_chars = []
        for box, has_space in zip(char_boxes, spaces_after):
            char_tensor_32 = prepare_character_crop_for_32x32_cnn(line_crop, box)
            pred_char, conf = predict_character(model, char_tensor_32, id_to_char, device)

            line_chars.append(pred_char)
            if has_space:
                line_chars.append(" ")

        reconstructed_lines.append("".join(line_chars))

    # Step 5: Join lines with newlines
    return "\n".join(reconstructed_lines)
```

### Academic Summary for Professor Presentation

* **Decoupling Geometry from Classification**: A deep neural network is exceptionally effective at learning scale- and shift-invariant representations of individual glyphs. However, training an end-to-end network to predict unrestricted 2D multi-line handwritten text requires immense datasets and complex sequence-to-sequence architectures (e.g., CRNN with CTC loss or Transformer decoders). By using classical, deterministic computer vision techniques (Projection Profiling and Connected-Component Analysis) to solve the geometric layout problem, the system isolates individual characters with high precision, allowing a lightweight $32\times32$ CNN to achieve **96.04% recognition accuracy** with sub-millisecond inference latency.
* **Why the $32\times32$ CNN Outperformed Alternative Paradigms**:
  1. *Versus SVM / Random Forest*: Traditional classifiers operate on flattened 1D pixel vectors ($1,024$ dimensions), discarding spatial topological relationships (edges, loops, stroke crossings) and failing when strokes vary in position by just 2–3 pixels.
  2. *Versus MLP*: Fully connected networks treat each pixel independently, requiring vastly more parameters to learn translation invariance.
  3. *Versus Higher CNN Resolutions ($64\times64$, $128\times128$)*: In the experimental matrix, the $32\times32$ CNN achieved **96.04%**, compared to 95.82% ($64\times64$) and 95.15% ($128\times128$). For isolated single-stroke characters, $32\times32$ provides sufficient spatial sampling while preventing overfitting and reducing compute overhead.

---

## E. Source References & Verification

### Code Traceability Matrix

Every algorithm and parameter in these educational examples maps directly to the actual production source files in GuessMyText:

| Component | Source File | Original Function / Symbol | Educational Simplification Made |
| :--- | :--- | :--- | :--- |
| **CNN Architecture** | `backend/model.py` | `class CNN(nn.Module)` (lines 15–41) | Preserved exactly as defined in the repository. |
| **Model Checkpoint** | `backend/models/CNN/original/` | `cnn_32_8x8_30epochs.pth` | Extracted the single $32\times32$ model (96.04% accuracy). Excluded 64×64 and 128×128 models and majority voting. |
| **Label Mapping** | `backend/models/CNN/original/` | `label_mapping.json` | 67 classes (0–9 digits, A–Z uppercase, a–z lowercase, +, -, *, ÷, %). Loaded via standard `json.load`. |
| **Image Preprocessing** | `backend/app.py` | `preprocess_image()` (lines 264–307) | Removed base64 preview generation and Flask request handling. Preserved grayscale conversion, `cv2.resize(..., (32, 32), INTER_AREA)`, and `> 100` thresholding. |
| **Single Inference** | `backend/app.py` | `run_inference()` (lines 314–359) | Removed CUDA timing synchronization and Top-3 dictionary formatting; preserved softmax and argmax. |
| **Connected Components** | `backend/line_segmentation.py` | `extract_initial_components()` (lines 94–109) | Preserved OpenCV 8-connectivity CCA and area filtering (`min_area=10`). |
| **Detached Stroke Merging** | `backend/line_segmentation.py` | `merge_detached_strokes()` (lines 112–200) | Preserved dot-of-'i' and vertical stacking rules; streamlined iterative loop for readability. |
| **Touching Splitting** | `backend/line_segmentation.py` | `split_touching_characters()` (lines 203–261) | Preserved vertical projection profile calculation, central 25%–75% search window, and 0.45 threshold. |
| **Square Framing** | `backend/line_segmentation.py` | `prepare_character_tensor_with_preview()` (lines 330–374) | Preserved 20% margin square padding, aspect-ratio preservation, and `(1, 1, 32, 32)` float32 tensor conversion. |
| **Horizontal Projection** | `backend/paragraph_segmentation.py` | `detect_line_regions()` (lines 119–244) | Preserved row-wise HPP sum, gap bridging ($\le 22$px), moving-average smoothing, and valley detection splitting. |
| **Line Ordering & Clustering**| `backend/paragraph_segmentation.py` | `segment_paragraph_lines()` (lines 247–279) | Preserved centroid clustering (`y + h/2`) and median-based top-to-bottom sorting. |

### Technical Verification Note

1. **Environment & Dependencies**:
   - The production GuessMyText backend specifies Python dependencies in `backend/requirements.txt`: `torch>=2.0`, `torchvision>=0.15`, `opencv-python-headless>=4.8`, `numpy>=1.24`, `Pillow>=10.0`.
   - The host system runs Windows with Python 3.14. Because official prebuilt PyTorch binary wheels are not yet distributed for Python 3.14 on Windows, standalone runtime execution of PyTorch was not performed in the local Windows Python 3.14 shell.
   - All code logic, tensor dimensions, threshold constants, and mathematical operations were statically traced and cross-verified against the repository's test suites (`backend/test_backend.py`, `backend/test_line_recognition.py`, and `backend/test_paragraph_recognition.py`).
2. **Exclusion of Ensembles & Web Components**:
   - The multi-model 3-CNN ensemble (`aggregate_predictions` and 3-model majority voting) from `backend/app.py` has been completely excluded from the single-character prediction pipeline as requested.
   - Flask routes, CORS headers, HTTP request parsers, React/TypeScript canvas components, and base64 preview encoding have been omitted so the presentation focuses strictly on Machine Learning and Computer Vision.
