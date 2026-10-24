"""
GuessMyText — V2 Line Segmentation and Reconstruction Pipeline

Provides modular components for handwritten line recognition:
  1. Canvas image decoding & empty detection
  2. Connected Component Analysis (CCA) with bounding boxes
  3. Detached stroke merging (dots on 'i'/'j', multi-stroke symbols: '=', '÷', '%')
  4. Left-to-right sorting of character candidates
  5. Touching character separation via vertical projection profile analysis
  6. Inter-character gap analysis and adaptive word-boundary (space) estimation
  7. Training-compatible square framing and multi-scale resizing
  8. Character recognition via 3-CNN ensemble with majority voting
  9. Line reconstruction with uncertainty marking
"""

from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import base64
import cv2
import numpy as np
import torch
import torch.nn.functional as F

THRESHOLD = 100


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


@dataclass
class CharacterSegment:
    index: int
    bbox: BoundingBox
    character: str
    is_uncertain: bool
    has_space_after: bool = False
    raw_predictions: Optional[List[str]] = None


def decode_and_binarize(image_bytes: bytes) -> np.ndarray:
    """
    Decodes PNG bytes and produces a single-channel binary image
    (0 = background, 255 = white handwriting).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Could not decode image from provided bytes.")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # Threshold at 100 — consistent with generate_dataset.py
    binary = np.where(gray > THRESHOLD, 255, 0).astype(np.uint8)
    return binary


def is_canvas_empty(binary: np.ndarray, min_pixels: int = 15) -> bool:
    """Returns True if foreground pixel count is below min_pixels."""
    return int(np.count_nonzero(binary)) < min_pixels


def extract_initial_components(binary: np.ndarray, min_area: int = 10) -> List[BoundingBox]:
    """
    Extracts bounding boxes of all connected components with area >= min_area.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append(BoundingBox(x=x, y=y, w=w, h=h))
    return boxes


def merge_detached_strokes(boxes: List[BoundingBox]) -> List[BoundingBox]:
    """
    Merges components that belong to the same character, such as:
      - The dot above 'i' or 'j'
      - Two horizontal bars of '=' or '-'
      - Division symbol '÷' (bar + two dots)
      - Percent sign '%' (slash and circular dots)
      - Colons or multi-stroke diacritics
    Strictly avoids merging adjacent side-by-side characters that share slight horizontal overlap.
    """
    if not boxes:
        return []

    # Dominant character height in this line
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

            current = merged[i]
            for j in range(i + 1, len(merged)):
                if j in skip_indices:
                    continue

                candidate = merged[j]

                # Horizontal overlap
                overlap_x = max(0, min(current.x2, candidate.x2) - max(current.x, candidate.x))
                min_w = min(current.w, candidate.w)
                max_w = max(current.w, candidate.w)
                overlap_ratio = (overlap_x / min_w) if min_w > 0 else 0.0

                # Horizontal center distance
                current_mid_x = current.x + current.w / 2.0
                cand_mid_x = candidate.x + candidate.w / 2.0
                dx_center = abs(current_mid_x - cand_mid_x)

                # Vertical gap
                if current.y2 < candidate.y:
                    v_gap = candidate.y - current.y2
                elif candidate.y2 < current.y:
                    v_gap = current.y - candidate.y2
                else:
                    v_gap = 0  # Vertically overlapping

                # Rule 0: If BOTH components are full-sized characters and their horizontal
                # centers are separated, they are distinct side-by-side characters (e.g. 'To', 'AV')
                # and must NEVER be merged.
                if current.h >= 0.50 * line_height and candidate.h >= 0.50 * line_height:
                    if dx_center > 0.35 * max_w:
                        continue

                # Rule 1: Small detached dot/stroke near a larger component (e.g. dot of 'i' or 'j')
                min_h = min(current.h, candidate.h)
                min_area = min(current.area, candidate.area)
                is_small_detached_dot = (
                    (min_h <= 0.45 * line_height and min_area <= line_height * 16)
                    and (overlap_ratio >= 0.15 or dx_center <= max_w * 0.60)
                    and v_gap <= line_height * 0.70
                )

                # Rule 2: Vertically stacked strokes of a symbol (e.g. '=', '÷', ':', '%')
                # Must have strong horizontal center alignment and one vertically stacked above the other
                is_vertically_stacked = (
                    dx_center <= max_w * 0.45
                    and overlap_ratio >= 0.30
                    and v_gap <= line_height * 0.60
                    and (current.h + candidate.h + v_gap <= line_height * 1.35)
                )

                if is_small_detached_dot or is_vertically_stacked:
                    current = current.union(candidate)
                    skip_indices.add(j)
                    changed = True

            new_merged.append(current)

        merged = new_merged

    return merged


def split_touching_characters(binary: np.ndarray, boxes: List[BoundingBox]) -> List[BoundingBox]:
    """
    Detects wide components that likely contain touching/connected characters
    and splits them at vertical projection valleys if a clear valley exists.
    Recalculates tight bounding boxes for each split character to avoid padding distortion.
    """
    if not boxes:
        return []

    widths = [b.w for b in boxes if b.w >= 10]
    if not widths:
        return boxes
    med_width = float(np.median(widths))

    result = []
    for box in boxes:
        # Check if box is wide relative to other characters or its own height
        is_wide = (box.w >= 1.35 * med_width and box.w >= 30) or (box.w >= 1.05 * box.h and box.w >= 35)
        if is_wide:
            crop = binary[box.y : box.y2, box.x : box.x2]
            # Column-wise sum of white pixels
            v_proj = np.sum(crop > 0, axis=0)

            # Look for valleys in the middle 25% to 75% range
            start_col = int(box.w * 0.25)
            end_col = int(box.w * 0.75)
            mid_proj = v_proj[start_col:end_col]

            if len(mid_proj) > 0:
                min_idx = int(np.argmin(mid_proj)) + start_col
                min_val = v_proj[min_idx]
                mean_val = float(np.mean(v_proj))

                # Split if valley is noticeably lower than the mean stroke height
                if min_val <= 0.45 * mean_val and min_val < box.h * 0.4:
                    # Slices
                    slice1 = crop[:, :min_idx]
                    slice2 = crop[:, min_idx:]

                    pts1 = cv2.findNonZero(slice1)
                    pts2 = cv2.findNonZero(slice2)

                    if pts1 is not None and pts2 is not None:
                        rx1, ry1, rw1, rh1 = cv2.boundingRect(pts1)
                        rx2, ry2, rw2, rh2 = cv2.boundingRect(pts2)

                        box1 = BoundingBox(x=box.x + rx1, y=box.y + ry1, w=rw1, h=rh1)
                        box2 = BoundingBox(x=box.x + min_idx + rx2, y=box.y + ry2, w=rw2, h=rh2)
                        result.extend([box1, box2])
                        continue
                    else:
                        box1 = BoundingBox(x=box.x, y=box.y, w=min_idx, h=box.h)
                        box2 = BoundingBox(x=box.x + min_idx, y=box.y, w=box.w - min_idx, h=box.h)
                        result.extend([box1, box2])
                        continue

        result.append(box)

    return result


def segment_characters(binary: np.ndarray) -> List[BoundingBox]:
    """
    Main segmentation entry point:
      1. Extract connected components
      2. Merge detached strokes (e.g. dots of 'i', parts of '÷', '%', '=')
      3. Attempt splitting of wide touching components
      4. Sort left-to-right
    """
    raw_boxes = extract_initial_components(binary)
    if not raw_boxes:
        return []

    merged_boxes = merge_detached_strokes(raw_boxes)
    refined_boxes = split_touching_characters(binary, merged_boxes)

    # Sort left to right by x
    refined_boxes.sort(key=lambda b: b.x)
    return refined_boxes


def estimate_word_boundaries(boxes: List[BoundingBox]) -> List[bool]:
    """
    Calculates horizontal gaps between consecutive character bounding boxes
    and determines where word spaces should be inserted.
    Returns a list of booleans of length len(boxes), where True at index i
    means a space should follow character i.
    Uses scale-adaptive thresholding based on handwriting width and gap distribution.
    """
    if len(boxes) <= 1:
        return [False] * len(boxes)

    gaps = []
    for i in range(len(boxes) - 1):
        gap = boxes[i + 1].x - boxes[i].x2
        gaps.append(max(0, gap))

    widths = [b.w for b in boxes if b.w >= 8]
    med_width = float(np.median(widths)) if widths else 40.0

    pos_gaps = [g for g in gaps if g > 2]
    med_gap = float(np.median(pos_gaps)) if pos_gaps else 10.0

    # Scale-adaptive space threshold:
    # Distinguishes ordinary intra-word character spacing from larger word gaps
    space_threshold = max(0.45 * med_width, 1.85 * med_gap, min(20.0, 0.55 * med_width))

    spaces = []
    for g in gaps:
        spaces.append(g >= space_threshold)
    spaces.append(False)  # Last character has no trailing space by default

    return spaces


def ndarray_to_png_data_url(img: np.ndarray) -> str:
    """Encodes a uint8 grayscale numpy array to a PNG data URL (data:image/png;base64,...)."""
    try:
        success, encoded = cv2.imencode(".png", img)
        if not success:
            return ""
        b64_str = base64.b64encode(encoded).decode("ascii")
        return f"data:image/png;base64,{b64_str}"
    except Exception:
        return ""


def prepare_character_tensor_with_preview(
    binary: np.ndarray, bbox: BoundingBox, resolution: int
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Crops the character from the binary image, frames it centered in a
    square canvas with 20% margin to preserve natural aspect ratio,
    resizes to (resolution, resolution) using cv2.INTER_AREA,
    and returns:
      - tensor: float32 tensor of shape (1, 1, resolution, resolution) for model
      - model_input_uint8: uint8 array (0 or 255) representing the exact binary pixels seen by CNN
      - raw_crop: uint8 array of the unpadded bounding box crop
    """
    # Crop from binary image (ensure within bounds)
    h_img, w_img = binary.shape
    y1 = max(0, bbox.y)
    y2 = min(h_img, bbox.y2)
    x1 = max(0, bbox.x)
    x2 = min(w_img, bbox.x2)

    crop = binary[y1:y2, x1:x2]
    h, w = crop.shape
    if h == 0 or w == 0:
        crop = np.zeros((10, 10), dtype=np.uint8)
        h, w = 10, 10

    # Frame into square with 20% margin
    s = max(h, w)
    pad = max(4, int(s * 0.20))
    sq_size = s + 2 * pad
    square = np.zeros((sq_size, sq_size), dtype=np.uint8)

    y_offset = (sq_size - h) // 2
    x_offset = (sq_size - w) // 2
    square[y_offset : y_offset + h, x_offset : x_offset + w] = crop

    # Resize to model resolution
    resized = cv2.resize(square, (resolution, resolution), interpolation=cv2.INTER_AREA)

    # Threshold at 100 → 0.0 or 1.0 float32
    bin_crop = (resized > THRESHOLD).astype(np.float32)
    tensor = torch.from_numpy(bin_crop).unsqueeze(0).unsqueeze(0)
    model_input_uint8 = (bin_crop * 255).astype(np.uint8)

    return tensor, model_input_uint8, crop


def prepare_character_tensor(
    binary: np.ndarray, bbox: BoundingBox, resolution: int
) -> torch.Tensor:
    """
    Backward-compatible helper returning solely the PyTorch tensor.
    Matches training dataset generation.
    """
    tensor, _, _ = prepare_character_tensor_with_preview(binary, bbox, resolution)
    return tensor


def classify_character(
    binary: np.ndarray,
    bbox: BoundingBox,
    models: dict,
    id_to_label: dict,
    device: torch.device,
    return_debug: bool = False,
) -> Any:
    """
    Runs the character crop through all three CNN models, applies
    majority voting, and returns:
      - (character, is_uncertain, predictions) if return_debug=False
      - (character, is_uncertain, predictions, debug_info) if return_debug=True
    Voting rules:
      - 3 agree: agreed character, is_uncertain=False
      - 2 agree: majority character, is_uncertain=False
      - all 3 differ: '[?]', is_uncertain=True
    If any model fails, raises RuntimeError (technical error).
    """
    predictions = []
    model_inputs = {}
    raw_crop_img = None

    for name, entry in models.items():
        res = entry["resolution"]
        model = entry["model"]
        tensor, model_input_uint8, crop = prepare_character_tensor_with_preview(binary, bbox, res)
        if raw_crop_img is None:
            raw_crop_img = crop

        model_inputs[str(res)] = {
            "resolution": res,
            "dimensions": f"{res}×{res}",
            "image": ndarray_to_png_data_url(model_input_uint8),
        }

        with torch.inference_mode():
            logits = model(tensor.to(device))
            pred_id = int(logits.argmax(dim=1).item())
            char = id_to_label[pred_id]
            predictions.append(char)

    if len(predictions) != 3:
        raise RuntimeError(f"Expected 3 model predictions, got {len(predictions)}")

    # Ensure all three standard model resolutions (32, 64, 128) are available for inspection
    for std_res in [32, 64, 128]:
        if str(std_res) not in model_inputs:
            _, std_uint8, _ = prepare_character_tensor_with_preview(binary, bbox, std_res)
            model_inputs[str(std_res)] = {
                "resolution": std_res,
                "dimensions": f"{std_res}×{std_res}",
                "image": ndarray_to_png_data_url(std_uint8),
            }

    counts = Counter(predictions)
    top_char, count = counts.most_common(1)[0]

    char = top_char if count >= 2 else "[?]"
    is_uncertain = count < 2

    if return_debug:
        crop_h, crop_w = (raw_crop_img.shape if raw_crop_img is not None else (bbox.h, bbox.w))
        debug_info = {
            "crop_image": ndarray_to_png_data_url(raw_crop_img) if raw_crop_img is not None else "",
            "crop_dimensions": {"width": int(crop_w), "height": int(crop_h)},
            "model_inputs": model_inputs,
        }
        return char, is_uncertain, predictions, debug_info

    return char, is_uncertain, predictions


def recognize_line(
    image_bytes: bytes,
    models: dict,
    id_to_label: dict,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Full pipeline for V2 handwritten line recognition:
      1. Decode & binarize PNG
      2. Check if empty
      3. Segment into character bounding boxes
      4. Estimate word boundaries (spaces)
      5. Classify each character with majority voting & capture debug previews
      6. Reconstruct line and return structured response
    """
    binary = decode_and_binarize(image_bytes)

    if is_canvas_empty(binary):
        return {
            "status": "empty",
            "text": "",
            "segments": [],
            "num_characters": 0,
            "num_words": 0,
            "has_uncertain": False,
            "message": "Canvas is empty. Please draw a line of text first.",
        }

    boxes = segment_characters(binary)
    if not boxes:
        return {
            "status": "no_characters",
            "text": "",
            "segments": [],
            "num_characters": 0,
            "num_words": 0,
            "has_uncertain": False,
            "message": "No characters detected. Please try drawing more clearly.",
        }

    spaces = estimate_word_boundaries(boxes)

    segments = []
    reconstructed_chars = []
    has_uncertain = False

    for idx, (box, has_space) in enumerate(zip(boxes, spaces)):
        char, is_unc, preds, debug_info = classify_character(
            binary, box, models, id_to_label, device, return_debug=True
        )
        if is_unc:
            has_uncertain = True

        reconstructed_chars.append(char)
        if has_space:
            reconstructed_chars.append(" ")

        segments.append({
            "index": idx,
            "character": char,
            "is_uncertain": is_unc,
            "bbox": {
                "x": int(box.x),
                "y": int(box.y),
                "width": int(box.w),
                "height": int(box.h),
            },
            "crop_dimensions": debug_info["crop_dimensions"],
            "has_space_after": has_space,
            "crop_image": debug_info["crop_image"],
            "model_inputs": debug_info["model_inputs"],
        })

    reconstructed_text = "".join(reconstructed_chars).strip()
    words = reconstructed_text.split()

    return {
        "status": "success",
        "text": reconstructed_text,
        "segments": segments,
        "num_characters": len(boxes),
        "num_words": len(words),
        "has_uncertain": has_uncertain,
        "message": (
            "Line recognized successfully with some uncertain characters."
            if has_uncertain
            else "Line recognized successfully."
        ),
    }
