"""
GuessMyText — V3 Paragraph Recognition Pipeline

Provides modular components for handwritten multi-line paragraph recognition:
  1. Full-canvas image decoding, binarization, empty canvas, and size validation
  2. Vertical line detection via horizontal projection profiling and component clustering
  3. Top-to-bottom line sorting and intra-line gap filtering
  4. Per-line character segmentation (reusing verified V2 CCA, detached-stroke merging,
     and touching-character splitting)
  5. Independent per-line word boundary (whitespace) estimation
  6. 3-CNN majority voting for each character across all lines
  7. Paragraph reconstruction preserving line breaks (\n) and uncertainty markers ([?])
  8. Line-by-line visual segmentation preview generation from actual inference intermediates
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import cv2

from line_segmentation import (
    BoundingBox,
    decode_and_binarize,
    is_canvas_empty,
    extract_initial_components,
    merge_detached_strokes,
    split_touching_characters,
    estimate_word_boundaries,
    classify_character,
    ndarray_to_png_data_url,
    THRESHOLD,
)

# Maximum canvas dimension / pixel limits to prevent OOM
MAX_CANVAS_DIM = 6000
MAX_CANVAS_PIXELS = 25_000_000


@dataclass
class LineRegion:
    line_index: int
    y_start: int
    y_end: int
    boxes: List[BoundingBox]

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    @property
    def bbox(self) -> BoundingBox:
        if not self.boxes:
            return BoundingBox(x=0, y=self.y_start, w=10, h=self.height)
        min_x = min(b.x for b in self.boxes)
        max_x = max(b.x2 for b in self.boxes)
        min_y = min(b.y for b in self.boxes)
        max_y = max(b.y2 for b in self.boxes)
        return BoundingBox(x=min_x, y=min_y, w=max_x - min_x, h=max_y - min_y)


def create_annotated_canvas(
    binary: np.ndarray,
    line_regions: List[LineRegion],
    per_line_boxes: List[List[BoundingBox]],
) -> str:
    """
    Renders an annotated visualization of the full exported canvas showing
    detected line bounding boxes and character crops for debugging and preview.
    """
    h, w = binary.shape
    # Create 3-channel BGR visualization (black background, white strokes)
    annotated = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for lr, boxes in zip(line_regions, per_line_boxes):
        lr_box = lr.bbox
        # Draw line bounding box in emerald green (BGR: 80, 200, 120), thickness 2
        cv2.rectangle(
            annotated,
            (lr_box.x, lr_box.y),
            (lr_box.x2, lr_box.y2),
            (80, 200, 120),
            2,
        )
        # Line label
        label_text = f"L{lr.line_index + 1}"
        cv2.putText(
            annotated,
            label_text,
            (max(5, lr_box.x), max(18, lr_box.y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 200, 120),
            2,
            cv2.LINE_AA,
        )

        for b in boxes:
            # Draw character bounding box in cyan/sky blue (BGR: 255, 190, 0), thickness 1
            cv2.rectangle(
                annotated,
                (b.x, b.y),
                (b.x2, b.y2),
                (255, 190, 0),
                1,
            )

    # Scale visualization down if canvas is very large (max dim > 1400) to keep payload light
    max_dim = max(h, w)
    if max_dim > 1400:
        scale = 1400.0 / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        annotated = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return ndarray_to_png_data_url(annotated)


def detect_line_regions(
    binary: np.ndarray,
    components: List[BoundingBox],
    min_line_gap: int = 15,
) -> List[LineRegion]:
    """
    Detects individual text lines in the complete exported canvas:
      1. Uses horizontal projection profiling (HPP) to find ink rows and empty inter-line rows.
      2. Merges small intra-line gaps (dots of 'i'/'j', accents, or bars of '=').
      3. Splits abnormally tall merged bands (where descenders meet ascenders) using HPP valleys.
      4. Assigns components to each line band based on their vertical center.
      5. Sorts lines strictly from top to bottom.
      6. Robust against empty rows, irregular line spacing, and expanded canvases.
    """
    if not components:
        return []

    h_img, w_img = binary.shape

    # 1. Compute HPP across all rows
    hpp = np.sum(binary > 0, axis=1)

    # Find raw ink row runs where hpp > 0
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

    # Estimate typical character height from components
    valid_heights = [c.h for c in components if c.h >= 20]
    med_h = float(np.median(valid_heights)) if valid_heights else 50.0

    # 2. Merge intra-line split bands in HPP
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

        # Bridging rules for intra-line vertical gaps:
        # A gap <= 22px is intra-line
        # If one band is small (dot/bar <= 30px), bridge up to 40px
        if gap <= 22 or ((prev_h <= 30 or curr_h <= 30) and gap <= 40):
            merged_bands[-1] = (prev_start, curr_end)
        else:
            merged_bands.append(r)

    # 3. Check if any merged band is unusually tall (two lines joined by descender/ascender)
    final_bands = []
    for b_start, b_end in merged_bands:
        band_h = b_end - b_start
        if band_h > 1.85 * med_h and band_h >= 135:
            # Check smoothed HPP for valley in the middle region
            sub_hpp = hpp[b_start:b_end].astype(float)
            # Smooth using simple 1D moving average or Gaussian
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

            if min_val <= 0.45 * max_val:
                split_row = b_start + min_rel_idx
                final_bands.append((b_start, split_row))
                final_bands.append((split_row, b_end))
                continue

        final_bands.append((b_start, b_end))

    # 4. Assign components to lines based on which band their vertical center falls in
    lines_comps: List[List[BoundingBox]] = [[] for _ in final_bands]
    for c in components:
        c_mid = c.y + c.h / 2.0
        best_line = -1
        min_dist = float("inf")

        for idx, (b_start, b_end) in enumerate(final_bands):
            if b_start <= c_mid <= b_end:
                best_line = idx
                break
            if c_mid < b_start:
                d = b_start - c_mid
            else:
                d = c_mid - b_end
            if d < min_dist:
                min_dist = d
                best_line = idx

        if best_line >= 0:
            lines_comps[best_line].append(c)

    # Filter out any empty lines and sort by median vertical center
    non_empty = [grp for grp in lines_comps if grp]
    non_empty.sort(key=lambda grp: np.median([b.y + b.h / 2.0 for b in grp]))

    # Build LineRegion objects
    line_regions = []
    for idx, grp in enumerate(non_empty):
        grp.sort(key=lambda b: b.x)
        y_start = max(0, min(b.y for b in grp) - 4)
        y_end = min(h_img, max(b.y2 for b in grp) + 4)
        line_regions.append(LineRegion(line_index=idx, y_start=y_start, y_end=y_end, boxes=grp))

    return line_regions


def segment_paragraph_lines(
    binary: np.ndarray,
) -> Tuple[List[LineRegion], List[List[BoundingBox]]]:
    """
    Extracts paragraph lines and character bounding boxes per line:
      1. Extract all connected components directly from the binary canvas.
      2. Cluster into top-to-bottom line regions.
      3. For EACH detected line independently:
         a. Merge detached strokes (dots of 'i'/'j', '=', '÷', '%') ONLY within this line's boxes.
            This strictly prevents cross-line stroke bleeding.
         b. Split wide touching characters at vertical projection valleys with tight bounding boxes.
         c. Sort character boxes strictly left-to-right.
    """
    initial_boxes = extract_initial_components(binary)
    if not initial_boxes:
        return [], []

    # Detect lines from initial components WITHOUT premature cross-line stroke merging
    line_regions = detect_line_regions(binary, initial_boxes)

    per_line_boxes: List[List[BoundingBox]] = []

    for lr in line_regions:
        # Merge detached strokes ONLY within this line
        merged_line_boxes = merge_detached_strokes(lr.boxes)
        # Split touching characters within this line
        refined_line_boxes = split_touching_characters(binary, merged_line_boxes)
        # Sort left to right by x
        refined_line_boxes.sort(key=lambda b: b.x)
        per_line_boxes.append(refined_line_boxes)

    return line_regions, per_line_boxes


def recognize_paragraph(
    image_bytes: bytes,
    models: dict,
    id_to_label: dict,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Full pipeline for V3 handwritten paragraph recognition:
      1. Decode & binarize PNG
      2. Validate dimensions and empty state
      3. Detect line regions (top-to-bottom)
      4. Segment characters and estimate word boundaries per line
      5. Run 3-model CNN ensemble with majority voting
      6. Reconstruct paragraph with preserved line breaks (\n)
      7. Return structured results, annotated full canvas, and diagnostics
    """
    binary = decode_and_binarize(image_bytes)
    h, w = binary.shape

    # Dimension and payload validation
    if h > MAX_CANVAS_DIM or w > MAX_CANVAS_DIM or (h * w) > MAX_CANVAS_PIXELS:
        raise ValueError(
            f"Canvas dimensions ({w}×{h}) exceed maximum allowed size "
            f"({MAX_CANVAS_DIM}×{MAX_CANVAS_DIM}px). Please reduce canvas size."
        )

    if is_canvas_empty(binary):
        return {
            "status": "empty",
            "text": "",
            "lines": [],
            "num_lines": 0,
            "num_characters": 0,
            "num_words": 0,
            "has_uncertain": False,
            "message": "Canvas is empty. Please draw handwritten text before predicting.",
        }

    line_regions, per_line_boxes = segment_paragraph_lines(binary)

    if not line_regions or not any(per_line_boxes):
        return {
            "status": "no_characters",
            "text": "",
            "lines": [],
            "num_lines": 0,
            "num_characters": 0,
            "num_words": 0,
            "has_uncertain": False,
            "message": "No characters detected. Please try drawing more clearly.",
        }

    lines_output = []
    reconstructed_lines_text = []
    total_characters = 0
    total_words = 0
    has_any_uncertain = False

    global_char_idx = 0

    for line_idx, (lr, boxes) in enumerate(zip(line_regions, per_line_boxes)):
        if not boxes:
            continue

        spaces = estimate_word_boundaries(boxes)
        line_chars = []
        line_segments = []
        line_has_uncertain = False

        for char_in_line_idx, (box, has_space) in enumerate(zip(boxes, spaces)):
            char, is_unc, preds, debug_info = classify_character(
                binary, box, models, id_to_label, device, return_debug=True
            )
            if is_unc:
                line_has_uncertain = True
                has_any_uncertain = True

            line_chars.append(char)
            if has_space:
                line_chars.append(" ")

            line_segments.append({
                "global_index": global_char_idx,
                "line_index": line_idx,
                "char_index": char_in_line_idx,
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
            global_char_idx += 1

        line_text = "".join(line_chars).strip()
        line_word_count = len(line_text.split())

        reconstructed_lines_text.append(line_text)
        total_characters += len(boxes)
        total_words += line_word_count

        lr_bbox = lr.bbox
        lines_output.append({
            "line_index": line_idx,
            "text": line_text,
            "num_characters": len(boxes),
            "num_words": line_word_count,
            "has_uncertain": line_has_uncertain,
            "bbox": {
                "x": int(lr_bbox.x),
                "y": int(lr_bbox.y),
                "width": int(lr_bbox.w),
                "height": int(lr_bbox.h),
            },
            "segments": line_segments,
        })

    # Join lines with newlines
    full_text = "\n".join(reconstructed_lines_text)

    # Generate full canvas visualization showing line & character bounding boxes
    annotated_canvas = create_annotated_canvas(binary, line_regions, per_line_boxes)

    # Diagnostic metadata for development debugging view
    diagnostics = {
        "canvas_dimensions": {"width": w, "height": h},
        "num_lines": len(lines_output),
        "total_characters": total_characters,
        "line_regions": [
            {
                "line_index": lr.line_index,
                "y_start": int(lr.y_start),
                "y_end": int(lr.y_end),
                "height": int(lr.height),
                "num_characters": len(b_list),
                "bbox": {
                    "x": int(lr.bbox.x),
                    "y": int(lr.bbox.y),
                    "width": int(lr.bbox.w),
                    "height": int(lr.bbox.h),
                },
            }
            for lr, b_list in zip(line_regions, per_line_boxes)
        ],
        "thresholds": {
            "binarization": THRESHOLD,
            "max_canvas_dim": MAX_CANVAS_DIM,
            "max_canvas_pixels": MAX_CANVAS_PIXELS,
        },
    }

    return {
        "status": "success",
        "text": full_text,
        "lines": lines_output,
        "num_lines": len(lines_output),
        "num_characters": total_characters,
        "num_words": total_words,
        "has_uncertain": has_any_uncertain,
        "annotated_canvas": annotated_canvas,
        "diagnostics": diagnostics,
        "message": (
            f"Paragraph recognized successfully ({len(lines_output)} lines) with some uncertain characters."
            if has_any_uncertain
            else f"Paragraph recognized successfully ({len(lines_output)} lines)."
        ),
    }
