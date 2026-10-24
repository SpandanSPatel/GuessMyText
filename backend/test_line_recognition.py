"""
GuessMyText — V2 Line Recognition Test Suite

Tests:
  1. Empty canvas detection
  2. BoundingBox union & properties
  3. Detached stroke merging (dot of 'i', '=', '÷')
  4. Touching character splitting
  5. Word-boundary / space estimation
  6. Square padding & tensor preprocessing (training compatible)
  7. Majority voting & uncertainty classification
  8. Line reconstruction
  9. Model failure handling (technical error, no partial fallback)
 10. Flask /api/predict-line endpoint (file upload & base64 JSON)
 11. End-to-end synthetic multi-word handwritten line recognition
"""

import base64
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from line_segmentation import (
    BoundingBox,
    decode_and_binarize,
    is_canvas_empty,
    extract_initial_components,
    merge_detached_strokes,
    split_touching_characters,
    segment_characters,
    estimate_word_boundaries,
    prepare_character_tensor,
    classify_character,
    recognize_line,
)
import app


class TestLineSegmentationAndRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.load_label_mapping()
        app.load_models()
        cls.client = app.app.test_client()

    def test_01_empty_canvas_detection(self):
        # Pure black canvas
        black = np.zeros((280, 1000), dtype=np.uint8)
        self.assertTrue(is_canvas_empty(black))

        # Canvas with a tiny 3-pixel noise spec
        black[50:53, 50] = 255
        self.assertTrue(is_canvas_empty(black, min_pixels=15))

        # Canvas with drawn strokes (>15 pixels)
        cv2.circle(black, (100, 100), 20, 255, thickness=4)
        self.assertFalse(is_canvas_empty(black, min_pixels=15))
        print("  [PASS] Test 1: Empty canvas detection")

    def test_02_bounding_box_logic(self):
        b1 = BoundingBox(x=10, y=20, w=30, h=40)
        b2 = BoundingBox(x=30, y=50, w=20, h=30)
        u = b1.union(b2)
        self.assertEqual(u.x, 10)
        self.assertEqual(u.y, 20)
        self.assertEqual(u.x2, 50)
        self.assertEqual(u.y2, 80)
        self.assertEqual(u.w, 40)
        self.assertEqual(u.h, 60)
        print("  [PASS] Test 2: BoundingBox union & properties")

    def test_03_detached_stroke_merging(self):
        # Simulate lowercase 'i': stem at x=100..120, y=80..150; dot at x=106..114, y=55..65
        stem = BoundingBox(x=100, y=80, w=20, h=70)
        dot = BoundingBox(x=106, y=55, w=8, h=10)

        # Another separate character 'o' at x=150..190, y=80..140
        o_char = BoundingBox(x=150, y=80, w=40, h=60)

        boxes = [stem, dot, o_char]
        merged = merge_detached_strokes(boxes)

        self.assertEqual(len(merged), 2)
        # First merged box should contain both stem and dot
        i_box = merged[0] if merged[0].x < merged[1].x else merged[1]
        self.assertEqual(i_box.x, 100)
        self.assertEqual(i_box.y, 55)
        self.assertEqual(i_box.h, 95)  # from y=55 to y=150
        print("  [PASS] Test 3: Detached stroke merging ('i' stem + dot)")

    def test_04_split_touching_characters(self):
        # Create a synthetic image of two tall strokes touching in the middle with a valley
        img = np.zeros((100, 120), dtype=np.uint8)
        # Left stroke
        img[10:90, 15:35] = 255
        # Right stroke
        img[10:90, 85:105] = 255
        # Thin bridge connecting them at bottom
        img[85:90, 35:85] = 255

        box = BoundingBox(x=10, y=10, w=100, h=80)
        # Give context of normal width ~35
        dummy_normal = BoundingBox(x=200, y=10, w=35, h=80)

        splits = split_touching_characters(img, [box, dummy_normal])
        # Box should be split into 2 candidates
        self.assertEqual(len(splits), 3)  # 2 from box + 1 dummy_normal
        print("  [PASS] Test 4: Touching character splitting via vertical projection profile")

    def test_05_word_boundary_estimation(self):
        # Simulate word 1 "HI" and word 2 "TO":
        # 'H': x=50..90 (w=40)
        # 'I': x=105..125 (w=20) -> intra-word gap = 15
        # 'T': x=200..240 (w=40) -> inter-word gap = 75 (200 - 125)
        # 'O': x=255..295 (w=40) -> intra-word gap = 15
        b_h = BoundingBox(x=50, y=50, w=40, h=80)
        b_i = BoundingBox(x=105, y=50, w=20, h=80)
        b_t = BoundingBox(x=200, y=50, w=40, h=80)
        b_o = BoundingBox(x=255, y=50, w=40, h=80)

        boxes = [b_h, b_i, b_t, b_o]
        spaces = estimate_word_boundaries(boxes)

        self.assertEqual(spaces, [False, True, False, False])
        print("  [PASS] Test 5: Word-boundary estimation (space after 'I')")

    def test_06_square_framing_and_tensor(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        # Draw a vertical stroke (w=20, h=100)
        img[50:150, 90:110] = 255
        bbox = BoundingBox(x=90, y=50, w=20, h=100)

        for res in [32, 64, 128]:
            tensor = prepare_character_tensor(img, bbox, resolution=res)
            self.assertEqual(tensor.shape, (1, 1, res, res))
            self.assertEqual(tensor.dtype, torch.float32)
            # Must be strictly binary 0.0 or 1.0
            unique_vals = set(tensor.unique().tolist())
            self.assertTrue(unique_vals.issubset({0.0, 1.0}))
            self.assertGreater(tensor.sum().item(), 0)
        print("  [PASS] Test 6: Square framing & multi-scale tensor preprocessing")

    def test_07_majority_voting_and_uncertainty(self):
        # Create synthetic character image
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, 255, thickness=8)
        bbox = BoundingBox(x=20, y=20, w=60, h=60)

        # Mock models dictionary
        def make_mock_model(predicted_char):
            m = MagicMock()
            logits = torch.zeros(1, 67)
            char_id = app.label_to_id[predicted_char]
            logits[0, char_id] = 10.0
            m.return_value = logits
            return m

        # Case A: 3 agree ('A', 'A', 'A')
        mock_models_a = {
            "M1": {"model": make_mock_model("A"), "resolution": 32},
            "M2": {"model": make_mock_model("A"), "resolution": 64},
            "M3": {"model": make_mock_model("A"), "resolution": 128},
        }
        char, is_unc, preds = classify_character(
            img, bbox, mock_models_a, app.id_to_label, torch.device("cpu")
        )
        self.assertEqual(char, "A")
        self.assertFalse(is_unc)

        # Case B: 2 agree ('A', 'A', 'B')
        mock_models_b = {
            "M1": {"model": make_mock_model("A"), "resolution": 32},
            "M2": {"model": make_mock_model("A"), "resolution": 64},
            "M3": {"model": make_mock_model("B"), "resolution": 128},
        }
        char, is_unc, preds = classify_character(
            img, bbox, mock_models_b, app.id_to_label, torch.device("cpu")
        )
        self.assertEqual(char, "A")
        self.assertFalse(is_unc)

        # Case C: all 3 differ ('A', 'B', 'C') -> uncertain
        mock_models_c = {
            "M1": {"model": make_mock_model("A"), "resolution": 32},
            "M2": {"model": make_mock_model("B"), "resolution": 64},
            "M3": {"model": make_mock_model("C"), "resolution": 128},
        }
        char, is_unc, preds = classify_character(
            img, bbox, mock_models_c, app.id_to_label, torch.device("cpu")
        )
        self.assertEqual(char, "[?]")
        self.assertTrue(is_unc)

        print("  [PASS] Test 7: Majority voting and uncertainty handling")

    def test_08_model_failure_raises_technical_error(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:80, 20:80] = 255
        bbox = BoundingBox(x=20, y=20, w=60, h=60)

        def failing_model(*args, **kwargs):
            raise RuntimeError("CUDA Out of Memory simulation")

        failing_models = {
            "M1": {"model": MagicMock(side_effect=failing_model), "resolution": 32},
            "M2": {"model": MagicMock(), "resolution": 64},
            "M3": {"model": MagicMock(), "resolution": 128},
        }

        with self.assertRaises(RuntimeError):
            classify_character(
                img, bbox, failing_models, app.id_to_label, torch.device("cpu")
            )
        print("  [PASS] Test 8: Required model failure triggers technical error (no silent fallback)")

    def test_09_api_predict_line_empty_and_valid(self):
        # 1. Empty image payload
        resp = self.client.post("/api/predict-line", json={"image": ""})
        self.assertEqual(resp.status_code, 400)

        # 2. Canvas with all black pixels
        black = np.zeros((280, 1000), dtype=np.uint8)
        _, png_buf = cv2.imencode(".png", black)
        resp = self.client.post(
            "/api/predict-line",
            data={"image": (io.BytesIO(png_buf.tobytes()), "empty.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "empty")
        self.assertEqual(data["text"], "")

        # 3. Canvas with real characters drawn
        canvas = np.zeros((280, 1000), dtype=np.uint8)
        # Draw digit '7': top bar and diagonal stroke
        cv2.line(canvas, (100, 60), (180, 60), 255, thickness=16)
        cv2.line(canvas, (180, 60), (120, 220), 255, thickness=16)
        # Draw digit '1': vertical bar
        cv2.line(canvas, (240, 60), (240, 220), 255, thickness=16)

        _, png_buf = cv2.imencode(".png", canvas)
        resp = self.client.post(
            "/api/predict-line",
            data={"image": (io.BytesIO(png_buf.tobytes()), "line.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertGreaterEqual(data["num_characters"], 1)
        self.assertIn("text", data)
        self.assertIn("segments", data)
        print("  [PASS] Test 9: Flask /api/predict-line endpoint (empty & line drawings)")

    def test_10_segmentation_preview_data(self):
        # Verify that segmented character crops and model input previews are generated
        canvas = np.zeros((280, 1000), dtype=np.uint8)
        # Draw 'A'
        cv2.line(canvas, (100, 200), (140, 60), 255, thickness=16)
        cv2.line(canvas, (140, 60), (180, 200), 255, thickness=16)
        cv2.line(canvas, (115, 140), (165, 140), 255, thickness=16)

        _, png_buf = cv2.imencode(".png", canvas)
        resp = self.client.post(
            "/api/predict-line",
            data={"image": (io.BytesIO(png_buf.tobytes()), "preview_test.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertGreaterEqual(len(data["segments"]), 1)

        first_seg = data["segments"][0]
        self.assertIn("crop_image", first_seg)
        self.assertTrue(first_seg["crop_image"].startswith("data:image/png;base64,"))
        self.assertIn("crop_dimensions", first_seg)
        self.assertGreater(first_seg["crop_dimensions"]["width"], 0)
        self.assertGreater(first_seg["crop_dimensions"]["height"], 0)

        self.assertIn("model_inputs", first_seg)
        for res_str in ["32", "64", "128"]:
            self.assertIn(res_str, first_seg["model_inputs"])
            entry = first_seg["model_inputs"][res_str]
            self.assertEqual(entry["resolution"], int(res_str))
            self.assertEqual(entry["dimensions"], f"{res_str}×{res_str}")
            self.assertTrue(entry["image"].startswith("data:image/png;base64,"))

            # Decode the base64 string to verify it is a valid image with correct dimensions
            b64_data = entry["image"].split(",", 1)[1]
            decoded_bytes = base64.b64decode(b64_data)
            nparr = np.frombuffer(decoded_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(img)
            self.assertEqual(img.shape, (int(res_str), int(res_str)))

        print("  [PASS] Test 10: Segmentation preview & model input tensor images verified")



if __name__ == "__main__":
    print("=" * 65)
    print("GuessMyText — V2 Line Recognition Pipeline Tests")
    print("=" * 65)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLineSegmentationAndRecognition)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("=" * 65)
        print("ALL TESTS PASSED ✅")
        print("=" * 65)
        sys.exit(0)
    else:
        print("=" * 65)
        print("SOME TESTS FAILED ❌")
        print("=" * 65)
        sys.exit(1)
