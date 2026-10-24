"""
Unit and integration tests for V3 Paragraph Recognition and V1/V2 regression.
Tests:
  1. Empty canvas detection
  2. Multi-line region detection and top-to-bottom sorting
  3. Paragraph recognition with preserved newlines (\n)
  4. Per-line word boundary estimation with space detection
  5. Maximum dimension / oversized image validation
  6. Model failure / technical error behavior
  7. Preview intermediate generation
  8. Flask endpoint POST /api/predict-paragraph via test client
  9. Regression test: V1 /api/predict and V2 /api/predict-line remain intact
"""

import io
import unittest
import numpy as np
import cv2
import torch

import app
from paragraph_segmentation import (
    recognize_paragraph,
    detect_line_regions,
    segment_paragraph_lines,
    MAX_CANVAS_DIM,
)
from line_segmentation import BoundingBox


def render_synthetic_char(binary: np.ndarray, x: int, y: int, char: str):
    """Draws a clean white character onto the black canvas using OpenCV Hershey fonts."""
    cv2.putText(
        binary,
        char,
        (x, y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        255,
        thickness=4,
        lineType=cv2.LINE_AA,
    )


class TestParagraphRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.load_label_mapping()
        app.load_models()
        cls.client = app.app.test_client()

    def test_empty_canvas(self):
        blank = np.zeros((600, 1200), dtype=np.uint8)
        _, png_bytes = cv2.imencode(".png", blank)
        result = recognize_paragraph(png_bytes.tobytes(), app.models, app.id_to_label, app.DEVICE)
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["num_lines"], 0)
        self.assertEqual(result["text"], "")

    def test_line_detection_top_to_bottom(self):
        # Create 2 distinct lines of text
        canvas = np.zeros((800, 1200), dtype=np.uint8)
        # Line 1 at y = 100
        render_synthetic_char(canvas, 100, 100, "A")
        render_synthetic_char(canvas, 180, 100, "B")

        # Line 2 at y = 400
        render_synthetic_char(canvas, 100, 400, "C")
        render_synthetic_char(canvas, 180, 400, "D")

        line_regions, per_line_boxes = segment_paragraph_lines(canvas)
        self.assertEqual(len(line_regions), 2)
        self.assertLess(line_regions[0].y_start, line_regions[1].y_start)
        self.assertEqual(len(per_line_boxes[0]), 2)
        self.assertEqual(len(per_line_boxes[1]), 2)

    def test_multiline_recognition_with_newlines(self):
        canvas = np.zeros((800, 1200), dtype=np.uint8)
        # Line 1
        render_synthetic_char(canvas, 100, 120, "7")
        render_synthetic_char(canvas, 180, 120, "+")
        render_synthetic_char(canvas, 260, 120, "1")

        # Line 2
        render_synthetic_char(canvas, 100, 450, "A")
        render_synthetic_char(canvas, 180, 450, "B")

        _, png_bytes = cv2.imencode(".png", canvas)
        result = recognize_paragraph(png_bytes.tobytes(), app.models, app.id_to_label, app.DEVICE)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["num_lines"], 2)
        self.assertIn("\n", result["text"])
        lines = result["text"].split("\n")
        self.assertEqual(len(lines), 2)
        self.assertGreater(result["num_characters"], 0)

    def test_word_spaces_within_lines(self):
        canvas = np.zeros((600, 1500), dtype=np.uint8)
        # Word 1: '1' at x=100, '2' at x=160
        render_synthetic_char(canvas, 100, 150, "1")
        render_synthetic_char(canvas, 160, 150, "2")
        # Word 2 (large gap): '3' at x=400, '4' at x=460
        render_synthetic_char(canvas, 400, 150, "3")
        render_synthetic_char(canvas, 460, 150, "4")

        _, png_bytes = cv2.imencode(".png", canvas)
        result = recognize_paragraph(png_bytes.tobytes(), app.models, app.id_to_label, app.DEVICE)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["num_lines"], 1)
        self.assertIn(" ", result["text"])
        self.assertEqual(result["num_words"], 2)

    def test_oversized_dimension_validation(self):
        # Image exceeding max dimension
        oversized = np.zeros((MAX_CANVAS_DIM + 100, 500), dtype=np.uint8)
        _, png_bytes = cv2.imencode(".png", oversized)
        with self.assertRaises(ValueError):
            recognize_paragraph(png_bytes.tobytes(), app.models, app.id_to_label, app.DEVICE)

    def test_preview_intermediates_structure(self):
        canvas = np.zeros((600, 1200), dtype=np.uint8)
        render_synthetic_char(canvas, 150, 150, "7")
        render_synthetic_char(canvas, 150, 350, "8")

        _, png_bytes = cv2.imencode(".png", canvas)
        result = recognize_paragraph(png_bytes.tobytes(), app.models, app.id_to_label, app.DEVICE)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["lines"]), 2)
        first_seg = result["lines"][0]["segments"][0]

        # Verify preview fields
        self.assertIn("crop_image", first_seg)
        self.assertTrue(first_seg["crop_image"].startswith("data:image/png;base64,"))
        self.assertIn("model_inputs", first_seg)
        self.assertIn("32", first_seg["model_inputs"])
        self.assertIn("64", first_seg["model_inputs"])
        self.assertIn("128", first_seg["model_inputs"])

    def test_flask_endpoint_predict_paragraph(self):
        canvas = np.zeros((600, 1200), dtype=np.uint8)
        render_synthetic_char(canvas, 150, 150, "A")
        render_synthetic_char(canvas, 150, 350, "B")
        _, png_bytes = cv2.imencode(".png", canvas)

        data = {
            "image": (io.BytesIO(png_bytes.tobytes()), "paragraph.png")
        }
        res = self.client.post(
            "/api/predict-paragraph",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        res_json = res.get_json()
        self.assertEqual(res_json["status"], "success")
        self.assertEqual(res_json["num_lines"], 2)

    def test_single_handwritten_line(self):
        canvas = np.zeros((400, 1200), dtype=np.uint8)
        render_synthetic_char(canvas, 100, 150, "H")
        render_synthetic_char(canvas, 180, 150, "E")
        render_synthetic_char(canvas, 260, 150, "L")
        render_synthetic_char(canvas, 340, 150, "L")
        render_synthetic_char(canvas, 420, 150, "O")

        _, png_bytes = cv2.imencode(".png", canvas)
        result = recognize_paragraph(png_bytes.tobytes(), app.models, app.id_to_label, app.DEVICE)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["num_lines"], 1)
        self.assertEqual(len(result["lines"][0]["segments"]), 5)

    def test_multiline_varying_spacing(self):
        canvas = np.zeros((1000, 1200), dtype=np.uint8)
        # Line 1 at y = 100
        render_synthetic_char(canvas, 100, 100, "A")
        # Line 2 at y = 240 (small gap ~80px)
        render_synthetic_char(canvas, 100, 240, "B")
        # Line 3 at y = 600 (large gap ~300px)
        render_synthetic_char(canvas, 100, 600, "C")

        line_regions, per_line_boxes = segment_paragraph_lines(canvas)
        self.assertEqual(len(line_regions), 3)
        self.assertLess(line_regions[0].y_start, line_regions[1].y_start)
        self.assertLess(line_regions[1].y_start, line_regions[2].y_start)

    def test_wide_and_narrow_characters(self):
        canvas = np.zeros((500, 1200), dtype=np.uint8)
        # Draw wide 'W' and narrow '1'
        render_synthetic_char(canvas, 100, 150, "W")
        render_synthetic_char(canvas, 220, 150, "1")
        render_synthetic_char(canvas, 300, 150, "M")

        line_regions, per_line_boxes = segment_paragraph_lines(canvas)
        self.assertEqual(len(line_regions), 1)
        # Verify 3 distinct character boxes (W not split, 1 not dropped)
        self.assertEqual(len(per_line_boxes[0]), 3)

    def test_disconnected_strokes_intra_line(self):
        canvas = np.zeros((700, 1000), dtype=np.uint8)
        # Line 1: 'i' (stem + dot)
        cv2.line(canvas, (100, 140), (100, 190), 255, 10)
        cv2.circle(canvas, (100, 115), 5, 255, 10)
        # Line 2: '=' (two bars directly below 'i')
        cv2.line(canvas, (90, 270), (130, 270), 255, 10)
        cv2.line(canvas, (90, 290), (130, 290), 255, 10)

        line_regions, per_line_boxes = segment_paragraph_lines(canvas)
        self.assertEqual(len(line_regions), 2)
        # Line 1 should have 1 merged 'i', Line 2 should have 1 merged '='
        self.assertEqual(len(per_line_boxes[0]), 1)
        self.assertEqual(len(per_line_boxes[1]), 1)
        # Check that 'i' dot was merged into 'i' stem
        self.assertLess(per_line_boxes[0][0].y, 120)
        self.assertGreater(per_line_boxes[0][0].y2, 185)

    def test_overlapping_side_by_side_characters_do_not_merge(self):
        # 'T' and 'o' where 'o' is partially tucked under T's horizontal bar
        box_T = BoundingBox(x=100, y=100, w=50, h=80)
        box_o = BoundingBox(x=135, y=130, w=35, h=50)

        from line_segmentation import merge_detached_strokes
        merged = merge_detached_strokes([box_T, box_o])
        self.assertEqual(len(merged), 2)

    def test_expanded_canvas_outside_initial_viewport(self):
        # Canvas size: 2400x1800 (expanded beyond initial 1200x600)
        canvas = np.zeros((1800, 2400), dtype=np.uint8)
        # Writing in expanded bottom-right region (x=1600, y=1400)
        render_synthetic_char(canvas, 1600, 1400, "X")
        render_synthetic_char(canvas, 1700, 1400, "Y")

        _, png_bytes = cv2.imencode(".png", canvas)
        result = recognize_paragraph(png_bytes.tobytes(), app.models, app.id_to_label, app.DEVICE)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["num_lines"], 1)
        self.assertGreater(result["lines"][0]["bbox"]["x"], 1500)
        self.assertGreater(result["lines"][0]["bbox"]["y"], 1300)

    def test_annotated_canvas_and_diagnostics_returned(self):
        canvas = np.zeros((600, 1200), dtype=np.uint8)
        render_synthetic_char(canvas, 150, 150, "A")
        render_synthetic_char(canvas, 150, 350, "B")
        _, png_bytes = cv2.imencode(".png", canvas)

        result = recognize_paragraph(png_bytes.tobytes(), app.models, app.id_to_label, app.DEVICE)
        self.assertIn("annotated_canvas", result)
        self.assertTrue(result["annotated_canvas"].startswith("data:image/png;base64,"))
        self.assertIn("diagnostics", result)
        self.assertEqual(result["diagnostics"]["num_lines"], 2)
        self.assertIn("thresholds", result["diagnostics"])

    def test_model_failure_technical_error(self):
        # Simulate model missing
        orig_models = app.models
        try:
            first_key = list(orig_models.keys())[0]
            app.models = {first_key: orig_models[first_key]}  # Only 1 model instead of 3
            canvas = np.zeros((300, 300), dtype=np.uint8)
            render_synthetic_char(canvas, 50, 50, "A")
            _, png_bytes = cv2.imencode(".png", canvas)

            data = {
                "image": (io.BytesIO(png_bytes.tobytes()), "paragraph.png")
            }
            res = self.client.post(
                "/api/predict-paragraph",
                data=data,
                content_type="multipart/form-data"
            )
            self.assertEqual(res.status_code, 500)
            res_json = res.get_json()
            self.assertEqual(res_json["status"], "error")
            self.assertIn("expected 3 loaded models", res_json["error"])
        finally:
            app.models = orig_models

    def test_regression_v1_and_v2(self):
        # Verify V1 endpoint
        canvas = np.zeros((200, 200), dtype=np.uint8)
        render_synthetic_char(canvas, 50, 50, "7")
        _, png_bytes = cv2.imencode(".png", canvas)

        data = {
            "image": (io.BytesIO(png_bytes.tobytes()), "char.png")
        }
        res_v1 = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res_v1.status_code, 200)
        v1_json = res_v1.get_json()
        self.assertIn("status", v1_json)
        self.assertIn("prediction", v1_json)

        # Verify V2 endpoint
        canvas_v2 = np.zeros((300, 1200), dtype=np.uint8)
        render_synthetic_char(canvas_v2, 100, 50, "7")
        render_synthetic_char(canvas_v2, 200, 50, "8")
        _, png_bytes_v2 = cv2.imencode(".png", canvas_v2)

        data_v2 = {
            "image": (io.BytesIO(png_bytes_v2.tobytes()), "line.png")
        }
        res_v2 = self.client.post("/api/predict-line", data=data_v2, content_type="multipart/form-data")
        self.assertEqual(res_v2.status_code, 200)
        v2_json = res_v2.get_json()
        self.assertEqual(v2_json["status"], "success")
        self.assertIn("segments", v2_json)


if __name__ == "__main__":
    unittest.main()
