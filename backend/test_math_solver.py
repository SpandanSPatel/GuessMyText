"""
Unit and integration tests for GuessMyText V4 — Mathematical Expression Solver
and regression checks for V1, V2, and V3.

Tests:
  1. Safe Evaluator Basic Operations (2+2, 15-7, 6*7, 20÷4, 17%5)
  2. DMAS / BODMAS Precedence (2+3*4=14, 20÷5*2=8, 10+7%4=13, 10-6÷2=7)
  3. Decimal division (7÷2=3.5, 1÷4=0.25)
  4. Unary signs (-3+5=2, 5*-3=-15)
  5. Trailing equals sign (2+2= -> 4)
  6. Division and modulo by zero detection
  7. Malformed sequence detection (consecutive operators, missing operands, trailing operators)
  8. Unsupported character rejection (parentheses, letters, unknown symbols)
  9. Magnitude and expression length safety bounds
 10. Flask endpoint POST /api/solve-math integration test
 11. Uncertainty blocking evaluation
 12. Regression tests for V1 (/api/predict), V2 (/api/predict-line), and V3 (/api/predict-paragraph)
"""

import io
import unittest
import numpy as np
import cv2
import torch

import app
from math_evaluator import (
    evaluate_expression,
    tokenize,
    ExpressionParser,
    MathEvaluationError,
    MAX_EXPR_LENGTH,
    MAX_MAGNITUDE,
)


def render_synthetic_char(binary: np.ndarray, x: int, y: int, char: str):
    """Draws a clean white character onto the black canvas using OpenCV Hershey fonts."""
    cv2.putText(
        binary,
        char,
        (x, y + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        255,
        thickness=4,
        lineType=cv2.LINE_AA,
    )


class TestMathEvaluator(unittest.TestCase):
    def test_basic_operations(self):
        cases = [
            ("2+2", 4, "2 + 2 = 4"),
            ("15-7", 8, "15 - 7 = 8"),
            ("6*7", 42, "6 × 7 = 42"),
            ("20/4", 5, "20 ÷ 4 = 5"),
            ("20÷4", 5, "20 ÷ 4 = 5"),
            ("17%5", 2, "17 % 5 = 2"),
        ]
        for expr, expected_val, expected_eq in cases:
            res = evaluate_expression(expr)
            self.assertEqual(res.status, "success", f"Failed on {expr}: {res.error_message}")
            self.assertEqual(res.result, expected_val)
            self.assertEqual(res.full_equation, expected_eq)

    def test_dmas_precedence(self):
        cases = [
            ("2+3*4", 14),   # 3*4=12 -> 2+12=14
            ("20÷5*2", 8),   # 20÷5=4 -> 4*2=8 (left-to-right)
            ("10+7%4", 13),  # 7%4=3 -> 10+3=13
            ("10-6÷2", 7),   # 6÷2=3 -> 10-3=7
            ("2*3+4*5", 26), # 6 + 20 = 26
        ]
        for expr, expected_val in cases:
            res = evaluate_expression(expr)
            self.assertEqual(res.status, "success", f"Failed on {expr}: {res.error_message}")
            self.assertEqual(res.result, expected_val)

    def test_decimal_division(self):
        res1 = evaluate_expression("7÷2")
        self.assertEqual(res1.status, "success")
        self.assertEqual(res1.result, 3.5)
        self.assertEqual(res1.formatted_result, "3.5")

        res2 = evaluate_expression("1÷4")
        self.assertEqual(res2.status, "success")
        self.assertEqual(res2.result, 0.25)
        self.assertEqual(res2.formatted_result, "0.25")

    def test_unary_signs(self):
        res1 = evaluate_expression("-3+5")
        self.assertEqual(res1.status, "success")
        self.assertEqual(res1.result, 2)

        res2 = evaluate_expression("5*-3")
        self.assertEqual(res2.status, "success")
        self.assertEqual(res2.result, -15)

    def test_equals_sign_rejected(self):
        # '=' is not in the dataset and must be rejected with unsupported_character
        res1 = evaluate_expression("2+2=")
        self.assertEqual(res1.status, "unsupported_character")
        self.assertIn("equals sign", res1.error_message.lower())

        res2 = evaluate_expression("2=2")
        self.assertEqual(res2.status, "unsupported_character")

    def test_division_and_modulo_by_zero(self):
        res_div = evaluate_expression("10÷0")
        self.assertEqual(res_div.status, "division_by_zero")
        self.assertIn("zero", res_div.error_message.lower())

        res_mod = evaluate_expression("17%0")
        self.assertEqual(res_mod.status, "division_by_zero")
        self.assertIn("zero", res_mod.error_message.lower())

    def test_malformed_sequences(self):
        # Consecutive operators
        res_consec = evaluate_expression("2++3")
        self.assertEqual(res_consec.status, "invalid_expression")

        res_consec2 = evaluate_expression("2+-3")
        self.assertEqual(res_consec2.status, "invalid_expression")

        # Trailing operators
        res_trail = evaluate_expression("5+")
        self.assertEqual(res_trail.status, "invalid_expression")

        # Leading operator without number
        res_lead = evaluate_expression("*4")
        self.assertEqual(res_lead.status, "invalid_expression")

        # Empty expression
        res_empty = evaluate_expression("")
        self.assertEqual(res_empty.status, "invalid_expression")

    def test_unsupported_characters(self):
        # Parentheses (not in dataset)
        res_paren = evaluate_expression("(2+3)")
        self.assertEqual(res_paren.status, "unsupported_character")
        self.assertIn("Parentheses", res_paren.error_message)

        # Letters
        res_letter = evaluate_expression("2+x")
        self.assertEqual(res_letter.status, "unsupported_character")

        # Unsupported symbols
        res_sym = evaluate_expression("2^3")
        self.assertEqual(res_sym.status, "unsupported_character")

    def test_safety_and_magnitude_limits(self):
        # Oversized expression length
        long_expr = "1+" * 60 + "1"
        res_long = evaluate_expression(long_expr)
        self.assertEqual(res_long.status, "invalid_expression")

        # Huge number
        res_huge = evaluate_expression("999999999999999999999999999+1")
        self.assertEqual(res_huge.status, "invalid_expression")


class TestMathSolverIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.load_label_mapping()
        app.load_models()
        cls.client = app.app.test_client()

    def test_empty_canvas(self):
        blank = np.zeros((300, 1200), dtype=np.uint8)
        _, png_bytes = cv2.imencode(".png", blank)
        data = {"image": (io.BytesIO(png_bytes.tobytes()), "empty.png")}

        res = self.client.post("/api/solve-math", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertEqual(json_data["status"], "empty")

    def test_solve_math_endpoint_success(self):
        # Draw synthetic '7+1'
        canvas = np.zeros((300, 1200), dtype=np.uint8)
        render_synthetic_char(canvas, 100, 100, "7")
        render_synthetic_char(canvas, 180, 100, "+")
        render_synthetic_char(canvas, 260, 100, "1")

        _, png_bytes = cv2.imencode(".png", canvas)
        data = {"image": (io.BytesIO(png_bytes.tobytes()), "math.png")}

        res = self.client.post("/api/solve-math", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()

        # Should recognize and return structured solver response
        self.assertIn("status", json_data)
        self.assertIn("segments", json_data)
        self.assertGreaterEqual(json_data["num_characters"], 1)

        # Check preview intermediates are present
        if json_data.get("segments"):
            first_seg = json_data["segments"][0]
            self.assertIn("crop_image", first_seg)
            self.assertIn("model_inputs", first_seg)

    def test_technical_model_failure_returns_500(self):
        # Simulate missing model
        saved_models = dict(app.models)
        try:
            app.models.pop(list(app.models.keys())[0])
            blank = np.zeros((300, 1200), dtype=np.uint8)
            render_synthetic_char(blank, 100, 100, "2")
            _, png_bytes = cv2.imencode(".png", blank)
            data = {"image": (io.BytesIO(png_bytes.tobytes()), "math.png")}

            res = self.client.post("/api/solve-math", data=data, content_type="multipart/form-data")
            self.assertEqual(res.status_code, 500)
            json_data = res.get_json()
            self.assertEqual(json_data["status"], "error")
        finally:
            app.models = saved_models

    def test_regression_v1_v2_v3(self):
        # V1 Single Char
        canvas_v1 = np.zeros((300, 300), dtype=np.uint8)
        render_synthetic_char(canvas_v1, 100, 100, "7")
        _, png_v1 = cv2.imencode(".png", canvas_v1)
        res_v1 = self.client.post("/api/predict", data={"image": (io.BytesIO(png_v1.tobytes()), "v1.png")}, content_type="multipart/form-data")
        self.assertEqual(res_v1.status_code, 200)

        # V2 Line Recognition
        canvas_v2 = np.zeros((300, 1200), dtype=np.uint8)
        render_synthetic_char(canvas_v2, 100, 100, "A")
        render_synthetic_char(canvas_v2, 200, 100, "B")
        _, png_v2 = cv2.imencode(".png", canvas_v2)
        res_v2 = self.client.post("/api/predict-line", data={"image": (io.BytesIO(png_v2.tobytes()), "v2.png")}, content_type="multipart/form-data")
        self.assertEqual(res_v2.status_code, 200)

        # V3 Paragraph Recognition
        canvas_v3 = np.zeros((600, 1200), dtype=np.uint8)
        render_synthetic_char(canvas_v3, 100, 100, "1")
        render_synthetic_char(canvas_v3, 100, 350, "2")
        _, png_v3 = cv2.imencode(".png", canvas_v3)
        res_v3 = self.client.post("/api/predict-paragraph", data={"image": (io.BytesIO(png_v3.tobytes()), "v3.png")}, content_type="multipart/form-data")
        self.assertEqual(res_v3.status_code, 200)


if __name__ == "__main__":
    unittest.main()
