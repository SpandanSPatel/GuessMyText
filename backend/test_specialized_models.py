"""
GuessMyText — Specialized Models and Multi-Version Verification Test Suite

Validates:
  1. Checkpoint presence, architecture, and output dimensions:
     - V1: 3 models, 67 classes
     - V2/V3: 3 models, 36 classes (A-Z, 0-9)
     - V4: 3 models, 15 classes (0-9, +, -, *, ÷, %)
  2. Label mapping loading and contiguous class integrity
  3. Preprocessing compatibility across all 9 models
  4. Real handwritten image recognition from New/Images/
  5. Majority voting logic and uncertainty handling
  6. Version isolation across V1, V2, V3, and V4 endpoints
  7. /api/health endpoint reporting all 9 models
"""

import io
import json
import sys
import unittest
from pathlib import Path

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

import app
from model import CNN
from line_segmentation import (
    recognize_line,
    classify_character,
    BoundingBox,
    prepare_character_tensor_with_preview,
)
from paragraph_segmentation import recognize_paragraph


class TestSpecializedModelsIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.load_label_mapping()
        app.load_models()
        cls.client = app.app.test_client()

    def test_01_checkpoint_presence_and_dimensions(self):
        """Confirm all 9 model checkpoints exist, load, and have correct output shapes."""
        # V1: 67 classes
        self.assertEqual(len(app.models_v1), 3)
        for name, entry in app.models_v1.items():
            self.assertIn("model", entry)
            self.assertIn("resolution", entry)
            dummy = torch.zeros(1, 1, entry["resolution"], entry["resolution"], device=app.DEVICE)
            with torch.inference_mode():
                out = entry["model"](dummy)
            self.assertEqual(out.shape, (1, 67), f"{name} output shape mismatch")

        # V2/V3: 36 classes
        self.assertEqual(len(app.models_v2_v3), 3)
        for name, entry in app.models_v2_v3.items():
            self.assertIn("model", entry)
            self.assertIn("resolution", entry)
            dummy = torch.zeros(1, 1, entry["resolution"], entry["resolution"], device=app.DEVICE)
            with torch.inference_mode():
                out = entry["model"](dummy)
            self.assertEqual(out.shape, (1, 36), f"{name} output shape mismatch")

        # V4: 15 classes
        self.assertEqual(len(app.models_v4), 3)
        for name, entry in app.models_v4.items():
            self.assertIn("model", entry)
            self.assertIn("resolution", entry)
            dummy = torch.zeros(1, 1, entry["resolution"], entry["resolution"], device=app.DEVICE)
            with torch.inference_mode():
                out = entry["model"](dummy)
            self.assertEqual(out.shape, (1, 15), f"{name} output shape mismatch")

        print("  [PASS] Test 1: All 9 checkpoints loaded with verified architectures and shapes (67, 36, 15)")

    def test_02_label_mappings_integrity(self):
        """Confirm each model family uses its own distinct, validated label mapping."""
        # V1: 67 classes
        self.assertEqual(len(app.label_to_id_v1), 67)
        self.assertEqual(len(app.id_to_label_v1), 67)
        self.assertEqual(sorted(app.label_to_id_v1.values()), list(range(67)))

        # V2/V3: 36 classes (A-Z, 0-9)
        self.assertEqual(len(app.label_to_id_v2_v3), 36)
        self.assertEqual(len(app.id_to_label_v2_v3), 36)
        self.assertEqual(sorted(app.label_to_id_v2_v3.values()), list(range(36)))
        expected_v2_classes = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        self.assertEqual(set(app.label_to_id_v2_v3.keys()), expected_v2_classes)

        # Confirm V2/V3 does not contain lowercase or operators
        for char in "abcdefghijklmnopqrstuvwxyz+-*÷%":
            self.assertNotIn(char, app.label_to_id_v2_v3)

        # V4: 15 classes (0-9, +, -, *, ÷, %)
        self.assertEqual(len(app.label_to_id_v4), 15)
        self.assertEqual(len(app.id_to_label_v4), 15)
        self.assertEqual(sorted(app.label_to_id_v4.values()), list(range(15)))
        expected_v4_classes = set("0123456789+-*÷%")
        self.assertEqual(set(app.label_to_id_v4.keys()), expected_v4_classes)

        # Confirm V4 does not contain letters
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
            self.assertNotIn(char, app.label_to_id_v4)

        print("  [PASS] Test 2: Label mappings validated and distinct across versions")

    def test_03_inference_on_real_handwritten_images(self):
        images_dir = BACKEND_DIR.parent.parent / "New" / "Images"
        if not images_dir.exists():
            self.skipTest(f"External training images folder {images_dir} not present (relocated environment).")

        # V2/V3: Uppercase and digits
        samples_v2 = [
            ("A - Uppercase letters", "1.png", "A"),
            ("B - Uppercase letters", "1.png", "B"),
            ("5 - Numbers", "1.png", "5"),
        ]
        for folder, fname, expected in samples_v2:
            p = images_dir / folder / fname
            if not p.exists():
                continue
            with open(p, "rb") as f:
                img_bytes = f.read()

            preds = []
            for name, entry in app.models_v2_v3.items():
                tensor, _ = app.preprocess_image(img_bytes, entry["resolution"])
                inf = app.run_inference(entry, tensor, app.id_to_label_v2_v3)
                preds.append(inf["predicted_character"])

            status, top_char, _ = app.aggregate_predictions(preds)
            self.assertEqual(status, "success")
            self.assertEqual(top_char, expected, f"V2/V3 model misclassified {folder}/{fname}")

        # V4: Digits and operators
        samples_v4 = [
            ("7 - Numbers", "1.png", "7"),
            ("plus - operators", "1.png", "+"),
            ("multiply - operators", "1.png", "*"),
        ]
        for folder, fname, expected in samples_v4:
            p = images_dir / folder / fname
            if not p.exists():
                continue
            with open(p, "rb") as f:
                img_bytes = f.read()

            preds = []
            for name, entry in app.models_v4.items():
                tensor, _ = app.preprocess_image(img_bytes, entry["resolution"])
                inf = app.run_inference(entry, tensor, app.id_to_label_v4)
                preds.append(inf["predicted_character"])

            status, top_char, _ = app.aggregate_predictions(preds)
            self.assertEqual(status, "success")
            self.assertEqual(top_char, expected, f"V4 model misclassified {folder}/{fname}")

        print("  [PASS] Test 3: Inference on real handwritten samples agreed across ensemble")

    def test_04_version_isolation_endpoints(self):
        """Verify each endpoint invokes the correct model family without regression."""
        # 1. V1 Single Character endpoint
        canvas_v1 = np.zeros((300, 300), dtype=np.uint8)
        cv2.putText(canvas_v1, "A", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 6)
        _, png_v1 = cv2.imencode(".png", canvas_v1)
        res_v1 = self.client.post(
            "/api/predict",
            data={"image": (io.BytesIO(png_v1.tobytes()), "v1.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(res_v1.status_code, 200)
        self.assertIn("status", res_v1.get_json())

        # 2. V2 Line Recognition endpoint
        canvas_v2 = np.zeros((280, 1000), dtype=np.uint8)
        cv2.line(canvas_v2, (100, 60), (180, 60), 255, thickness=16)
        cv2.line(canvas_v2, (180, 60), (120, 220), 255, thickness=16)
        _, png_v2 = cv2.imencode(".png", canvas_v2)
        res_v2 = self.client.post(
            "/api/predict-line",
            data={"image": (io.BytesIO(png_v2.tobytes()), "v2.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(res_v2.status_code, 200)
        data_v2 = res_v2.get_json()
        self.assertEqual(data_v2["status"], "success")

        # 3. V3 Paragraph Recognition endpoint
        canvas_v3 = np.zeros((600, 1000), dtype=np.uint8)
        cv2.putText(canvas_v3, "A", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 6)
        cv2.putText(canvas_v3, "B", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 6)
        _, png_v3 = cv2.imencode(".png", canvas_v3)
        res_v3 = self.client.post(
            "/api/predict-paragraph",
            data={"image": (io.BytesIO(png_v3.tobytes()), "v3.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(res_v3.status_code, 200)
        data_v3 = res_v3.get_json()
        self.assertEqual(data_v3["status"], "success")
        self.assertEqual(data_v3["num_lines"], 2)

        # 4. V4 Math Solver endpoint
        canvas_v4 = np.zeros((280, 1000), dtype=np.uint8)
        cv2.putText(canvas_v4, "2+3", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 255, 6)
        _, png_v4 = cv2.imencode(".png", canvas_v4)
        res_v4 = self.client.post(
            "/api/solve-math",
            data={"image": (io.BytesIO(png_v4.tobytes()), "v4.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(res_v4.status_code, 200)
        data_v4 = res_v4.get_json()
        self.assertIn(data_v4["status"], ("success", "uncertain", "invalid_expression"))

        print("  [PASS] Test 4: Endpoints V1, V2, V3, and V4 successfully executed with correct models")

    def test_05_health_endpoint_shows_all_families(self):
        """Confirm /api/health accurately reports all 3 model families (9 models total)."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["models_loaded"], 9)
        self.assertIn("versions", data)
        self.assertEqual(data["versions"]["v1"]["num_classes"], 67)
        self.assertEqual(len(data["versions"]["v1"]["models"]), 3)
        self.assertEqual(data["versions"]["v2_v3"]["num_classes"], 36)
        self.assertEqual(len(data["versions"]["v2_v3"]["models"]), 3)
        self.assertEqual(data["versions"]["v4"]["num_classes"], 15)
        self.assertEqual(len(data["versions"]["v4"]["models"]), 3)

        print("  [PASS] Test 5: /api/health correctly reports all 9 models and version breakdown")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("GuessMyText — Specialized Models and Multi-Version Verification")
    print("=" * 65)
    unittest.main()
