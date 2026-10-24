"""
GuessMyText — Prediction Aggregation and Voting Logic Tests

Verifies deterministic majority voting rules and Flask API responses:
  1. Three identical predictions return common character (Case A).
  2. Two matching predictions and one different return majority character (Case B).
  3. Three different predictions return no final character (Case C) with friendly redraw message.
  4. Model failure during inference triggers a technical error (500) and does not manufacture a majority.
  5. Invalid/empty payload handling.
  6. Real forward pass on loaded CNN checkpoints.
"""

import base64
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure UTF-8 output on Windows consoles
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
from app import aggregate_predictions, app as flask_app


def test_voting_three_agree():
    """Case A: All three predictions agree."""
    status, pred, msg = aggregate_predictions(["T", "T", "T"])
    assert status == "success"
    assert pred == "T"
    assert msg is None

    status, pred, msg = aggregate_predictions(["5", "5", "5"])
    assert status == "success"
    assert pred == "5"
    assert msg is None

    status, pred, msg = aggregate_predictions(["+", "+", "+"])
    assert status == "success"
    assert pred == "+"
    assert msg is None
    print("  [PASS] Case A: Three identical predictions")


def test_voting_two_agree():
    """Case B: Exactly two predictions agree (all permutations)."""
    # First two agree
    status, pred, msg = aggregate_predictions(["T", "T", "7"])
    assert status == "success"
    assert pred == "T"
    assert msg is None

    # First and third agree
    status, pred, msg = aggregate_predictions(["T", "7", "T"])
    assert status == "success"
    assert pred == "T"
    assert msg is None

    # Last two agree
    status, pred, msg = aggregate_predictions(["7", "T", "T"])
    assert status == "success"
    assert pred == "T"
    assert msg is None

    # Another character pair
    status, pred, msg = aggregate_predictions(["B", "8", "B"])
    assert status == "success"
    assert pred == "B"
    assert msg is None
    print("  [PASS] Case B: Exactly two predictions agree")


def test_voting_all_differ():
    """Case C: All three predictions differ."""
    status, pred, msg = aggregate_predictions(["T", "7", "J"])
    assert status == "no_consensus"
    assert pred is None
    assert msg == "Couldn't recognize the character. Please try drawing it again."

    status, pred, msg = aggregate_predictions(["0", "O", "o"])
    assert status == "no_consensus"
    assert pred is None
    assert msg == "Couldn't recognize the character. Please try drawing it again."
    print("  [PASS] Case C: All three predictions differ (friendly message)")


def test_voting_invalid_length():
    """Voting requires exactly 3 predictions."""
    try:
        aggregate_predictions(["T", "T"])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        aggregate_predictions(["T", "T", "T", "T"])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  [PASS] Invalid input length validation")


def test_api_case_a_three_agree():
    """Flask API: Case A returns status='success' with character."""
    client = flask_app.test_client()

    mock_results = [
        {"predicted_character": "T", "confidence": 99.0, "inference_time_ms": 1.0},
        {"predicted_character": "T", "confidence": 98.0, "inference_time_ms": 1.0},
        {"predicted_character": "T", "confidence": 97.0, "inference_time_ms": 1.0},
    ]

    with patch.object(app, "models", {"M1": {"resolution": 32}, "M2": {"resolution": 64}, "M3": {"resolution": 128}}):
        with patch.object(app, "preprocess_image", return_value=(torch.zeros(1, 1, 32, 32), "preview")):
            with patch.object(app, "run_inference", side_effect=mock_results):
                resp = client.post(
                    "/api/predict",
                    data={"image": (io.BytesIO(b"fake_png"), "test.png")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200
                data = resp.get_json()
                assert data["status"] == "success"
                assert data["prediction"] == "T"
    print("  [PASS] API Case A: 3 agree -> {status: 'success', prediction: 'T'}")


def test_api_case_b_two_agree():
    """Flask API: Case B returns majority character."""
    client = flask_app.test_client()

    mock_results = [
        {"predicted_character": "T", "confidence": 85.0, "inference_time_ms": 1.0},
        {"predicted_character": "7", "confidence": 92.0, "inference_time_ms": 1.0},
        {"predicted_character": "T", "confidence": 88.0, "inference_time_ms": 1.0},
    ]

    with patch.object(app, "models", {"M1": {"resolution": 32}, "M2": {"resolution": 64}, "M3": {"resolution": 128}}):
        with patch.object(app, "preprocess_image", return_value=(torch.zeros(1, 1, 32, 32), "preview")):
            with patch.object(app, "run_inference", side_effect=mock_results):
                resp = client.post(
                    "/api/predict",
                    data={"image": (io.BytesIO(b"fake_png"), "test.png")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200
                data = resp.get_json()
                assert data["status"] == "success"
                assert data["prediction"] == "T"
    print("  [PASS] API Case B: 2 agree -> {status: 'success', prediction: 'T'}")


def test_api_case_c_no_majority():
    """Flask API: Case C returns no_consensus with friendly redraw message."""
    client = flask_app.test_client()

    mock_results = [
        {"predicted_character": "T", "confidence": 60.0, "inference_time_ms": 1.0},
        {"predicted_character": "7", "confidence": 55.0, "inference_time_ms": 1.0},
        {"predicted_character": "J", "confidence": 70.0, "inference_time_ms": 1.0},
    ]

    with patch.object(app, "models", {"M1": {"resolution": 32}, "M2": {"resolution": 64}, "M3": {"resolution": 128}}):
        with patch.object(app, "preprocess_image", return_value=(torch.zeros(1, 1, 32, 32), "preview")):
            with patch.object(app, "run_inference", side_effect=mock_results):
                resp = client.post(
                    "/api/predict",
                    data={"image": (io.BytesIO(b"fake_png"), "test.png")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200
                data = resp.get_json()
                assert data["status"] == "no_consensus"
                assert data["prediction"] is None
                assert "Couldn't recognize the character" in data["message"]
    print("  [PASS] API Case C: all differ -> {status: 'no_consensus', prediction: null, message: '...'}")


def test_api_model_failure_triggers_technical_error():
    """Flask API: One model fails during inference -> returns HTTP 500 technical error, no silent vote."""
    client = flask_app.test_client()

    def mock_inference_fail(entry, tensor):
        if entry["resolution"] == 64:
            raise RuntimeError("CUDA device out of memory simulation")
        return {"predicted_character": "T", "confidence": 95.0, "inference_time_ms": 1.0}

    with patch.object(app, "models", {"M1": {"resolution": 32}, "M2": {"resolution": 64}, "M3": {"resolution": 128}}):
        with patch.object(app, "preprocess_image", return_value=(torch.zeros(1, 1, 32, 32), "preview")):
            with patch.object(app, "run_inference", side_effect=mock_inference_fail):
                resp = client.post(
                    "/api/predict",
                    data={"image": (io.BytesIO(b"fake_png"), "test.png")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 500
                data = resp.get_json()
                assert data["status"] == "error"
                assert "Technical inference failure" in data["error"]
    print("  [PASS] API Model failure -> 500 technical error (no silent majority manufactured)")


def test_api_empty_payload():
    """Flask API: Empty payload -> returns HTTP 400 error."""
    client = flask_app.test_client()
    resp = client.post("/api/predict", data={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    print("  [PASS] API Empty payload -> 400 error")


def test_real_model_inference_pipeline():
    """Tests real model loading and pipeline with a synthetic drawing."""
    # Ensure models are loaded
    if not app.models:
        app.load_label_mapping()
        app.load_models()

    assert len(app.models) == 3

    # Generate a simple synthetic image with a white circle
    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(img, (200, 200), 100, 255, 20)
    _, buf = cv2.imencode(".png", img)
    image_bytes = buf.tobytes()

    client = flask_app.test_client()
    resp = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(image_bytes), "circle.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] in ("success", "no_consensus")
    if data["status"] == "success":
        assert isinstance(data["prediction"], str)
        print(f"  [PASS] Real CNN Pipeline -> consensus prediction: '{data['prediction']}'")
    else:
        assert data["prediction"] is None
        print(f"  [PASS] Real CNN Pipeline -> no consensus: '{data['message']}'")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("GuessMyText — Prediction Logic & API Verification")
    print("=" * 65)

    test_voting_three_agree()
    test_voting_two_agree()
    test_voting_all_differ()
    test_voting_invalid_length()
    test_api_case_a_three_agree()
    test_api_case_b_two_agree()
    test_api_case_c_no_majority()
    test_api_model_failure_triggers_technical_error()
    test_api_empty_payload()
    test_real_model_inference_pipeline()

    print("=" * 65)
    print("ALL TESTS PASSED ✅")
    print("=" * 65 + "\n")
