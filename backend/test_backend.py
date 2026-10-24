"""
GuessMyText — Backend Verification Script

Tests model loading, class mapping, preprocessing, and inference
without starting the Flask server.
"""

import json
import sys
import time
from pathlib import Path

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

# Add backend dir to path
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from model import CNN

MODELS_DIR = BACKEND_DIR / "models" / "CNN" / "original"
LABEL_MAPPING_PATH = MODELS_DIR / "label_mapping.json"

THRESHOLD = 100  # matches generate_dataset.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_label_mapping():
    """Test 1: Verify label_mapping.json."""
    print("=" * 60)
    print("TEST 1: Label Mapping")
    print("=" * 60)

    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        label_to_id = {k: int(v) for k, v in json.load(f).items()}

    assert len(label_to_id) == 67, f"Expected 67 classes, got {len(label_to_id)}"
    assert sorted(label_to_id.values()) == list(range(67)), "Class IDs not contiguous 0-66"

    id_to_label = {v: k for k, v in label_to_id.items()}

    # Print first and last few
    print(f"  Total classes: {len(label_to_id)}")
    print(f"  Class 0: '{id_to_label[0]}' (%)")
    print(f"  Class 66: '{id_to_label[66]}' (÷)")
    print(f"  Digits: {[id_to_label[i] for i in range(4, 14)]}")
    print(f"  First uppercase: '{id_to_label[14]}' (A)")
    print(f"  First lowercase: '{id_to_label[40]}' (a)")
    print("  ✅ PASSED\n")

    return label_to_id, id_to_label


def test_model_loading(label_to_id, id_to_label):
    """Test 2: Load all three checkpoints, verify architecture and output."""
    print("=" * 60)
    print("TEST 2: Model Loading & Verification")
    print("=" * 60)

    configs = [
        {"name": "CNN 32×32", "resolution": 32, "file": "cnn_32_8x8_30epochs.pth"},
        {"name": "CNN 64×64", "resolution": 64, "file": "cnn_64_8x8_30epochs.pth"},
        {"name": "CNN 128×128", "resolution": 128, "file": "cnn_128_8x8_30epochs.pth"},
    ]

    expected_classes = [
        char for char, _ in sorted(label_to_id.items(), key=lambda item: item[1])
    ]

    loaded_models = {}

    for cfg in configs:
        path = MODELS_DIR / cfg["file"]
        assert path.exists(), f"Checkpoint not found: {path}"

        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        # Verify metadata
        assert checkpoint["resolution"] == cfg["resolution"], \
            f"Resolution mismatch: {checkpoint['resolution']} != {cfg['resolution']}"
        assert checkpoint["classes"] == expected_classes, \
            f"Class ordering mismatch in {cfg['name']}"
        assert len(checkpoint["classes"]) == 67

        # Load model
        model = CNN(num_classes=67)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(DEVICE)
        model.eval()

        # Verify output shape
        with torch.inference_mode():
            dummy = torch.zeros(1, 1, cfg["resolution"], cfg["resolution"], device=DEVICE)
            out = model(dummy)
            assert out.shape == (1, 67), f"Output shape {out.shape} != (1, 67)"

        test_acc = checkpoint.get("test_accuracy", "N/A")
        print(f"  {cfg['name']}: loaded, output (1,67) ✓, test acc: {test_acc}%")

        loaded_models[cfg["name"]] = {
            "model": model,
            "resolution": cfg["resolution"],
        }

    print(f"  Device: {DEVICE}")
    print("  ✅ PASSED\n")
    return loaded_models


def test_preprocessing():
    """Test 3: Create a synthetic white-on-black image and preprocess it."""
    print("=" * 60)
    print("TEST 3: Preprocessing Pipeline")
    print("=" * 60)

    # Create a synthetic 400×400 black image with white circle (simulating drawn char)
    canvas = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(canvas, (200, 200), 80, 255, thickness=20)

    # Encode as PNG (simulating canvas export)
    _, png_buf = cv2.imencode(".png", canvas)
    png_bytes = png_buf.tobytes()

    for resolution in [32, 64, 128]:
        # Decode
        nparr = np.frombuffer(png_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        assert image is not None, "Failed to decode"

        # Grayscale (already grayscale from our synthetic)
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize
        resized = cv2.resize(gray, (resolution, resolution), interpolation=cv2.INTER_AREA)

        # Threshold
        binary = (resized > THRESHOLD).astype(np.uint8)

        # To tensor
        tensor = torch.from_numpy(binary.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        assert tensor.shape == (1, 1, resolution, resolution), \
            f"Shape {tensor.shape} != (1, 1, {resolution}, {resolution})"
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0

        unique_vals = set(tensor.unique().tolist())
        assert unique_vals.issubset({0.0, 1.0}), f"Non-binary values: {unique_vals}"

        white_pixels = tensor.sum().item()
        print(f"  {resolution}×{resolution}: shape={tensor.shape}, "
              f"dtype={tensor.dtype}, white_px={int(white_pixels)}")

    print("  ✅ PASSED\n")


def test_inference(loaded_models, id_to_label):
    """Test 4: Run inference with a synthetic image through all models."""
    print("=" * 60)
    print("TEST 4: Inference")
    print("=" * 60)

    # Create synthetic image
    canvas = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(canvas, (200, 200), 80, 255, thickness=20)
    _, png_buf = cv2.imencode(".png", canvas)
    png_bytes = png_buf.tobytes()

    for name, entry in loaded_models.items():
        model = entry["model"]
        resolution = entry["resolution"]

        # Preprocess
        nparr = np.frombuffer(png_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (resolution, resolution), interpolation=cv2.INTER_AREA)
        binary = (resized > THRESHOLD).astype(np.uint8)
        tensor = torch.from_numpy(binary.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Inference
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            logits = model(tensor)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Softmax
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0)
        top3_probs, top3_ids = torch.topk(probs, 3)

        top3 = [
            (id_to_label[idx.item()], round(prob.item() * 100, 2))
            for prob, idx in zip(top3_probs, top3_ids)
        ]

        print(f"  {name}: pred='{top3[0][0]}' ({top3[0][1]}%), "
              f"top3={top3}, time={elapsed_ms:.2f}ms")

    print("  ✅ PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GuessMyText — Backend Verification")
    print("=" * 60 + "\n")

    try:
        label_to_id, id_to_label = test_label_mapping()
        loaded_models = test_model_loading(label_to_id, id_to_label)
        test_preprocessing()
        test_inference(loaded_models, id_to_label)

        print("=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
