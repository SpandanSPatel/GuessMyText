"""
GuessMyText — Flask Inference API

Loads three CNN checkpoints at startup, receives a canvas PNG,
applies training-compatible preprocessing, runs inference on all
three models, and returns predictions with timing.

Preprocessing pipeline (matches generate_dataset.py):
  1. Decode PNG from memory
  2. Convert to grayscale
  3. Resize to each model's resolution using INTER_AREA
  4. Threshold at 100 → binary 0/1
  5. Convert to float32 tensor of shape (1, 1, H, W)
  No normalization. No centering, padding, or cropping.
"""

import base64
from collections import Counter
import io
import json
import sys
import time
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from model import CNN
from line_segmentation import recognize_line
from paragraph_segmentation import recognize_paragraph
from math_evaluator import evaluate_expression

# ---------------------------------------------------------------------------
# Path & Model Configurations
# ---------------------------------------------------------------------------
# Resolve relative to this file's location so it works regardless of cwd.
BACKEND_DIR = Path(__file__).resolve().parent

# Dedicated backend model root directory — completely self-contained for deployment & portability
MODELS_DIR = BACKEND_DIR / "models" / "CNN"

# Training threshold — must match generate_dataset.py THRESHOLD = 100
THRESHOLD = 100

# V1 — 67-class universal model family (Digits 0-9, Uppercase A-Z, Lowercase a-z, Operators +, -, *, ÷, %)
V1_MODELS_DIR = MODELS_DIR / "original"
V1_MAPPING_PATH = V1_MODELS_DIR / "label_mapping.json"
V1_MODEL_CONFIGS = [
    {"name": "CNN 32×32 (V1)",  "resolution": 32,  "file": "cnn_32_8x8_30epochs.pth"},
    {"name": "CNN 64×64 (V1)",  "resolution": 64,  "file": "cnn_64_8x8_30epochs.pth"},
    {"name": "CNN 128×128 (V1)", "resolution": 128, "file": "cnn_128_8x8_30epochs.pth"},
]

# V2 & V3 — 36-class specialized model family (Uppercase letters A-Z, Digits 0-9)
V2_V3_MODELS_DIR = MODELS_DIR / "uppercase_digits"
V2_V3_MAPPING_PATH = V2_V3_MODELS_DIR / "uppercase_digits_label_mapping.json"
V2_V3_MODEL_CONFIGS = [
    {"name": "CNN 32×32 (V2/V3)",  "resolution": 32,  "file": "cnn_32_uppercase_digits_8x8_30epochs.pth"},
    {"name": "CNN 64×64 (V2/V3)",  "resolution": 64,  "file": "cnn_64_uppercase_digits_8x8_30epochs.pth"},
    {"name": "CNN 128×128 (V2/V3)", "resolution": 128, "file": "cnn_128_uppercase_digits_8x8_30epochs.pth"},
]

# V4 — 15-class specialized model family (Digits 0-9, Operators +, -, *, ÷, %)
V4_MODELS_DIR = MODELS_DIR / "math_symbols"
V4_MAPPING_PATH = V4_MODELS_DIR / "math_symbols_label_mapping.json"
V4_MODEL_CONFIGS = [
    {"name": "CNN 32×32 (V4)",  "resolution": 32,  "file": "cnn_32_math_symbols_8x8_30epochs.pth"},
    {"name": "CNN 64×64 (V4)",  "resolution": 64,  "file": "cnn_64_math_symbols_8x8_30epochs.pth"},
    {"name": "CNN 128×128 (V4)", "resolution": 128, "file": "cnn_128_math_symbols_8x8_30epochs.pth"},
]

# Backward compatibility aliases
LABEL_MAPPING_PATH = V1_MAPPING_PATH
MODEL_CONFIGS = V1_MODEL_CONFIGS

# ---------------------------------------------------------------------------
# Global state (populated on startup)
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_v1: dict = {}       # name → {"model": CNN, "resolution": int, "test_accuracy": float}
id_to_label_v1: dict = {}  # int → str (class index → character)
label_to_id_v1: dict = {}  # str → int

models_v2_v3: dict = {}    # name → {"model": CNN, "resolution": int, "test_accuracy": float}
id_to_label_v2_v3: dict = {}
label_to_id_v2_v3: dict = {}

models_v4: dict = {}       # name → {"model": CNN, "resolution": int, "test_accuracy": float}
id_to_label_v4: dict = {}
label_to_id_v4: dict = {}

# Backward compatibility module-level aliases pointing to V1
models = models_v1
id_to_label = id_to_label_v1
label_to_id = label_to_id_v1

# ---------------------------------------------------------------------------
# Startup — load label mappings and models once
# ---------------------------------------------------------------------------

def load_mapping_file(path: Path, expected_classes: int) -> tuple[dict, dict]:
    """
    Load and validate a label mapping file.
    Supports both flat JSON {char: int_id} and structured JSON {class_to_id, id_to_class, classes}.
    """
    if not path.exists():
        raise FileNotFoundError(f"Label mapping not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "class_to_id" in data and "id_to_class" in data:
        l2i = {str(char): int(cid) for char, cid in data["class_to_id"].items()}
        i2l = {int(cid): str(char) for cid, char in data["id_to_class"].items()}
    else:
        l2i = {str(char): int(cid) for char, cid in data.items()}
        i2l = {v: k for k, v in l2i.items()}

    if len(l2i) != expected_classes or sorted(l2i.values()) != list(range(expected_classes)):
        raise ValueError(
            f"Invalid label mapping in {path.name} — expected {expected_classes} contiguous classes (0–{expected_classes - 1}), "
            f"got {len(l2i)} classes."
        )

    return l2i, i2l


def load_label_mapping():
    """Load and validate label mappings for V1, V2/V3, and V4."""
    global id_to_label_v1, label_to_id_v1
    global id_to_label_v2_v3, label_to_id_v2_v3
    global id_to_label_v4, label_to_id_v4
    global id_to_label, label_to_id

    label_to_id_v1, id_to_label_v1 = load_mapping_file(V1_MAPPING_PATH, 67)
    label_to_id_v2_v3, id_to_label_v2_v3 = load_mapping_file(V2_V3_MAPPING_PATH, 36)
    label_to_id_v4, id_to_label_v4 = load_mapping_file(V4_MAPPING_PATH, 15)

    label_to_id = label_to_id_v1
    id_to_label = id_to_label_v1

    print(f"[startup] Loaded V1 label mapping: {len(label_to_id_v1)} classes")
    print(f"[startup] Loaded V2/V3 label mapping: {len(label_to_id_v2_v3)} classes")
    print(f"[startup] Loaded V4 label mapping: {len(label_to_id_v4)} classes")


def load_checkpoint_family(
    family_name: str,
    configs: list[dict],
    models_dir: Path,
    expected_classes_count: int,
    label_to_id_map: dict,
) -> dict:
    """
    Instantiate CNN, load state dict, verify resolution and class mapping,
    run dummy forward pass, and set eval mode for a model family.
    """
    family_models = {}
    expected_classes = [
        char for char, _ in sorted(label_to_id_map.items(), key=lambda item: item[1])
    ]

    for cfg in configs:
        path = models_dir / cfg["file"]
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found for {family_name}: {path}")

        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        # Verify resolution
        ckpt_resolution = checkpoint.get("resolution")
        if ckpt_resolution != cfg["resolution"]:
            raise ValueError(
                f"{family_name} ({cfg['name']}): checkpoint resolution {ckpt_resolution} ≠ expected {cfg['resolution']}"
            )

        # Verify class count
        ckpt_classes = checkpoint.get("classes", [])
        if len(ckpt_classes) != expected_classes_count:
            raise ValueError(
                f"{family_name} ({cfg['name']}): checkpoint has {len(ckpt_classes)} classes, expected {expected_classes_count}"
            )

        # Verify class ordering matches label mapping
        if ckpt_classes != expected_classes:
            raise ValueError(
                f"{family_name} ({cfg['name']}): class ordering in checkpoint doesn't match label mapping"
            )

        # Instantiate model with exact class count
        model = CNN(num_classes=expected_classes_count)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(DEVICE)
        model.eval()

        # Verify output shape with dummy forward pass
        with torch.inference_mode():
            dummy = torch.zeros(1, 1, cfg["resolution"], cfg["resolution"], device=DEVICE)
            out = model(dummy)
            if out.shape != (1, expected_classes_count):
                raise ValueError(
                    f"{family_name} ({cfg['name']}): output shape {out.shape}, expected (1, {expected_classes_count})"
                )

        test_acc = checkpoint.get("test_accuracy", "N/A")
        family_models[cfg["name"]] = {
            "model": model,
            "resolution": cfg["resolution"],
            "test_accuracy": test_acc,
        }
        print(
            f"[startup] Loaded {family_name} {cfg['name']} from {cfg['file']} "
            f"(test acc: {test_acc}%, device: {DEVICE})"
        )

    return family_models


def load_models():
    """Instantiate CNNs for all model families (V1, V2/V3, V4), verify, and set eval mode."""
    global models_v1, models_v2_v3, models_v4, models

    print(f"[startup] Loading V1 67-class models...")
    models_v1 = load_checkpoint_family("V1", V1_MODEL_CONFIGS, V1_MODELS_DIR, 67, label_to_id_v1)

    print(f"[startup] Loading V2/V3 36-class specialized models (uppercase letters + digits)...")
    models_v2_v3 = load_checkpoint_family("V2/V3", V2_V3_MODEL_CONFIGS, V2_V3_MODELS_DIR, 36, label_to_id_v2_v3)

    print(f"[startup] Loading V4 15-class specialized models (digits + math symbols)...")
    models_v4 = load_checkpoint_family("V4", V4_MODEL_CONFIGS, V4_MODELS_DIR, 15, label_to_id_v4)

    # Keep backward-compatible alias
    models = models_v1

    total_loaded = len(models_v1) + len(models_v2_v3) + len(models_v4)
    print(f"[startup] All {total_loaded} models across 3 families loaded successfully on {DEVICE}")


# ---------------------------------------------------------------------------
# Preprocessing — matches generate_dataset.py binary_pixels()
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes, resolution: int) -> tuple[torch.Tensor, str]:
    """
    Decode PNG from bytes, apply training-compatible preprocessing,
    return (tensor, base64_preview).

    Pipeline:
      1. Decode image (may be RGBA from canvas)
      2. Convert to grayscale
      3. Resize to (resolution, resolution) using INTER_AREA
      4. Threshold at 100 → binary 0/1
      5. Convert to float32 tensor (1, 1, H, W)
    """
    # Decode from bytes using OpenCV (same as training used cv2.imread)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("Could not decode image")

    # Grayscale conversion — matches generate_dataset.py binary_pixels()
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # Resize to target resolution — matches training
    resized = cv2.resize(gray, (resolution, resolution), interpolation=cv2.INTER_AREA)

    # Threshold → binary 0/1 — matches training (resized > THRESHOLD).astype("uint8")
    binary = (resized > THRESHOLD).astype(np.uint8)

    # Generate preview (scale binary 0/1 to 0/255 for display)
    preview_img = (binary * 255).astype(np.uint8)
    _, preview_buf = cv2.imencode(".png", preview_img)
    preview_b64 = base64.b64encode(preview_buf).decode("utf-8")

    # Convert to float32 tensor: shape (1, 1, H, W), values 0.0 or 1.0
    tensor = torch.from_numpy(binary.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    return tensor, preview_b64


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model_entry: dict, tensor: torch.Tensor, id_to_label_map: dict | None = None) -> dict:
    """
    Run a single model's inference with timing.
    Returns dict with prediction, confidence, top3, and time.
    """
    mapping = id_to_label_map if id_to_label_map is not None else id_to_label
    model = model_entry["model"]
    tensor_device = tensor.to(DEVICE)

    # Synchronize before timing (for CUDA)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        logits = model(tensor_device)

    # Synchronize after inference (for CUDA)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Softmax probabilities
    probs = F.softmax(logits, dim=1).squeeze(0)

    # Top-3 predictions
    k = min(3, probs.shape[0])
    top_probs, top_ids = torch.topk(probs, k)

    top3 = []
    for prob, idx in zip(top_probs.tolist(), top_ids.tolist()):
        top3.append({
            "character": mapping[idx],
            "confidence": round(prob * 100, 2),
        })

    predicted_idx = top_ids[0].item()

    return {
        "predicted_character": mapping[predicted_idx],
        "confidence": round(top_probs[0].item() * 100, 2),
        "top3": top3,
        "inference_time_ms": round(elapsed_ms, 2),
    }


# ---------------------------------------------------------------------------
# Prediction Aggregation Logic (Majority Rule)
# ---------------------------------------------------------------------------

def aggregate_predictions(predictions: list[str]) -> tuple[str, str | None, str | None]:
    """
    Combines exactly 3 model predictions into a single final character:
      - Case A (all 3 agree): return ('success', character, None)
      - Case B (exactly 2 agree): return ('success', majority_character, None)
      - Case C (all 3 differ): return ('no_consensus', None, friendly_message)
    """
    if len(predictions) != 3:
        raise ValueError(f"Expected exactly 3 predictions for aggregation, got {len(predictions)}")

    counts = Counter(predictions)
    most_common = counts.most_common()
    top_char, top_count = most_common[0]

    if top_count >= 2:
        return "success", top_char, None
    else:
        return (
            "no_consensus",
            None,
            "Couldn't recognize the character. Please try drawing it again.",
        )


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=str(BACKEND_DIR / "static"), static_url_path="/static")

# Configure CORS via ALLOWED_ORIGINS env variable (defaults to allow all origins)
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*").strip()
if allowed_origins_env == "*":
    CORS(app, resources={r"/*": {"origins": "*"}})
else:
    origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    CORS(app, resources={r"/*": {"origins": origins}})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts a PNG image (as file upload or base64 in JSON),
    runs all three V1 67-class models, applies majority voting, and returns
    a single final character or a friendly redraw message.
    """
    try:
        # Accept image as file upload
        if "image" in request.files:
            image_bytes = request.files["image"].read()
        # Accept image as base64 in JSON body
        elif request.is_json and "image" in request.get_json(silent=True, force=True):
            data = request.get_json(force=True)
            image_b64 = data["image"]
            # Strip data URI prefix if present
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_b64)
        else:
            return jsonify({
                "status": "error",
                "error": "No image provided. Send as 'image' file or base64 in JSON."
            }), 400

        if len(image_bytes) == 0:
            return jsonify({
                "status": "error",
                "error": "Empty image data."
            }), 400

        # Technical verification: ensure all 3 V1 models are loaded
        active_v1_models = models_v1 if len(models_v1) == 3 else models
        active_v1_mapping = id_to_label_v1 if id_to_label_v1 else id_to_label

        if len(active_v1_models) != 3:
            err_msg = f"System configuration error: expected 3 loaded models for V1, found {len(active_v1_models)}."
            print(f"[predict error] {err_msg}", file=sys.stderr)
            return jsonify({"status": "error", "error": err_msg}), 500

        model_results = []
        for name, entry in active_v1_models.items():
            resolution = entry["resolution"]
            try:
                # Preprocess for this model's resolution
                tensor, _ = preprocess_image(image_bytes, resolution)
                # Run inference using V1 67-class mapping
                try:
                    prediction = run_inference(entry, tensor, id_to_label_map=active_v1_mapping)
                except TypeError:
                    prediction = run_inference(entry, tensor)
                prediction["model_name"] = name
                model_results.append(prediction)
            except Exception as e:
                err_msg = f"Inference failed on model '{name}': {str(e)}"
                print(f"[predict error] {err_msg}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                return jsonify({
                    "status": "error",
                    "error": f"Technical inference failure on {name}."
                }), 500

        # Extract predictions for voting
        predicted_chars = [r["predicted_character"] for r in model_results]
        status, final_char, message = aggregate_predictions(predicted_chars)

        # Internal backend diagnostic logging (remains hidden from normal UI)
        diagnostic_summary = [
            f"{r['model_name']}='{r['predicted_character']}' ({r['confidence']}%, {r['inference_time_ms']}ms)"
            for r in model_results
        ]
        print(f"[predict] Model predictions: {', '.join(diagnostic_summary)} -> status={status}, result={final_char}")

        if status == "success":
            return jsonify({
                "status": "success",
                "prediction": final_char
            })
        else:
            return jsonify({
                "status": "no_consensus",
                "prediction": None,
                "message": message
            })

    except ValueError as e:
        return jsonify({"status": "error", "error": f"Image processing error: {str(e)}"}), 400
    except Exception as e:
        print(f"[predict error] Technical inference failure: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"Inference error: {str(e)}"}), 500


@app.route("/api/predict-line", methods=["POST"])
def predict_line():
    """
    Accepts a PNG image (as file upload or base64 in JSON),
    segments character candidates, estimates word boundaries,
    runs the 36-class specialized uppercase-and-digit CNN ensemble
    with majority voting per character, and returns the reconstructed text
    and character segment metadata.
    """
    try:
        # Accept image as file upload
        if "image" in request.files:
            image_bytes = request.files["image"].read()
        # Accept image as base64 in JSON body
        elif request.is_json and "image" in request.get_json(silent=True, force=True):
            data = request.get_json(force=True)
            image_b64 = data["image"]
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_b64)
        else:
            return jsonify({
                "status": "error",
                "error": "No image provided. Send as 'image' file or base64 in JSON."
            }), 400

        if len(image_bytes) == 0:
            return jsonify({
                "status": "error",
                "error": "Empty image data."
            }), 400

        # Technical verification: ensure all 3 V2/V3 models are loaded
        active_models = models_v2_v3 if len(models) == 3 else models
        if len(active_models) != 3:
            err_msg = f"System configuration error: expected 3 loaded models for V2/V3, found {len(active_models)}."
            print(f"[predict-line error] {err_msg}", file=sys.stderr)
            return jsonify({"status": "error", "error": err_msg}), 500

        result = recognize_line(image_bytes, models_v2_v3, id_to_label_v2_v3, DEVICE)

        # Internal diagnostic logging
        print(
            f"[predict-line] Segments: {result.get('num_characters')}, "
            f"Words: {result.get('num_words')}, "
            f"Has uncertain: {result.get('has_uncertain')} -> Text: '{result.get('text')}'"
        )

        return jsonify(result)

    except ValueError as e:
        return jsonify({"status": "error", "error": f"Image processing error: {str(e)}"}), 400
    except Exception as e:
        print(f"[predict-line error] Technical failure: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"Line recognition error: {str(e)}"}), 500


@app.route("/api/predict-paragraph", methods=["POST"])
def predict_paragraph_endpoint():
    """
    POST /api/predict-paragraph
    Accepts:
      - multipart form-data: file field named 'image'
      - JSON: {"image": "data:image/png;base64,..."} or raw base64 string
    Returns:
      JSON matching LinePredictionResponse / ParagraphPredictionResponse
      with reconstructed paragraph text (retaining newlines),
      lines array, character counts, and preview intermediates.
      Uses the 36-class specialized uppercase-and-digit CNN ensemble.
    """
    try:
        image_bytes = None

        if "image" in request.files:
            file = request.files["image"]
            image_bytes = file.read()
        elif request.is_json:
            data = request.get_json(silent=True) or {}
            b64_str = data.get("image", "")
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            if b64_str:
                image_bytes = base64.b64decode(b64_str)

        if image_bytes is None:
            return jsonify({
                "status": "error",
                "error": "No image provided. Send as 'image' file or base64 in JSON."
            }), 400

        if len(image_bytes) == 0:
            return jsonify({
                "status": "error",
                "error": "Empty image data."
            }), 400

        # Technical verification: ensure all 3 V3 models are loaded
        active_models = models_v2_v3 if len(models) == 3 else models
        if len(active_models) != 3:
            err_msg = f"System configuration error: expected 3 loaded models for V3, found {len(active_models)}."
            print(f"[predict-paragraph error] {err_msg}", file=sys.stderr)
            return jsonify({"status": "error", "error": err_msg}), 500

        result = recognize_paragraph(image_bytes, models_v2_v3, id_to_label_v2_v3, DEVICE)

        print(
            f"[predict-paragraph] Lines: {result.get('num_lines')}, "
            f"Characters: {result.get('num_characters')}, "
            f"Words: {result.get('num_words')}, "
            f"Has uncertain: {result.get('has_uncertain')} -> Lines recognized: {len(result.get('lines', []))}"
        )

        return jsonify(result)

    except ValueError as e:
        return jsonify({"status": "error", "error": f"Image processing error: {str(e)}"}), 400
    except Exception as e:
        print(f"[predict-paragraph error] Technical failure: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"Paragraph recognition error: {str(e)}"}), 500


@app.route("/api/solve-math", methods=["POST"])
def solve_math_endpoint():
    """
    POST /api/solve-math
    Accepts:
      - multipart form-data: file field named 'image'
      - JSON: {"image": "data:image/png;base64,..."} or raw base64 string
    Returns:
      JSON with recognized mathematical expression, computed arithmetic result,
      status ('success', 'uncertain', 'division_by_zero', 'invalid_expression', 'unsupported_character', 'empty', 'no_characters'),
      character segments with preview intermediates, and message.
      Uses the 15-class specialized digit-and-operator CNN ensemble.
    """
    try:
        image_bytes = None

        if "image" in request.files:
            file = request.files["image"]
            image_bytes = file.read()
        elif request.is_json:
            data = request.get_json(silent=True) or {}
            b64_str = data.get("image", "")
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            if b64_str:
                image_bytes = base64.b64decode(b64_str)

        if image_bytes is None:
            return jsonify({
                "status": "error",
                "error": "No image provided. Send as 'image' file or base64 in JSON."
            }), 400

        if len(image_bytes) == 0:
            return jsonify({
                "status": "error",
                "error": "Empty image data."
            }), 400

        # Technical verification: ensure all 3 V4 models are loaded
        active_models = models_v4 if len(models) == 3 else models
        if len(active_models) != 3:
            err_msg = f"System configuration error: expected 3 loaded models for V4, found {len(active_models)}."
            print(f"[solve-math error] {err_msg}", file=sys.stderr)
            return jsonify({"status": "error", "error": err_msg}), 500

        # 1. Run single-line character recognition and voting using V4 specialized models
        recog_result = recognize_line(image_bytes, models_v4, id_to_label_v4, DEVICE)

        # Handle empty canvas or no characters detected
        if recog_result.get("status") in ("empty", "no_characters"):
            return jsonify({
                "status": recog_result.get("status"),
                "text": "",
                "expression": "",
                "result": None,
                "formatted_result": None,
                "full_equation": None,
                "has_uncertain": False,
                "segments": [],
                "num_characters": 0,
                "message": recog_result.get("message", "No characters detected."),
            })

        # 2. Uncertainty handling: any uncertain character blocks evaluation
        if recog_result.get("has_uncertain"):
            return jsonify({
                "status": "uncertain",
                "text": recog_result.get("text", ""),
                "expression": recog_result.get("text", ""),
                "result": None,
                "formatted_result": None,
                "full_equation": None,
                "has_uncertain": True,
                "segments": recog_result.get("segments", []),
                "num_characters": recog_result.get("num_characters", 0),
                "message": "Some characters could not be recognized with confidence (no consensus among models). Please redraw uncertain characters marked with [?].",
            })

        # 3. Evaluate expression safely using math_evaluator
        recog_text = recog_result.get("text", "")
        eval_res = evaluate_expression(recog_text)

        if eval_res.status == "success":
            return jsonify({
                "status": "success",
                "text": recog_text,
                "expression": eval_res.expression,
                "result": eval_res.result,
                "formatted_result": eval_res.formatted_result,
                "full_equation": eval_res.full_equation,
                "has_uncertain": False,
                "segments": recog_result.get("segments", []),
                "num_characters": recog_result.get("num_characters", 0),
                "message": f"Solved successfully: {eval_res.full_equation}",
            })
        else:
            return jsonify({
                "status": eval_res.status,
                "text": recog_text,
                "expression": eval_res.expression,
                "result": None,
                "formatted_result": None,
                "full_equation": None,
                "error_message": eval_res.error_message,
                "has_uncertain": False,
                "segments": recog_result.get("segments", []),
                "num_characters": recog_result.get("num_characters", 0),
                "message": eval_res.error_message or "Invalid expression.",
            })

    except ValueError as e:
        return jsonify({"status": "error", "error": f"Image processing error: {str(e)}"}), 400
    except Exception as e:
        print(f"[solve-math error] Technical failure: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"Math solving error: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check — confirms models are loaded across all versions."""
    total_models = len(models_v1) + len(models_v2_v3) + len(models_v4)
    all_names = list(models_v1.keys()) + list(models_v2_v3.keys()) + list(models_v4.keys())
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "models_loaded": total_models,
        "model_names": all_names,
        "num_classes": len(id_to_label_v1),
        "versions": {
            "v1": {
                "models": list(models_v1.keys()),
                "num_classes": len(id_to_label_v1),
            },
            "v2_v3": {
                "models": list(models_v2_v3.keys()),
                "num_classes": len(id_to_label_v2_v3),
            },
            "v4": {
                "models": list(models_v4.keys()),
                "num_classes": len(id_to_label_v4),
            },
        },
    })


CHARACTER_REF_DIR = BACKEND_DIR / "static" / "character-references"
MANIFEST_PATH = CHARACTER_REF_DIR / "manifest.json"


@app.route("/api/character-references", methods=["GET"])
def get_character_references():
    """
    Returns the central 67-class Character Reference manifest.
    Validates manifest existence and schema integrity.
    """
    if not MANIFEST_PATH.exists():
        return jsonify({
            "status": "error",
            "error": "Character reference manifest file not found on server."
        }), 404

    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        return jsonify({
            "status": "success",
            "metadata": manifest_data.get("metadata", {}),
            "characters": manifest_data.get("characters", {})
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Failed to load character reference manifest: {str(e)}"
        }), 500


# ---------------------------------------------------------------------------
# Pre-warm models & Startup
# ---------------------------------------------------------------------------
# Load label mappings and models once on import so both 'python app.py'
# and WSGI production servers (e.g. Gunicorn, uWSGI) have models pre-warmed.
load_label_mapping()
load_models()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 60)
    print("GuessMyText — Flask Inference API")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Backend directory: {BACKEND_DIR}")
    print(f"Models root directory: {MODELS_DIR}")
    print(f"V1 Models dir: {V1_MODELS_DIR}")
    print(f"V2/V3 Models dir: {V2_V3_MODELS_DIR}")
    print(f"V4 Models dir: {V4_MODELS_DIR}")
    print()
    print(f"Starting Flask server on http://{host}:{port}")
    print("=" * 60)
    app.run(host=host, port=port, debug=False)
