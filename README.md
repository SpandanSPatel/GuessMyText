# GuessMyText — V1: Single Character Recognition

GuessMyText is a web application for handwritten character and mathematical symbol recognition. It reuses the frontend design, layout, and drawing canvas from **CaptureMyText**, with a centralized `#89CFF0` light blue primary accent theme variable. Predictions from three trained multi-scale CNN models (`32×32`, `64×64`, and `128×128`) are aggregated using majority voting into a single prominent result displayed in the horizontal center of a minimal three-section status card.

---

## Architecture Overview

```text
GuessMyText/
├── docs/
│   ├── PORTABILITY_AND_HOSTING.md  # Relocation, hosting & model architecture guide
│   ├── IMPLEMENTATION_CHARACTER_REFERENCE.md
│   └── ...
├── backend/
│   ├── app.py                      # Multi-version Flask API (V1-V4, /character-references)
│   ├── model.py                    # PyTorch CNN architecture matching checkpoints
│   ├── requirements.txt            # Python dependencies
│   ├── models/                     # Self-contained runtime model directory (19.7 MB)
│   │   └── CNN/
│   │       ├── original/           # V1: 3 CNN models + label_mapping.json (67 classes)
│   │       ├── uppercase_digits/   # V2/V3: 3 CNN models + mapping (36 classes)
│   │       └── math_symbols/       # V4: 3 CNN models + mapping (15 classes)
│   ├── static/                     # Self-contained static assets & reference characters
│   ├── test_backend.py             # Baseline checkpoint & mapping verification script
│   ├── test_specialized_models.py  # Multi-version model verification test suite
│   └── test_*.py                   # Feature & regression test suites
├── frontend/
│   ├── src/
│   │   ├── components/             # Reusable UI components & canvas
│   │   ├── pages/                  # V1, V2, V3, V4, and Character Reference pages
│   │   └── App.tsx                 # Routing configuration
│   ├── package.json
│   └── vite.config.ts              # Vite config with /api proxy to Flask (:5000)
└── README.md
```

---

## Prediction Aggregation Logic

When the user submits a drawing for recognition, inference runs across all three CNN models, and predictions are aggregated using the following rules:

1. **Case A (All 3 models agree)**:
   - *Example:* CNN 32 predicts `T`, CNN 64 predicts `T`, CNN 128 predicts `T`.
   - *Result:* Returns `T` as the final predicted character.
2. **Case B (Exactly 2 models agree)**:
   - *Example:* CNN 32 predicts `T`, CNN 64 predicts `T`, CNN 128 predicts `7`.
   - *Result:* Returns `T` as the final predicted character (majority rule).
3. **Case C (All 3 models differ)**:
   - *Example:* CNN 32 predicts `T`, CNN 64 predicts `7`, CNN 128 predicts `J`.
   - *Result:* No consensus character. The UI displays the friendly message:
     `"Couldn't recognize the character. Please try drawing it again."`
   - The drawing remains on the canvas for user inspection and manual redrawing.

### Internal Model Abstraction
- The three models remain an internal implementation detail.
- The UI displays only the final predicted character (or redraw message).
- Individual CNN resolutions, confidence percentages, agreement ratios (e.g. `2/3`), and per-model latencies are kept strictly in backend diagnostic logs.
- If any model fails during inference, a technical error (HTTP 500) is returned; the system never silently manufactures a majority prediction from fewer models.

---

## Supported Classes (67 Total)

- **Digits (10)**: `0` through `9`
- **Uppercase Letters (26)**: `A` through `Z`
- **Lowercase Letters (26)**: `a` through `z`
- **Math Operators (5)**: `+`, `-`, `*`, `÷`, `%`

---

## Installation & Setup

### 1. Backend Setup

Prerequisites: Python 3.10+ with CUDA or CPU support.

```bash
cd GuessMyText/backend
pip install -r requirements.txt
```

### 2. Frontend Setup

Prerequisites: Node.js 18+ and npm.

```bash
cd GuessMyText/frontend
npm install
```

---

## Running the Application

### 1. Start the Flask Backend

```bash
cd GuessMyText/backend
python app.py
```
*Server runs on `http://localhost:5000`.*

### 2. Start the Frontend Dev Server

```bash
cd GuessMyText/frontend
npm run dev
```
*Frontend opens at `http://localhost:5173/`.*

---

## Testing & Verification

### Prediction Logic & Voting Verification

Run the test suite verifying Cases A, B, C, model failure handling, and API endpoints:

```bash
cd GuessMyText/backend
python test_prediction_logic.py
```

### Frontend Typecheck & Build

```bash
cd GuessMyText/frontend
npm run build
```
