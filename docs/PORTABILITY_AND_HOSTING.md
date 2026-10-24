# GuessMyText — Backend Portability, Hosting & Model Directory Guide

## 1. Executive Summary

The GuessMyText application is now **100% self-contained and portable**. All nine CNN model checkpoints and three label mappings required by versions V1, V2, V3, and V4 have been copied directly into the backend under `GuessMyText/backend/models/CNN/`.

The application no longer depends on the external `New/` training directory or any hardcoded host paths at runtime. It can be moved to another directory, transferred to another computer, or packaged into a container for cloud deployment with zero external dependencies.

---

## 2. Dedicated Backend Model Directory Layout

All checkpoints and mappings are organized by model family within `GuessMyText/backend/models/CNN/`:

```text
GuessMyText/
└── backend/
    ├── app.py
    ├── model.py
    ├── requirements.txt
    └── models/
        └── CNN/
            ├── original/
            │   ├── cnn_32_8x8_30epochs.pth              # 2,212,661 bytes
            │   ├── cnn_64_8x8_30epochs.pth              # 2,212,661 bytes
            │   ├── cnn_128_8x8_30epochs.pth             # 2,212,675 bytes
            │   └── label_mapping.json                   # 932 bytes (67 classes)
            │
            ├── uppercase_digits/
            │   ├── cnn_32_uppercase_digits_8x8_30epochs.pth   # 2,196,963 bytes
            │   ├── cnn_64_uppercase_digits_8x8_30epochs.pth   # 2,196,963 bytes
            │   ├── cnn_128_uppercase_digits_8x8_30epochs.pth  # 2,196,977 bytes
            │   └── uppercase_digits_label_mapping.json        # 1,494 bytes (36 classes)
            │
            └── math_symbols/
                ├── cnn_32_math_symbols_8x8_30epochs.pth       # 2,185,771 bytes
                ├── cnn_64_math_symbols_8x8_30epochs.pth       # 2,185,771 bytes
                ├── cnn_128_math_symbols_8x8_30epochs.pth      # 2,185,785 bytes
                └── math_symbols_label_mapping.json            # 669 bytes (15 classes)
```

---

## 3. Version-to-Model Family Mapping

| Application Version | Route / Task | Model Family Directory | Checkpoints | Classes | Mapping File | Character Classes |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **V1** | `/api/predict`<br>(Single Character) | `models/CNN/original/` | `cnn_32_8x8_30epochs.pth`<br>`cnn_64_8x8_30epochs.pth`<br>`cnn_128_8x8_30epochs.pth` | 67 | `label_mapping.json` | Digits `0-9`, Uppercase `A-Z`, Lowercase `a-z`, Operators `+`, `-`, `*`, `÷`, `%` |
| **V2** | `/api/predict-line`<br>(Line Recognition) | `models/CNN/uppercase_digits/` | `cnn_32_uppercase_digits_8x8_30epochs.pth`<br>`cnn_64_uppercase_digits_8x8_30epochs.pth`<br>`cnn_128_uppercase_digits_8x8_30epochs.pth` | 36 | `uppercase_digits_label_mapping.json` | Uppercase `A-Z`, Digits `0-9` |
| **V3** | `/api/predict-paragraph`<br>(Paragraph Recognition) | `models/CNN/uppercase_digits/` | `cnn_32_uppercase_digits_8x8_30epochs.pth`<br>`cnn_64_uppercase_digits_8x8_30epochs.pth`<br>`cnn_128_uppercase_digits_8x8_30epochs.pth` | 36 | `uppercase_digits_label_mapping.json` | Uppercase `A-Z`, Digits `0-9` |
| **V4** | `/api/solve-math`<br>(Math Expression Solver) | `models/CNN/math_symbols/` | `cnn_32_math_symbols_8x8_30epochs.pth`<br>`cnn_64_math_symbols_8x8_30epochs.pth`<br>`cnn_128_math_symbols_8x8_30epochs.pth` | 15 | `math_symbols_label_mapping.json` | Digits `0-9`, Operators `+`, `-`, `*`, `÷`, `%` |

---

## 4. Path Resolution Architecture

All model checkpoints, label mappings, and static assets are resolved dynamically using Python's `pathlib.Path`:

```python
# Resolved relative to backend/app.py location:
BACKEND_DIR = Path(__file__).resolve().parent

# Dedicated models root directory:
MODELS_DIR = BACKEND_DIR / "models" / "CNN"

# V1 Paths
V1_MODELS_DIR = MODELS_DIR / "original"
V1_MAPPING_PATH = V1_MODELS_DIR / "label_mapping.json"

# V2 / V3 Paths
V2_V3_MODELS_DIR = MODELS_DIR / "uppercase_digits"
V2_V3_MAPPING_PATH = V2_V3_MODELS_DIR / "uppercase_digits_label_mapping.json"

# V4 Paths
V4_MODELS_DIR = MODELS_DIR / "math_symbols"
V4_MAPPING_PATH = V4_MODELS_DIR / "math_symbols_label_mapping.json"
```

### Guarantees
1. **Working-Directory Independent:** Launching `app.py` from any working directory (`C:\`, `~`, `/var/www`, or `backend/`) correctly resolves all paths.
2. **Zero External References:** The runtime no longer uses `../New/` or looks outside `GuessMyText/`.
3. **Fail-Fast Safety:** If any model or mapping file is missing, the backend immediately raises an explicit `FileNotFoundError` during startup before binding to port 5000.

---

## 5. Storage and Deployment Footprint

| Item | Count | Size |
| :--- | :---: | :---: |
| V1 Original Models & Mapping | 3 `.pth` + 1 `.json` | 6.64 MB |
| V2/V3 Uppercase & Digit Models & Mapping | 3 `.pth` + 1 `.json` | 6.59 MB |
| V4 Math Symbol Models & Mapping | 3 `.pth` + 1 `.json` | 6.56 MB |
| **Total Models & Mappings Footprint** | **9 `.pth` + 3 `.json`** | **19.79 MB** |

### Deployment Implications
- **Git Friendly:** Total model footprint is under 20 MB, well within GitHub's 100 MB single-file and 2 GB repository limits. Git LFS is **not** required.
- **Docker / Cloud Friendly:** Fits easily in lightweight container images (e.g., `python:3.11-slim`) on platforms like Render, Railway, Fly.io, AWS Elastic Beanstalk, or Google Cloud Run.
- **Memory Consumption:** Loading all 9 models into memory simultaneously requires < 150 MB of RAM (CPU or GPU).

---

## 6. How to Relocate and Run GuessMyText

To move the project to a new machine or folder:

### Step 1: Copy the Project Folder
Copy the entire `GuessMyText/` directory to the target location:
```text
cp -r GuessMyText/ /path/to/destination/GuessMyText/
```

### Step 2: Backend Setup
```bash
cd GuessMyText/backend

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run verification suite (optional sanity check)
python -m unittest test_backend.py test_specialized_models.py test_line_recognition.py test_paragraph_recognition.py test_math_solver.py

# Start Flask backend
python app.py
```
The backend starts at `http://localhost:5000` with all 9 models pre-loaded.

### Step 3: Frontend Setup
```bash
cd GuessMyText/frontend

# Install node dependencies
npm install

# Start Vite dev server
npm run dev
```
Open `http://localhost:5173` in your browser. All versions (V1–V4 and Character Reference) are fully operational.

---

## 7. Deploying on Render (Step-by-Step)

The project includes environment configuration and a root [`render.yaml`](file:///d:/DataAndEntertainment/N/Degree/College-Work-Degree/Sem5/ML/Project/GuessMyText/render.yaml) blueprint for deploying on [Render](https://render.com).

### Method A: One-Click Render Blueprint (Recommended)

1. Push your repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com), click **New +** → **Blueprint**.
3. Select your `GuessMyText` repository.
4. Render automatically detects `render.yaml` and provisions:
   - **`guessmytext-backend`** (Python Web Service): Installs dependencies and runs `python app.py` on `$PORT`.
   - **`guessmytext-frontend`** (Static Site): Builds the Vite bundle and automatically sets `VITE_API_BASE_URL` to your backend's live URL.
5. Click **Apply**. Both services will build and deploy automatically.

### Method B: Manual Deployment on Render

#### 1. Backend Web Service
1. **New +** → **Web Service** → Connect your GitHub repo.
2. **Root Directory:** `backend`
3. **Runtime:** `Python 3`
4. **Build Command:** `pip install -r requirements.txt`
5. **Start Command:** `python app.py` (or `gunicorn app:app --bind 0.0.0.0:$PORT`)
6. **Environment Variables:**
   - `PORT`: `10000` (Render default)
   - `PYTHON_VERSION`: `3.11.9`
   - `ALLOWED_ORIGINS`: `*` (or your frontend's Render URL)
7. Once deployed, copy your backend URL (e.g. `https://guessmytext-backend.onrender.com`).

#### 2. Frontend Static Site
1. **New +** → **Static Site** → Connect your GitHub repo.
2. **Root Directory:** `frontend`
3. **Build Command:** `npm install && npm run build`
4. **Publish Directory:** `dist`
5. **Environment Variables:**
   - `VITE_API_BASE_URL`: `https://guessmytext-backend.onrender.com` (your backend URL from step 1)
6. **Redirects/Rewrites:**
   - Under **Settings** → **Redirects/Rewrites**, add a rewrite:
     - **Source:** `/*`
     - **Destination:** `/index.html`
     - **Action:** `Rewrite`
   *(This ensures client-side React Router routing works on `/v1`, `/v2`, `/v3`, `/v4`, and `/character-reference` on page refresh).*

