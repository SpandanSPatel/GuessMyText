# Technical Investigation Report: Machine Learning System Integration in GuessMyText

**Course:** Machine Learning Essentials  
**Subject:** End-to-End Architectural Investigation, Model Integration, and Empirical Evaluation of the GuessMyText System  
**Methodology:** Read-only static code inspection and empirical verification across backend, frontend, training pipelines, and evaluation suites.

---

## 1. Project Architecture and Relevant File Map

### 1.1 High-Level Architecture Overview

GuessMyText is an end-to-end optical character recognition and handwriting computing system. It is decoupled into two independent execution environments:
1. **Frontend Client:** A Single-Page Application (SPA) built with React 18, TypeScript, and Vite. It provides an interactive drawing canvas with fixed-square, horizontally expandable, and 2D-expandable modes.
2. **Backend Inference Engine:** A Python Flask microservice that hosts nine Convolutional Neural Network (CNN) checkpoints grouped into three specialized model families, accompanied by Connected Component Analysis (CCA), projection profile segmentation, and a recursive-descent arithmetic parser.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Frontend Client (React + Vite)                         │
│  - Single Character Canvas (400×400)      - Line Canvas (Expandable 1200×300)   │
│  - Paragraph Canvas (2D 1200×600)         - Math Solver Canvas (1200×300)       │
│  - Character Reference Dictionary (67 classes, 79 dataset samples)              │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ HTTP POST (Multipart PNG / JSON base64)
                                        │ HTTP GET (JSON / Static Assets)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Backend Inference API (Flask)                         │
│   [app.py] Routing, Model Pre-warming, Ensemble Voting, Health Monitoring        │
├───────────────────────┬─────────────────────────┬───────────────────────────────┤
│  Line Segmentation    │  Paragraph Segmentation │      Math Expression Parser   │
│  [line_segmentation.py]│  [paragraph_segment...] │      [math_evaluator.py]      │
│  - CCA (connectivity=8)│  - Horizontal Projection│      - Tokenizer (DMAS rules) │
│  - Detached merging   │  - Line band clustering │      - Recursive descent      │
│  - Vertical projection│  - Intra-line splitting │      - Uncertainty blocking   │
│  - Word gap estimation│  - Newline preservation │      - Zero eval()/exec()     │
├───────────────────────┴─────────────────────────┴───────────────────────────────┤
│                             PyTorch CNN Inference                               │
│                  Architecture: Conv-Conv-AdaptiveAvgPool-MLP                    │
│   Family 1: Universal 67-Class | Family 2: Text 36-Class | Family 3: Math 15-Class│
│   (32×32, 64×64, 128×128)     | (32×32, 64×64, 128×128)  | (32×32, 64×64, 128×128)   │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ File System Access
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                 Self-Contained Model Checkpoints & Static Assets                │
│   GuessMyText/backend/models/CNN/{original, uppercase_digits, math_symbols}     │
│   GuessMyText/backend/static/character-references/{manifest.json, images}       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 File Map of System Components

| Component | File Path | Primary Responsibility |
| :--- | :--- | :--- |
| **Model Definition** | `backend/model.py` | Defines the uniform PyTorch `CNN` architecture with `AdaptiveAvgPool2d((8, 8))`. |
| **Application Controller** | `backend/app.py` | API endpoints (`/api/predict`, `/api/predict-line`, `/api/predict-paragraph`, `/api/solve-math`, `/api/character-references`, `/api/health`), model loading, pre-warming, majority voting. |
| **Line Segmentation** | `backend/line_segmentation.py` | CCA component extraction, detached stroke merging, projection valley splitting, adaptive space thresholding, square crop framing. |
| **Paragraph Segmentation** | `backend/paragraph_segmentation.py` | Horizontal projection profile (HPP) line clustering, intra-line gap bridging, tall band splitting, line sorting, newline reconstruction. |
| **Math Evaluator** | `backend/math_evaluator.py` | Safe arithmetic expression tokenizer and recursive-descent precedence parser enforcing standard mathematical order of operations. |
| **Drawing Surface** | `frontend/src/components/DrawingCanvas.tsx` | HTML5 2D canvas with pointer capture, stroke preservation via offscreen blitting, dynamic expansion, and PNG blob export. |
| **Router & Shell** | `frontend/src/App.tsx`, `frontend/src/components/AppLayout.tsx` | Client-side routing (`/v1`, `/v2`, `/v3`, `/v4`, `/character-reference`) and responsive sidebar navigation. |
| **Training Pipeline** | `New/generate_dataset.py` | Generates CSV datasets from raw PNGs across 6 resolutions with collector-balanced train/test splits. |
| **V2/V3 Training** | `New/notebooks/train_v2_v3_uppercase_digits_cnn.py` | Chunked streaming training script for the 36-class uppercase-and-digit CNN ensemble. |
| **V4 Training** | `New/notebooks/train_v4_math_symbols_cnn.py` | Training script for the 15-class digit-and-operator CNN ensemble. |

### 1.3 Communication Protocols

The client and server communicate via standard HTTP/1.1 REST calls:
* **Image Transmission:** The canvas exports an uncompressed `image/png` binary blob via `canvas.toBlob()`. It is submitted via `POST` as a multipart form-data payload with field key `'image'` (JSON base64 strings are supported as a fallback in `backend/app.py:L417`).
* **Response Serialization:** Responses are serialized as JSON containing character glyphs, consensus classifications, bounding box coordinates, word boundary flags, and base64-encoded PNG previews of segmentation intermediates.

---

## 2. Single-Character Recognition Workflow

```
[User Canvas (400×400)]
         │
         ▼ exportToBlob() (image/png)
[POST /api/predict]
         │
         ▼ cv2.imdecode & cv2.cvtColor(BGR/BGRA -> GRAY)
   [Grayscale Image]
         │
         ├──────────────────────────┬──────────────────────────┐
         ▼                          ▼                          ▼
cv2.resize(32×32, INTER_AREA) cv2.resize(64×64, INTER_AREA) cv2.resize(128×128, INTER_AREA)
         │                          │                          │
Threshold > 100 → Binary    Threshold > 100 → Binary    Threshold > 100 → Binary
Tensor (1, 1, 32, 32)       Tensor (1, 1, 64, 64)       Tensor (1, 1, 128, 128)
         │                          │                          │
         ▼                          ▼                          ▼
CNN 32×32 (67 classes)      CNN 64×64 (67 classes)      CNN 128×128 (67 classes)
         │                          │                          │
    Prediction 1               Prediction 2               Prediction 3
         └──────────────────────────┼──────────────────────────┘
                                    │
                                    ▼
                    [aggregate_predictions(preds)]
                      - 3 Agree  → Success (Character)
                      - 2 Agree  → Success (Majority Character)
                      - 0 Agree  → No Consensus (Redraw prompt)
```

### 2.1 Trace from Canvas to Prediction
1. **Canvas Input:** The user draws on a fixed-square canvas configured at $400 \times 400$ internal resolution (`frontend/src/pages/SingleCharPage.tsx:L175-L176`). Background is `#000000`; brush is solid white `#FFFFFF` with round line caps/joins and width $400 \times 0.06 = 24\text{ px}$.
2. **Payload Submission:** Clicking **Predict** invokes `exportToBlob()`, serializing the canvas into `drawing.png`, transmitted via `multipart/form-data` to `/api/predict`.
3. **Preprocessing Pipeline (`backend/app.py:L264-L307`):**
   * Decodes bytes with `cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)`.
   * Channels normalized: 4-channel BGRA or 3-channel BGR converted to 1-channel grayscale via `cv2.COLOR_BGRA2GRAY` or `cv2.COLOR_BGR2GRAY`.
   * **Multi-scale Resizing:** Directly resizes the full canvas image to each target resolution ($32 \times 32$, $64 \times 64$, $128 \times 128$) using `cv2.INTER_AREA`.
   * **Binarization:** Binarized via `(resized > 100).astype(np.uint8)`.
   * **Tensor Shaping:** Converted to PyTorch tensor `torch.from_numpy(binary.astype(np.float32)).unsqueeze(0).unsqueeze(0)`, yielding tensor shape `(1, 1, resolution, resolution)`.
   * *Critical Architectural Note:* In single-character recognition, **no cropping, bounding box detection, or centering padding is performed**. The full canvas is resized directly to the square model inputs, precisely replicating the training procedure defined in `New/generate_dataset.py:L149-L150`.
4. **Ensemble Execution & Decision Rule (`backend/app.py:L366-L388`):**
   * Each of the three models runs forward inference inside `torch.inference_mode()`.
   * Softmax probabilities are calculated (`F.softmax(logits, dim=1)`), generating top-3 predictions and confidence values.
   * Voting logic is evaluated via `Counter(predictions)`:
     * **Case A (Unanimous):** All 3 models predict the same character $\rightarrow$ returned with `status: 'success'`.
     * **Case B (Majority):** 2 models agree $\rightarrow$ majority character returned with `status: 'success'`.
     * **Case C (Total Disagreement):** All 3 models predict different characters $\rightarrow$ returned with `status: 'no_consensus'`, `prediction: null`, and message: *"Couldn't recognize the character. Please try drawing it again."*
5. **Class Configuration:** 67 contiguous classes mapped via `backend/models/CNN/original/label_mapping.json`:
   * Class 0: `%`, Class 1: `*`, Class 2: `+`, Class 3: `-`
   * Classes 4–13: Digits `0` through `9`
   * Classes 14–39: Uppercase letters `A` through `Z`
   * Classes 40–65: Lowercase letters `a` through `z`
   * Class 66: Division operator `÷`

---

## 3. Line Recognition Workflow & Character Segmentation Pipeline

Line recognition processes continuous handwritten text across a horizontally expandable canvas ($1200 \times 300\text{ px}$, expandable in $+600\text{ px}$ increments).

### 3.1 Step-by-Step Segmentation Algorithm

The segmentation algorithm is implemented in `backend/line_segmentation.py` and executes the following sequence:

```
[Raw Canvas PNG] ──► [Decode & Threshold at 100] ──► [cv2.connectedComponentsWithStats]
                                                              │ (Area ≥ 10 px)
                                                              ▼
                                                 [Initial Bounding Boxes]
                                                              │
                                                              ▼
                                                 [merge_detached_strokes]
                                                 - Dot of 'i'/'j'
                                                 - Equality '=' bars
                                                 - Division '÷' & Modulo '%'
                                                              │
                                                              ▼
                                                 [split_touching_characters]
                                                 - Wide box check (w ≥ 1.35 med_w)
                                                 - Vertical Projection Profile (HPP)
                                                 - Valley detection in 25%–75% zone
                                                              │
                                                              ▼
                                                 [Left-to-Right Sort (box.x)]
                                                              │
                                                              ▼
                                                 [estimate_word_boundaries]
                                                 - Gap > Adaptive Space Threshold
                                                              │
                                                              ▼
                                                 [Square Framing (20% pad) & Resizing]
                                                              │
                                                              ▼
                                                 [3-CNN Ensemble Majority Vote]
```

#### Step 1: Binarization and Initial Connected Components
* Image is thresholded at pixel intensity 100 into a binary mask ($0 = \text{background}$, $255 = \text{ink}$).
* Connected components are extracted using 8-way connectivity via `cv2.connectedComponentsWithStats(binary, connectivity=8)` (`line_segmentation.py:L98`).
* Components with an area $< 10\text{ pixels}$ are pruned as noise.

#### Step 2: Detached Stroke Merging (`merge_detached_strokes`, L112–L200)
Handwritten characters frequently consist of multiple disconnected strokes (the dot above `i` or `j`, the two parallel bars of `=`, the horizontal bar and dots of `÷`, or the strokes of `%`). To prevent these from being classified as independent characters, an iterative merging pass is applied:
* Baseline line height is computed: $\text{line\_height} = \max(40.0, \max(\text{box.h}))$.
* **Rule 0 (Anti-merge guard):** If two components both have height $\ge 0.50 \times \text{line\_height}$ and their horizontal centers are separated by $> 0.35 \times \max(w_1, w_2)$, they are identified as distinct side-by-side glyphs (e.g., `"To"` or `"AV"`) and **must not be merged**.
* **Rule 1 (Small detached dot/stroke):** If $\min(h_1, h_2) \le 0.45 \times \text{line\_height}$, area $\le 16 \times \text{line\_height}$, horizontal overlap ratio $\ge 0.15$ (or horizontal center distance $\le 0.60 \times \max(w_1, w_2)$), and vertical gap $\le 0.70 \times \text{line\_height}$, the components are unified into their bounding box union.
* **Rule 2 (Vertically stacked strokes):** If horizontal center distance $\le 0.45 \times \max(w_1, w_2)$, horizontal overlap $\ge 0.30$, vertical gap $\le 0.60 \times \text{line\_height}$, and total combined height $\le 1.35 \times \text{line\_height}$, they are merged (handling symbols like `=`, `:`, `÷`).

#### Step 3: Separation of Touching Characters (`split_touching_characters`, L203–L261)
When handwriting contains cursive joins or touching letters, bounding boxes become abnormally wide.
* A box is identified as candidate touching glyphs if:
  $$w \ge 1.35 \times \text{median\_width} \quad (\text{with } w \ge 30\text{ px}) \quad \lor \quad w \ge 1.05 \times h \quad (\text{with } w \ge 35\text{ px})$$
* **Vertical Projection Profile:** A 1D column-wise sum of ink pixels is computed:
  $$V_{\text{proj}}(x) = \sum_{y=y_1}^{y_2} \mathbb{I}(\text{pixel}(x, y) > 0)$$
* The algorithm searches for the minimum valley within the central 25% to 75% width region.
* If the valley intensity satisfies $\min(V) \le 0.45 \times \text{mean}(V)$ and $\min(V) < 0.40 \times h$, a split cut is executed at the valley column.
* Tight bounding boxes for both slices are recomputed using `cv2.findNonZero` and `cv2.boundingRect`.

#### Step 4: Left-to-Right Sorting
The refined bounding boxes are sorted along the horizontal axis:
`refined_boxes.sort(key=lambda b: b.x)` (`line_segmentation.py:L280`).

#### Step 5: Adaptive Word-Boundary Estimation (`estimate_word_boundaries`, L284–L316)
Inter-character horizontal gaps are computed as $G_i = x_{i+1} - (x_i + w_i)$. Because writing styles vary in scale, a fixed pixel threshold causes severe errors. An adaptive space threshold is calculated:
$$\text{threshold}_{\text{space}} = \max\Big(0.45 \times \text{median}(W), \, 1.85 \times \text{median}(G_{>2}), \, \min(20.0, \, 0.55 \times \text{median}(W))\Big)$$
If $G_i \ge \text{threshold}_{\text{space}}$, a space character (`' '`) is appended after character $i$.

#### Step 6: Square Crop Framing & Preprocessing (`prepare_character_tensor_with_preview`, L330–L374)
Unlike single-character recognition (which resizes the full canvas), segmented characters have varying aspect ratios (e.g., tall thin `'1'` vs wide `'M'`). Direct non-uniform scaling to square dimensions would distort the character representation:
1. The character bounding box is cropped from the binary mask.
2. The maximum dimension is computed: $s = \max(h, w)$.
3. A uniform 20% margin is added: $\text{pad} = \max(4, \lfloor s \times 0.20 \rfloor)$.
4. The crop is centered on a solid black square canvas of dimension $s + 2 \times \text{pad}$.
5. The square is resized to target model resolution ($32 \times 32$, $64 \times 64$, or $128 \times 128$) using `cv2.INTER_AREA`.
6. Thresholded at $100$ and converted to tensor shape `(1, 1, resolution, resolution)`.

#### Step 7: Classification and Voting
Each framed character tensor is evaluated across the **36-class specialized ensemble** (`CNN 32×32`, `CNN 64×64`, `CNN 128×128`). If two or three models agree, the character is confirmed. If all three disagree, the character is assigned `[?]` and flagged with `is_uncertain: true`.

### 3.2 36-Class Specialized Configuration
Line recognition specifically uses the 36-class vocabulary (`backend/models/CNN/uppercase_digits/uppercase_digits_label_mapping.json`):
* **Class IDs 0–25:** Uppercase `A` through `Z`
* **Class IDs 26–35:** Digits `0` through `9`
* Lowercase letters and mathematical symbols are excluded from this vocabulary to avoid classification confusion (e.g., `'O'` vs `'o'`, `'C'` vs `'c'`, or `'+'` vs `'t'`).

---

## 4. Paragraph Recognition Workflow & Multi-Line Segmentation

Paragraph recognition extends line recognition to multi-line documents on an expandable 2D canvas ($1200 \times 600\text{ px}$, expandable in four directions).

```
[Full Canvas Binary Mask]
           │
           ▼
[detect_line_regions]
  1. Horizontal Projection Profile (HPP) across all rows
  2. Detect contiguous ink row runs
  3. Bridge intra-line gaps (≤ 22 px, or dot bridges ≤ 40 px)
  4. Detect & split merged lines (height > 1.85 med_h via smoothed HPP valley)
  5. Assign components to lines by vertical center (y + h/2)
  6. Sort lines strictly Top-to-Bottom
           │
           ▼
[For Each Line Independently (segment_paragraph_lines)]
  - merge_detached_strokes (isolated to current line)
  - split_touching_characters (isolated to current line)
  - Left-to-Right Sort
  - estimate_word_boundaries
  - Square framing & 3-CNN classification
           │
           ▼
[Reconstruct Paragraph with '\n' and Uncertainty Flags]
```

### 4.1 Multi-Line Detection Algorithm (`detect_line_regions`, `backend/paragraph_segmentation.py:L119-L245`)
1. **Safety Size Validation:** Checks that width and height $\le 6000\text{ px}$ and total area $\le 25,000,000\text{ pixels}$ (`paragraph_segmentation.py:L36-L37`) to prevent Out-Of-Memory (OOM) failures on expanded canvases.
2. **Horizontal Projection Profiling (HPP):** Calculates total ink density per row:
   $$\text{HPP}(y) = \sum_{x=0}^{W-1} \mathbb{I}(\text{pixel}(x, y) > 0)$$
3. **Contiguous Run Extraction:** Groups continuous runs where $\text{HPP}(y) > 0$.
4. **Intra-Line Band Merging:** Disconnected marks (such as accents or dots of letters) create small isolated HPP runs above a line. Runs are merged if their vertical gap $\le 22\text{ px}$, or if either run has height $\le 30\text{ px}$ and gap $\le 40\text{ px}$.
5. **Detection and Splitting of Merged Lines:** When an ascender on line $k+1$ touches a descender on line $k$, HPP merges the two lines into one tall band. If $\text{band\_height} > 1.85 \times \text{median\_height}$ and $\text{band\_height} \ge 135\text{ px}$:
   * The HPP slice is smoothed using a 1D moving average filter: $\text{kernel\_size} = \max(5, \lfloor 0.25 \times \text{median\_height} \rfloor)$.
   * It searches for a deep valley in the middle 25%–75% height region.
   * If $\min(\text{HPP}_{\text{smooth}}) \le 0.45 \times \max(\text{HPP}_{\text{smooth}})$, the band is split at the minimum index.
6. **Component Clustering:** Connected components from CCA are assigned to line bands based on their vertical center coordinate:
   $$y_{\text{center}} = y_i + \frac{h_i}{2}$$
7. **Top-to-Bottom Line Ordering:** Lines are sorted strictly by the median vertical center of their constituent components.

### 4.2 Intra-Line Isolation & Paragraph Assembly
* **Cross-Line Bleed Prevention:** Detached stroke merging is explicitly applied **after** line assignment and restricted to the components of that specific line (`paragraph_segmentation.py:L269-L276`). This prevents the dot of an `'i'` or accent on line 2 from erroneously merging with a character on line 1.
* **Text Assembly:** Word boundaries are estimated per line, individual characters are classified using the 36-class specialized ensemble, and the text lines are joined with explicit newline characters:
  `full_text = "\n".join(reconstructed_lines_text)` (`paragraph_segmentation.py:L405`).

---

## 5. Math Solver Workflow & Expression Evaluation

The Math Solver provides handwritten arithmetic calculation.

```
[Canvas PNG: e.g., "7 + 2 * 3"]
             │
             ▼
[recognize_line with 15-class Math Models]
             │
             ▼
    Uncertainty Check
    - Any character is '[?]' ───────────► HALT: Return status="uncertain"
             │                                   (Blocks evaluation)
             ▼ No uncertain characters
   [math_evaluator.py]
             │
             ▼
   [tokenize(expression_str)]
   - Lex digits & floats (e.g., '7', '2', '3')
   - Lex operators ('+', '-', '*', '÷', '%')
   - Reject parentheses '(', ')' ──────► Error: Unsupported Character
   - Reject '=' sign ──────────────────► Error: Instruction to click Solve
   - Enforce limits: length ≤ 100, tokens ≤ 50, magnitude ≤ 1e12
             │
             ▼
   [ExpressionParser (Recursive-Descent)]
   - Grammar: expr -> term (('+' | '-') term)*
              term -> factor (('*' | '÷' | '%') factor)*
              factor -> ('+' | '-')? NUMBER
   - Enforce DMAS precedence (*, ÷, % before +, -)
   - Left-to-right evaluation
   - Trap division/modulo by zero ─────► Error: status="division_by_zero"
             │
             ▼
   [format_number & Output Assembly]
   e.g., "7 + 2 × 3 = 13" (status="success")
```

### 5.1 Pipeline Execution Path
1. **Recognition Phase:** The input canvas image is segmented using `recognize_line()` (`backend/app.py:L672`), passing the **15-class specialized math model ensemble** (`models_v4`).
2. **Uncertainty Gatekeeper (`backend/app.py:L690-L703`):**
   * If any recognized character has `is_uncertain: true` (models disagreed, returning `[?]`), **expression parsing is immediately aborted**.
   * The API returns `status: "uncertain"` with an explicit error: *"Some characters could not be recognized with confidence (no consensus among models). Please redraw uncertain characters marked with [?]..."*
   * This design prevents arbitrary or erroneous arithmetic operations on ambiguous handwriting.
3. **Safe Parsing and Evaluation Phase (`backend/math_evaluator.py`):**
   * **Security Policy:** **Zero use of `eval()`, `exec()`, or Python AST evaluation**. The evaluator is a sandboxed recursive-descent parser.
   * **Tokenization (`tokenize`, L64–L158):**
     * Multi-digit numbers and decimal floats are lexed (e.g., `"25"`, `"3.14"`).
     * Supported operators: `+`, `-`, `*` (or `×`), `/` (or `÷`), `%`.
     * Unsupported characters: Parentheses `(` and `)` raise `unsupported_character`: *"Parentheses are not supported in arithmetic expressions."*
     * Equals sign `=`: Raises an explicit prompt explaining that `=` is not in the dataset and the user should click **Solve** directly.
     * Length and size constraints: Expression length $\le 100$, token count $\le 50$, numeric values $\le 10^{12}$.
   * **Grammar & Precedence (`ExpressionParser`, L161–L288):**
     $$\begin{aligned}
     \text{expr}   &\to \text{term} \;\Big(\big(\text{'+'} \mid \text{'-'}\big) \;\text{term}\Big)^* \\
     \text{term}   &\to \text{factor} \;\Big(\big(\text{'*'} \mid \text{'÷'} \mid \text{'\%'}\big) \;\text{factor}\Big)^* \\
     \text{factor} &\to (\text{'+'} \mid \text{'-'})? \;\text{NUMBER}
     \end{aligned}$$
     * **DMAS / BODMAS Precedence:** Multiplicative operators (`*`, `÷`, `%`) share higher precedence and evaluate strictly left-to-right. Additive operators (`+`, `-`) evaluate second.
     * **Modulo Operator (`%`):** Evaluates mathematical remainder ($17 \% 5 = 2$).
     * **Division (`÷`):** Yields exact decimal floating-point numbers ($7 \div 2 = 3.5$, $1 \div 4 = 0.25$).
     * **Unary Signs:** Supported at expression start or after multiplicative operators (e.g., `-3 + 5 = 2`, $5 \times -3 = -15$).
     * **Consecutive Operators:** Consecutive additive operators (e.g., `2 + + 3` or `2 + - 3`) are rejected as invalid syntax.
     * **Zero Division Guard:** Catches division or modulo by zero and flags `status: "division_by_zero"`.

### 5.2 15-Class Specialized Configuration
Configured in `backend/models/CNN/math_symbols/math_symbols_label_mapping.json`:
* **Class IDs 0–9:** Digits `0` through `9`
* **Class ID 10:** `+`
* **Class ID 11:** `-`
* **Class ID 12:** `*`
* **Class ID 13:** `÷`
* **Class ID 14:** `%`
* All alphabetic characters (uppercase and lowercase) are excluded, preventing cross-domain misclassifications (such as recognizing a `+` as `t` or `x` as `*`).

---

## 6. Character Reference Implementation

The Character Reference module serves as an interactive reference dictionary showing supported character classes and sample handwriting styles.

### 6.1 Server Architecture
* Manifest: `backend/static/character-references/manifest.json` stores metadata for all 67 classes.
* Endpoint: `GET /api/character-references` (`backend/app.py:L777-L803`) returns the validated manifest JSON.
* Asset Serving: Flask serves individual sample images from `backend/static/character-references/<category>/<char>/<id>.png` using Flask's static handler.

### 6.2 Frontend Architecture (`CharacterReferencePage.tsx`)
* Tab Filtering: Groups characters into **All (67)**, **Uppercase (26)**, **Lowercase (26)**, **Digits (10)**, and **Operators (5)**.
* Live Search: Supports multi-field filtering across character symbol, name, category, and sample ID (e.g., searching `#134`).
* Modal Inspection: Clicking any character card triggers an overlay showing high-resolution renderings of the reference samples, class index, category tag, and recommended drawing advice.

### 6.3 Relationship to the Training Dataset
* The 67 classes in the manifest match the 67 classes of the universal training dataset.
* The 79 reference sample images are actual ground-truth handwriting images taken from the human-collected dataset (`New/Images/`).
* The filenames preserve their original collector dataset sample IDs (e.g., `Sample #4`, `Sample #128`, `Sample #208`), representing collectors 1, 2, and 3.

---

## 7. Model Training and Inference Details

### 7.1 Experimental Models vs. Deployed Models

The academic project conducted extensive experiments across four machine learning paradigms before deploying the final system. All experimental training artifacts were recorded in `New/notebooks/` and saved in `New/models/`.

| Model Family | Algorithm Details | Evaluated Resolutions | Recorded Test Accuracy (67 Classes) | Deployment Status | Reason for Deployment Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | Support Vector Classifier with RBF kernel (`New/notebooks/SVM.ipynb`) | $32\times32$<br>$64\times64$<br>$128\times128$ | $32\times32$: **83.88%**<br>$64\times64$: **84.70%**<br>$128\times128$: **84.43%** | Experimental only | Lower accuracy; high inference latency and large memory footprint during prediction on raw pixel vectors. |
| **Random Forest** | 100 Estimators, Gini impurity (`New/notebooks/RandomForest.ipynb`) | $32\times32$<br>$64\times64$<br>$128\times128$<br>$256\times256$ | $32\times32$: **84.03%**<br>$64\times64$: **83.36%**<br>$128\times128$: **82.59%**<br>$256\times256$: **82.29%** | Experimental only | Accuracy plateaus around ~84%; struggles with stroke variations and pixel misalignments. |
| **MLP** | Fully Connected Neural Network: `Linear-ReLU-Dropout-Linear` (`New/notebooks/MLP.ipynb`) | $32\times32$<br>$64\times64$<br>$128\times128$<br>$256\times256$ | $32\times32$: **82.49%**<br>$64\times64$: **86.29%**<br>$128\times128$: **87.64%**<br>$256\times256$: **86.64%** | Experimental only | Outperformed classical methods but lags behind CNNs by 8–9%; cannot leverage local 2D spatial hierarchies. |
| **CNN** | Deep Convolutional Network: 2 Conv layers + Adaptive Pooling + FC classifier (`New/notebooks/CNN.ipynb`) | $32\times32$<br>$64\times64$<br>$128\times128$ | $32\times32$: **96.04%**<br>$64\times64$: **95.82%**<br>$128\times128$: **95.15%** | **DEPLOYED (All features)** | **Highest performance across all metrics** (~96–97%); translation invariance; fast inference (<1.5 ms on GPU). |

### 7.2 Deployed Model Catalog (All 9 Production Checkpoints)

The production deployment hosts nine CNN checkpoints organized into three task-specific families located under `backend/models/CNN/`:

```
backend/models/CNN/
├── original/
│   ├── cnn_32_8x8_30epochs.pth           (Universal, 67 classes, 32×32,   acc: 96.04%)
│   ├── cnn_64_8x8_30epochs.pth           (Universal, 67 classes, 64×64,   acc: 95.82%)
│   ├── cnn_128_8x8_30epochs.pth          (Universal, 67 classes, 128×128, acc: 95.15%)
│   └── label_mapping.json
├── uppercase_digits/
│   ├── cnn_32_uppercase_digits_8x8_30epochs.pth   (Text, 36 classes, 32×32,   acc: 97.36%)
│   ├── cnn_64_uppercase_digits_8x8_30epochs.pth   (Text, 36 classes, 64×64,   acc: 97.36%)
│   ├── cnn_128_uppercase_digits_8x8_30epochs.pth  (Text, 36 classes, 128×128, acc: 96.30%)
│   └── uppercase_digits_label_mapping.json
└── math_symbols/
    ├── cnn_32_math_symbols_8x8_30epochs.pth       (Math, 15 classes, 32×32,   acc: 96.44%)
    ├── cnn_64_math_symbols_8x8_30epochs.pth       (Math, 15 classes, 64×64,   acc: 97.11%)
    ├── cnn_128_math_symbols_8x8_30epochs.pth      (Math, 15 classes, 128×128, acc: 96.11%)
    └── math_symbols_label_mapping.json
```

### 7.3 Detailed Architecture Specification

All nine models share the architecture defined in `backend/model.py`:

$$\begin{aligned}
\mathbf{x} &\in \mathbb{R}^{B \times 1 \times H \times W} \quad (H, W \in \{32, 64, 128\}) \\
\mathbf{h}_1 &= \text{MaxPool2d}\Big(\text{ReLU}\big(\text{Conv2d}(1 \to 32, k=3, p=1)(\mathbf{x})\big), k=2\Big) \\
\mathbf{h}_2 &= \text{MaxPool2d}\Big(\text{ReLU}\big(\text{Conv2d}(32 \to 64, k=3, p=1)(\mathbf{h}_1)\big), k=2\Big) \\
\mathbf{h}_{\text{pooled}} &= \text{AdaptiveAvgPool2d}((8, 8))(\mathbf{h}_2) \quad \in \mathbb{R}^{B \times 64 \times 8 \times 8} \\
\mathbf{z} &= \text{Flatten}(\mathbf{h}_{\text{pooled}}) \quad \in \mathbb{R}^{B \times 4096} \\
\mathbf{f}_1 &= \text{Dropout}(p=0.2)\Big(\text{ReLU}\big(\text{Linear}(4096 \to 128)(\mathbf{z})\big)\Big) \\
\hat{\mathbf{y}} &= \text{Linear}(128 \to C)(\mathbf{f}_1) \quad (C \in \{67, 36, 15\})
\end{aligned}$$

**Key Architectural Insight:** The inclusion of `AdaptiveAvgPool2d((8, 8))` normalizes the feature map spatial dimension to $8 \times 8$ regardless of whether the input image was $32 \times 32$, $64 \times 64$, or $128 \times 128$. This keeps the classifier dense parameter count identical across all resolutions:
* Conv layer 1: $(1 \times 3 \times 3 \times 32) + 32 = 320$ parameters.
* Conv layer 2: $(32 \times 3 \times 3 \times 64) + 64 = 18,496$ parameters.
* Dense layer 1: $(4096 \times 128) + 128 = 524,416$ parameters.
* Dense layer 2 (Output): $(128 \times C) + C$ parameters ($8,643$ for $C=67$; $4,644$ for $C=36$; $1,935$ for $C=15$).
* Total model weights: $\approx 544\text{k to }552\text{k}$ parameters ($\approx 2.19\text{ MB}$ per file).

### 7.4 Training Procedure & Preprocessing Comparison

| Hyperparameter / Parameter | Value in Training Pipeline |
| :--- | :--- |
| **Optimizer** | `AdamW(lr=0.001)` (`train_v2_v3_uppercase_digits_cnn.py:L256`) |
| **Loss Function** | `CrossEntropyLoss()` |
| **Batch Size** | 128 |
| **Epochs** | 30 |
| **Hardware** | NVIDIA CUDA GPU acceleration |
| **Weight Decay / Regularization** | Default AdamW weight decay ($10^{-2}$), Dropout $0.20$ in penultimate dense layer |

#### Training vs. Inference Preprocessing Comparison:
* **Training Preprocessing (`New/generate_dataset.py:L133-L150`):**
  1. Load raw collected PNG image from disk (`cv2.imread`).
  2. Convert to grayscale (`cv2.COLOR_BGR2GRAY`).
  3. Resize uncropped image directly to $(R \times R)$ using `cv2.INTER_AREA`.
  4. Apply hard binary threshold: $\text{pixel} = 1 \text{ if } \text{pixel} > 100 \text{ else } 0$.
  5. Flatten to 1D binary vector of length $R^2$ and save to CSV.
* **Inference Preprocessing:**
  * *Single Character:* Exactly matches training preprocessing. Full canvas decoded $\rightarrow$ Grayscale $\rightarrow$ Resized directly to $(R \times R)$ via `cv2.INTER_AREA` $\rightarrow$ Threshold $> 100 \rightarrow$ Converted to float32 tensor `(1, 1, R, R)`.
  * *Line, Paragraph, and Math Segmented Characters:* Differs deliberately from training to prevent distortion: Character is cropped along tight bounding box $\rightarrow$ Centered onto square canvas with 20% margin $\rightarrow$ Resized to $(R \times R)$ via `cv2.INTER_AREA` $\rightarrow$ Threshold $> 100$.

---

## 8. Class Mappings and Supported Character Sets

The application maintains three distinct label mappings corresponding to its model families:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Universal 67-Class Mapping (Single Character)                          │
│ 0-3: %, *, +, - | 4-13: 0-9 | 14-39: A-Z | 40-65: a-z | 66: ÷         │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │ Subset                         │ Subset
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│ Text 36-Class Mapping (Line/Paragraph)│  │ Math 15-Class Mapping (Math) │
│ 0-25: A-Z | 26-35: 0-9               │  │ 0-9: 0-9 | 10-14: +, -, *, ÷,%│
└──────────────────────────────────────┘  └──────────────────────────────┘
```

| Dimension | Universal Family (V1) | Text Family (V2 & V3) | Math Family (V4) |
| :--- | :--- | :--- | :--- |
| **Target Route** | `Single Character (/v1)` | `Line (/v2)` & `Paragraph (/v3)` | `Math Solver (/v4)` |
| **Total Classes** | **67** | **36** | **15** |
| **Mapping File** | `backend/models/CNN/original/label_mapping.json` | `backend/models/CNN/uppercase_digits/uppercase_digits_label_mapping.json` | `backend/models/CNN/math_symbols/math_symbols_label_mapping.json` |
| **File Format** | Flat `{char: int_id}` | Structured `{class_to_id, id_to_class, classes}` | Structured `{class_to_id, id_to_class, classes}` |
| **Digits (0–9)** | IDs 4–13 | IDs 26–35 | IDs 0–9 |
| **Uppercase (A–Z)**| IDs 14–39 | IDs 0–25 | *Not Supported* |
| **Lowercase (a–z)**| IDs 40–65 | *Not Supported* | *Not Supported* |
| **Operators** | `%` (0), `*` (1), `+` (2), `-` (3), `÷` (66) | *Not Supported* | `+` (10), `-` (11), `*` (12), `÷` (13), `%` (14) |

---

## 9. Existing Testing and Evaluation Evidence

The backend includes a comprehensive automated test suite consisting of **52 unit and integration tests** implemented across seven test files.

### 9.1 Test File Breakdown

| Test File | Total Tests | Execution Target & Primary Assertions |
| :--- | :---: | :--- |
| `backend/test_backend.py` | 4 | Verifies 67-class label mapping integrity, model loading on CUDA/CPU, multi-scale tensor output shapes `(1, 67)`, and live inference. |
| `backend/test_prediction_logic.py` | 6 | Validates majority voting cases: Case A (3 agree $\to$ success), Case B (2 agree $\to$ majority), Case C (all differ $\to$ no consensus), and verifies that model failure produces HTTP 500 without fabricating a prediction. |
| `backend/test_specialized_models.py` | 5 | Validates all 9 checkpoints across the 3 families (67, 36, 15 classes), label mapping separation, real image inference from `New/Images/`, version endpoint isolation, and `/api/health`. |
| `backend/test_line_recognition.py` | 10 | Validates empty canvas detection, BoundingBox math, detached stroke merging, touching character splitting via vertical projection, adaptive word-boundary gaps, majority voting, and `/api/predict-line`. |
| `backend/test_paragraph_recognition.py` | 9 | Tests HPP multi-line detection, top-to-bottom line sorting, intra-line gap bridging, newline (`\n`) preservation, canvas size safety limits, preview intermediates, and `/api/predict-paragraph`. |
| `backend/test_math_solver.py` | 13 | Tests arithmetic operations, DMAS precedence, decimal division, unary signs, division by zero trapping, malformed syntax detection, safety limits, uncertainty evaluation blocking, and `/api/solve-math`. |
| `backend/test_character_reference.py` | 5 | Validates `manifest.json` schema, 67-class distribution (10 digits, 26 uppercase, 26 lowercase, 5 operators), `GET /api/character-references`, and static sample assets. |
| **Total Automated Tests** | **52** | **100% Passing (Verified via `python -m unittest discover`)** |

### 9.2 Empirically Verified Model Accuracies

These accuracies are recorded in the production checkpoints and verified during model startup:

```
Universal Models (67 classes):
  - CNN 32×32:   96.04%
  - CNN 64×64:   95.82%
  - CNN 128×128: 95.15%

Text Specialized Models (36 classes):
  - CNN 32×32:   97.36%
  - CNN 64×64:   97.36%
  - CNN 128×128: 96.30%

Math Specialized Models (15 classes):
  - CNN 32×32:   96.44%
  - CNN 64×64:   97.11%
  - CNN 128×128: 96.11%
```

---

## 10. Implementation Details, Limitations, and Academic Considerations

### 10.1 Key Findings & Architectural Rationale

1. **Resolution Ensemble and Majority Voting:**
   * Rather than deploying a single monolithic network, the architecture uses an ensemble of three identical models trained at three different spatial resolutions: $32 \times 32$, $64 \times 64$, and $128 \times 128$.
   * **Why Multi-Resolution?** $32 \times 32$ captures coarse stroke topologies and global character structure while ignoring minor drawing jitter. Conversely, $128 \times 128$ preserves fine details needed to distinguish similar characters (such as `'E'` vs `'F'` or `'O'` vs `'Q'`).
   * The majority voting rule provides an uncertainty detection mechanism: when handwriting is illegible or ambiguous, the models diverge across scales, triggering a `no_consensus` response rather than presenting an incorrect prediction.
2. **Domain Specialization via Vocabulary Pruning:**
   * In continuous line recognition and paragraph recognition, human handwriting exhibits high ambiguity between cases (e.g., `'c'` vs `'C'`, `'s'` vs `'S'`, `'o'` vs `'O'`) and between symbols and letters (`'+'` vs `'t'`, `'-'` vs `'_'`, `'*'` vs `'x'`).
   * By training specialized 36-class models (A–Z, 0–9) for text and 15-class models (0–9, +, -, *, ÷, %) for arithmetic, the system eliminated cross-category classification errors, raising test set accuracy from $\approx 95.8\%$ to **$97.36\%$**.
3. **Aspect Ratio Preservation in Segmentation:**
   * Single-character recognition assumes the user utilizes the full square drawing surface and resizes the full canvas directly.
   * In line and paragraph recognition, characters have natural non-square aspect ratios (e.g., `'I'` is tall and narrow; `'W'` is wide). If a narrow `'I'` were squeezed directly into a $32 \times 32$ square without padding, the stroke would expand horizontally and resemble a rectangle or square block. The square framing step (20% margin) in `line_segmentation.py` preserves natural stroke proportions prior to classification.

### 10.2 System Limitations & Open Research Questions

1. **Absence of Parentheses and Equals Sign:**
   * The underlying dataset collected via `CaptureMyText` did not include open/close parentheses or an equals sign. Consequently, the Math Solver cannot evaluate expressions with parentheses like `(2 + 3) * 4`. The system handles this gracefully at runtime by catching them in the tokenizer and returning actionable user feedback, but expanding the training dataset to include grouping symbols is an area for future work.
2. **Connected Cursive Handwriting Limits:**
   * While the vertical projection profile split (`split_touching_characters`) handles simple touching characters with clear vertical valleys, heavily overlapping cursive handwriting where strokes cross diagonally cannot be separated cleanly by vertical projection profiling alone. More complex segmentation algorithms (such as watershed segmentation, contour analysis, or recurrent CTC-loss architectures like CRNN) would be required for continuous cursive script.
3. **Multi-Column Paragraph Layouts:**
   * The paragraph line detection algorithm uses horizontal projection profiling across the full width of the canvas. This assumes single-column text. If a user draws multi-column text side-by-side, HPP merges the lines across columns. A pre-clustering step along the vertical axis would be necessary to support multi-column layouts.
4. **Execution Safety Policy:**
   * The math evaluator avoids the standard pitfall of using Python's `eval()` on user-supplied strings. The custom tokenizer and recursive-descent parser ensure execution safety against code injection vulnerabilities.

---

## 11. Verification and Reproduction Guide

To independently verify all findings in this report, execute the following commands in the project directory:

```bash
# 1. Run the complete backend test suite (all 52 automated tests)
cd GuessMyText/backend
python -m unittest discover

# 2. Verify all 9 model checkpoints and metadata
python test_specialized_models.py

# 3. Verify single-character voting logic
python test_prediction_logic.py

# 4. Verify line recognition and segmentation
python test_line_recognition.py

# 5. Verify paragraph multi-line recognition
python test_paragraph_recognition.py

# 6. Verify mathematical expression evaluator
python test_math_solver.py

# 7. Verify frontend production compilation
cd ../frontend
npm run build
```
