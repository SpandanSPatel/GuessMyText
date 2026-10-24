# GuessMyText V4 — Mathematical Expression Solver Implementation

**Status:** Implemented, Decoupled & Verified ✅  
**Scope:** Single-line handwritten arithmetic expression solver  
**Base:** V2 Line Recognition Pipeline  
**Compatibility:** V1, V2, and V3 remain functional and behaviorally unchanged. `CaptureMyText_Copy/` remains untouched.

---

## 1. Overview & Objective

GuessMyText V4 extends the handwritten recognition system with an isolated arithmetic solver. Users draw a single-line mathematical expression using digits and supported operators. Upon clicking **Solve** (exclusive to V4), the application:
1. Segments characters and runs 3-CNN majority voting using V2's established pipeline.
2. Checks for character uncertainty (`[?]`). If any character is uncertain (all 3 models disagreed), evaluation is strictly blocked to prevent misleading calculations.
3. Evaluates the arithmetic expression with standard DMAS/BODMAS operator precedence using a safe, isolated recursive-descent parser (zero use of Python's `eval()` or `exec()`).
4. Displays the recognized expression alongside the calculated result (e.g., `2 + 3 × 4 = 14`), while retaining the user's drawing on the canvas and preserving the full segmentation preview.

---

## 2. Actual Files & Routes

### Backend
- **`GuessMyText/backend/math_evaluator.py`**:
  - `TokenType` & `Token`: Typed lexical tokens (`NUMBER`, `PLUS`, `MINUS`, `MULTIPLY`, `DIVIDE`, `MODULO`, `EOF`).
  - `tokenize(expression_str)`: Strict tokenizer enforcing character whitelist, length limits ($\le 100$), token count limits ($\le 50$), and numeric magnitude ($\le 10^{12}$). Rejects letters and parentheses.
  - `ExpressionParser`: Recursive-descent precedence parser implementing strict DMAS/BODMAS order of operations with left-to-right evaluation.
  - `evaluate_expression(raw_text)`: High-level evaluation entrypoint returning structured `EvaluationResult`.
- **`GuessMyText/backend/app.py`**:
  - `POST /api/solve-math`: Dedicated endpoint accepting canvas PNG. Calls `recognize_line()`, verifies certainty, invokes `evaluate_expression()`, and returns expression, answer, segments, and preview data.
- **`GuessMyText/backend/test_math_solver.py`**:
  - 13 automated unit and integration tests covering basic operations, precedence, decimal division, unary signs, trailing equals, division-by-zero, malformed sequences, unsupported characters, safety limits, Flask endpoint integration, and V1–V3 regression checks.

### Frontend
- **`GuessMyText/frontend/src/pages/MathSolverPage.tsx`**:
  - Mounted at route `/v4`.
  - Reuses `DrawingCanvas` in horizontal mode (`internalWidth=1200`, `internalHeight=300`, `scrollable=true`).
  - Primary button: **Solve** (V2 retains **Predict**).
  - Secondary button: **Clear** (clears drawing, expression, answer, and preview).
  - Distinct state cards for empty canvas, solving progress, success, uncertain characters, division by zero, invalid expression, and unsupported characters.
  - Full Segmentation Preview showing raw crops and model input tensors across $32\times32$, $64\times64$, and $128\times128$.
- **`GuessMyText/frontend/src/App.tsx`**:
  - Maps route `/v4` to `<MathSolverPage />`.
- **`GuessMyText/frontend/src/components/AppLayout.tsx`**:
  - Sidebar links `/v4` as "V4 — Math Solver" with `MathOperations` Phosphor icon.

---

## 3. Grammar, Precedence & Numeric Semantics

### Whitelist & Supported Characters
- **Digits:** `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`.
- **Decimal Point:** `.` (e.g., `3.5`).
- **Operators:**
  - Addition: `+`
  - Subtraction: `-`
  - Multiplication: `*`, `×`
  - Division: `÷`, `/`
  - Modulo / Remainder: `%`
- **Equals Sign NOT Supported:** The equals sign `=` is NOT one of the 67 classes in the CNN dataset and cannot be recognized. Users must not write `=` on the canvas. Simply writing the expression (e.g. `2 + 3 * 4`) and clicking **Solve** calculates and displays the answer. Any handwritten `=` is rejected with status `unsupported_character`.

### Precedence & Associativity
- **Multiplicative Tier (`*`, `÷`, `%`):** Evaluated first, strictly **left-to-right**:
  - `20 ÷ 5 * 2` $\rightarrow$ $(20 \div 5) \times 2 = 4 \times 2 = \mathbf{8}$.
  - `10 + 7 % 4` $\rightarrow$ $10 + (7 \pmod 4) = 10 + 3 = \mathbf{13}$.
  - `10 - 6 ÷ 2` $\rightarrow$ $10 - (6 \div 2) = 10 - 3 = \mathbf{7}$.
  - `2 + 3 * 4` $\rightarrow$ $2 + (3 \times 4) = 2 + 12 = \mathbf{14}$.
- **Additive Tier (`+`, `-`):** Evaluated second, strictly **left-to-right**:
  - `15 - 7 + 2` $\rightarrow$ $(15 - 7) + 2 = 8 + 2 = \mathbf{10}$.

### Modulo `%` Semantics
- `%` represents the mathematical **remainder / modulo** operation, not percentage:
  - `17 % 5 = 2`.
  - `10 % 0` triggers a structured `division_by_zero` error: *"Modulo by zero is undefined."*

### Unary Signs
- Leading unary minus or plus is fully supported:
  - `-3 + 5 = 2`.
  - `+7 * 2 = 14`.
- Unary signs immediately following a multiplicative operator are supported:
  - `5 * -3 = -15`.
  - `20 ÷ -4 = -5`.
- Consecutive additive operators (e.g., `2++3`, `2+-3`, `5--2`) are strictly rejected with `invalid_expression` to match standard mathematical notation.

### Decimal Division & Formatting
- Division computes exact floating-point values where appropriate:
  - `7 ÷ 2 = 3.5`.
  - `1 ÷ 4 = 0.25`.
- Clean formatting:
  - Integers and whole floats display without decimal points (`14`, not `14.0`).
  - Non-integer results display cleanly, rounded to at most 6 decimal places with trailing zeros stripped (e.g., `10 ÷ 3 = 3.333333`).

### Non-Supported Characters & Safety Limits
- **Equals & Parentheses:** Neither `=` nor parentheses (`(`, `)`) exist in the CNN training dataset (67 classes). Expressions containing `=` or parentheses are rejected with status `unsupported_character`. Users should write expressions like `2 + 3 * 4` and click **Solve**.
- **Letters & Symbols:** Alphabetic characters (`x`, `a`, `B`) and unknown symbols are rejected with status `unsupported_character`.
- **Limits:** Max expression length is 100 characters; max token count is 50; max numeric magnitude is $10^{12}$ to prevent denial-of-service / overflow.

---

## 4. Decoupled Architecture for Future Model Replacement

The architecture strictly maintains a clean separation of concerns:
```
[ User Drawing on Canvas ]
            │
            ▼
[ Recognition Layer (line_segmentation.py) ]
  ├── Connected Component Analysis & Valley Splitting
  ├── 3-CNN Inference (32×32, 64×64, 128×128)
  └── Majority Voting (All 3 agree / 2 agree / no consensus)
            │
            ├─► has_uncertain == True ──► [ Block Evaluation, Prompt Redraw ]
            │
            ▼ has_uncertain == False
[ Evaluation Layer (math_evaluator.py) ]
  ├── Lexical Tokenizer & Whitelist Validator
  ├── Recursive-Descent Precedence Parser (DMAS)
  └── Formatted Equation & Arithmetic Result Generator
            │
            ▼
[ Presentation Layer (MathSolverPage.tsx) ]
  ├── Prominent Equation & Computed Result Card
  ├── Distinct Error & Uncertainty Diagnostic States
  └── Segmentation Preview with Multi-Resolution CNN Tensors
```

**Future Model Replacement:**
When a dedicated 15-class math model (digits 0–9 and `+`, `-`, `*`, `÷`, `%`) is trained in the future, it can directly replace the 3 CNNs in the Recognition Layer without changing a single line of `math_evaluator.py`.

---

## 5. API Contract (`POST /api/solve-math`)

### Request
- **Header:** `Content-Type: multipart/form-data` or `application/json`
- **Body:**
  - Form: `image: <binary PNG blob>`
  - JSON: `{"image": "data:image/png;base64,..."}`

### Responses

#### Success (`200 OK`)
```json
{
  "status": "success",
  "text": "2+3*4",
  "expression": "2 + 3 × 4",
  "result": 14,
  "formatted_result": "14",
  "full_equation": "2 + 3 × 4 = 14",
  "has_uncertain": false,
  "num_characters": 5,
  "segments": [ ... ],
  "message": "Solved successfully: 2 + 3 × 4 = 14"
}
```

#### Uncertainty (`200 OK`)
```json
{
  "status": "uncertain",
  "text": "2+[?]4",
  "expression": "2+[?]4",
  "result": null,
  "formatted_result": null,
  "full_equation": null,
  "has_uncertain": true,
  "num_characters": 5,
  "segments": [ ... ],
  "message": "Some characters could not be recognized with confidence..."
}
```

#### Math / Syntax Error (`200 OK`)
```json
{
  "status": "division_by_zero",  // or "invalid_expression", "unsupported_character"
  "text": "10÷0",
  "expression": "10 ÷ 0",
  "result": null,
  "formatted_result": null,
  "full_equation": null,
  "error_message": "Division by zero is undefined.",
  "has_uncertain": false,
  "segments": [ ... ],
  "message": "Division by zero is undefined."
}
```

#### Technical Model Failure (`500 Internal Server Error`)
```json
{
  "status": "error",
  "error": "System configuration error: expected 3 loaded models, found 2."
}
```

---

## 6. Verification & Test Results

### Automated Test Suite (`test_math_solver.py`)
Ran 13 automated tests covering:
1. `test_basic_operations`: Verified `2+2=4`, `15-7=8`, `6*7=42`, `20÷4=5`, `17%5=2`. (Passed)
2. `test_dmas_precedence`: Verified `2+3*4=14`, `20÷5*2=8`, `10+7%4=13`, `10-6÷2=7`. (Passed)
3. `test_decimal_division`: Verified `7÷2=3.5` and `1÷4=0.25`. (Passed)
4. `test_unary_signs`: Verified `-3+5=2` and `5*-3=-15`. (Passed)
5. `test_equals_sign_rejected`: Verified that `=` is rejected with status `unsupported_character`. (Passed)
6. `test_division_and_modulo_by_zero`: Caught `10÷0` and `17%0`. (Passed)
7. `test_malformed_sequences`: Caught `2++3`, `2+-3`, `5+`, `*4`, empty string. (Passed)
8. `test_unsupported_characters`: Rejected parentheses `(2+3)`, letters `2+x`, symbols `2^3`. (Passed)
9. `test_safety_and_magnitude_limits`: Rejected $>100$ chars and $>10^{12}$ numbers. (Passed)
10. `test_empty_canvas`: Returns `status: "empty"`. (Passed)
11. `test_solve_math_endpoint_success`: Verified end-to-end endpoint with synthetic image and preview intermediates. (Passed)
12. `test_technical_model_failure_returns_500`: Verified HTTP 500 when models fail without partial voting. (Passed)
13. `test_regression_v1_v2_v3`: Verified `/api/predict`, `/api/predict-line`, and `/api/predict-paragraph` are intact. (Passed)

### Full Regression Suite
- `test_paragraph_recognition.py`: 16/16 tests passed.
- `test_line_recognition.py`: 10/10 tests passed.
- `test_backend.py` & `test_prediction_logic.py`: 12/12 tests passed.
- Frontend build: `npm run build` completed in 621ms with 0 errors.

### Browser End-to-End Verification
- Verified `/v4` loads with "Mathematical Expression Solver" heading and `#89CFF0` theme.
- Verified primary button on V4 says **Solve** while V2 strictly says **Predict**.
- Verified canvas drawing, expansion controls (+600px), solving action, equation card, and Segmentation Preview.