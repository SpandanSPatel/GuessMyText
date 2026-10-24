/**
 * GuessMyText — V4 Mathematical Expression Solver Page
 *
 * Implements single-line handwritten arithmetic expression recognition and evaluation:
 *   1. Reuses V2's drawing canvas with horizontal scrolling and 600px expansion.
 *   2. Reuses V2's 3-CNN ensemble recognition and segmentation preview.
 *   3. Evaluates recognized arithmetic expressions with DMAS/BODMAS precedence.
 *   4. Displays recognized expression and calculated answer (e.g. 2 + 3 × 4 = 14).
 *   5. Blocks evaluation if any character is uncertain [?].
 *   6. Handles division by zero, invalid expressions, and unsupported characters.
 */

import { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  MathOperations,
  Eraser,
  Equals,
  Warning,
  XCircle,
  BookOpen,
  Info,
  X,
  Copy,
  Check,
  CheckCircle,
  SquaresFour,
  Plus,
  ArrowCounterClockwise,
  CaretLeft,
  CaretRight,
} from '@phosphor-icons/react';
import { DrawingCanvas } from '../components/DrawingCanvas';
import type { DrawingCanvasRef } from '../components/DrawingCanvas';
import { apiUrl } from '../config/api';
import toast from 'react-hot-toast';
import { SHOW_SEGMENTATION_PREVIEW } from '../config/features';

interface ModelInputPreview {
  resolution: number;
  dimensions: string;
  image: string;
}

interface CharacterSegment {
  index: number;
  character: string;
  is_uncertain: boolean;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  crop_dimensions?: {
    width: number;
    height: number;
  };
  has_space_after: boolean;
  crop_image?: string;
  model_inputs?: Record<string, ModelInputPreview>;
}

interface MathSolverResponse {
  status:
    | 'success'
    | 'uncertain'
    | 'division_by_zero'
    | 'invalid_expression'
    | 'unsupported_character'
    | 'empty'
    | 'no_characters'
    | 'error';
  text?: string;
  expression?: string;
  result?: number | null;
  formatted_result?: string | null;
  full_equation?: string | null;
  error_message?: string;
  has_uncertain?: boolean;
  segments?: CharacterSegment[];
  num_characters?: number;
  message?: string;
  error?: string;
}

export function MathSolverPage() {
  const [isCanvasEmpty, setIsCanvasEmpty] = useState(true);
  const [isSolving, setIsSolving] = useState(false);
  const [canvasWidth, setCanvasWidth] = useState(1200);
  const [result, setResult] = useState<MathSolverResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [selectedResolution, setSelectedResolution] = useState<'32' | '64' | '128'>('64');

  // Dismissible guidance banner state (persists across sessions)
  const [showGuidance, setShowGuidance] = useState(() => {
    try {
      return localStorage.getItem('gmt_v4_dismiss_guidance_v2') !== 'true';
    } catch {
      return true;
    }
  });

  const canvasRef = useRef<DrawingCanvasRef>(null);

  const handleDismissGuidance = () => {
    setShowGuidance(false);
    try {
      localStorage.setItem('gmt_v4_dismiss_guidance_v2', 'true');
    } catch {
      // Ignore local storage error
    }
  };

  const handleClear = () => {
    canvasRef.current?.clear();
    setIsCanvasEmpty(true);
    setResult(null);
    setError(null);
    setCopied(false);
  };

  const handleExpandLeft = () => {
    canvasRef.current?.expandLeft(600);
  };

  const handleExpandRight = () => {
    canvasRef.current?.expandRight(600);
  };

  const handleResetWidth = () => {
    canvasRef.current?.resetDimensions(1200);
  };

  const handleScrollLeft = () => {
    canvasRef.current?.scrollToLeft();
  };

  const handleScrollRight = () => {
    canvasRef.current?.scrollToRight();
  };

  const handleSolve = async () => {
    if (!canvasRef.current || isSolving) return;

    // Empty canvas check: do not submit request
    if (isCanvasEmpty || canvasRef.current.isEmpty()) {
      setIsCanvasEmpty(true);
      toast.error('Please write an arithmetic expression first.');
      return;
    }

    const blob = await canvasRef.current.exportToBlob();
    if (!blob) return;

    setIsSolving(true);
    setError(null);
    setResult(null);
    setCopied(false);

    try {
      const formData = new FormData();
      formData.append('image', blob, 'math_expression.png');

      const response = await fetch(apiUrl('/api/solve-math'), {
        method: 'POST',
        body: formData,
      });

      const data: MathSolverResponse = await response.json();

      if (!response.ok || data.status === 'error') {
        const msg = data.error || data.message || `Server error: ${response.status}`;
        setError(msg);
        toast.error(msg);
        return;
      }

      setResult(data);

      if (data.status === 'empty' || data.status === 'no_characters') {
        toast(data.message || 'No characters detected.', { icon: 'ℹ️' });
      } else if (data.status === 'success') {
        toast.success(`Solved: ${data.full_equation}`);
      } else if (data.status === 'uncertain') {
        toast('Some characters are uncertain. Please redraw clearly.', { icon: '⚠️' });
      } else if (data.status === 'division_by_zero') {
        toast.error(data.error_message || 'Division by zero is undefined.');
      } else if (data.status === 'unsupported_character') {
        toast.error(data.error_message || 'Unsupported character in expression.');
      } else if (data.status === 'invalid_expression') {
        toast.error(data.error_message || 'Invalid arithmetic expression.');
      }
    } catch (err) {
      const msg =
        err instanceof Error
          ? `Connection error: ${err.message}`
          : 'Failed to connect to the prediction server.';
      setError(msg);
      toast.error(msg);
    } finally {
      setIsSolving(false);
    }
  };

  const handleCopy = async () => {
    const textToCopy = result?.full_equation || result?.formatted_result?.toString() || result?.text;
    if (!textToCopy) return;
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      toast.success('Copied equation to clipboard!');
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error('Failed to copy text.');
    }
  };

  return (
    <div className="max-w-5xl mx-auto flex flex-col gap-6">
      {/* Page Header */}
      <div className="flex items-center gap-3">
        <MathOperations size={28} weight="duotone" className="text-primary" />
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">Mathematical Expression Solver</h1>
          <p className="text-xs text-text-muted mt-0.5">
            Write a single-line arithmetic expression using digits 0–9 and +, -, *, ÷, % to solve.
          </p>
        </div>
      </div>

      {/* Dismissible Math Guidance Banner */}
      {showGuidance && (
        <div className="bg-primary/5 border border-primary/20 rounded-xl p-4 md:p-5 relative text-text-primary text-sm shadow-sm animate-in fade-in transition-all">
          <div className="flex items-start gap-3 pr-8">
            <Info size={22} weight="duotone" className="text-primary shrink-0 mt-0.5" />
            <div className="flex flex-col gap-2 leading-relaxed">
              <h2 className="font-semibold text-text-primary text-base">
                Tips for better recognition
              </h2>
              <div className="flex flex-col gap-2 text-xs md:text-sm text-text-secondary leading-relaxed">
                <p>
                  GuessMyText V4 recognizes handwritten digits (0–9) and the operators <code className="px-1 py-0.5 rounded bg-base-200 text-text-primary font-mono">+</code>, <code className="px-1 py-0.5 rounded bg-base-200 text-text-primary font-mono">-</code>, <code className="px-1 py-0.5 rounded bg-base-200 text-text-primary font-mono">*</code>, <code className="px-1 py-0.5 rounded bg-base-200 text-text-primary font-mono">÷</code>, and <code className="px-1 py-0.5 rounded bg-base-200 text-text-primary font-mono">%</code>.
                </p>
                <p>
                  Write each digit and operator clearly, leaving enough space to distinguish neighboring characters.
                </p>
                <p className="text-text-muted">
                  The <code className="px-1 py-0.5 rounded bg-base-200 text-text-primary font-mono">%</code> symbol represents modulo/remainder, not percentage. Parentheses are not supported in this version.
                </p>
              </div>
            </div>
          </div>
          <button
            onClick={handleDismissGuidance}
            className="absolute top-3.5 right-3.5 p-1 rounded-lg text-text-muted hover:text-text-primary hover:bg-base-200 transition-colors cursor-pointer"
            title="Dismiss tips"
            aria-label="Dismiss tips"
          >
            <X size={18} />
          </button>
        </div>
      )}

      {/* Three-Section Status Card */}
      <div className="bg-surface rounded-xl border border-border px-4 py-3 md:px-6 md:py-3.5 shadow-sm min-h-[64px] flex items-center">
        <div className="w-full grid grid-cols-[auto_1fr_auto] items-center gap-4">
          {/* Left: Character Reference link */}
          <div className="flex items-center justify-start">
            <Link
              to="/character-reference"
              className="btn btn-sm border border-border bg-base-200/60 hover:bg-primary-light hover:text-primary hover:border-primary/40 text-text-secondary font-medium transition-colors gap-2"
              title="Open Character Reference"
            >
              <BookOpen size={16} weight="duotone" className="text-primary shrink-0" />
              <span className="hidden sm:inline">Character Reference</span>
              <span className="sm:hidden">Reference</span>
            </Link>
          </div>

          {/* Center: Solver Status / Feedback */}
          <div className="flex items-center justify-center text-center">
            {isSolving ? (
              <div className="flex items-center gap-2 text-sm font-medium text-text-secondary">
                <span className="loading loading-spinner loading-xs text-primary"></span>
                <span>Recognizing characters & evaluating expression...</span>
              </div>
            ) : error ? (
              <div className="flex items-center gap-1.5 text-error text-xs md:text-sm font-medium text-center">
                <XCircle size={18} weight="bold" className="shrink-0 text-error" />
                <span className="leading-snug truncate max-w-xs md:max-w-md">{error}</span>
              </div>
            ) : result?.status === 'success' ? (
              <div className="flex items-center gap-2 text-xs md:text-sm font-medium text-text-secondary">
                <CheckCircle size={18} weight="fill" className="text-success shrink-0" />
                <span className="font-semibold text-text-primary">
                  {result.full_equation}
                </span>
              </div>
            ) : result?.status === 'uncertain' ? (
              <div className="flex items-center gap-1.5 text-warning text-xs md:text-sm font-medium">
                <Warning size={18} weight="fill" className="shrink-0 text-warning" />
                <span>Uncertain characters detected. Evaluation blocked.</span>
              </div>
            ) : result?.status === 'division_by_zero' ||
              result?.status === 'invalid_expression' ||
              result?.status === 'unsupported_character' ? (
              <div className="flex items-center gap-1.5 text-error text-xs md:text-sm font-medium">
                <XCircle size={18} weight="bold" className="shrink-0 text-error" />
                <span>{result.error_message || result.message}</span>
              </div>
            ) : result?.status === 'no_characters' || result?.status === 'empty' ? (
              <div className="flex items-center gap-1.5 text-warning text-xs md:text-sm font-medium">
                <Warning size={18} weight="fill" className="shrink-0 text-warning" />
                <span>{result.message}</span>
              </div>
            ) : (
              <span className="text-xs text-text-muted hidden md:inline">
                Write an arithmetic expression e.g. 2 + 3 * 4 or 20 ÷ 5
              </span>
            )}
          </div>

          {/* Right: Ready indicator */}
          <div className="flex items-center justify-end">
            {isSolving ? (
              <div className="flex items-center gap-2 text-sm font-medium text-primary">
                <span className="loading loading-spinner loading-xs text-primary"></span>
                <span>Solving...</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm font-medium text-text-primary">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-success"></span>
                <span>Ready</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Canvas Area with Left & Right Expansion Controls */}
      <div className="flex flex-col gap-3 w-full">
        {/* Canvas Toolbar: Dimensions & Quick Scroll Controls */}
        <div className="w-full flex flex-wrap items-center justify-between text-xs text-text-muted px-1 gap-2">
          <div className="flex items-center gap-2">
            <span className="font-medium text-text-secondary">Canvas:</span>
            <span className="font-mono bg-base-200 px-2.5 py-0.5 rounded-md text-text-primary border border-border font-semibold">
              {canvasWidth} × 300 px
            </span>
            {canvasWidth > 1200 && (
              <button
                type="button"
                onClick={handleResetWidth}
                disabled={isSolving}
                className="btn btn-xs btn-ghost text-primary hover:bg-primary/10 gap-1 font-medium transition-colors"
                title="Reset canvas width to initial 1200px"
              >
                <ArrowCounterClockwise size={12} weight="bold" />
                <span>Reset to 1200px</span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[11px] text-text-muted hidden sm:inline">
              +600px per expansion
            </span>
            <div className="flex items-center gap-1 border border-border rounded-lg bg-base-200/50 p-0.5">
              <button
                type="button"
                onClick={handleScrollLeft}
                className="btn btn-xs btn-ghost px-2 text-text-secondary hover:text-primary gap-1"
                title="Scroll canvas to start"
              >
                <CaretLeft size={12} weight="bold" />
                <span className="hidden md:inline text-[11px]">Left</span>
              </button>
              <button
                type="button"
                onClick={handleScrollRight}
                className="btn btn-xs btn-ghost px-2 text-text-secondary hover:text-primary gap-1"
                title="Scroll canvas to end"
              >
                <span className="hidden md:inline text-[11px]">Right</span>
                <CaretRight size={12} weight="bold" />
              </button>
            </div>
          </div>
        </div>

        {/* Flanked Drawing Canvas Row: Left [+] | Scroll Container | Right [+] */}
        <div className="w-full flex items-start gap-2 sm:gap-3">
          {/* Left Expansion Button */}
          <button
            type="button"
            onClick={handleExpandLeft}
            disabled={isSolving}
            className="h-[300px] w-10 sm:w-14 rounded-xl border-2 border-dashed border-border bg-surface hover:bg-primary-light hover:border-primary/60 text-text-secondary hover:text-primary flex flex-col items-center justify-center transition-all group shrink-0 shadow-sm active:scale-[0.98] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            title="Expand canvas left by 600px (+600px)"
            aria-label="Expand canvas left by 600px"
          >
            <Plus size={22} weight="bold" className="group-hover:scale-125 transition-transform text-primary" />
            <span className="text-[10px] font-bold mt-1 text-primary hidden sm:block">+600</span>
            <span className="text-[9px] text-text-muted uppercase tracking-wider font-semibold hidden sm:block">Left</span>
          </button>

          {/* Horizontally Scrollable Canvas Viewport */}
          <div className="flex-1 min-w-0 shadow-sm rounded-xl overflow-hidden">
            <DrawingCanvas
              ref={canvasRef}
              internalWidth={1200}
              internalHeight={300}
              brushSizeRatio={0.06}
              scrollable={true}
              onWidthChange={(w) => setCanvasWidth(w)}
              onDrawChange={(empty) => setIsCanvasEmpty(empty)}
            />
          </div>

          {/* Right Expansion Button */}
          <button
            type="button"
            onClick={handleExpandRight}
            disabled={isSolving}
            className="h-[300px] w-10 sm:w-14 rounded-xl border-2 border-dashed border-border bg-surface hover:bg-primary-light hover:border-primary/60 text-text-secondary hover:text-primary flex flex-col items-center justify-center transition-all group shrink-0 shadow-sm active:scale-[0.98] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            title="Expand canvas right by 600px (+600px)"
            aria-label="Expand canvas right by 600px"
          >
            <Plus size={22} weight="bold" className="group-hover:scale-125 transition-transform text-primary" />
            <span className="text-[10px] font-bold mt-1 text-primary hidden sm:block">+600</span>
            <span className="text-[9px] text-text-muted uppercase tracking-wider font-semibold hidden sm:block">Right</span>
          </button>
        </div>

        {/* Action Buttons — Clear and Solve */}
        <div className="flex gap-4 w-full mt-2">
          <button
            className="btn btn-outline btn-error flex-1"
            onClick={handleClear}
            disabled={(isCanvasEmpty && !result && !error) || isSolving}
          >
            <Eraser size={20} />
            Clear
          </button>
          <button
            className="btn btn-primary flex-1 gap-2 font-semibold text-base shadow-sm"
            onClick={handleSolve}
            disabled={isCanvasEmpty || isSolving}
          >
            {isSolving ? (
              <span className="loading loading-spinner loading-sm"></span>
            ) : (
              <Equals size={20} weight="bold" />
            )}
            {isSolving ? 'Solving...' : 'Solve'}
          </button>
        </div>
      </div>

      {/* Solver Result Area */}
      <div className="bg-surface rounded-xl border border-border p-5 md:p-6 shadow-sm flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-text-primary flex items-center gap-2">
            <span>Calculation Result</span>
            {result?.status === 'success' && (
              <span className="badge badge-sm badge-success text-white font-medium">
                Solved
              </span>
            )}
          </h2>

          {result?.status === 'success' && (
            <button
              onClick={handleCopy}
              className="btn btn-sm btn-ghost gap-1.5 text-text-secondary hover:text-primary transition-colors"
              title="Copy result"
            >
              {copied ? (
                <>
                  <Check size={16} weight="bold" className="text-success" />
                  <span className="text-success text-xs">Copied</span>
                </>
              ) : (
                <>
                  <Copy size={16} />
                  <span className="text-xs">Copy Equation</span>
                </>
              )}
            </button>
          )}
        </div>

        {/* Dynamic Result State */}
        {isSolving ? (
          <div className="py-8 flex flex-col items-center justify-center gap-3 text-text-secondary">
            <span className="loading loading-spinner loading-md text-primary"></span>
            <span className="text-sm font-medium">Segmenting characters & solving arithmetic...</span>
          </div>
        ) : error ? (
          <div className="p-4 rounded-lg bg-error/10 border border-error/20 flex items-center gap-3 text-error text-sm">
            <XCircle size={20} weight="bold" className="shrink-0" />
            <span>{error}</span>
          </div>
        ) : result?.status === 'success' ? (
          <div className="flex flex-col gap-4">
            {/* Prominent equation & answer card */}
            <div className="p-5 md:p-7 bg-base-200/50 rounded-2xl border border-border/80 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex flex-col gap-1">
                <span className="text-xs uppercase tracking-wider font-semibold text-text-muted">
                  Recognized Expression
                </span>
                <div className="font-mono font-bold text-2xl md:text-3xl text-text-primary tracking-tight">
                  {result.expression}
                </div>
              </div>

              <div className="flex items-center gap-3 self-start md:self-auto bg-surface px-5 py-3 rounded-xl border border-border shadow-xs">
                <span className="font-mono text-xl font-bold text-text-muted">=</span>
                <span className="font-mono text-3xl md:text-4xl font-extrabold text-primary">
                  {result.formatted_result}
                </span>
              </div>
            </div>

            {/* Plain equation text */}
            <div className="flex items-center gap-2 text-xs text-text-muted bg-base-100 p-2.5 rounded-lg border border-border">
              <span className="font-semibold text-text-secondary shrink-0">Full Equation:</span>
              <input
                type="text"
                readOnly
                value={result.full_equation || ''}
                className="w-full bg-transparent font-mono text-sm text-text-primary outline-none select-all font-semibold"
              />
            </div>
          </div>
        ) : result?.status === 'uncertain' ? (
          <div className="flex flex-col gap-3">
            <div className="p-4 rounded-lg bg-warning/10 border border-warning/30 flex items-start gap-3 text-warning-content">
              <Warning size={20} weight="fill" className="shrink-0 text-warning mt-0.5" />
              <div className="flex flex-col gap-1 text-sm">
                <span className="font-semibold text-text-primary">
                  Evaluation Blocked — Uncertain Character(s) Detected
                </span>
                <p className="text-xs text-text-secondary">
                  One or more characters could not be recognized with majority consensus (all 3 models disagreed). {SHOW_SEGMENTATION_PREVIEW ? 'Please inspect the segmentation preview below and redraw the uncertain characters marked with ' : 'Please redraw the uncertain characters marked with '}<strong>[?]</strong>.
                </p>
              </div>
            </div>

            {/* Character chips */}
            <div className="p-4 bg-base-200/50 rounded-xl border border-border flex flex-wrap items-center gap-2 min-h-[64px]">
              {result.segments?.map((seg, idx) => (
                <span key={idx} className="inline-flex items-center">
                  {seg.is_uncertain ? (
                    <span
                      className="inline-flex items-center justify-center px-2.5 py-1 rounded-md bg-warning/20 border border-warning/50 text-warning-content font-mono font-bold text-lg shadow-xs"
                      title="Uncertain character — all 3 models disagreed"
                    >
                      [?]
                    </span>
                  ) : (
                    <span className="font-mono font-bold text-xl text-text-primary px-1">
                      {seg.character}
                    </span>
                  )}
                  {seg.has_space_after && <span className="inline-block w-3" />}
                </span>
              ))}
            </div>
          </div>
        ) : result?.status === 'division_by_zero' ? (
          <div className="p-4 rounded-xl bg-error/10 border border-error/30 flex items-start gap-3 text-error">
            <XCircle size={20} weight="bold" className="shrink-0 mt-0.5" />
            <div className="flex flex-col gap-1 text-sm">
              <span className="font-semibold">Math Error: Division by Zero</span>
              <p className="text-xs text-text-secondary">
                {result.error_message || 'Division or modulo by zero is undefined in mathematics.'}
              </p>
              {result.expression && (
                <span className="font-mono text-xs mt-1 bg-surface px-2 py-0.5 rounded border border-border self-start text-text-primary">
                  Expression: {result.expression}
                </span>
              )}
            </div>
          </div>
        ) : result?.status === 'unsupported_character' ? (
          <div className="p-4 rounded-xl bg-warning/10 border border-warning/30 flex items-start gap-3 text-warning-content">
            <Warning size={20} weight="fill" className="shrink-0 text-warning mt-0.5" />
            <div className="flex flex-col gap-1 text-sm">
              <span className="font-semibold text-text-primary">Unsupported Character</span>
              <p className="text-xs text-text-secondary">
                {result.error_message || 'V4 Math Solver only supports digits 0–9 and +, -, *, ÷, %.'}
              </p>
            </div>
          </div>
        ) : result?.status === 'invalid_expression' ? (
          <div className="p-4 rounded-xl bg-error/10 border border-error/30 flex items-start gap-3 text-error">
            <XCircle size={20} weight="bold" className="shrink-0 mt-0.5" />
            <div className="flex flex-col gap-1 text-sm">
              <span className="font-semibold">Invalid Mathematical Syntax</span>
              <p className="text-xs text-text-secondary">
                {result.error_message || 'Check for missing operands, trailing operators, or malformed sequences.'}
              </p>
              {result.expression && (
                <span className="font-mono text-xs mt-1 bg-surface px-2 py-0.5 rounded border border-border self-start text-text-primary">
                  Expression: {result.expression}
                </span>
              )}
            </div>
          </div>
        ) : result?.status === 'no_characters' || result?.status === 'empty' ? (
          <div className="py-6 text-center text-text-secondary text-sm">
            {result.message}
          </div>
        ) : (
          <div className="py-8 text-center text-text-muted text-sm border border-dashed border-border rounded-xl">
            Draw a single-line math expression above and click <strong>Solve</strong> to calculate the result.
          </div>
        )}
      </div>

      {/* Segmentation Preview Debugging Section */}
      {SHOW_SEGMENTATION_PREVIEW && result && result.segments && result.segments.length > 0 && (
        <div className="bg-surface rounded-xl border border-border p-5 md:p-6 shadow-sm flex flex-col gap-4 animate-in fade-in transition-all">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
            <div className="flex items-center gap-2">
              <SquaresFour size={20} weight="duotone" className="text-primary" />
              <div>
                <h3 className="text-sm font-semibold text-text-primary">Segmentation Preview</h3>
                <p className="text-xs text-text-muted">
                  Inspect raw segmented character crops and exact CNN model inputs
                </p>
              </div>
            </div>

            {/* Resolution selector for model input visualization */}
            <div className="flex items-center gap-2 self-start sm:self-auto">
              <span className="text-xs text-text-muted font-medium">Model Input:</span>
              <div className="join border border-border rounded-lg bg-base-200/50 p-0.5">
                {(['32', '64', '128'] as const).map((res) => (
                  <button
                    key={res}
                    type="button"
                    onClick={() => setSelectedResolution(res)}
                    className={`btn btn-xs join-item font-mono transition-colors ${
                      selectedResolution === res
                        ? 'btn-primary font-semibold'
                        : 'btn-ghost text-text-secondary hover:text-primary'
                    }`}
                  >
                    {res}×{res}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Cards for each character segment */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {result.segments.map((seg) => {
              const modelInput = seg.model_inputs?.[selectedResolution];

              return (
                <div
                  key={seg.index}
                  className={`flex flex-col p-2.5 rounded-xl border bg-base-100 transition-all ${
                    seg.is_uncertain
                      ? 'border-warning/60 ring-2 ring-warning/20'
                      : 'border-border hover:border-primary/40'
                  }`}
                >
                  {/* Segment header: index & prediction badge */}
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[11px] font-mono text-text-muted font-medium">
                      #{seg.index + 1}
                    </span>
                    {seg.is_uncertain ? (
                      <span className="badge badge-xs badge-warning font-mono font-bold px-1.5 py-0.5">
                        [?]
                      </span>
                    ) : (
                      <span className="badge badge-xs badge-primary font-mono font-bold px-1.5 py-0.5">
                        '{seg.character}'
                      </span>
                    )}
                  </div>

                  {/* Side-by-side or stacked preview: Crop vs Model Input */}
                  <div className="grid grid-cols-2 gap-1.5 bg-black/90 p-1.5 rounded-lg border border-border/80">
                    {/* Raw Segmented Crop */}
                    <div className="flex flex-col items-center gap-1">
                      <span className="text-[9px] text-white/60 font-sans tracking-wider uppercase font-semibold">
                        Crop
                      </span>
                      <div className="w-14 h-14 bg-black rounded flex items-center justify-center overflow-hidden border border-white/10">
                        {seg.crop_image ? (
                          <img
                            src={seg.crop_image}
                            alt={`Segment ${seg.index + 1}`}
                            className="max-w-full max-h-full object-contain"
                            style={{ imageRendering: 'pixelated' }}
                          />
                        ) : (
                          <span className="text-[10px] text-white/40">N/A</span>
                        )}
                      </div>
                      <span className="text-[9px] text-white/50 font-mono">
                        {seg.crop_dimensions
                          ? `${seg.crop_dimensions.width}×${seg.crop_dimensions.height}`
                          : '—'}
                      </span>
                    </div>

                    {/* Preprocessed Model Input Tensor Image */}
                    <div className="flex flex-col items-center gap-1">
                      <span className="text-[9px] text-primary/80 font-sans tracking-wider uppercase font-semibold">
                        Model
                      </span>
                      <div className="w-14 h-14 bg-black rounded flex items-center justify-center overflow-hidden border border-primary/30">
                        {modelInput?.image ? (
                          <img
                            src={modelInput.image}
                            alt={`Model Input ${selectedResolution}×${selectedResolution}`}
                            className="w-full h-full object-contain"
                            style={{ imageRendering: 'pixelated' }}
                          />
                        ) : (
                          <span className="text-[10px] text-white/40">N/A</span>
                        )}
                      </div>
                      <span className="text-[9px] text-primary/70 font-mono">
                        {selectedResolution}×{selectedResolution}
                      </span>
                    </div>
                  </div>

                  {/* Segment Bounding Box Info */}
                  <div className="mt-2 pt-1.5 border-t border-border/60 flex items-center justify-between text-[10px] font-mono text-text-muted">
                    <span>
                      x:{seg.bbox.x} y:{seg.bbox.y}
                    </span>
                    {seg.has_space_after ? (
                      <span className="text-primary font-semibold text-[9px] bg-primary/10 px-1 rounded">
                        Space
                      </span>
                    ) : (
                      <span className="text-text-muted/60 text-[9px]">No space</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
