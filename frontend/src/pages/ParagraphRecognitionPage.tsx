/**
 * GuessMyText — V3 Paragraph Recognition Page
 *
 * Implements handwritten paragraph recognition across multiple lines:
 *   1. 2D scrollable expandable canvas (initial 1200×600 px) with 4-way expansion controls.
 *   2. Top-to-bottom line detection and independent per-line word boundary estimation.
 *   3. 3-CNN majority voting for each character across all detected lines.
 *   4. Reconstructs full paragraph preserving line breaks (\n).
 *   5. Comprehensive Segmentation Preview grouped by line, showing exact crops and model-input tensors.
 */

import { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  Article,
  Eraser,
  MagnifyingGlass,
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
  Crosshair,
  Faders,
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
  global_index: number;
  line_index: number;
  char_index: number;
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

interface LineOutput {
  line_index: number;
  text: string;
  num_characters: number;
  num_words: number;
  has_uncertain: boolean;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  segments: CharacterSegment[];
}

interface DiagnosticLineRegion {
  line_index: number;
  y_start: number;
  y_end: number;
  height: number;
  num_characters: number;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

interface SegmentationDiagnostics {
  canvas_dimensions: { width: number; height: number };
  num_lines: number;
  total_characters: number;
  line_regions: DiagnosticLineRegion[];
  thresholds: {
    binarization: number;
    max_canvas_dim: number;
    max_canvas_pixels: number;
  };
}

interface ParagraphPredictionResponse {
  status: 'success' | 'empty' | 'no_characters' | 'error';
  text?: string;
  lines?: LineOutput[];
  num_lines?: number;
  num_characters?: number;
  num_words?: number;
  has_uncertain?: boolean;
  annotated_canvas?: string;
  diagnostics?: SegmentationDiagnostics;
  message?: string;
  error?: string;
}

export function ParagraphRecognitionPage() {
  const [isCanvasEmpty, setIsCanvasEmpty] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [canvasDimensions, setCanvasDimensions] = useState({ width: 1200, height: 600 });
  const [result, setResult] = useState<ParagraphPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [selectedResolution, setSelectedResolution] = useState<'32' | '64' | '128'>('64');
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  // Dismissible handwriting guidance banner state (persists across sessions)
  const [showGuidance, setShowGuidance] = useState(() => {
    try {
      return localStorage.getItem('gmt_v3_dismiss_guidance_v2') !== 'true';
    } catch {
      return true;
    }
  });

  const canvasRef = useRef<DrawingCanvasRef>(null);

  const handleDismissGuidance = () => {
    setShowGuidance(false);
    try {
      localStorage.setItem('gmt_v3_dismiss_guidance_v2', 'true');
    } catch {
      // Ignore local storage errors
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

  const handleExpandTop = () => {
    canvasRef.current?.expandTop(600);
  };

  const handleExpandBottom = () => {
    canvasRef.current?.expandBottom(600);
  };

  const handleResetDimensions = () => {
    canvasRef.current?.resetDimensions(1200, 600);
  };

  const handleScrollToOrigin = () => {
    canvasRef.current?.scrollToOrigin();
  };

  const handlePredict = async () => {
    if (!canvasRef.current || isPredicting) return;

    if (isCanvasEmpty || canvasRef.current.isEmpty()) {
      setIsCanvasEmpty(true);
      toast.error('Please write a paragraph before predicting.');
      return;
    }

    const blob = await canvasRef.current.exportToBlob();
    if (!blob) return;

    setIsPredicting(true);
    setError(null);
    setResult(null);
    setCopied(false);

    try {
      const formData = new FormData();
      formData.append('image', blob, 'paragraph_drawing.png');

      const response = await fetch(apiUrl('/api/predict-paragraph'), {
        method: 'POST',
        body: formData,
      });

      const data: ParagraphPredictionResponse = await response.json();

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
        if (data.has_uncertain) {
          toast(`Recognized ${data.num_lines} lines with some uncertain characters.`, { icon: '⚠️' });
        } else {
          toast.success(`Recognized ${data.num_lines} lines (${data.num_characters} characters)!`);
        }
      }
    } catch (err) {
      const msg =
        err instanceof Error
          ? `Connection error: ${err.message}`
          : 'Failed to connect to the prediction server.';
      setError(msg);
      toast.error(msg);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleCopy = async () => {
    if (!result?.text) return;
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
      toast.success('Copied paragraph to clipboard!');
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error('Failed to copy text.');
    }
  };

  const isExpanded =
    canvasDimensions.width > 1200 || canvasDimensions.height > 600;

  return (
    <div className="max-w-5xl mx-auto flex flex-col gap-6">
      {/* Page Header */}
      <div className="flex items-center gap-3">
        <Article size={28} weight="duotone" className="text-primary" />
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">Paragraph Recognition</h1>
          <p className="text-xs text-text-muted mt-0.5">
            Write multi-line handwritten text with 2D scrolling and four-way expansion.
          </p>
        </div>
      </div>

      {/* Dismissible Notice Banner (exact wording per requirement) */}
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
                  GuessMyText recognizes uppercase letters (A–Z) and digits (0–9) across multiple lines.
                </p>
                <p>
                  Write characters clearly, leave sufficient space between words, and leave clear vertical space between lines. Connected or overlapping handwriting and touching lines can be harder to segment.
                </p>
                <p className="text-text-muted">
                  Lowercase letters and mathematical operators are not supported.
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

          {/* Center: Recognition Status / Feedback */}
          <div className="flex items-center justify-center text-center">
            {isPredicting ? (
              <div className="flex items-center gap-2 text-sm font-medium text-text-secondary">
                <span className="loading loading-spinner loading-xs text-primary"></span>
                <span>Segmenting lines & recognizing paragraph...</span>
              </div>
            ) : error ? (
              <div className="flex items-center gap-1.5 text-error text-xs md:text-sm font-medium text-center">
                <XCircle size={18} weight="bold" className="shrink-0 text-error" />
                <span className="leading-snug truncate max-w-xs md:max-w-md">{error}</span>
              </div>
            ) : result?.status === 'success' ? (
              <div className="flex items-center gap-2 text-xs md:text-sm font-medium text-text-secondary">
                <CheckCircle size={18} weight="fill" className="text-success shrink-0" />
                <span>
                  {result.num_lines} {result.num_lines === 1 ? 'line' : 'lines'}
                  {' · '}
                  {result.num_words} {result.num_words === 1 ? 'word' : 'words'}
                  {' · '}
                  {result.num_characters} {result.num_characters === 1 ? 'character' : 'characters'}
                  {result.has_uncertain && ' (with uncertain positions)'}
                </span>
              </div>
            ) : result?.status === 'no_characters' || result?.status === 'empty' ? (
              <div className="flex items-center gap-1.5 text-warning text-xs md:text-sm font-medium">
                <Warning size={18} weight="fill" className="shrink-0 text-warning" />
                <span>{result.message}</span>
              </div>
            ) : (
              <span className="text-xs text-text-muted hidden md:inline">
                Write multi-line handwritten text inside the canvas
              </span>
            )}
          </div>

          {/* Right: Ready indicator */}
          <div className="flex items-center justify-end">
            {isPredicting ? (
              <div className="flex items-center gap-2 text-sm font-medium text-primary">
                <span className="loading loading-spinner loading-xs text-primary"></span>
                <span>Predicting...</span>
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

      {/* Canvas Area with Four-Way Expansion Controls outside drawing surface */}
      <div className="flex flex-col gap-2.5 w-full">
        {/* Canvas Toolbar: Dimensions & Quick Scroll Controls */}
        <div className="w-full flex flex-wrap items-center justify-between text-xs text-text-muted px-1 gap-2">
          <div className="flex items-center gap-2">
            <span className="font-medium text-text-secondary">Canvas:</span>
            <span className="font-mono bg-base-200 px-2.5 py-0.5 rounded-md text-text-primary border border-border font-semibold">
              {canvasDimensions.width} × {canvasDimensions.height} px
            </span>
            {isExpanded && (
              <button
                id="btn-reset-dimensions"
                type="button"
                onClick={handleResetDimensions}
                disabled={isPredicting}
                className="btn btn-xs btn-ghost text-primary hover:bg-primary/10 gap-1 font-medium transition-colors"
                title="Reset canvas dimensions to initial 1200 × 600 px"
              >
                <ArrowCounterClockwise size={12} weight="bold" />
                <span>Reset to 1200 × 600</span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[11px] text-text-muted hidden sm:inline">
              +600px per expansion control
            </span>
            <button
              id="btn-scroll-origin"
              type="button"
              onClick={handleScrollToOrigin}
              className="btn btn-xs btn-ghost border border-border bg-base-200/50 hover:bg-base-200 text-text-secondary hover:text-primary gap-1"
              title="Scroll viewport to top-left origin (0, 0)"
            >
              <Crosshair size={12} weight="bold" />
              <span className="text-[11px]">Scroll to Top-Left</span>
            </button>
          </div>
        </div>

        {/* 4-Way Expansion Layout: Top [+] Bar | [Left [+] | Viewport | Right [+]] | Bottom [+] Bar */}
        <div className="flex flex-col gap-2 w-full">
          {/* Top Expansion Button */}
          <button
            id="btn-expand-top"
            type="button"
            onClick={handleExpandTop}
            disabled={isPredicting}
            className="w-full h-8 rounded-lg border-2 border-dashed border-border bg-surface hover:bg-primary-light hover:border-primary/60 text-text-secondary hover:text-primary flex items-center justify-center gap-2 transition-all group shrink-0 shadow-xs active:scale-[0.99] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            title="Add space above canvas (+600px top)"
            aria-label="Add space above canvas"
          >
            <Plus size={16} weight="bold" className="group-hover:scale-125 transition-transform text-primary" />
            <span className="text-xs font-semibold text-text-secondary group-hover:text-primary">
              +600px Above
            </span>
          </button>

          {/* Middle Row: Left [+] | Canvas Viewport | Right [+] */}
          <div className="w-full flex items-start gap-2">
            {/* Left Expansion Button */}
            <button
              id="btn-expand-left"
              type="button"
              onClick={handleExpandLeft}
              disabled={isPredicting}
              className="h-[480px] w-10 sm:w-12 rounded-xl border-2 border-dashed border-border bg-surface hover:bg-primary-light hover:border-primary/60 text-text-secondary hover:text-primary flex flex-col items-center justify-center transition-all group shrink-0 shadow-sm active:scale-[0.98] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              title="Add space to the left and shift strokes right (+600px left)"
              aria-label="Add space to the left"
            >
              <Plus size={20} weight="bold" className="group-hover:scale-125 transition-transform text-primary" />
              <span className="text-[10px] font-bold mt-1 text-primary hidden sm:block">+600</span>
              <span className="text-[9px] text-text-muted uppercase tracking-wider font-semibold hidden sm:block">Left</span>
            </button>

            {/* Viewport Canvas */}
            <div className="flex-1 min-w-0 shadow-sm rounded-xl overflow-hidden">
              <DrawingCanvas
                ref={canvasRef}
                internalWidth={1200}
                internalHeight={600}
                brushSizePx={18}
                scrollable="2d"
                onDimensionsChange={(w, h) => setCanvasDimensions({ width: w, height: h })}
                onDrawChange={(empty) => setIsCanvasEmpty(empty)}
              />
            </div>

            {/* Right Expansion Button */}
            <button
              id="btn-expand-right"
              type="button"
              onClick={handleExpandRight}
              disabled={isPredicting}
              className="h-[480px] w-10 sm:w-12 rounded-xl border-2 border-dashed border-border bg-surface hover:bg-primary-light hover:border-primary/60 text-text-secondary hover:text-primary flex flex-col items-center justify-center transition-all group shrink-0 shadow-sm active:scale-[0.98] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              title="Add space to the right (+600px right)"
              aria-label="Add space to the right"
            >
              <Plus size={20} weight="bold" className="group-hover:scale-125 transition-transform text-primary" />
              <span className="text-[10px] font-bold mt-1 text-primary hidden sm:block">+600</span>
              <span className="text-[9px] text-text-muted uppercase tracking-wider font-semibold hidden sm:block">Right</span>
            </button>
          </div>

          {/* Bottom Expansion Button */}
          <button
            id="btn-expand-bottom"
            type="button"
            onClick={handleExpandBottom}
            disabled={isPredicting}
            className="w-full h-8 rounded-lg border-2 border-dashed border-border bg-surface hover:bg-primary-light hover:border-primary/60 text-text-secondary hover:text-primary flex items-center justify-center gap-2 transition-all group shrink-0 shadow-xs active:scale-[0.99] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            title="Add space below canvas (+600px bottom)"
            aria-label="Add space below canvas"
          >
            <Plus size={16} weight="bold" className="group-hover:scale-125 transition-transform text-primary" />
            <span className="text-xs font-semibold text-text-secondary group-hover:text-primary">
              +600px Below
            </span>
          </button>
        </div>

        {/* Action Buttons — Clear and Predict */}
        <div className="flex gap-4 w-full mt-2">
          <button
            id="btn-clear-paragraph"
            className="btn btn-outline btn-error flex-1"
            onClick={handleClear}
            disabled={(isCanvasEmpty && !result && !error) || isPredicting}
          >
            <Eraser size={20} />
            Clear
          </button>
          <button
            id="btn-predict-paragraph"
            className="btn btn-primary flex-1"
            onClick={handlePredict}
            disabled={isCanvasEmpty || isPredicting}
          >
            {isPredicting ? (
              <span className="loading loading-spinner loading-sm"></span>
            ) : (
              <MagnifyingGlass size={20} weight="bold" />
            )}
            {isPredicting ? 'Recognizing...' : 'Predict'}
          </button>
        </div>
      </div>

      {/* Recognition Result Area */}
      <div className="bg-surface rounded-xl border border-border p-5 md:p-6 shadow-sm flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-text-primary flex items-center gap-2">
            <span>Recognized Paragraph</span>
            {result?.status === 'success' && (
              <span className="badge badge-sm badge-outline text-text-muted font-normal">
                {result.num_lines} lines · {result.num_words} words · {result.num_characters} chars
              </span>
            )}
          </h2>

          {result?.status === 'success' && result.text && (
            <button
              id="btn-copy-paragraph"
              onClick={handleCopy}
              className="btn btn-sm btn-ghost gap-1.5 text-text-secondary hover:text-primary transition-colors"
              title="Copy recognized paragraph"
            >
              {copied ? (
                <>
                  <Check size={16} weight="bold" className="text-success" />
                  <span className="text-success text-xs">Copied</span>
                </>
              ) : (
                <>
                  <Copy size={16} />
                  <span className="text-xs">Copy Text</span>
                </>
              )}
            </button>
          )}
        </div>

        {/* Dynamic Result State */}
        {isPredicting ? (
          <div className="py-8 flex flex-col items-center justify-center gap-3 text-text-secondary">
            <span className="loading loading-spinner loading-md text-primary"></span>
            <span className="text-sm font-medium">Detecting lines, segmenting characters & running CNN ensemble...</span>
          </div>
        ) : error ? (
          <div className="p-4 rounded-lg bg-error/10 border border-error/20 flex items-center gap-3 text-error text-sm">
            <XCircle size={20} weight="bold" className="shrink-0" />
            <span>{error}</span>
          </div>
        ) : result?.status === 'success' ? (
          <div className="flex flex-col gap-4">
            {/* Visual multi-line display with line breaks preserved */}
            <div className="p-4 md:p-6 bg-base-200/50 rounded-xl border border-border/80 flex flex-col gap-4 min-h-[96px]">
              {result.lines && result.lines.length > 0 ? (
                result.lines.map((line) => (
                  <div key={line.line_index} className="flex flex-wrap items-center gap-y-2 gap-x-1 border-b border-border/40 pb-2.5 last:border-b-0 last:pb-0">
                    <span className="text-[10px] font-mono text-text-muted mr-2 shrink-0 select-none bg-base-100 px-1.5 py-0.5 rounded border border-border/60">
                      L{line.line_index + 1}
                    </span>
                    {line.segments.map((seg) => (
                      <span key={seg.global_index} className="inline-flex items-center">
                        {seg.is_uncertain ? (
                          <span
                            className="inline-flex items-center justify-center px-1.5 py-0.5 rounded bg-warning/20 border border-warning/50 text-warning-content font-mono font-bold text-lg shadow-xs"
                            title="Uncertain character — all 3 models disagreed"
                          >
                            [?]
                          </span>
                        ) : (
                          <span className="font-mono font-bold text-2xl text-text-primary px-0.5 tracking-tight">
                            {seg.character}
                          </span>
                        )}
                        {seg.has_space_after && <span className="inline-block w-3 md:w-5" />}
                      </span>
                    ))}
                  </div>
                ))
              ) : (
                <pre className="font-mono text-xl text-text-primary whitespace-pre-wrap">
                  {result.text}
                </pre>
              )}
            </div>

            {/* Plain selectable textarea for easy copy-paste */}
            <div className="flex flex-col gap-1.5 text-xs text-text-muted bg-base-100 p-3 rounded-lg border border-border">
              <span className="font-semibold text-text-secondary">Plain text (with preserved line breaks):</span>
              <textarea
                id="paragraph-text-plain"
                readOnly
                rows={Math.max(2, Math.min(8, (result.num_lines || 1) + 1))}
                value={result.text}
                className="w-full bg-transparent font-mono text-sm text-text-primary outline-none select-all resize-y"
              />
            </div>

            {/* Consensus & Guidance Note */}
            {result.has_uncertain ? (
              <div className="flex items-start gap-2 p-3 rounded-lg bg-warning/10 border border-warning/20 text-warning-content text-xs">
                <Warning size={16} weight="fill" className="shrink-0 mt-0.5 text-warning" />
                <span>
                  Some character positions are marked as <strong>[?]</strong> because all 3 CNN
                  models predicted different characters (no consensus). Try redrawing those
                  characters more clearly with separated strokes.
                </span>
              </div>
            ) : SHOW_SEGMENTATION_PREVIEW ? (
              <div className="flex items-center gap-2 text-xs text-text-muted">
                <CheckCircle size={16} weight="fill" className="text-success shrink-0" />
                <span>All characters recognized with majority or unanimous consensus across all lines.</span>
              </div>
            ) : null}
          </div>
        ) : result?.status === 'no_characters' || result?.status === 'empty' ? (
          <div className="py-6 text-center text-text-secondary text-sm">
            {result.message}
          </div>
        ) : (
          <div className="py-8 text-center text-text-muted text-sm border border-dashed border-border rounded-xl">
            Write a multi-line paragraph above and click <strong>Predict</strong> to recognize.
          </div>
        )}
      </div>

      {/* Segmentation Preview Debugging Section */}
      {SHOW_SEGMENTATION_PREVIEW && result?.status === 'success' && result.lines && result.lines.length > 0 && (
        <div className="bg-surface rounded-xl border border-border p-5 md:p-6 shadow-sm flex flex-col gap-5 animate-in fade-in transition-all">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
            <div>
              <h2 className="text-base font-semibold text-text-primary flex items-center gap-2">
                <SquaresFour size={20} weight="duotone" className="text-primary" />
                <span>Segmentation Preview</span>
              </h2>
              <p className="text-xs text-text-muted mt-0.5">
                Inspect detected line regions, segmented character crops, and exact binary tensors fed to the CNN models.
              </p>
            </div>

            {/* Resolution Selector for Model Input Image */}
            <div className="flex flex-wrap items-center gap-2 self-start sm:self-auto">
              <button
                type="button"
                onClick={() => setShowDiagnostics((prev) => !prev)}
                className={`px-2.5 py-1 text-xs font-medium rounded-lg border transition-all flex items-center gap-1.5 ${
                  showDiagnostics
                    ? 'bg-base-200 border-primary/40 text-primary font-semibold'
                    : 'border-border text-text-muted hover:text-text-primary hover:bg-base-100'
                }`}
                title="Toggle development diagnostic view"
              >
                <Faders size={14} weight="duotone" />
                <span>{showDiagnostics ? 'Hide Diagnostics' : 'Diagnostics'}</span>
              </button>

              <div className="flex items-center gap-1.5 bg-base-200/80 p-1 rounded-lg border border-border">
                <span className="text-[11px] font-medium text-text-muted px-2 hidden md:inline">Model Resolution:</span>
                {(['32', '64', '128'] as const).map((res) => (
                  <button
                    key={res}
                    type="button"
                    onClick={() => setSelectedResolution(res)}
                    className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${
                      selectedResolution === res
                        ? 'bg-primary text-white shadow-xs'
                        : 'text-text-secondary hover:text-text-primary hover:bg-base-100'
                    }`}
                    title={`View model input at ${res}×${res} resolution`}
                  >
                    {res}×{res}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Full Canvas Line & Character Segmentation Overview */}
          {result.annotated_canvas && (
            <div className="flex flex-col gap-2 p-4 bg-base-200/40 rounded-xl border border-border/80">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-text-primary">Canvas Segmentation Overview</span>
                  <span className="text-[11px] text-text-muted font-mono bg-base-100 px-1.5 py-0.5 rounded border border-border">
                    {result.diagnostics?.canvas_dimensions.width ?? canvasDimensions.width} × {result.diagnostics?.canvas_dimensions.height ?? canvasDimensions.height} px
                  </span>
                </div>
                <div className="flex items-center gap-3 text-[11px] text-text-muted font-medium">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-xs bg-[#50c878] inline-block border border-black/30" />
                    <span>Detected Lines</span>
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-xs bg-[#00beff] inline-block border border-black/30" />
                    <span>Character Crops</span>
                  </span>
                </div>
              </div>

              <div className="w-full bg-black rounded-lg border border-border overflow-hidden flex items-center justify-center p-2 min-h-[140px] max-h-[380px]">
                <img
                  src={result.annotated_canvas}
                  alt="Annotated canvas showing detected lines and character bounding boxes"
                  className="max-h-[360px] w-auto max-w-full object-contain"
                  style={{ imageRendering: 'auto' }}
                />
              </div>
            </div>
          )}

          {/* Development Diagnostic View */}
          {showDiagnostics && result.diagnostics && (
            <div className="p-4 bg-base-100 rounded-xl border border-primary/30 flex flex-col gap-3 text-xs animate-in fade-in transition-all">
              <div className="flex items-center justify-between border-b border-border pb-2">
                <span className="font-semibold text-primary flex items-center gap-1.5">
                  <Faders size={15} weight="duotone" />
                  <span>Development Diagnostics (Line Boundaries & Thresholds)</span>
                </span>
                <span className="text-[10px] text-text-muted font-mono">
                  Binarization: {result.diagnostics.thresholds.binarization} · Limits: {result.diagnostics.thresholds.max_canvas_dim}px
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {result.diagnostics.line_regions.map((lr) => (
                  <div key={lr.line_index} className="p-2.5 bg-base-200/50 rounded-lg border border-border/70 flex flex-col gap-1 font-mono text-[11px]">
                    <div className="flex items-center justify-between font-bold text-text-primary">
                      <span>Line {lr.line_index + 1}</span>
                      <span className="text-text-muted">{lr.num_characters} chars</span>
                    </div>
                    <div className="text-text-secondary text-[10px]">
                      y-span: [{lr.y_start} → {lr.y_end}] (height: {lr.height}px)
                    </div>
                    <div className="text-text-muted text-[10px]">
                      bbox: x:{lr.bbox.x}, y:{lr.bbox.y}, {lr.bbox.width}×{lr.bbox.height}px
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Line-by-Line Character Cards */}
          <div className="flex flex-col gap-6">
            {result.lines.map((line) => (
              <div key={line.line_index} className="flex flex-col gap-3 p-4 bg-base-200/30 rounded-xl border border-border/80">
                {/* Line Region Metadata Header */}
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-text-primary font-mono bg-base-200 px-2 py-0.5 rounded border border-border">
                      Line {line.line_index + 1}
                    </span>
                    <span className="text-text-muted">
                      Region: x:{line.bbox.x}, y:{line.bbox.y}, {line.bbox.width}×{line.bbox.height}px
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-text-muted font-mono text-[11px]">
                    <span>{line.num_characters} characters</span>
                    <span>·</span>
                    <span>{line.num_words} words</span>
                  </div>
                </div>

                {/* Character Previews Grid for this line */}
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
                  {line.segments.map((seg) => {
                    const cropW = seg.crop_dimensions?.width ?? seg.bbox.width;
                    const cropH = seg.crop_dimensions?.height ?? seg.bbox.height;
                    const modelInput = seg.model_inputs?.[selectedResolution];

                    return (
                      <div
                        key={seg.global_index}
                        className={`p-3 rounded-xl border transition-all flex flex-col gap-2.5 shadow-xs ${seg.is_uncertain
                            ? 'bg-warning/5 border-warning/40'
                            : 'bg-base-100/70 border-border hover:border-primary/40'
                          }`}
                      >
                        {/* Header: Line/Char index and character badge */}
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-semibold text-text-secondary uppercase tracking-wider">
                            L{seg.line_index + 1} : C{seg.char_index + 1}
                          </span>
                          {seg.is_uncertain ? (
                            <span className="badge badge-xs badge-warning font-mono font-bold" title="All 3 models differed">
                              [?] Uncertain
                            </span>
                          ) : (
                            <span className="badge badge-xs badge-primary text-white font-mono font-bold">
                              '{seg.character}'
                            </span>
                          )}
                        </div>

                        {/* Side-by-Side Images: Crop vs Model Input */}
                        <div className="grid grid-cols-2 gap-2 text-center">
                          {/* Segmented Crop */}
                          <div className="flex flex-col items-center gap-1">
                            <span className="text-[10px] uppercase font-bold text-text-muted tracking-tight truncate w-full" title="Segmented Crop">
                              Crop
                            </span>
                            <div
                              className="w-full aspect-square bg-black rounded-lg border border-border flex items-center justify-center p-1 overflow-hidden"
                              title={`Segmented Crop: ${cropW}×${cropH} px`}
                            >
                              {seg.crop_image ? (
                                <img
                                  src={seg.crop_image}
                                  alt={`Crop for line ${seg.line_index + 1} char ${seg.char_index + 1}`}
                                  className="w-full h-full object-contain"
                                  style={{ imageRendering: 'pixelated' }}
                                />
                              ) : (
                                <span className="text-[10px] text-text-muted">No crop</span>
                              )}
                            </div>
                            <span className="text-[10px] font-mono text-text-muted">
                              {cropW}×{cropH}
                            </span>
                          </div>

                          {/* Model Input */}
                          <div className="flex flex-col items-center gap-1">
                            <span className="text-[10px] uppercase font-bold text-primary tracking-tight truncate w-full" title="Model Input Tensor">
                              Model Input
                            </span>
                            <div
                              className="w-full aspect-square bg-black rounded-lg border border-primary/30 flex items-center justify-center p-1 overflow-hidden shadow-inner"
                              title={`Preprocessed Tensor Input (${selectedResolution}×${selectedResolution} px)`}
                            >
                              {modelInput?.image ? (
                                <img
                                  src={modelInput.image}
                                  alt={`Model input ${selectedResolution}×${selectedResolution} for line ${seg.line_index + 1} char ${seg.char_index + 1}`}
                                  className="w-full h-full object-contain"
                                  style={{ imageRendering: 'pixelated' }}
                                />
                              ) : (
                                <span className="text-[10px] text-text-muted">No tensor</span>
                              )}
                            </div>
                            <span className="text-[10px] font-mono text-primary font-medium">
                              {selectedResolution}×{selectedResolution}
                            </span>
                          </div>
                        </div>

                        {/* Footer: Position & Space info */}
                        <div className="flex items-center justify-between pt-1.5 border-t border-border/60 text-[10px] text-text-muted">
                          <span title={`Origin at (${seg.bbox.x}, ${seg.bbox.y})`}>
                            x:{seg.bbox.x} y:{seg.bbox.y}
                          </span>
                          {seg.has_space_after ? (
                            <span className="badge badge-xs bg-base-200 border border-border text-text-secondary font-medium" title="Word space follows this character">
                              ␣ Space
                            </span>
                          ) : (
                            <span>No space</span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
