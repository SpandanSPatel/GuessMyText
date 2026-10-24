/**
 * GuessMyText — V2 Line Recognition Page
 *
 * Implements handwritten line recognition using the existing 3 CNN models.
 * Segments characters via Connected Component Analysis + projection valley splitting,
 * estimates word boundaries from inter-character gap distributions,
 * classifies each character using 3-model majority voting, and
 * reconstructs the line while preserving and marking any uncertain positions.
 *
 * Includes a visual Segmentation Preview section below the prediction
 * allowing inspection of raw segmented character crops and the exact
 * preprocessed tensors supplied to the 32×32, 64×64, and 128×128 CNN models.
 */

import { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  TextAlignLeft,
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

interface LinePredictionResponse {
  status: 'success' | 'empty' | 'no_characters' | 'error';
  text?: string;
  segments?: CharacterSegment[];
  num_characters?: number;
  num_words?: number;
  has_uncertain?: boolean;
  message?: string;
  error?: string;
}

export function LineRecognitionPage() {
  const [isCanvasEmpty, setIsCanvasEmpty] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [canvasWidth, setCanvasWidth] = useState(1200);
  const [result, setResult] = useState<LinePredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [selectedResolution, setSelectedResolution] = useState<'32' | '64' | '128'>('64');

  // Dismissible guidance banner state (persists across sessions)
  const [showGuidance, setShowGuidance] = useState(() => {
    try {
      return localStorage.getItem('gmt_v2_dismiss_guidance_v2') !== 'true';
    } catch {
      return true;
    }
  });

  const canvasRef = useRef<DrawingCanvasRef>(null);

  const handleDismissGuidance = () => {
    setShowGuidance(false);
    try {
      localStorage.setItem('gmt_v2_dismiss_guidance_v2', 'true');
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

  const handlePredict = async () => {
    if (!canvasRef.current || isPredicting) return;

    // Empty canvas check: do not submit request
    if (isCanvasEmpty || canvasRef.current.isEmpty()) {
      setIsCanvasEmpty(true);
      toast.error('Please draw a line of text first.');
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
      formData.append('image', blob, 'line_drawing.png');

      const response = await fetch(apiUrl('/api/predict-line'), {
        method: 'POST',
        body: formData,
      });

      const data: LinePredictionResponse = await response.json();

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
          toast('Recognized with some uncertain characters.', { icon: '⚠️' });
        } else {
          toast.success(`Recognized: "${data.text}"`);
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
      toast.success('Copied to clipboard!');
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error('Failed to copy text.');
    }
  };

  return (
    <div className="max-w-5xl mx-auto flex flex-col gap-6">
      {/* Page Header */}
      <div className="flex items-center gap-3">
        <TextAlignLeft size={28} weight="duotone" className="text-primary" />
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">Line Recognition</h1>
          <p className="text-xs text-text-muted mt-0.5">
            Write a line of handwritten text (uppercase letters A–Z and digits 0–9).
          </p>
        </div>
      </div>

      {/* Dismissible Handwriting Guidance Banner */}
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
                  GuessMyText recognizes uppercase letters (A–Z) and digits (0–9).
                </p>
                <p>
                  Write characters clearly and leave sufficient space between words. Connected or overlapping handwriting can be harder to segment and may produce inaccurate results.
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

      {/* Three-Section Status Card — minimal horizontal layout */}
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
                <span>Segmenting & recognizing line...</span>
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
                  {result.num_characters} {result.num_characters === 1 ? 'character' : 'characters'}
                  {' · '}
                  {result.num_words} {result.num_words === 1 ? 'word' : 'words'}
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
                Draw characters and words across the canvas
              </span>
            )}
          </div>

          {/* Right: Ready indicator (shows Predicting... during inference) */}
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
                disabled={isPredicting}
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
            disabled={isPredicting}
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
            disabled={isPredicting}
            className="h-[300px] w-10 sm:w-14 rounded-xl border-2 border-dashed border-border bg-surface hover:bg-primary-light hover:border-primary/60 text-text-secondary hover:text-primary flex flex-col items-center justify-center transition-all group shrink-0 shadow-sm active:scale-[0.98] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            title="Expand canvas right by 600px (+600px)"
            aria-label="Expand canvas right by 600px"
          >
            <Plus size={22} weight="bold" className="group-hover:scale-125 transition-transform text-primary" />
            <span className="text-[10px] font-bold mt-1 text-primary hidden sm:block">+600</span>
            <span className="text-[9px] text-text-muted uppercase tracking-wider font-semibold hidden sm:block">Right</span>
          </button>
        </div>

        {/* Action Buttons — Clear and Predict */}
        <div className="flex gap-4 w-full mt-2">
          <button
            className="btn btn-outline btn-error flex-1"
            onClick={handleClear}
            disabled={(isCanvasEmpty && !result && !error) || isPredicting}
          >
            <Eraser size={20} />
            Clear
          </button>
          <button
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
            <span>Recognition Result</span>
            {result?.status === 'success' && (
              <span className="badge badge-sm badge-outline text-text-muted font-normal">
                {result.num_words} words · {result.num_characters} chars
              </span>
            )}
          </h2>

          {result?.status === 'success' && result.text && (
            <button
              onClick={handleCopy}
              className="btn btn-sm btn-ghost gap-1.5 text-text-secondary hover:text-primary transition-colors"
              title="Copy recognized line"
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
            <span className="text-sm font-medium">Segmenting characters & running ensemble models...</span>
          </div>
        ) : error ? (
          <div className="p-4 rounded-lg bg-error/10 border border-error/20 flex items-center gap-3 text-error text-sm">
            <XCircle size={20} weight="bold" className="shrink-0" />
            <span>{error}</span>
          </div>
        ) : result?.status === 'success' ? (
          <div className="flex flex-col gap-4">
            {/* Prominent recognized text display */}
            <div className="p-4 md:p-6 bg-base-200/50 rounded-xl border border-border/80 flex flex-wrap items-center gap-y-3 gap-x-1.5 min-h-[72px]">
              {result.segments && result.segments.length > 0 ? (
                result.segments.map((seg, idx) => (
                  <span key={idx} className="inline-flex items-center">
                    {seg.is_uncertain ? (
                      <span
                        className="inline-flex items-center justify-center px-2 py-0.5 rounded bg-warning/20 border border-warning/50 text-warning-content font-mono font-bold text-lg md:text-xl shadow-xs"
                        title="Uncertain character — all 3 models disagreed"
                      >
                        [?]
                      </span>
                    ) : (
                      <span className="font-mono font-bold text-2xl md:text-3xl text-text-primary px-0.5 tracking-tight">
                        {seg.character}
                      </span>
                    )}
                    {seg.has_space_after && <span className="inline-block w-4 md:w-6" />}
                  </span>
                ))
              ) : (
                <span className="font-mono font-bold text-2xl text-text-primary">
                  {result.text}
                </span>
              )}
            </div>

            {/* Plain selectable text container for easy copy-paste */}
            <div className="flex items-center gap-2 text-xs text-text-muted bg-base-100 p-2.5 rounded-lg border border-border">
              <span className="font-semibold text-text-secondary shrink-0">Plain text:</span>
              <input
                type="text"
                readOnly
                value={result.text}
                className="w-full bg-transparent font-mono text-sm text-text-primary outline-none select-all"
              />
            </div>

            {/* Consensus & Guidance Note */}
            {result.has_uncertain ? (
              <div className="flex items-start gap-2 p-3 rounded-lg bg-warning/10 border border-warning/20 text-warning-content text-xs">
                <Warning size={16} weight="fill" className="shrink-0 mt-0.5 text-warning" />
                <span>
                  Some character positions are marked as <strong>[?]</strong> because all 3 CNN
                  models predicted different characters (no consensus). Try redrawing those
                  characters more clearly or with separated strokes.
                </span>
              </div>
            ) : SHOW_SEGMENTATION_PREVIEW ? (
              <div className="flex items-center gap-2 text-xs text-text-muted">
                <CheckCircle size={16} weight="fill" className="text-success shrink-0" />
                <span>All characters recognized with majority or unanimous consensus.</span>
              </div>
            ) : null}
          </div>
        ) : result?.status === 'no_characters' || result?.status === 'empty' ? (
          <div className="py-6 text-center text-text-secondary text-sm">
            {result.message}
          </div>
        ) : (
          <div className="py-8 text-center text-text-muted text-sm border border-dashed border-border rounded-xl">
            Draw a line of handwritten text above and click <strong>Predict</strong> to recognize.
          </div>
        )}
      </div>

      {/* Segmentation Preview Debugging Section */}
      {SHOW_SEGMENTATION_PREVIEW && result?.status === 'success' && result.segments && result.segments.length > 0 && (
        <div className="bg-surface rounded-xl border border-border p-5 md:p-6 shadow-sm flex flex-col gap-4 animate-in fade-in transition-all">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
            <div>
              <h2 className="text-base font-semibold text-text-primary flex items-center gap-2">
                <SquaresFour size={20} weight="duotone" className="text-primary" />
                <span>Segmentation Preview</span>
              </h2>
              <p className="text-xs text-text-muted mt-0.5">
                Inspect how each character was cropped and the exact preprocessed binary image supplied to the CNN models.
              </p>
            </div>

            {/* Resolution Selector for Model Input Image */}
            <div className="flex items-center gap-1.5 bg-base-200/80 p-1 rounded-lg border border-border self-start sm:self-auto">
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

          {/* Character Previews Grid (ordered left-to-right) */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3.5">
            {result.segments.map((seg) => {
              const cropW = seg.crop_dimensions?.width ?? seg.bbox.width;
              const cropH = seg.crop_dimensions?.height ?? seg.bbox.height;
              const modelInput = seg.model_inputs?.[selectedResolution];

              return (
                <div
                  key={seg.index}
                  className={`p-3.5 rounded-xl border transition-all flex flex-col gap-3 shadow-xs ${
                    seg.is_uncertain
                      ? 'bg-warning/5 border-warning/40'
                      : 'bg-base-100/60 border-border hover:border-primary/40'
                  }`}
                >
                  {/* Card Header: Position & Predicted Character */}
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
                      Character {seg.index + 1}
                    </span>
                    {seg.is_uncertain ? (
                      <span className="badge badge-sm badge-warning font-mono font-bold text-xs" title="All 3 models differed">
                        [?] Uncertain
                      </span>
                    ) : (
                      <span className="badge badge-sm badge-primary text-white font-mono font-bold text-xs">
                        '{seg.character}'
                      </span>
                    )}
                  </div>

                  {/* Side-by-Side Images: Segmented Crop vs Model Input */}
                  <div className="grid grid-cols-2 gap-2 text-center">
                    {/* 1. Segmented Crop (from canvas) */}
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
                            alt={`Crop for character ${seg.index + 1}`}
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

                    {/* 2. Model Input (exact CNN tensor representation) */}
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
                            alt={`Model input ${selectedResolution}×${selectedResolution} for character ${seg.index + 1}`}
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

                  {/* Card Footer: Bounding box position & Word Boundary status */}
                  <div className="flex items-center justify-between pt-2 border-t border-border/60 text-[11px] text-text-muted">
                    <span title={`Origin at (${seg.bbox.x}, ${seg.bbox.y})`}>
                      x:{seg.bbox.x} y:{seg.bbox.y}
                    </span>
                    {seg.has_space_after ? (
                      <span className="badge badge-xs bg-base-200 border border-border text-text-secondary font-medium" title="Word boundary space follows this character">
                        ␣ Space after
                      </span>
                    ) : (
                      <span className="text-[10px] text-text-muted">
                        No space
                      </span>
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
