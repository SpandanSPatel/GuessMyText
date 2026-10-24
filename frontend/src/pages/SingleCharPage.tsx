/**
 * GuessMyText — V1 Single Character Recognition Page
 *
 * Combines model predictions into a single final character:
 *   - Case A (3 agree): Prominently displays the recognized character
 *   - Case B (2 agree): Prominently displays the majority character
 *   - Case C (all differ): Displays friendly redraw message
 *
 * Internal model details (CNN resolutions, individual predictions,
 * confidence scores, agreement counts, timings) are kept internal.
 */

import { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  PencilLine,
  Eraser,
  MagnifyingGlass,
  Warning,
  XCircle,
  BookOpen,
} from '@phosphor-icons/react';
import { DrawingCanvas } from '../components/DrawingCanvas';
import type { DrawingCanvasRef } from '../components/DrawingCanvas';
import { apiUrl } from '../config/api';
import toast from 'react-hot-toast';

interface PredictionResponse {
  status: 'success' | 'no_consensus' | 'error';
  prediction?: string | null;
  message?: string;
  error?: string;
}

export function SingleCharPage() {
  const [isCanvasEmpty, setIsCanvasEmpty] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canvasRef = useRef<DrawingCanvasRef>(null);

  const handleClear = () => {
    canvasRef.current?.clear();
    setIsCanvasEmpty(true);
    setPrediction(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!canvasRef.current || isPredicting) return;

    // Empty canvas check: do not submit request
    if (isCanvasEmpty || canvasRef.current.isEmpty()) {
      setIsCanvasEmpty(true);
      toast.error('Please draw a character first.');
      return;
    }

    const blob = await canvasRef.current.exportToBlob();
    if (!blob) return;

    setIsPredicting(true);
    setError(null);
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append('image', blob, 'drawing.png');

      const response = await fetch(apiUrl('/api/predict'), {
        method: 'POST',
        body: formData,
      });

      const data: PredictionResponse = await response.json();

      if (!response.ok || data.status === 'error') {
        const msg = data.error || data.message || `Server error: ${response.status}`;
        setError(msg);
        toast.error(msg);
        return;
      }

      setPrediction(data);
      if (data.status === 'success' && data.prediction) {
        toast.success(`Recognized: "${data.prediction}"`);
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

  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-6">
      {/* Page Header */}
      <div className="flex items-center gap-3">
        <PencilLine size={28} weight="duotone" className="text-primary" />
        <h1 className="text-2xl font-semibold text-text-primary">
          Single Character Recognition
        </h1>
      </div>

      {/* Three-Section Status Card — minimal horizontal layout */}
      <div className="bg-surface rounded-xl border border-border px-4 py-3 md:px-6 md:py-3.5 shadow-sm min-h-[68px] flex items-center">
        <div className="w-full grid grid-cols-[auto_1fr_auto] items-center gap-4">
          {/* Left: Character Reference button */}
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

          {/* Center: Predicted Character / Predicting... / Disagreement or Error Message (empty before prediction) */}
          <div className="flex items-center justify-center min-w-[56px] text-center">
            {isPredicting ? (
              <div className="flex items-center gap-2 text-sm font-medium text-text-secondary">
                <span className="loading loading-spinner loading-xs text-primary"></span>
                <span>Predicting...</span>
              </div>
            ) : error ? (
              <div className="flex items-center gap-1.5 text-error text-xs md:text-sm font-medium text-center max-w-xs md:max-w-md">
                <XCircle size={18} weight="bold" className="shrink-0 text-error" />
                <span className="leading-snug">{error}</span>
              </div>
            ) : prediction?.status === 'success' && prediction.prediction ? (
              <div className="w-12 h-12 md:w-14 md:h-14 rounded-xl bg-primary-light text-primary font-bold text-2xl md:text-3xl flex items-center justify-center border border-primary/30 shadow-sm animate-in fade-in">
                {prediction.prediction}
              </div>
            ) : prediction?.status === 'no_consensus' ? (
              <div className="flex items-center gap-1.5 text-warning text-xs md:text-sm font-medium text-center max-w-xs md:max-w-md">
                <Warning size={18} weight="fill" className="shrink-0 text-warning" />
                <span className="leading-snug">
                  {prediction.message || "Couldn't recognize the character. Please try drawing it again."}
                </span>
              </div>
            ) : null}
          </div>

          {/* Right: Ready indicator (shows Predicting... while inference runs) */}
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

      {/* Canvas Area — centered max-w-2xl */}
      <div className="flex flex-col items-center gap-6">
        <div className="w-full max-w-2xl shadow-xl rounded-xl">
          <DrawingCanvas
            ref={canvasRef}
            internalWidth={400}
            internalHeight={400}
            brushSizeRatio={0.06}
            onDrawChange={(empty) => setIsCanvasEmpty(empty)}
          />
        </div>

        {/* Action Buttons — Clear and Predict */}
        <div className="flex gap-4 w-full max-w-2xl">
          <button
            className="btn btn-outline btn-error flex-1"
            onClick={handleClear}
            disabled={(isCanvasEmpty && !prediction && !error) || isPredicting}
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
            {isPredicting ? 'Predicting...' : 'Predict'}
          </button>
        </div>
      </div>
    </div>
  );
}
