/**
 * GuessMyText — Unified Drawing Canvas Component
 *
 * Direct adaptation and extension of CaptureMyText's original DrawingCanvas.tsx.
 * Supports three operating modes across GuessMyText versions:
 *   1. Fixed Square (V1): scrollable=false (default), aspect-square, centered drawing surface.
 *   2. Horizontal Expandable (V2): scrollable=true or 'horizontal', left/right +600px expansion, custom draggable scrollbar.
 *   3. 2D Expandable (V3): scrollable='2d', 4-way +600px expansion (Left, Right, Top, Bottom), 2D scrolling viewport.
 *
 * All modes use the same core HTML5 Canvas 2D rendering technology, pointer capture,
 * stroke preservation via offscreen blitting, brush calculations, and PNG blob export.
 */

import React, {
  useRef,
  useState,
  useLayoutEffect,
  useEffect,
  useImperativeHandle,
  forwardRef,
  useCallback,
} from 'react';

export interface DrawingCanvasRef {
  clear: () => void;
  isEmpty: () => boolean;
  exportToBlob: () => Promise<Blob | null>;
  expandLeft: (amountPx?: number) => void;
  expandRight: (amountPx?: number) => void;
  expandTop: (amountPx?: number) => void;
  expandBottom: (amountPx?: number) => void;
  resetDimensions: (targetWidth?: number, targetHeight?: number) => void;
  getWidth: () => number;
  getHeight: () => number;
  getDimensions: () => { width: number; height: number };
  scrollToLeft: () => void;
  scrollToRight: () => void;
  scrollToOrigin: () => void;
}

interface DrawingCanvasProps {
  internalWidth: number;
  internalHeight: number;
  brushSizeRatio?: number;
  brushSizePx?: number;
  scrollable?: boolean | 'horizontal' | '2d';
  onWidthChange?: (width: number) => void;
  onDimensionsChange?: (width: number, height: number) => void;
  onDrawChange?: (isEmpty: boolean) => void;
  className?: string;
}

interface PendingRestore {
  snapshot: HTMLCanvasElement;
  dx: number;
  dy: number;
  targetScroll: 'left' | 'right' | 'top' | 'bottom' | null;
}

export const DrawingCanvas = forwardRef<DrawingCanvasRef, DrawingCanvasProps>(
  (
    {
      internalWidth,
      internalHeight,
      brushSizeRatio,
      brushSizePx,
      scrollable = false,
      onWidthChange,
      onDimensionsChange,
      onDrawChange,
      className,
    },
    ref
  ) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const scrollContainerRef = useRef<HTMLDivElement>(null);

    const is2D = scrollable === '2d';
    const isHorizontal = scrollable === true || scrollable === 'horizontal';

    const [currentWidth, setCurrentWidth] = useState(internalWidth);
    const [currentHeight, setCurrentHeight] = useState(internalHeight);

    const [scrollState, setScrollState] = useState({
      scrollLeft: 0,
      scrollTop: 0,
      scrollWidth: internalWidth,
      scrollHeight: internalHeight,
      clientWidth: internalWidth,
      clientHeight: internalHeight,
    });
    const [isDraggingThumb, setIsDraggingThumb] = useState(false);

    const isDrawing = useRef(false);
    const hasDrawing = useRef(false);
    const isInitializedRef = useRef(false);
    const needsClearRef = useRef(false);
    const pendingRestoreRef = useRef<PendingRestore | null>(null);

    // Draggable scrollbar refs (for horizontal mode)
    const trackRef = useRef<HTMLDivElement>(null);
    const isDraggingThumbRef = useRef(false);
    const dragStartXRef = useRef(0);
    const dragStartScrollLeftRef = useRef(0);

    // Draggable scrollbar refs (for vertical 2D mode)
    const verticalTrackRef = useRef<HTMLDivElement>(null);
    const [isDraggingVerticalThumb, setIsDraggingVerticalThumb] = useState(false);
    const isDraggingVerticalThumbRef = useRef(false);
    const dragStartYRef = useRef(0);
    const dragStartScrollTopRef = useRef(0);

    // Previous scaled coordinates for stroke interpolation
    const lastPos = useRef<{ x: number; y: number } | null>(null);

    // Keep latest callbacks without triggering effect re-runs
    const onDrawChangeRef = useRef(onDrawChange);
    useEffect(() => {
      onDrawChangeRef.current = onDrawChange;
    }, [onDrawChange]);

    const onWidthChangeRef = useRef(onWidthChange);
    useEffect(() => {
      onWidthChangeRef.current = onWidthChange;
    }, [onWidthChange]);

    const onDimensionsChangeRef = useRef(onDimensionsChange);
    useEffect(() => {
      onDimensionsChangeRef.current = onDimensionsChange;
    }, [onDimensionsChange]);

    // Update scroll position metrics
    const updateScrollState = useCallback(() => {
      const container = scrollContainerRef.current;
      if (!container) return;
      setScrollState({
        scrollLeft: container.scrollLeft,
        scrollTop: container.scrollTop,
        scrollWidth: container.scrollWidth,
        scrollHeight: container.scrollHeight,
        clientWidth: container.clientWidth,
        clientHeight: container.clientHeight,
      });
    }, []);

    // Observe container resizing and dimension changes
    useEffect(() => {
      const container = scrollContainerRef.current;
      if (!container || !scrollable) return;

      updateScrollState();

      const resizeObserver = new ResizeObserver(() => {
        updateScrollState();
      });
      resizeObserver.observe(container);

      return () => {
        resizeObserver.disconnect();
      };
    }, [scrollable, currentWidth, currentHeight, updateScrollState]);

    // Calculate fixed brush thickness based on internalHeight and ratio (or explicit brushSizePx)
    const brushPixelSize = brushSizePx ?? internalHeight * (brushSizeRatio ?? 0.06);

    // Helper to configure stroke styling on 2D context
    const setupContext = useCallback(
      (ctx: CanvasRenderingContext2D) => {
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = brushPixelSize;
        ctx.strokeStyle = '#FFFFFF';
      },
      [brushPixelSize]
    );

    // Initialize or restore canvas after DOM updates
    useLayoutEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      if (pendingRestoreRef.current) {
        const { snapshot, dx, dy, targetScroll } = pendingRestoreRef.current;
        pendingRestoreRef.current = null;

        // Fill background with solid black
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw preserved strokes from snapshot at offset (dx, dy)
        ctx.drawImage(snapshot, dx, dy);

        // Reconfigure context stroke styles
        setupContext(ctx);

        // Auto-scroll toward newly added space
        if (scrollContainerRef.current && targetScroll) {
          const container = scrollContainerRef.current;
          requestAnimationFrame(() => {
            if (targetScroll === 'right') {
              container.scrollTo({ left: container.scrollWidth, behavior: 'smooth' });
            } else if (targetScroll === 'left') {
              container.scrollTo({ left: 0, behavior: 'smooth' });
            } else if (targetScroll === 'bottom') {
              container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
            } else if (targetScroll === 'top') {
              container.scrollTo({ top: 0, behavior: 'smooth' });
            }
          });
        }
      } else if (needsClearRef.current || !isInitializedRef.current) {
        needsClearRef.current = false;
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        setupContext(ctx);
        hasDrawing.current = false;
        onDrawChangeRef.current?.(true);
        isInitializedRef.current = true;

        if (scrollContainerRef.current) {
          scrollContainerRef.current.scrollTo({ left: 0, top: 0, behavior: 'smooth' });
        }
      }
    }, [currentWidth, currentHeight, setupContext]);

    // Clear canvas across the entire current dimensions and reset scroll position
    const clearCanvas = useCallback(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      setupContext(ctx);

      hasDrawing.current = false;
      onDrawChangeRef.current?.(true);

      if (scrollContainerRef.current) {
        scrollContainerRef.current.scrollTo({ left: 0, top: 0, behavior: 'smooth' });
      }
    }, [setupContext]);

    // Snapshot current canvas bitmap to offscreen canvas
    const takeSnapshot = useCallback((): HTMLCanvasElement | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      if (tempCtx) {
        tempCtx.drawImage(canvas, 0, 0);
      }
      return tempCanvas;
    }, []);

    // 1. Expand canvas on the right by amountPx (default 600)
    const expandRight = useCallback(
      (amountPx: number = 600) => {
        const snapshot = takeSnapshot();
        if (!snapshot) return;

        const newW = currentWidth + amountPx;
        pendingRestoreRef.current = {
          snapshot,
          dx: 0,
          dy: 0,
          targetScroll: 'right',
        };

        setCurrentWidth(newW);
        onWidthChangeRef.current?.(newW);
        onDimensionsChangeRef.current?.(newW, currentHeight);
      },
      [currentWidth, currentHeight, takeSnapshot]
    );

    // 2. Expand canvas on the left by amountPx (default 600)
    const expandLeft = useCallback(
      (amountPx: number = 600) => {
        const snapshot = takeSnapshot();
        if (!snapshot) return;

        const newW = currentWidth + amountPx;
        pendingRestoreRef.current = {
          snapshot,
          dx: amountPx,
          dy: 0,
          targetScroll: 'left',
        };

        setCurrentWidth(newW);
        onWidthChangeRef.current?.(newW);
        onDimensionsChangeRef.current?.(newW, currentHeight);
      },
      [currentWidth, currentHeight, takeSnapshot]
    );

    // 3. Expand canvas below by amountPx (default 600)
    const expandBottom = useCallback(
      (amountPx: number = 600) => {
        const snapshot = takeSnapshot();
        if (!snapshot) return;

        const newH = currentHeight + amountPx;
        pendingRestoreRef.current = {
          snapshot,
          dx: 0,
          dy: 0,
          targetScroll: 'bottom',
        };

        setCurrentHeight(newH);
        onDimensionsChangeRef.current?.(currentWidth, newH);
      },
      [currentWidth, currentHeight, takeSnapshot]
    );

    // 4. Expand canvas above by amountPx (default 600)
    const expandTop = useCallback(
      (amountPx: number = 600) => {
        const snapshot = takeSnapshot();
        if (!snapshot) return;

        const newH = currentHeight + amountPx;
        pendingRestoreRef.current = {
          snapshot,
          dx: 0,
          dy: amountPx,
          targetScroll: 'top',
        };

        setCurrentHeight(newH);
        onDimensionsChangeRef.current?.(currentWidth, newH);
      },
      [currentWidth, currentHeight, takeSnapshot]
    );

    // Explicitly reset canvas dimensions back to target (or initial)
    const resetDimensions = useCallback(
      (targetWidth?: number, targetHeight?: number) => {
        const targetW = targetWidth ?? internalWidth;
        const targetH = targetHeight ?? internalHeight;
        pendingRestoreRef.current = null;
        needsClearRef.current = true;
        setCurrentWidth(targetW);
        setCurrentHeight(targetH);
        onWidthChangeRef.current?.(targetW);
        onDimensionsChangeRef.current?.(targetW, targetH);
      },
      [internalWidth, internalHeight]
    );

    const scrollToLeft = useCallback(() => {
      scrollContainerRef.current?.scrollTo({ left: 0, behavior: 'smooth' });
    }, []);

    const scrollToRight = useCallback(() => {
      if (scrollContainerRef.current) {
        scrollContainerRef.current.scrollTo({
          left: scrollContainerRef.current.scrollWidth,
          behavior: 'smooth',
        });
      }
    }, []);

    const scrollToOrigin = useCallback(() => {
      scrollContainerRef.current?.scrollTo({ left: 0, top: 0, behavior: 'smooth' });
    }, []);

    useImperativeHandle(ref, () => ({
      clear: clearCanvas,
      isEmpty: () => {
        if (!hasDrawing.current) return true;

        const canvas = canvasRef.current;
        if (!canvas) return true;
        const ctx = canvas.getContext('2d');
        if (!ctx) return true;

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
          if (data[i] > 0 || data[i + 1] > 0 || data[i + 2] > 0) {
            return false;
          }
        }
        return true;
      },
      exportToBlob: () => {
        return new Promise<Blob | null>((resolve) => {
          if (!canvasRef.current) {
            resolve(null);
            return;
          }
          canvasRef.current.toBlob((blob) => {
            resolve(blob);
          }, 'image/png');
        });
      },
      expandLeft,
      expandRight,
      expandTop,
      expandBottom,
      resetDimensions,
      getWidth: () => currentWidth,
      getHeight: () => currentHeight,
      getDimensions: () => ({ width: currentWidth, height: currentHeight }),
      scrollToLeft,
      scrollToRight,
      scrollToOrigin,
    }));

    const getScaledCoordinates = (e: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };

      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;

      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
      };
    };

    const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
      e.currentTarget.setPointerCapture(e.pointerId);
      isDrawing.current = true;

      const coords = getScaledCoordinates(e);
      lastPos.current = coords;

      draw(coords, coords);
    };

    const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!isDrawing.current || !lastPos.current) return;
      const currentPos = getScaledCoordinates(e);
      draw(lastPos.current, currentPos);
      lastPos.current = currentPos;
    };

    const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
      isDrawing.current = false;
      lastPos.current = null;
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // Ignore if pointer capture already released
      }

      if (!hasDrawing.current) {
        hasDrawing.current = true;
        onDrawChangeRef.current?.(false);
      }
    };

    const handlePointerCancel = (e: React.PointerEvent<HTMLCanvasElement>) => {
      handlePointerUp(e);
    };

    const draw = (start: { x: number; y: number }, end: { x: number; y: number }) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      setupContext(ctx);
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(end.x, end.y);
      ctx.stroke();
    };

    // Scrollbar track click to jump directly to clicked ratio (horizontal mode)
    const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
      const track = trackRef.current;
      const container = scrollContainerRef.current;
      if (!track || !container) return;

      const { scrollWidth, clientWidth } = container;
      const maxScroll = scrollWidth - clientWidth;
      if (maxScroll <= 0) return;

      const rect = track.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const trackWidth = rect.width;
      if (trackWidth <= 0) return;

      const clickRatio = Math.max(0, Math.min(1, clickX / trackWidth));
      const targetScroll = clickRatio * maxScroll;
      container.scrollTo({ left: targetScroll, behavior: 'smooth' });
    };

    // Thumb dragging event handlers (horizontal mode)
    const handleThumbPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
      e.stopPropagation();
      const container = scrollContainerRef.current;
      if (!container) return;

      const { scrollWidth, clientWidth } = container;
      const maxScroll = scrollWidth - clientWidth;
      if (maxScroll <= 0) return;

      isDraggingThumbRef.current = true;
      setIsDraggingThumb(true);
      dragStartXRef.current = e.clientX;
      dragStartScrollLeftRef.current = container.scrollLeft;
      e.currentTarget.setPointerCapture(e.pointerId);
    };

    const handleThumbPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
      if (!isDraggingThumbRef.current) return;
      const track = trackRef.current;
      const container = scrollContainerRef.current;
      if (!track || !container) return;

      const { scrollWidth, clientWidth } = container;
      const maxScroll = scrollWidth - clientWidth;
      if (maxScroll <= 0) return;

      const trackRect = track.getBoundingClientRect();
      const trackWidth = trackRect.width;
      if (trackWidth <= 0) return;

      const rawRatio = clientWidth / scrollWidth;
      const thumbWidthPercent = Math.max(10, Math.min(100, rawRatio * 100));
      const thumbWidthPx = (thumbWidthPercent / 100) * trackWidth;
      const trackTravelPx = trackWidth - thumbWidthPx;
      if (trackTravelPx <= 0) return;

      const deltaX = e.clientX - dragStartXRef.current;
      const deltaScroll = (deltaX / trackTravelPx) * maxScroll;
      const newScroll = Math.max(0, Math.min(maxScroll, dragStartScrollLeftRef.current + deltaScroll));
      container.scrollLeft = newScroll;
    };

    const handleThumbPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
      if (isDraggingThumbRef.current) {
        isDraggingThumbRef.current = false;
        setIsDraggingThumb(false);
        try {
          e.currentTarget.releasePointerCapture(e.pointerId);
        } catch {
          // Ignore
        }
      }
    };

    // Vertical scrollbar track click to jump directly to clicked ratio (2D mode)
    const handleVerticalTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
      const track = verticalTrackRef.current;
      const container = scrollContainerRef.current;
      if (!track || !container) return;

      const { scrollHeight, clientHeight } = container;
      const maxScroll = scrollHeight - clientHeight;
      if (maxScroll <= 0) return;

      const rect = track.getBoundingClientRect();
      const clickY = e.clientY - rect.top;
      const trackHeight = rect.height;
      if (trackHeight <= 0) return;

      const clickRatio = Math.max(0, Math.min(1, clickY / trackHeight));
      const targetScroll = clickRatio * maxScroll;
      container.scrollTo({ top: targetScroll, behavior: 'smooth' });
    };

    // Vertical thumb dragging event handlers (2D mode)
    const handleVerticalThumbPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
      e.stopPropagation();
      const container = scrollContainerRef.current;
      if (!container) return;

      const { scrollHeight, clientHeight } = container;
      const maxScroll = scrollHeight - clientHeight;
      if (maxScroll <= 0) return;

      isDraggingVerticalThumbRef.current = true;
      setIsDraggingVerticalThumb(true);
      dragStartYRef.current = e.clientY;
      dragStartScrollTopRef.current = container.scrollTop;
      e.currentTarget.setPointerCapture(e.pointerId);
    };

    const handleVerticalThumbPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
      if (!isDraggingVerticalThumbRef.current) return;
      const track = verticalTrackRef.current;
      const container = scrollContainerRef.current;
      if (!track || !container) return;

      const { scrollHeight, clientHeight } = container;
      const maxScroll = scrollHeight - clientHeight;
      if (maxScroll <= 0) return;

      const trackRect = track.getBoundingClientRect();
      const trackHeight = trackRect.height;
      if (trackHeight <= 0) return;

      const rawRatio = clientHeight / scrollHeight;
      const thumbHeightPercent = Math.max(10, Math.min(100, rawRatio * 100));
      const thumbHeightPx = (thumbHeightPercent / 100) * trackHeight;
      const trackTravelPx = trackHeight - thumbHeightPx;
      if (trackTravelPx <= 0) return;

      const deltaY = e.clientY - dragStartYRef.current;
      const deltaScroll = (deltaY / trackTravelPx) * maxScroll;
      const newScroll = Math.max(0, Math.min(maxScroll, dragStartScrollTopRef.current + deltaScroll));
      container.scrollTop = newScroll;
    };

    const handleVerticalThumbPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
      if (isDraggingVerticalThumbRef.current) {
        isDraggingVerticalThumbRef.current = false;
        setIsDraggingVerticalThumb(false);
        try {
          e.currentTarget.releasePointerCapture(e.pointerId);
        } catch {
          // Ignore
        }
      }
    };

    // MODE 3: 2D Scrollable & 4-Way Expandable (V3 Paragraph)
    if (is2D) {
      const { scrollLeft, scrollTop, scrollWidth, scrollHeight, clientWidth, clientHeight } = scrollState;
      const maxScrollX = Math.max(0, scrollWidth - clientWidth);
      const maxScrollY = Math.max(0, scrollHeight - clientHeight);

      // Horizontal thumb calculations
      let thumbWidthPercent = 100;
      let thumbLeftPercent = 0;
      if (scrollWidth > 0 && maxScrollX > 0) {
        const rawRatioX = clientWidth / scrollWidth;
        thumbWidthPercent = Math.max(10, Math.min(100, rawRatioX * 100));
        const scrollRatioX = Math.max(0, Math.min(1, scrollLeft / maxScrollX));
        thumbLeftPercent = scrollRatioX * (100 - thumbWidthPercent);
      }

      // Vertical thumb calculations
      let thumbHeightPercent = 100;
      let thumbTopPercent = 0;
      if (scrollHeight > 0 && maxScrollY > 0) {
        const rawRatioY = clientHeight / scrollHeight;
        thumbHeightPercent = Math.max(10, Math.min(100, rawRatioY * 100));
        const scrollRatioY = Math.max(0, Math.min(1, scrollTop / maxScrollY));
        thumbTopPercent = scrollRatioY * (100 - thumbHeightPercent);
      }

      const visibleStartX = Math.round(scrollLeft);
      const visibleEndX = Math.round(Math.min(scrollWidth, scrollLeft + clientWidth));
      const visibleStartY = Math.round(scrollTop);
      const visibleEndY = Math.round(Math.min(scrollHeight, scrollTop + clientHeight));

      return (
        <div className="flex flex-col gap-2 w-full">
          {/* Viewport Row: [Canvas Viewport] + [Vertical Draggable Scrollbar Track] */}
          <div className="flex items-stretch gap-2 w-full">
            {/* 2D Scrollable Drawing Viewport (Native scrollbar hidden with no-scrollbar) */}
            <div
              id="paragraph-canvas-viewport"
              ref={scrollContainerRef}
              onScroll={updateScrollState}
              className={
                className ||
                'flex-1 min-w-0 h-[480px] border-2 border-border rounded-xl overflow-auto bg-canvas-bg shadow-sm no-scrollbar'
              }
              style={{ overscrollBehavior: 'contain' }}
            >
              <canvas
                id="paragraph-canvas"
                ref={canvasRef}
                width={currentWidth}
                height={currentHeight}
                style={{
                  width: `${currentWidth}px`,
                  height: `${currentHeight}px`,
                  minWidth: `${currentWidth}px`,
                  minHeight: `${currentHeight}px`,
                  touchAction: 'none',
                }}
                className="block touch-none"
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerCancel={handlePointerCancel}
              />
            </div>

            {/* Vertical Draggable Scrollbar Track (touch/mouse friendly) */}
            <div
              ref={verticalTrackRef}
              onClick={handleVerticalTrackClick}
              className={`relative w-5 sm:w-6 h-[480px] bg-base-200/90 rounded-xl border border-border/90 p-0.5 cursor-pointer shadow-inner hover:border-primary/50 transition-all select-none overflow-hidden shrink-0 ${
                maxScrollY > 0 ? 'opacity-100' : 'opacity-40 cursor-default pointer-events-none'
              }`}
              style={{ touchAction: 'none' }}
              title={maxScrollY > 0 ? "Click anywhere to jump, or drag thumb to scroll vertically" : "Canvas height fits within viewport"}
            >
              {/* Subtle vertical track guide marks */}
              <div className="absolute inset-0 flex flex-col items-center justify-between py-4 pointer-events-none opacity-20">
                <div className="h-px w-2 bg-text-secondary"></div>
                <div className="h-px w-2 bg-text-secondary"></div>
                <div className="h-px w-2 bg-text-secondary"></div>
              </div>

              {/* Vertical Draggable Active Thumb */}
              <div
                onPointerDown={handleVerticalThumbPointerDown}
                onPointerMove={handleVerticalThumbPointerMove}
                onPointerUp={handleVerticalThumbPointerUp}
                onPointerCancel={handleVerticalThumbPointerUp}
                className={`absolute left-0.5 right-0.5 rounded-lg bg-primary hover:bg-primary-hover active:bg-primary cursor-grab active:cursor-grabbing shadow-sm border border-white/60 flex flex-col items-center justify-center transition-shadow group-hover:shadow-md ${
                  isDraggingVerticalThumb ? 'cursor-grabbing shadow-md ring-2 ring-primary/50 scale-x-105' : ''
                }`}
                style={{
                  top: `${thumbTopPercent}%`,
                  height: `${thumbHeightPercent}%`,
                  touchAction: 'none',
                }}
                title="Drag vertically to scroll canvas up/down"
              >
                {/* Grip marks (horizontal lines) */}
                <div className="flex flex-col items-center justify-center gap-1 pointer-events-none py-1">
                  <span className="h-0.5 w-2 bg-white/80 rounded-full shadow-xs"></span>
                  <span className="h-0.5 w-2.5 bg-white rounded-full shadow-xs"></span>
                  <span className="h-0.5 w-2 bg-white/80 rounded-full shadow-xs"></span>
                </div>
              </div>
            </div>
          </div>

          {/* Horizontal Draggable Scrollbar Track (touch/mouse friendly) */}
          <div className="w-full flex flex-col gap-1.5 px-0.5 select-none animate-in fade-in transition-all">
            <div
              ref={trackRef}
              onClick={handleTrackClick}
              className={`relative w-full h-5.5 bg-base-200/90 rounded-full border border-border/90 p-0.5 cursor-pointer shadow-inner hover:border-primary/50 transition-all group overflow-hidden ${
                maxScrollX > 0 ? 'opacity-100' : 'opacity-40 cursor-default pointer-events-none'
              }`}
              style={{ touchAction: 'none' }}
              title={maxScrollX > 0 ? "Click anywhere to jump, or drag thumb to scroll horizontally" : "Canvas width fits within viewport"}
            >
              {/* Subtle track guide lines */}
              <div className="absolute inset-0 flex items-center justify-between px-4 pointer-events-none opacity-20">
                <div className="w-px h-2 bg-text-secondary"></div>
                <div className="w-px h-2 bg-text-secondary"></div>
                <div className="w-px h-2 bg-text-secondary"></div>
              </div>

              {/* Horizontal Draggable Active Thumb */}
              <div
                onPointerDown={handleThumbPointerDown}
                onPointerMove={handleThumbPointerMove}
                onPointerUp={handleThumbPointerUp}
                onPointerCancel={handleThumbPointerUp}
                className={`absolute top-0.5 bottom-0.5 rounded-full bg-primary hover:bg-primary-hover active:bg-primary cursor-grab active:cursor-grabbing shadow-sm border border-white/60 flex items-center justify-center transition-shadow group-hover:shadow-md ${
                  isDraggingThumb ? 'cursor-grabbing shadow-md ring-2 ring-primary/50 scale-y-105' : ''
                }`}
                style={{
                  left: `${thumbLeftPercent}%`,
                  width: `${thumbWidthPercent}%`,
                  touchAction: 'none',
                }}
                title="Drag horizontally to scroll canvas left/right"
              >
                {/* 3 white grip bars */}
                <div className="flex items-center justify-center gap-1 pointer-events-none px-1">
                  <span className="w-0.5 h-2 bg-white/80 rounded-full shadow-xs"></span>
                  <span className="w-0.5 h-2.5 bg-white rounded-full shadow-xs"></span>
                  <span className="w-0.5 h-2 bg-white/80 rounded-full shadow-xs"></span>
                </div>
              </div>
            </div>
          </div>

          {/* Viewport & Canvas Dimension Metrics */}
          <div className="flex flex-wrap items-center justify-between text-xs text-text-muted px-1 font-mono select-none gap-2">
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1 font-sans text-text-secondary text-[11px]">
                <span className="inline-block w-2 h-2 rounded-full bg-primary shrink-0 animate-pulse"></span>
                <span>Visible:</span>
              </span>
              <span className="text-[11px] font-semibold text-text-primary bg-base-200/80 px-2 py-0.5 rounded border border-border/80">
                H: {visibleStartX}–{visibleEndX}px ({scrollWidth}px)
                {' · '}
                V: {visibleStartY}–{visibleEndY}px ({scrollHeight}px)
              </span>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-[11px] text-text-muted hidden sm:inline">
                Drag scrollbars to pan canvas on mobile
              </span>
              <span className="text-[11px] font-semibold text-text-secondary bg-base-200/80 px-2 py-0.5 rounded border border-border/80">
                Total: {currentWidth} × {currentHeight} px
              </span>
            </div>
          </div>
        </div>
      );
    }

    // MODE 2: Horizontally Scrollable & Expandable (V2 Line)
    if (isHorizontal) {
      const { scrollLeft, scrollWidth, clientWidth } = scrollState;
      const maxScroll = Math.max(0, scrollWidth - clientWidth);

      let thumbWidthPercent = 100;
      let thumbLeftPercent = 0;

      if (scrollWidth > 0 && maxScroll > 0) {
        const rawRatio = clientWidth / scrollWidth;
        thumbWidthPercent = Math.max(10, Math.min(100, rawRatio * 100));
        const scrollRatio = Math.max(0, Math.min(1, scrollLeft / maxScroll));
        thumbLeftPercent = scrollRatio * (100 - thumbWidthPercent);
      }

      const visibleStart = Math.round(scrollLeft);
      const visibleEnd = Math.round(Math.min(scrollWidth, scrollLeft + clientWidth));
      const visiblePercent = scrollWidth > 0 ? Math.round((clientWidth / scrollWidth) * 100) : 100;

      return (
        <div className="flex flex-col gap-2.5 w-full">
          {/* Main Drawing Viewport (Native scrollbar hidden with no-scrollbar) */}
          <div
            ref={scrollContainerRef}
            onScroll={updateScrollState}
            className={
              className ||
              'w-full h-[300px] border-2 border-border rounded-xl overflow-x-auto overflow-y-hidden bg-canvas-bg shadow-sm no-scrollbar'
            }
          >
            <canvas
              ref={canvasRef}
              width={currentWidth}
              height={currentHeight}
              style={{
                width: `${currentWidth}px`,
                height: `${currentHeight}px`,
                minWidth: `${currentWidth}px`,
                touchAction: 'none',
              }}
              className="block touch-none"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerCancel}
            />
          </div>

          {/* Custom Draggable Scrollbar & Position Cue */}
          <div className="w-full flex flex-col gap-1.5 px-0.5 select-none animate-in fade-in transition-all">
            {/* Visual Queue & Length Header */}
            <div className="flex items-center justify-between text-xs text-text-muted px-1 font-mono">
              <span className="flex items-center gap-1.5 font-sans text-text-secondary font-medium text-[11px]">
                <span className="inline-block w-2 h-2 rounded-full bg-primary shrink-0 animate-pulse"></span>
                <span>Visible View:</span>
                <span className="font-mono font-semibold text-text-primary bg-base-200/80 px-1.5 py-0.5 rounded border border-border/80">
                  {visibleStart}px – {visibleEnd}px
                </span>
              </span>

              <span className="text-[11px] text-text-muted hidden sm:inline">
                {maxScroll > 0 ? (
                  <span>Drag thumb or click bar to scroll ({visiblePercent}% visible)</span>
                ) : (
                  <span>Full canvas visible (100%)</span>
                )}
              </span>

              <span className="text-[11px] font-mono font-semibold text-text-secondary bg-base-200/80 px-1.5 py-0.5 rounded border border-border/80">
                {scrollWidth}px total length
              </span>
            </div>

            {/* Draggable Scrollbar Track */}
            <div
              ref={trackRef}
              onClick={handleTrackClick}
              className="relative w-full h-5 bg-base-200/90 rounded-full border border-border/90 p-0.5 cursor-pointer shadow-inner hover:border-primary/50 transition-all group overflow-hidden"
              title="Click anywhere to jump, or drag thumb to scroll"
            >
              {/* Subtle track guide lines */}
              <div className="absolute inset-0 flex items-center justify-between px-4 pointer-events-none opacity-20">
                <div className="w-px h-2 bg-text-secondary"></div>
                <div className="w-px h-2 bg-text-secondary"></div>
                <div className="w-px h-2 bg-text-secondary"></div>
              </div>

              {/* Draggable Active Thumb */}
              <div
                onPointerDown={handleThumbPointerDown}
                onPointerMove={handleThumbPointerMove}
                onPointerUp={handleThumbPointerUp}
                onPointerCancel={handleThumbPointerUp}
                className={`absolute top-0.5 bottom-0.5 rounded-full bg-primary hover:bg-primary-hover active:bg-primary cursor-grab active:cursor-grabbing shadow-sm border border-white/60 flex items-center justify-center transition-shadow group-hover:shadow-md ${
                  isDraggingThumb ? 'cursor-grabbing shadow-md ring-2 ring-primary/50 scale-y-105' : ''
                }`}
                style={{
                  left: `${thumbLeftPercent}%`,
                  width: `${thumbWidthPercent}%`,
                  touchAction: 'none',
                }}
                title="Drag to scroll canvas"
              >
                {/* 3 white grip bars */}
                <div className="flex items-center justify-center gap-1 pointer-events-none px-1">
                  <span className="w-0.5 h-2 bg-white/80 rounded-full shadow-xs"></span>
                  <span className="w-0.5 h-2.5 bg-white rounded-full shadow-xs"></span>
                  <span className="w-0.5 h-2 bg-white/80 rounded-full shadow-xs"></span>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }

    // MODE 1: Fixed Square (V1 Single Character) — Identical to CaptureMyText DrawingCanvas
    return (
      <div
        ref={scrollContainerRef}
        className={
          className ||
          'w-full max-w-[55vh] max-h-[55vh] aspect-square border-2 border-border rounded-xl overflow-hidden bg-canvas-bg touch-none mx-auto'
        }
      >
        <canvas
          ref={canvasRef}
          width={internalWidth}
          height={internalHeight}
          className="w-full h-full block touch-none"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerCancel}
          style={{ touchAction: 'none' }}
        />
      </div>
    );
  }
);
