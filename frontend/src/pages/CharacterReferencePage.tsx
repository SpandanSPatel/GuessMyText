/**
 * GuessMyText — Character Reference Page
 *
 * Displays reference images and handwriting styles for the characters in the
 * original 67-class dataset:
 *   - Digits: 0–9 (10 classes)
 *   - Uppercase letters: A–Z (26 classes)
 *   - Lowercase letters: a–z (26 classes)
 *   - Mathematical operators: +, -, *, ÷, % (5 classes)
 *
 * Supports dynamic reference images per character (zero, one, or multiple),
 * search, category filtering, and an enlarged image preview dialog.
 */

import { useState, useEffect, useMemo } from 'react';
import {
  BookOpen,
  MagnifyingGlass,
  X,
  Eye,
  WarningCircle,
  Hash,
  Sparkle,
} from '@phosphor-icons/react';
import { apiUrl } from '../config/api';

export type CharacterCategory = 'all' | 'digit' | 'uppercase' | 'lowercase' | 'operator';

export interface ReferenceImage {
  path: string;
  label?: string;
  source_number?: number;
}

export interface CharacterEntry {
  id: string;
  character: string;
  name: string;
  category: 'digit' | 'uppercase' | 'lowercase' | 'operator';
  folder: string;
  images: ReferenceImage[];
}

export interface ManifestMetadata {
  total_classes: number;
  categories: {
    digit: number;
    uppercase: number;
    lowercase: number;
    operator: number;
  };
  description?: string;
}

// Fallback initial 67 classes metadata to guarantee rendering even if offline
const INITIAL_DIGITS: CharacterEntry[] = Array.from({ length: 10 }, (_, i) => ({
  id: String(i),
  character: String(i),
  name: `Digit ${i}`,
  category: 'digit',
  folder: `digits/${i}`,
  images: [],
}));

const INITIAL_UPPERCASE: CharacterEntry[] = Array.from({ length: 26 }, (_, i) => {
  const char = String.fromCharCode(65 + i);
  return {
    id: char,
    character: char,
    name: `Uppercase ${char}`,
    category: 'uppercase',
    folder: `uppercase/${char}`,
    images: [],
  };
});

const INITIAL_LOWERCASE: CharacterEntry[] = Array.from({ length: 26 }, (_, i) => {
  const char = String.fromCharCode(97 + i);
  return {
    id: char,
    character: char,
    name: `Lowercase ${char}`,
    category: 'lowercase',
    folder: `lowercase/${char}`,
    images: [],
  };
});

const INITIAL_OPERATORS: CharacterEntry[] = [
  { id: 'plus', character: '+', name: 'Addition (+)', category: 'operator', folder: 'operators/plus', images: [] },
  { id: 'minus', character: '-', name: 'Subtraction (-)', category: 'operator', folder: 'operators/minus', images: [] },
  { id: 'multiply', character: '*', name: 'Multiplication (*)', category: 'operator', folder: 'operators/multiply', images: [] },
  { id: 'divide', character: '÷', name: 'Division (÷)', category: 'operator', folder: 'operators/divide', images: [] },
  { id: 'modulo', character: '%', name: 'Modulo (%)', category: 'operator', folder: 'operators/modulo', images: [] },
];

const DEFAULT_CHARACTERS: CharacterEntry[] = [
  ...INITIAL_DIGITS,
  ...INITIAL_UPPERCASE,
  ...INITIAL_LOWERCASE,
  ...INITIAL_OPERATORS,
];

export function CharacterReferencePage() {
  const [characters, setCharacters] = useState<CharacterEntry[]>(DEFAULT_CHARACTERS);
  const [metadata, setMetadata] = useState<ManifestMetadata>({
    total_classes: 67,
    categories: { digit: 10, uppercase: 26, lowercase: 26, operator: 5 },
  });
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Filters
  const [activeCategory, setActiveCategory] = useState<CharacterCategory>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Enlarged preview modal state
  const [previewModal, setPreviewModal] = useState<{
    character: CharacterEntry;
    image: ReferenceImage;
    index: number;
  } | null>(null);

  // Track images that failed to load (e.g. 404) to render graceful placeholder
  const [failedImagePaths, setFailedImagePaths] = useState<Set<string>>(new Set());

  // Fetch manifest from backend
  useEffect(() => {
    let isMounted = true;
    async function fetchManifest() {
      try {
        setLoading(true);
        const res = await fetch(apiUrl('/api/character-references'));
        if (!res.ok) {
          throw new Error(`Server returned HTTP ${res.status}`);
        }
        const data = await res.json();
        if (data.status === 'success' && data.characters) {
          if (isMounted) {
            const list: CharacterEntry[] = Object.values(data.characters);
            setCharacters(list);
            if (data.metadata) {
              setMetadata(data.metadata);
            }
            setLoadError(null);
          }
        } else {
          throw new Error(data.error || 'Failed to parse manifest');
        }
      } catch (err) {
        console.warn('Could not load live character reference manifest, using default catalog:', err);
        if (isMounted) {
          setLoadError(err instanceof Error ? err.message : 'Could not reach server');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }
    fetchManifest();
    return () => {
      isMounted = false;
    };
  }, []);

  // Keyboard navigation: Escape key closes image preview
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        setPreviewModal(null);
      }
    }
    if (previewModal) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [previewModal]);

  // Filtered characters list
  const filteredCharacters = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    return characters.filter((c) => {
      // Category filter
      if (activeCategory !== 'all' && c.category !== activeCategory) {
        return false;
      }
      // Search filter
      if (!query) return true;
      return (
        c.character.toLowerCase() === query ||
        c.character.toLowerCase().includes(query) ||
        c.name.toLowerCase().includes(query) ||
        c.id.toLowerCase().includes(query)
      );
    });
  }, [characters, activeCategory, searchQuery]);

  const handleImageError = (path: string) => {
    setFailedImagePaths((prev) => new Set(prev).add(path));
  };

  const getCategoryBadgeClass = (category: CharacterEntry['category']) => {
    switch (category) {
      case 'digit':
        return 'bg-blue-500/10 text-blue-600 border-blue-500/30';
      case 'uppercase':
        return 'bg-indigo-500/10 text-indigo-600 border-indigo-500/30';
      case 'lowercase':
        return 'bg-emerald-500/10 text-emerald-600 border-emerald-500/30';
      case 'operator':
        return 'bg-amber-500/10 text-amber-600 border-amber-500/30';
      default:
        return 'bg-base-200 text-text-secondary border-border';
    }
  };

  const getCategoryLabel = (category: CharacterEntry['category']) => {
    switch (category) {
      case 'digit':
        return 'Digit';
      case 'uppercase':
        return 'Uppercase';
      case 'lowercase':
        return 'Lowercase';
      case 'operator':
        return 'Operator';
    }
  };

  return (
    <div className="max-w-6xl mx-auto flex flex-col gap-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-2 border-b border-border/80">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary-light flex items-center justify-center shrink-0 border border-primary/30 mt-0.5">
            <BookOpen size={24} weight="duotone" className="text-primary" />
          </div>
          <div className="flex flex-col">
            <h1 className="text-2xl font-bold text-text-primary tracking-tight">
              Character Reference
            </h1>
            <p className="text-sm text-text-secondary mt-0.5 max-w-2xl">
              Explore the characters supported by GuessMyText and view examples of the handwriting
              styles represented in its recognition dataset.
            </p>
          </div>
        </div>

        {/* Status / Loading indicator */}
        {loading && (
          <div className="flex items-center gap-2 text-xs text-text-muted self-start md:self-center">
            <span className="loading loading-spinner loading-xs text-primary"></span>
            <span>Syncing catalog...</span>
          </div>
        )}
      </div>

      {/* Summary Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        <div className="bg-surface rounded-xl border border-border p-3.5 shadow-xs flex flex-col gap-1">
          <div className="flex items-center justify-between text-text-muted text-xs font-medium">
            <span>Total Classes</span>
            <Sparkle size={14} className="text-primary" weight="fill" />
          </div>
          <span className="text-2xl font-bold text-text-primary font-mono">
            {metadata.total_classes}
          </span>
          <span className="text-[11px] text-text-muted">Original dataset classes</span>
        </div>

        <div className="bg-surface rounded-xl border border-border p-3.5 shadow-xs flex flex-col gap-1">
          <div className="flex items-center justify-between text-text-muted text-xs font-medium">
            <span>Digits</span>
            <Hash size={14} className="text-blue-500" />
          </div>
          <span className="text-2xl font-bold text-text-primary font-mono">
            {metadata.categories.digit}
          </span>
          <span className="text-[11px] text-text-muted">Numbers 0 through 9</span>
        </div>

        <div className="bg-surface rounded-xl border border-border p-3.5 shadow-xs flex flex-col gap-1">
          <div className="flex items-center justify-between text-text-muted text-xs font-medium">
            <span>Uppercase</span>
            <span className="text-xs font-bold font-mono text-indigo-500">A–Z</span>
          </div>
          <span className="text-2xl font-bold text-text-primary font-mono">
            {metadata.categories.uppercase}
          </span>
          <span className="text-[11px] text-text-muted">Capital letters</span>
        </div>

        <div className="bg-surface rounded-xl border border-border p-3.5 shadow-xs flex flex-col gap-1">
          <div className="flex items-center justify-between text-text-muted text-xs font-medium">
            <span>Lowercase</span>
            <span className="text-xs font-bold font-mono text-emerald-500">a–z</span>
          </div>
          <span className="text-2xl font-bold text-text-primary font-mono">
            {metadata.categories.lowercase}
          </span>
          <span className="text-[11px] text-text-muted">Small letters</span>
        </div>

        <div className="bg-surface rounded-xl border border-border p-3.5 shadow-xs flex flex-col gap-1 col-span-2 sm:col-span-1">
          <div className="flex items-center justify-between text-text-muted text-xs font-medium">
            <span>Operators</span>
            <span className="text-xs font-bold font-mono text-amber-500">+−×</span>
          </div>
          <span className="text-2xl font-bold text-text-primary font-mono">
            {metadata.categories.operator}
          </span>
          <span className="text-[11px] text-text-muted">+, −, *, ÷, %</span>
        </div>
      </div>

      {/* Controls Bar: Search & Category Filter Pills */}
      <div className="bg-surface rounded-xl border border-border p-4 shadow-sm flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
        {/* Category Filter Pills */}
        <div className="flex flex-wrap items-center gap-1.5 order-2 md:order-1">
          {(
            [
              { id: 'all', label: 'All', count: metadata.total_classes },
              { id: 'digit', label: 'Digits', count: metadata.categories.digit },
              { id: 'uppercase', label: 'Uppercase', count: metadata.categories.uppercase },
              { id: 'lowercase', label: 'Lowercase', count: metadata.categories.lowercase },
              { id: 'operator', label: 'Operators', count: metadata.categories.operator },
            ] as const
          ).map((cat) => {
            const isActive = activeCategory === cat.id;
            return (
              <button
                key={cat.id}
                type="button"
                onClick={() => setActiveCategory(cat.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                  isActive
                    ? 'bg-primary text-white shadow-xs font-semibold'
                    : 'bg-base-200/60 text-text-secondary hover:text-text-primary hover:bg-base-200'
                }`}
              >
                <span>{cat.label}</span>
                <span
                  className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono ${
                    isActive ? 'bg-white/25 text-white' : 'bg-base-300 text-text-muted'
                  }`}
                >
                  {cat.count}
                </span>
              </button>
            );
          })}
        </div>

        {/* Search Input Field */}
        <div className="relative order-1 md:order-2 w-full md:w-72">
          <MagnifyingGlass
            size={18}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none"
          />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search characters..."
            className="w-full pl-9 pr-8 py-2 rounded-lg bg-base-200/50 border border-border text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-primary focus:bg-base-100 transition-colors"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery('')}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary p-0.5 rounded"
              title="Clear search"
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Results Header Info */}
      <div className="flex items-center justify-between text-xs text-text-muted px-1">
        <span>
          Showing <strong>{filteredCharacters.length}</strong> of{' '}
          <strong>{metadata.total_classes}</strong> character classes
        </span>
        {searchQuery && (
          <span className="italic">
            Filtered by: &quot;{searchQuery}&quot;
          </span>
        )}
      </div>

      {/* Character Cards Gallery Grid */}
      {filteredCharacters.length === 0 ? (
        <div className="bg-surface rounded-xl border border-dashed border-border p-12 text-center flex flex-col items-center justify-center gap-3">
          <WarningCircle size={36} className="text-text-muted opacity-60" />
          <span className="text-base font-semibold text-text-primary">No characters found</span>
          <p className="text-xs text-text-secondary max-w-sm">
            No character class matched your search &quot;{searchQuery}&quot; in the &quot;{activeCategory}&quot; category.
          </p>
          <button
            type="button"
            onClick={() => {
              setSearchQuery('');
              setActiveCategory('all');
            }}
            className="btn btn-sm btn-outline border-border text-xs mt-2"
          >
            Reset Filters
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {filteredCharacters.map((char) => {
            const hasImages = char.images && char.images.length > 0;
            const validImages = hasImages
              ? char.images.filter((img) => !failedImagePaths.has(img.path))
              : [];

            return (
              <div
                key={char.id}
                className="bg-surface rounded-xl border border-border hover:border-primary/40 transition-all p-4 shadow-xs flex flex-col gap-3 group"
              >
                {/* Card Top: Character Glyph & Category / Count Badges */}
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2.5">
                    <div className="w-12 h-12 rounded-xl bg-base-200/60 border border-border flex items-center justify-center shrink-0 shadow-inner group-hover:border-primary/30 transition-colors">
                      <span className="font-mono font-bold text-2xl text-text-primary select-all">
                        {char.character}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-xs font-semibold text-text-primary leading-tight">
                        {char.name}
                      </span>
                      <span className="text-[10px] text-text-muted font-mono mt-0.5">
                        id: {char.id}
                      </span>
                    </div>
                  </div>

                  <span
                    className={`badge badge-xs text-[10px] font-semibold border ${getCategoryBadgeClass(
                      char.category
                    )}`}
                  >
                    {getCategoryLabel(char.category)}
                  </span>
                </div>

                {/* References Count Header */}
                <div className="flex items-center justify-between text-[11px] text-text-muted pt-2 border-t border-border/60">
                  <span>Reference Variants:</span>
                  <span className="font-semibold text-text-secondary font-mono">
                    {char.images.length} {char.images.length === 1 ? 'variant' : 'variants'}
                  </span>
                </div>

                {/* Card Center: Dynamic Reference Images or Neutral Placeholder */}
                {hasImages && validImages.length > 0 ? (
                  <div className={validImages.length === 1 ? "flex justify-center pt-1" : "grid grid-cols-2 gap-2 pt-1"}>
                    {validImages.map((img, idx) => {
                      const imageSrc = img.path.startsWith('http')
                        ? img.path
                        : apiUrl(img.path.startsWith('/') ? img.path : `/static/character-references/${img.path}`);
                      const variantLabel = img.label || `Variant ${idx + 1}`;

                      return (
                        <div
                          key={idx}
                          className={`flex flex-col items-center gap-1 cursor-pointer group/img ${
                            validImages.length === 1 ? 'w-full max-w-[130px]' : ''
                          }`}
                          onClick={() =>
                            setPreviewModal({ character: char, image: img, index: idx })
                          }
                          title={`Click to enlarge ${variantLabel} for '${char.character}'`}
                        >
                          <div className="relative w-full aspect-square bg-black rounded-lg border border-border overflow-hidden p-1 flex items-center justify-center group-hover/img:border-primary transition-colors shadow-xs">
                            <img
                              src={imageSrc}
                              alt={`${char.name} reference ${variantLabel}`}
                              onError={() => handleImageError(img.path)}
                              className="w-full h-full object-contain"
                              style={{ imageRendering: 'pixelated' }}
                            />
                            <div className="absolute inset-0 bg-primary/20 opacity-0 group-hover/img:opacity-100 transition-opacity flex items-center justify-center">
                              <Eye size={18} weight="bold" className="text-white drop-shadow" />
                            </div>
                          </div>
                          <span className="text-[10px] text-text-muted font-medium truncate max-w-full">
                            {variantLabel}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="py-5 px-3 rounded-lg border border-dashed border-border/80 bg-base-200/30 flex flex-col items-center justify-center gap-1.5 text-center transition-colors">
                    <span className="text-xs text-text-muted font-medium">
                      References not added yet.
                    </span>
                    <span className="text-[10px] text-text-muted/70 font-mono">
                      Target: {char.folder}/
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Enlarged Image Preview Modal */}
      {previewModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4 animate-in fade-in"
          onClick={() => setPreviewModal(null)}
          role="dialog"
          aria-modal="true"
          aria-label={`Enlarged preview for ${previewModal.character.name}`}
        >
          <div
            className="bg-surface rounded-2xl border border-border shadow-2xl max-w-md w-full p-5 flex flex-col gap-4 animate-in zoom-in-95"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between pb-3 border-b border-border">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-lg bg-base-200 flex items-center justify-center font-mono font-bold text-xl text-text-primary border border-border">
                  {previewModal.character.character}
                </div>
                <div>
                  <h3 className="text-base font-bold text-text-primary leading-tight">
                    {previewModal.character.name}
                  </h3>
                  <span className="text-xs text-text-muted">
                    {previewModal.image.label || `Variant ${previewModal.index + 1}`}
                  </span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setPreviewModal(null)}
                className="btn btn-sm btn-ghost btn-circle text-text-muted hover:text-text-primary"
                aria-label="Close image preview"
              >
                <X size={20} />
              </button>
            </div>

            {/* Modal Image Display */}
            <div className="w-full aspect-square bg-black rounded-xl border border-border p-4 flex items-center justify-center overflow-hidden shadow-inner">
              <img
                src={
                  previewModal.image.path.startsWith('http')
                    ? previewModal.image.path
                    : apiUrl(previewModal.image.path.startsWith('/') ? previewModal.image.path : `/static/character-references/${previewModal.image.path}`)
                }
                alt={`${previewModal.character.name} enlarged reference`}
                className="w-full h-full object-contain"
                style={{ imageRendering: 'pixelated' }}
              />
            </div>

            {/* Modal Footer: Metadata & Close button */}
            <div className="flex items-center justify-between pt-1 text-xs text-text-muted">
              <span className="font-mono text-[11px] truncate max-w-[240px]">
                {previewModal.image.path}
              </span>
              <button
                type="button"
                onClick={() => setPreviewModal(null)}
                className="btn btn-sm btn-primary px-4 text-white"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Guidance Note at Bottom */}
      <div className="p-4 rounded-xl bg-base-200/40 border border-border text-xs text-text-muted flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <span>
          To add reference images for a character class, copy selected sample images into{' '}
          <code className="text-text-primary font-mono bg-base-100 px-1 py-0.5 rounded border border-border">
            backend/static/character-references/
          </code>{' '}
          and register their paths in{' '}
          <code className="text-text-primary font-mono bg-base-100 px-1 py-0.5 rounded border border-border">
            manifest.json
          </code>.
        </span>
        {loadError && (
          <span className="text-warning text-xs font-medium shrink-0">
            Offline mode (serving default catalog)
          </span>
        )}
      </div>
    </div>
  );
}
