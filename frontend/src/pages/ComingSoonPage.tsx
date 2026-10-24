/**
 * GuessMyText — Coming Soon Page
 *
 * Simple placeholder for V2–V4 routes, styled in accordance with
 * CaptureMyText's page layout, typography, and card conventions.
 */

import { Clock } from '@phosphor-icons/react';

interface ComingSoonPageProps {
  title: string;
  version: string;
  description?: string;
}

export function ComingSoonPage({ title, version, description }: ComingSoonPageProps) {
  return (
    <div className="max-w-4xl mx-auto flex flex-col gap-6">
      {/* Page Header */}
      <div className="flex items-center gap-3">
        <Clock size={28} weight="duotone" className="text-primary" />
        <h1 className="text-2xl font-semibold text-text-primary">{title}</h1>
      </div>

      {/* Content Card */}
      <div className="bg-surface rounded-xl border border-border p-12 text-center flex flex-col items-center gap-4 shadow-sm">
        <div className="w-16 h-16 rounded-2xl bg-primary-light flex items-center justify-center border border-primary/20">
          <Clock size={32} weight="duotone" className="text-primary" />
        </div>

        <div>
          <span className="badge bg-primary-light text-primary border-primary/20 font-semibold mb-2">
            {version} • Coming Soon
          </span>
          <h2 className="text-xl font-semibold text-text-primary mb-1">{title}</h2>
          <p className="text-sm text-text-secondary max-w-md mx-auto">
            {description ||
              'This recognition module is planned for a future release. Single Character Recognition is available now in V1.'}
          </p>
        </div>
      </div>
    </div>
  );
}
