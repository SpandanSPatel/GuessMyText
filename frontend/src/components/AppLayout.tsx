/**
 * GuessMyText — App Layout
 *
 * Reused from CaptureMyText's CollectorLayout.tsx.
 * Same sidebar + main content structure with responsive padding.
 *
 * Nav items: Character Reference
 */

import { Outlet } from 'react-router-dom';
import { BookOpen } from '@phosphor-icons/react';
import { Sidebar } from './Sidebar';
import type { NavItem } from './Sidebar';

const navItems: NavItem[] = [
  { label: 'Character Reference', path: '/character-reference', icon: BookOpen },
];

export function AppLayout() {
  return (
    <div className="min-h-screen bg-bg-main">
      <Sidebar navItems={navItems} />

      {/* Main content area — offset by sidebar width at each breakpoint */}
      <main className="pt-14 md:pt-0 md:pl-16 lg:pl-60 min-h-screen">
        <div className="p-4 md:p-6 lg:p-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
