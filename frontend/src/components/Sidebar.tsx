/**
 * GuessMyText — Sidebar Navigation
 *
 * Reused from CaptureMyText's Sidebar.tsx with identical layout and responsive behavior:
 *   Desktop (≥1024px): Full sidebar with icon + label
 *   Tablet (768–1023px): Collapsed icon-only rail
 *   Mobile (<768px): Top bar + drawer overlay
 *
 * Changed branding from CaptureMyText to GuessMyText and accent color to #89CFF0.
 */

import { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'motion/react';
import { List, X, ScribbleLoop } from '@phosphor-icons/react';
import type { Icon } from '@phosphor-icons/react';

export interface NavItem {
  label: string;
  path: string;
  icon: Icon;
}

interface SidebarProps {
  navItems: NavItem[];
}

export function Sidebar({ navItems }: SidebarProps) {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  // Close mobile drawer on navigation
  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  // Close drawer on Escape key
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') setMobileOpen(false);
    }
    if (mobileOpen) {
      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }
  }, [mobileOpen]);

  const linkClasses = (isActive: boolean) =>
    `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
      isActive
        ? 'bg-primary-light text-primary font-semibold'
        : 'text-text-secondary hover:bg-base-200 hover:text-text-primary'
    }`;

  const railLinkClasses = (isActive: boolean) =>
    `flex items-center justify-center p-2.5 rounded-lg transition-colors ${
      isActive
        ? 'bg-primary-light text-primary'
        : 'text-text-secondary hover:bg-base-200 hover:text-text-primary'
    }`;

  const brandDesktop = (
    <div className="flex items-center gap-2.5">
      <div className="w-8 h-8 rounded-lg bg-primary-light flex items-center justify-center shrink-0 border border-primary/30">
        <ScribbleLoop size={20} weight="bold" className="text-primary" />
      </div>
      <div className="flex flex-col">
        <span className="text-base font-bold text-text-primary tracking-tight leading-none">
          Guess<span className="text-primary">My</span>Text
        </span>
        <span className="text-[10px] text-text-muted uppercase tracking-wider font-medium">
          ML Character AI
        </span>
      </div>
    </div>
  );

  const brandTablet = (
    <div className="w-8 h-8 rounded-lg bg-primary-light flex items-center justify-center shrink-0 border border-primary/30">
      <ScribbleLoop size={18} weight="bold" className="text-primary" />
    </div>
  );

  return (
    <>
      {/* ---- Mobile Top Bar ---- */}
      <div className="md:hidden fixed top-0 left-0 right-0 h-14 bg-surface border-b border-border flex items-center px-4 z-40">
        <button
          onClick={() => setMobileOpen(true)}
          className="p-1.5 rounded-lg hover:bg-base-200 text-text-secondary"
          aria-label="Open navigation menu"
        >
          <List size={24} />
        </button>
        <div className="ml-3">{brandDesktop}</div>
      </div>

      {/* ---- Mobile Drawer Overlay ---- */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="md:hidden fixed inset-0 bg-black/40 z-50"
            onClick={() => setMobileOpen(false)}
            aria-hidden="true"
          />
        )}
      </AnimatePresence>

      {/* ---- Mobile Drawer ---- */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.aside
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="md:hidden fixed top-0 left-0 bottom-0 w-64 bg-surface border-r border-border z-50 flex flex-col"
          >
            <div className="flex items-center justify-between px-4 h-14 border-b border-border">
              {brandDesktop}
              <button
                onClick={() => setMobileOpen(false)}
                className="p-1.5 rounded-lg hover:bg-base-200 text-text-secondary"
                aria-label="Close navigation menu"
              >
                <X size={20} />
              </button>
            </div>

            <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
              {navItems.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => linkClasses(isActive)}
                >
                  <item.icon size={20} weight="duotone" />
                  <span>{item.label}</span>
                </NavLink>
              ))}
            </nav>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* ---- Tablet Rail (md – lg) ---- */}
      <aside className="hidden md:flex lg:hidden fixed top-0 left-0 bottom-0 w-16 bg-surface border-r border-border flex-col z-30">
        <div className="flex items-center justify-center h-14 border-b border-border">
          {brandTablet}
        </div>

        <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              title={item.label}
              className={({ isActive }) => railLinkClasses(isActive)}
            >
              <item.icon size={22} weight="duotone" />
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* ---- Desktop Full Sidebar (lg+) ---- */}
      <aside className="hidden lg:flex fixed top-0 left-0 bottom-0 w-60 bg-surface border-r border-border flex-col z-30">
        <div className="flex items-center px-4 h-14 border-b border-border">
          {brandDesktop}
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => linkClasses(isActive)}
            >
              <item.icon size={20} weight="duotone" />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </aside>
    </>
  );
}
