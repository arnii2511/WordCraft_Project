import React, { useState } from 'react';
import type { UserProfile } from '../types';

interface TopBarProps {
  context: string;
  setContext: (value: string) => void;
  mode: 'write' | 'edit' | 'rewrite';
  setMode: (value: 'write' | 'edit' | 'rewrite') => void;
  user: UserProfile | null;
  isAuthenticated: boolean;
  onLogout: () => void;
  onLogin?: () => void;
  onSave?: () => void;
  onToggleTools?: () => void;
  onOpenDocs?: () => void;
  onOpenHistory?: () => void;
  onOpenFavorites?: () => void;
}

const TopBar = ({
  context,
  setContext,
  mode,
  setMode,
  user,
  isAuthenticated,
  onLogout,
  onLogin = () => {},
  onSave = () => {},
  onToggleTools = () => {},
  onOpenDocs = () => {},
  onOpenHistory = () => {},
  onOpenFavorites = () => {},
}: TopBarProps) => {
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [showMenu, setShowMenu] = useState(false);

  const contexts = [
    'neutral',
    'hopeful',
    'horror',
    'nostalgia',
    'academic',
    'romantic',
    'joyful',
    'melancholic',
    'mysterious',
    'formal',
  ];

  const modes = [
    { id: 'write', label: 'Draft' },
    { id: 'edit', label: 'Polish' },
    { id: 'rewrite', label: 'Transform' },
  ];

  const modeDescriptions: Record<'write' | 'edit' | 'rewrite', string> = {
    write: 'Suggests tone-matching words and phrases.',
    edit: 'Fixes clarity, grammar, repetition.',
    rewrite: 'Generates rewrite versions on click.',
  };

  const formatLabel = (value: string) =>
    value ? value.charAt(0).toUpperCase() + value.slice(1) : 'Creative';

  return (
    <header className="topbar">
      <div className="topbar-left">
        <h1 className="topbar-title">WordCraft</h1>
      </div>

      <div className="topbar-center">
        <nav className="topbar-tabs">
          {modes.map((m) => (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className={`topbar-tab ${mode === m.id ? 'is-active' : ''}`}
              title={m.label}
            >
              {m.label}
            </button>
          ))}
        </nav>
        <div className="mode-description">{modeDescriptions[mode]}</div>
      </div>

      <div className="topbar-right">
        <div className="relative">
          <button
            onClick={() => setShowContextMenu(!showContextMenu)}
            className="context-button"
          >
            {formatLabel(context)}
            <span className="context-caret">▾</span>
          </button>

          {showContextMenu && (
            <div className="context-menu">
              {contexts.map((ctx) => (
                <button
                  key={ctx}
                  onClick={() => {
                    setContext(ctx);
                    setShowContextMenu(false);
                  }}
                  className={`context-item ${context === ctx ? 'is-selected' : ''}`}
                >
                  {formatLabel(ctx)}
                </button>
              ))}
            </div>
          )}
        </div>

        {isAuthenticated ? (
          <>
            <button onClick={onSave} className="btn-outline">
              Save
            </button>
            <div className="menu-wrapper">
              <button
                type="button"
                className="menu-button"
                onClick={() => setShowMenu((prev) => !prev)}
                aria-label="Open menu"
              >
                <span />
                <span />
                <span />
              </button>
              {showMenu && (
                <div className="menu-dropdown">
                  <div className="menu-profile">
                    <div className="menu-avatar">
                      {user?.username?.charAt(0).toUpperCase() || 'W'}
                    </div>
                    <div>
                      <div className="menu-name">{user?.username || 'Writer'}</div>
                      <div className="menu-email">{user?.email}</div>
                    </div>
                  </div>
                  <button
                    className="menu-item"
                    type="button"
                    onClick={() => {
                      onToggleTools();
                      setShowMenu(false);
                    }}
                  >
                    Tools
                  </button>
                  <button
                    className="menu-item"
                    type="button"
                    onClick={() => {
                      onOpenDocs();
                      setShowMenu(false);
                    }}
                  >
                    My Docs
                  </button>
                  <button
                    className="menu-item"
                    type="button"
                    onClick={() => {
                      onOpenHistory();
                      setShowMenu(false);
                    }}
                  >
                    History
                  </button>
                  <button
                    className="menu-item"
                    type="button"
                    onClick={() => {
                      onOpenFavorites();
                      setShowMenu(false);
                    }}
                  >
                    My Vocabulary
                  </button>
                  <button
                    className="menu-item danger"
                    type="button"
                    onClick={() => {
                      onLogout();
                      setShowMenu(false);
                    }}
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          </>
        ) : (
          <>
            <button type="button" className="btn-outline" onClick={onSave} disabled title="Login to save">
              Save
            </button>
            <button type="button" className="btn-outline" onClick={onToggleTools}>
              Tools
            </button>
            <button type="button" className="btn-outline" onClick={onLogin}>
              Login/Register to Save content
            </button>
          </>
        )}
      </div>
    </header>
  );
};

export default TopBar;
