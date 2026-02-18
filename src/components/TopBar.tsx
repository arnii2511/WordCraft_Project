import React from 'react';
import type { UserProfile } from '../types';

interface TopBarProps {
  user: UserProfile | null;
  isAuthenticated: boolean;
  onLogin?: () => void;
  onToggleTools?: () => void;
  onOpenProfile?: () => void;
}

const TopBar = ({
  user,
  isAuthenticated,
  onLogin = () => {},
  onToggleTools = () => {},
  onOpenProfile = () => {},
}: TopBarProps) => {
  const profileLabel = user?.username?.trim().charAt(0).toUpperCase() || 'P';

  return (
    <header className="tools-home-topbar topbar">
      <div className="topbar-left">
        <h1 className="topbar-title">WordCraft</h1>
      </div>

      <div className="tools-home-top-actions topbar-right">
        <button type="button" className="btn-outline" onClick={onToggleTools}>
          Tools
        </button>
        <button type="button" className="btn-outline is-active" aria-current="page">
          Editor
        </button>

        {isAuthenticated ? (
          <button
            type="button"
            className="profile-shortcut"
            onClick={onOpenProfile}
            title="Open profile"
          >
            {profileLabel}
          </button>
        ) : (
          <button type="button" className="btn-outline" onClick={onLogin}>
            Login / Register
          </button>
        )}
      </div>
    </header>
  );
};

export default TopBar;
