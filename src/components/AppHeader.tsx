import React from 'react';
import { useNavigate } from 'react-router-dom';
import type { UserProfile } from '../types';

type HeaderPage = 'tools' | 'editor' | 'profile';

interface AppHeaderProps {
  activePage: HeaderPage;
  isAuthenticated: boolean;
  user: UserProfile | null;
  onRequireAuth: () => void;
}

const AppHeader = ({ activePage, isAuthenticated, user, onRequireAuth }: AppHeaderProps) => {
  const navigate = useNavigate();
  const profileLabel = user?.username?.trim().charAt(0).toUpperCase() || 'P';

  const toolsActive = activePage === 'tools';
  const editorActive = activePage === 'editor';

  return (
    <header className="tools-home-topbar topbar">
      <h1 className="tools-home-brand topbar-title">WordCraft</h1>
      <div className="tools-home-top-actions topbar-right">
        <button
          type="button"
          className={`btn-outline ${toolsActive ? 'is-active' : ''}`}
          onClick={() => {
            if (!toolsActive) navigate('/');
          }}
          aria-current={toolsActive ? 'page' : undefined}
        >
          Tools
        </button>
        <button
          type="button"
          className={`btn-outline ${editorActive ? 'is-active' : ''}`}
          onClick={() => {
            if (!editorActive) navigate('/editor');
          }}
          aria-current={editorActive ? 'page' : undefined}
        >
          Editor
        </button>
        {isAuthenticated ? (
          <button
            type="button"
            className="profile-shortcut"
            onClick={() => navigate('/profile')}
            title="Open profile"
            aria-current={activePage === 'profile' ? 'page' : undefined}
          >
            {profileLabel}
          </button>
        ) : (
          <button type="button" className="btn-outline" onClick={onRequireAuth}>
            Login / Register
          </button>
        )}
      </div>
    </header>
  );
};

export default AppHeader;
