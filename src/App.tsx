import React, { useEffect, useState } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { authAPI } from './services/api';
import { clearAllSnapshots } from './services/sessionHistory';
import Login from './components/Login';
import EditorPage from './pages/EditorPage';
import ProfilePage from './pages/ProfilePage';
import ToolsHome from './pages/ToolsHome';
import type { AuthResponse, UserProfile } from './types';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<UserProfile | null>(null);
  const [context, setContext] = useState('neutral');
  const [mode, setMode] = useState<'write' | 'edit' | 'rewrite'>('write');
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showAuth, setShowAuth] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      authAPI
        .getMe()
        .then((profile) => {
          setIsAuthenticated(true);
          setUser(profile);
        })
        .catch(() => {
          authAPI.logout();
          setIsAuthenticated(false);
          setUser(null);
        });
    }
    const onboarded = localStorage.getItem('wordcraft_onboarded');
    if (!onboarded) {
      setShowOnboarding(true);
    }
  }, []);

  const handleLogin = (userData: UserProfile) => {
    setIsAuthenticated(true);
    setUser(userData);
    setShowOnboarding(false);
  };

  const handleLogout = () => {
    authAPI.logout();
    clearAllSnapshots();
    setIsAuthenticated(false);
    setUser(null);
    localStorage.removeItem('active_document_id');
  };

  const handleUserUpdate = (profile: UserProfile) => {
    setUser(profile);
  };

  return (
    <div className="app-shell">
      {showAuth && (
        <Login
          onSuccess={(userData: AuthResponse) => {
            handleLogin(userData.user);
            setShowAuth(false);
          }}
        />
      )}

      {showOnboarding && (
        <div className="onboarding-overlay">
          <div className="onboarding-card">
            <h2>Welcome to WordCraft</h2>
            <ul>
              <li>Home is tools-first for quick lexical wins</li>
              <li>Editor mode is your deep drafting workspace</li>
              <li>Use Start Writing anytime to move into `/editor`</li>
            </ul>
            <button
              className="btn-accept"
              onClick={() => {
                localStorage.setItem('wordcraft_onboarded', 'true');
                setShowOnboarding(false);
              }}
            >
              Got it
            </button>
          </div>
        </div>
      )}

      <Routes>
        <Route
          path="/"
          element={
            <ToolsHome
              context={context}
              setContext={setContext}
              isAuthenticated={isAuthenticated}
              user={user}
              onRequireAuth={() => setShowAuth(true)}
            />
          }
        />
        <Route
          path="/editor"
          element={
            <EditorPage
              context={context}
              setContext={setContext}
              mode={mode}
              setMode={setMode}
              user={user}
              isAuthenticated={isAuthenticated}
              onLogout={handleLogout}
              onRequireAuth={() => setShowAuth(true)}
            />
          }
        />
        <Route
          path="/profile"
          element={
            isAuthenticated ? (
              <ProfilePage
                user={user}
                isAuthenticated={isAuthenticated}
                onRequireAuth={() => setShowAuth(true)}
                onLogout={handleLogout}
                onUserUpdate={handleUserUpdate}
              />
            ) : (
              <Navigate to="/" replace />
            )
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>

      <footer className="app-footer">
        <span>WordCraft © 2026</span>
      </footer>
    </div>
  );
}

export default App;
