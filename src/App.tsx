import React, { useEffect, useState } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { authAPI } from './services/api';
import { clearAllSnapshots } from './services/sessionHistory';
import Login from './components/Login';
import EditorPage from './pages/EditorPage';
import ProfilePage from './pages/ProfilePage';
import ToolsHome from './pages/ToolsHome';
import type { AuthResponse, UserProfile, VocabularyPreference } from './types';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(() => authAPI.isAuthenticated());
  const [user, setUser] = useState<UserProfile | null>(() => authAPI.getProfile());
  const [context, setContext] = useState('neutral');
  const [vocabularyPreference, setVocabularyPreference] =
    useState<VocabularyPreference>('balanced');
  const [mode, setMode] = useState<'write' | 'edit' | 'rewrite'>('write');
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showAuth, setShowAuth] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      const cachedProfile = authAPI.getProfile();
      if (cachedProfile) {
        setIsAuthenticated(true);
        setUser(cachedProfile);
      }
      authAPI
        .getMe()
        .then((profile) => {
          setIsAuthenticated(true);
          setUser(profile);
        })
        .catch((error) => {
          const statusCode = error?.response?.status;
          if (statusCode === 401 || statusCode === 403) {
            authAPI.logout();
            setIsAuthenticated(false);
            setUser(null);
            return;
          }
          if (cachedProfile) {
            setIsAuthenticated(true);
            setUser(cachedProfile);
          }
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

  useEffect(() => {
    if (!showAuth) return;
    window.history.pushState({ wordcraftAuthModal: true }, '');
    const handlePopState = () => {
      setShowAuth(false);
    };
    window.addEventListener('popstate', handlePopState);
    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  }, [showAuth]);

  const closeAuthModal = () => {
    const state = window.history.state as { wordcraftAuthModal?: boolean } | null;
    if (state?.wordcraftAuthModal) {
      window.history.back();
      return;
    }
    setShowAuth(false);
  };

  return (
    <div className="app-shell">
      {showAuth && (
        <Login
          onClose={closeAuthModal}
          onSuccess={(userData: AuthResponse) => {
            handleLogin(userData.user);
            closeAuthModal();
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
              vocabularyPreference={vocabularyPreference}
              setVocabularyPreference={setVocabularyPreference}
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
              vocabularyPreference={vocabularyPreference}
              setVocabularyPreference={setVocabularyPreference}
              mode={mode}
              setMode={setMode}
              user={user}
              isAuthenticated={isAuthenticated}
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
