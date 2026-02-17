import React, { useState } from 'react';
import { authAPI } from '../services/api';
import type { AuthResponse } from '../types';

interface LoginProps {
  onSuccess: (response: AuthResponse) => void;
}

const Login = ({ onSuccess }: LoginProps) => {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [phone, setPhone] = useState('');
  const [bio, setBio] = useState('');
  const [interests, setInterests] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      let response: AuthResponse | null = null;
      if (isRegister) {
        response = await authAPI.register({
          email,
          password,
          username,
          phone,
          bio,
          interests,
        });
      } else {
        response = await authAPI.login(email, password);
      }
      if (response) {
        onSuccess(response);
      }
    } catch (err: any) {
      const apiMessage = err?.response?.data?.detail;
      if (apiMessage) {
        setError(apiMessage);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Login failed');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-header">
          <h1 className="auth-title">WordCraft</h1>
          <p className="auth-subtitle">Your writing companion</p>
        </div>

        <div className="auth-tabs">
          <button
            type="button"
            className={`auth-tab ${!isRegister ? 'is-active' : ''}`}
            onClick={() => setIsRegister(false)}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`auth-tab ${isRegister ? 'is-active' : ''}`}
            onClick={() => setIsRegister(true)}
          >
            Register
          </button>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          {isRegister && (
            <div className="auth-field">
              <label className="auth-label">Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="yourname"
                className="auth-input"
                required
              />
            </div>
          )}

          <div className={`auth-field ${!isRegister ? 'full' : ''}`}>
            <label className="auth-label">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="auth-input"
              required
            />
          </div>

          {isRegister && (
            <div className="auth-field">
              <label className="auth-label">Phone</label>
              <input
                type="tel"
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
                placeholder="+1 555 123 4567"
                className="auth-input"
              />
            </div>
          )}

          <div className={`auth-field ${!isRegister ? 'full' : ''}`}>
            <label className="auth-label">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="auth-input"
              required
            />
          </div>

          {isRegister && (
            <>
              <div className="auth-field full">
                <label className="auth-label">Bio</label>
                <textarea
                  value={bio}
                  onChange={(e) => setBio(e.target.value)}
                  placeholder="Tell us about your writing style."
                  className="auth-textarea"
                />
              </div>
              <div className="auth-field full">
                <label className="auth-label">Interests</label>
                <input
                  type="text"
                  value={interests}
                  onChange={(e) => setInterests(e.target.value)}
                  placeholder="Poetry, Fantasy, Essays"
                  className="auth-input"
                />
              </div>
            </>
          )}

          {error && <div className="auth-error">{error}</div>}

          <div className="auth-actions">
            <button type="submit" className="auth-submit" disabled={loading}>
              {loading
                ? isRegister
                  ? 'Creating...'
                  : 'Signing in...'
                : isRegister
                  ? 'Create Account'
                  : 'Sign In'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Login;
