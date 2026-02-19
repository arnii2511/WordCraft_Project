import React, { useState } from 'react';
import { authAPI } from '../services/api';
import type { AuthResponse } from '../types';

interface LoginProps {
  onSuccess: (response: AuthResponse) => void;
}

const Login = ({ onSuccess }: LoginProps) => {
  const [isRegister, setIsRegister] = useState(false);

  const [signInEmail, setSignInEmail] = useState('');
  const [signInPassword, setSignInPassword] = useState('');

  const [registerEmail, setRegisterEmail] = useState('');
  const [registerPassword, setRegisterPassword] = useState('');
  const [registerUsername, setRegisterUsername] = useState('');
  const [registerPhone, setRegisterPhone] = useState('');
  const [registerBio, setRegisterBio] = useState('');
  const [registerInterests, setRegisterInterests] = useState('');

  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showRegisterPrompt, setShowRegisterPrompt] = useState(false);
  const [loading, setLoading] = useState(false);

  const emailPattern = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
  const strongPasswordPattern =
    /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$/;
  const indiaPhonePattern = /^\+91\s[6-9]\d{9}$/;

  const switchTab = (registerMode: boolean) => {
    setIsRegister(registerMode);
    setError('');
    setSuccess('');
    setShowRegisterPrompt(false);
  };

  const validateSignIn = (): string | null => {
    const email = signInEmail.trim().toLowerCase();
    if (!emailPattern.test(email)) {
      return 'Enter a valid email address.';
    }
    if (!signInPassword) {
      return 'Enter your password.';
    }
    return null;
  };

  const validateRegister = (): string | null => {
    const email = registerEmail.trim().toLowerCase();
    if (registerUsername.trim().length < 2) {
      return 'Username must be at least 2 characters.';
    }
    if (!emailPattern.test(email)) {
      return 'Enter a valid email address.';
    }
    if (!indiaPhonePattern.test(registerPhone.trim())) {
      return 'Phone must be in India format: +91 9876543210.';
    }
    if (!strongPasswordPattern.test(registerPassword)) {
      return 'Password must be 8+ chars with uppercase, lowercase, number, and special character.';
    }
    return null;
  };

  const clearRegisterForm = () => {
    setRegisterEmail('');
    setRegisterPassword('');
    setRegisterUsername('');
    setRegisterPhone('');
    setRegisterBio('');
    setRegisterInterests('');
  };

  const resolveAuthError = (err: any, registerMode: boolean): string => {
    const statusCode = err?.response?.status;
    const apiMessage = err?.response?.data?.detail;

    if (!err?.response) {
      return 'Unable to reach the server. Please try again.';
    }

    if (statusCode === 400) {
      return registerMode
        ? 'Could not create account. Please check your details and try again.'
        : 'Could not sign in with those details.';
    }
    if (statusCode === 401) {
      return 'Incorrect password. Please try again.';
    }
    if (statusCode === 404) {
      return registerMode
        ? 'Registration is unavailable right now. Please try again shortly.'
        : 'Account not found. Please register first.';
    }
    if (statusCode === 409) {
      return 'This account already exists. Please sign in instead.';
    }
    if (statusCode >= 500) {
      return 'Server error. Please try again in a moment.';
    }

    if (typeof apiMessage === 'string') {
      const cleaned = apiMessage.trim();
      if (cleaned && cleaned.toLowerCase() !== 'not found') {
        return cleaned;
      }
    }

    if (Array.isArray(apiMessage) && apiMessage.length > 0) {
      const firstDetail = apiMessage[0]?.msg;
      if (typeof firstDetail === 'string' && firstDetail.trim()) {
        return firstDetail;
      }
    }

    return registerMode ? 'Account creation failed.' : 'Sign in failed.';
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setShowRegisterPrompt(false);

    const validationError = isRegister ? validateRegister() : validateSignIn();
    if (validationError) {
      setError(validationError);
      return;
    }

    setLoading(true);
    try {
      if (isRegister) {
        await authAPI.register({
          email: registerEmail.trim().toLowerCase(),
          password: registerPassword,
          username: registerUsername.trim(),
          phone: registerPhone.trim(),
          bio: registerBio.trim(),
          interests: registerInterests.trim(),
        });

        clearRegisterForm();
        setSignInEmail('');
        setSignInPassword('');
        setIsRegister(false);
        setSuccess('Account created successfully. Please sign in.');
        return;
      }

      const response: AuthResponse = await authAPI.login(
        signInEmail.trim().toLowerCase(),
        signInPassword,
      );
      onSuccess(response);
    } catch (err: any) {
      const statusCode = err?.response?.status;
      if (!isRegister && statusCode === 404) {
        setError(resolveAuthError(err, false));
        setShowRegisterPrompt(true);
      } else if (err instanceof Error) {
        setError(resolveAuthError(err, isRegister));
      } else {
        setError(resolveAuthError(err, isRegister));
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
            onClick={() => switchTab(false)}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`auth-tab ${isRegister ? 'is-active' : ''}`}
            onClick={() => switchTab(true)}
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
                value={registerUsername}
                onChange={(e) => setRegisterUsername(e.target.value)}
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
              value={isRegister ? registerEmail : signInEmail}
              onChange={(e) =>
                isRegister
                  ? setRegisterEmail(e.target.value)
                  : setSignInEmail(e.target.value)
              }
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
                value={registerPhone}
                onChange={(e) => setRegisterPhone(e.target.value)}
                placeholder="+91 9876543210"
                className="auth-input"
                required
              />
            </div>
          )}

          <div className={`auth-field ${!isRegister ? 'full' : ''}`}>
            <label className="auth-label">Password</label>
            <input
              type="password"
              value={isRegister ? registerPassword : signInPassword}
              onChange={(e) =>
                isRegister
                  ? setRegisterPassword(e.target.value)
                  : setSignInPassword(e.target.value)
              }
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
                  value={registerBio}
                  onChange={(e) => setRegisterBio(e.target.value)}
                  placeholder="Tell us about your writing style."
                  className="auth-textarea"
                />
              </div>
              <div className="auth-field full">
                <label className="auth-label">Interests</label>
                <input
                  type="text"
                  value={registerInterests}
                  onChange={(e) => setRegisterInterests(e.target.value)}
                  placeholder="Poetry, Fantasy, Essays"
                  className="auth-input"
                />
              </div>
            </>
          )}

          {error && <div className="auth-error">{error}</div>}
          {success && <div className="auth-success">{success}</div>}

          <div className="auth-actions">
            {showRegisterPrompt && !isRegister && (
              <button
                type="button"
                className="panel-link"
                onClick={() => switchTab(true)}
              >
                Go to Register
              </button>
            )}
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
