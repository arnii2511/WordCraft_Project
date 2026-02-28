import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { authAPI } from '../services/api';
import AppHeader from '../components/AppHeader';
import type { UserProfile } from '../types';

interface ProfilePageProps {
  user: UserProfile | null;
  isAuthenticated: boolean;
  onRequireAuth: () => void;
  onLogout: () => void;
  onUserUpdate: (profile: UserProfile) => void;
}

const emailPattern = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
const indiaPhonePattern = /^\+91\s[6-9]\d{9}$/;
const strongPasswordPattern =
  /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$/;

const ProfilePage = ({
  user,
  isAuthenticated,
  onRequireAuth,
  onLogout,
  onUserUpdate,
}: ProfilePageProps) => {
  const navigate = useNavigate();
  const [username, setUsername] = useState(user?.username ?? '');
  const [email, setEmail] = useState(user?.email ?? '');
  const [phone, setPhone] = useState(user?.phone ?? '');
  const [bio, setBio] = useState(user?.bio ?? '');
  const [interests, setInterests] = useState(user?.interests ?? '');

  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const [loadingProfile, setLoadingProfile] = useState(false);
  const [loadingPassword, setLoadingPassword] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    if (!isAuthenticated) {
      onRequireAuth();
      navigate('/');
      return;
    }
    let active = true;
    void authAPI
      .getMe()
      .then((profile) => {
        if (!active) return;
        onUserUpdate(profile);
        setUsername(profile.username ?? '');
        setEmail(profile.email ?? '');
        setPhone(profile.phone ?? '');
        setBio(profile.bio ?? '');
        setInterests(profile.interests ?? '');
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, [isAuthenticated]);

  const validateProfile = (): string | null => {
    if (!username.trim() || username.trim().length < 2) {
      return 'Username must be at least 2 characters.';
    }
    if (!emailPattern.test(email.trim().toLowerCase())) {
      return 'Enter a valid email address.';
    }
    if (phone.trim() && !indiaPhonePattern.test(phone.trim())) {
      return 'Phone must be in India format: +91 9876543210.';
    }
    return null;
  };

  const validatePassword = (): string | null => {
    if (!currentPassword) {
      return 'Enter current password.';
    }
    if (!strongPasswordPattern.test(newPassword)) {
      return 'New password must be 8+ chars with uppercase, lowercase, number, and special character.';
    }
    if (newPassword !== confirmPassword) {
      return 'Confirm password does not match new password.';
    }
    return null;
  };

  const handleSaveProfile = async () => {
    setError('');
    setSuccess('');
    const validationError = validateProfile();
    if (validationError) {
      setError(validationError);
      return;
    }

    setLoadingProfile(true);
    try {
      const profile = await authAPI.updateMe({
        username: username.trim(),
        email: email.trim().toLowerCase(),
        phone: phone.trim(),
        bio: bio.trim(),
        interests: interests.trim(),
      });
      onUserUpdate(profile);
      setSuccess('Profile updated successfully.');
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to update profile.');
    } finally {
      setLoadingProfile(false);
    }
  };

  const handleChangePassword = async () => {
    setError('');
    setSuccess('');
    const validationError = validatePassword();
    if (validationError) {
      setError(validationError);
      return;
    }

    setLoadingPassword(true);
    try {
      await authAPI.changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      });
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setSuccess('Password changed successfully.');
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Unable to change password.');
    } finally {
      setLoadingPassword(false);
    }
  };

  return (
    <div className="profile-page">
      <AppHeader
        activePage="profile"
        isAuthenticated={isAuthenticated}
        user={user}
        onRequireAuth={onRequireAuth}
      />

      <section className="profile-card">
        <h2>Account Details</h2>
        <p className="profile-subtext">Manage your profile and security settings.</p>

        <div className="profile-grid">
          <label className="auth-field">
            <span className="auth-label">Username</span>
            <input
              className="auth-input"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              placeholder="yourname"
            />
          </label>

          <label className="auth-field">
            <span className="auth-label">Email</span>
            <input
              className="auth-input"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
            />
          </label>

          <label className="auth-field">
            <span className="auth-label">Phone</span>
            <input
              className="auth-input"
              value={phone}
              onChange={(event) => setPhone(event.target.value)}
              placeholder="+91 9876543210"
            />
          </label>

          <label className="auth-field">
            <span className="auth-label">Interests</span>
            <input
              className="auth-input"
              value={interests}
              onChange={(event) => setInterests(event.target.value)}
              placeholder="Poetry, Fantasy, Essays"
            />
          </label>

          <label className="auth-field full">
            <span className="auth-label">Bio</span>
            <textarea
              className="auth-textarea"
              value={bio}
              onChange={(event) => setBio(event.target.value)}
              placeholder="Tell us about your writing style."
            />
          </label>
        </div>

        <div className="profile-section">
          <h3>Change Password</h3>
          <div className="profile-grid">
            <label className="auth-field">
              <span className="auth-label">Current Password</span>
              <input
                className="auth-input"
                type="password"
                value={currentPassword}
                onChange={(event) => setCurrentPassword(event.target.value)}
                placeholder="Current password"
              />
            </label>

            <label className="auth-field">
              <span className="auth-label">New Password</span>
              <input
                className="auth-input"
                type="password"
                value={newPassword}
                onChange={(event) => setNewPassword(event.target.value)}
                placeholder="New strong password"
              />
            </label>

            <label className="auth-field full">
              <span className="auth-label">Confirm New Password</span>
              <input
                className="auth-input"
                type="password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
                placeholder="Re-enter new password"
              />
            </label>
          </div>
        </div>

        {error && <div className="auth-error">{error}</div>}
        {success && <div className="auth-success">{success}</div>}

        <div className="profile-actions">
          <button
            type="button"
            className="btn-accept"
            disabled={loadingProfile}
            onClick={() => void handleSaveProfile()}
          >
            {loadingProfile ? 'Saving...' : 'Save Profile'}
          </button>
          <button
            type="button"
            className="btn-outline"
            disabled={loadingPassword}
            onClick={() => void handleChangePassword()}
          >
            {loadingPassword ? 'Updating...' : 'Change Password'}
          </button>
          <button
            type="button"
            className="btn-ghost"
            onClick={() => {
              onLogout();
              navigate('/');
            }}
          >
            Logout
          </button>
        </div>
      </section>
    </div>
  );
};

export default ProfilePage;
