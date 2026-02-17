import React, { useEffect, useState } from 'react';
import { favoritesAPI } from '../services/api';
import type { FavoriteEntry } from '../types';

interface FavoritesPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onInsertFavorite: (content: string) => void;
}

const FavoritesPanel = ({ isOpen, onClose, onInsertFavorite }: FavoritesPanelProps) => {
  const [favorites, setFavorites] = useState<FavoriteEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadFavorites();
    }
  }, [isOpen]);

  const loadFavorites = async () => {
    setLoading(true);
    try {
      const data = await favoritesAPI.getFavorites();
      setFavorites(data);
    } catch (error) {
      console.error('Error loading favorites:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: number | string) => {
    try {
      await favoritesAPI.deleteFavorite(id);
      setFavorites(favorites.filter((fav) => fav.id !== id));
    } catch (error) {
      console.error('Error deleting favorite:', error);
    }
  };

  const words = favorites.filter((fav) => fav.type !== 'rewrite');
  const rewrites = favorites.filter((fav) => fav.type === 'rewrite');

  return (
    <div
      className={`overlay ${isOpen ? 'is-open' : ''}`}
      onClick={onClose}
    >
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="panel-header">
          <h2 className="panel-title">My Vocabulary</h2>
          <button onClick={onClose} className="panel-close" type="button">
            ×
          </button>
        </div>

        <div className="panel-body">
          {loading ? (
            <p className="panel-muted">Loading...</p>
          ) : favorites.length === 0 ? (
            <p className="panel-muted">
              No saved words yet. Save words as you write to build your vocabulary.
            </p>
          ) : (
            <div className="panel-section">
              {words.length > 0 && (
                <div className="panel-block">
                  <h3 className="panel-subtitle">Words</h3>
                  <div className="panel-chip-row">
                    {words.map((word) => (
                      <div key={word.id} className="panel-chip">
                        <button
                          onClick={() => onInsertFavorite(word.word)}
                          className="panel-chip-button"
                        >
                          {word.word}
                        </button>
                        <span className="panel-item-date">{word.type}</span>
                        <button
                          onClick={() => handleDelete(word.id)}
                          className="panel-chip-remove"
                          title="Remove"
                          type="button"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {rewrites.length > 0 && (
                <div className="panel-block">
                  <h3 className="panel-subtitle">Rewrites</h3>
                  <div className="panel-list">
                    {rewrites.map((rewrite) => (
                      <div key={rewrite.id} className="panel-item-card">
                        <p className="panel-item-text">{rewrite.word}</p>
                        <div className="panel-actions">
                          <button
                            onClick={() => onInsertFavorite(rewrite.word)}
                            className="panel-link"
                            type="button"
                          >
                            Insert
                          </button>
                          <button
                            onClick={() => handleDelete(rewrite.id)}
                            className="panel-link danger"
                            type="button"
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FavoritesPanel;
