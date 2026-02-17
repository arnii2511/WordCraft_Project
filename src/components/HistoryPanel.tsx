import React, { useEffect, useState } from 'react';
import {
  clearSnapshots,
  deleteSnapshot,
  loadSnapshots,
} from '../services/sessionHistory';
import type { SessionSnapshot } from '../services/sessionHistory';

interface HistoryPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onRestoreSnapshot: (content: string) => void;
  documentId: string | null;
}

const HistoryPanel = ({
  isOpen,
  onClose,
  onRestoreSnapshot,
  documentId,
}: HistoryPanelProps) => {
  const [history, setHistory] = useState<SessionSnapshot[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadHistory();
    }
  }, [isOpen, documentId]);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const data = loadSnapshots(documentId);
      setHistory(data);
    } catch (error) {
      console.error('Error loading history:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteSnapshot = async (id: string) => {
    deleteSnapshot(documentId, id);
    setHistory((prev) => prev.filter((item) => item.id !== id));
  };

  const handleClearAll = async () => {
    clearSnapshots(documentId);
    setHistory([]);
  };

  return (
    <div
      className={`overlay ${isOpen ? 'is-open' : ''}`}
      onClick={onClose}
    >
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="panel-header">
          <h2 className="panel-title">Autosave History</h2>
          <button onClick={onClose} className="panel-close" type="button">
            ×
          </button>
        </div>

        <div className="panel-body">
          {loading ? (
            <p className="panel-muted">Loading...</p>
          ) : history.length === 0 ? (
            <p className="panel-muted">No autosaves yet</p>
          ) : (
            <>
              <div className="panel-actions">
                <button type="button" className="panel-link danger" onClick={handleClearAll}>
                  Clear all
                </button>
              </div>
              <div className="panel-list">
                {history.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => {
                      onRestoreSnapshot(item.contentHtml);
                      onClose();
                    }}
                    className="panel-item"
                  >
                    <div className="panel-item-row">
                      <p className="panel-item-text">
                        {(item.contentText || item.contentHtml || '')
                          .replace(/<[^>]+>/g, '')
                          .substring(0, 80)}
                        ...
                      </p>
                      <button
                        className="panel-link danger"
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          handleDeleteSnapshot(item.id);
                        }}
                      >
                        Delete
                      </button>
                    </div>
                    <p className="panel-item-date">
                      {item.ts ? new Date(item.ts).toLocaleString() : ''}
                    </p>
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default HistoryPanel;
