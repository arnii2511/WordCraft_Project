import React, { useEffect, useState } from 'react';
import { documentsAPI } from '../services/api';
import type { DocumentEntry } from '../types';

interface DocumentsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectDocument: (doc: DocumentEntry) => void;
  onDeleteDocument: (id: string) => void;
}

const DocumentsPanel = ({
  isOpen,
  onClose,
  onSelectDocument,
  onDeleteDocument,
}: DocumentsPanelProps) => {
  const [documents, setDocuments] = useState<DocumentEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadDocuments();
    }
  }, [isOpen]);

  const loadDocuments = async () => {
    setLoading(true);
    try {
      const data = await documentsAPI.getDocuments();
      setDocuments(data);
    } catch (error) {
      console.error('Error loading documents:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`overlay ${isOpen ? 'is-open' : ''}`} onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="panel-header">
          <h2 className="panel-title">My Docs</h2>
          <button onClick={onClose} className="panel-close" type="button">
            ×
          </button>
        </div>

        <div className="panel-body">
          {loading ? (
            <p className="panel-muted">Loading...</p>
          ) : documents.length === 0 ? (
            <p className="panel-muted">No documents yet. Click Save to create one.</p>
          ) : (
            <div className="panel-list">
              {documents.map((doc) => (
                <button
                  key={doc.id}
                  onClick={() => {
                    onSelectDocument(doc);
                    onClose();
                  }}
                  className="panel-item"
                >
                  <div className="panel-item-row">
                    <p className="panel-item-text">{doc.title || 'Untitled'}</p>
                    <button
                      className="panel-link danger"
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        onDeleteDocument(doc.id);
                        setDocuments((prev) => prev.filter((item) => item.id !== doc.id));
                      }}
                    >
                      Delete
                    </button>
                  </div>
                  <p className="panel-item-date">
                    {doc.updatedAt
                      ? new Date(doc.updatedAt).toLocaleDateString()
                      : doc.createdAt
                        ? new Date(doc.createdAt).toLocaleDateString()
                        : ''}
                    {doc.context ? ` · ${doc.context}` : ''}
                    {doc.mode ? ` · ${doc.mode}` : ''}
                  </p>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentsPanel;
