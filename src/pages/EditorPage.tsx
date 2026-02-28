import React, { useEffect, useRef, useState } from 'react';
import { Editor as TipTapEditor } from '@tiptap/react';
import { documentsAPI } from '../services/api';
import Editor from '../components/Editor';
import SuggestionSidebar from '../components/SuggestionSidebar';
import DocumentsPanel from '../components/DocumentsPanel';
import FavoritesPanel from '../components/FavoritesPanel';
import HistoryPanel from '../components/HistoryPanel';
import AppHeader from '../components/AppHeader';
import type { DocumentEntry, SelectionPayload, SuggestResponse, UserProfile } from '../types';

const PENDING_INSERT_KEY = 'wordcraft_pending_insert';
const MODE_CARDS: Array<{
  id: 'write' | 'edit' | 'rewrite';
  title: string;
  subtitle: string;
}> = [
  {
    id: 'write',
    title: 'Draft',
    subtitle: 'Light suggestions while writing',
  },
  {
    id: 'edit',
    title: 'Polish',
    subtitle: 'Clarity and grammar refinement',
  },
  {
    id: 'rewrite',
    title: 'Transform',
    subtitle: 'Generate rewrites on click',
  },
];

interface EditorPageProps {
  context: string;
  setContext: (value: string) => void;
  mode: 'write' | 'edit' | 'rewrite';
  setMode: (value: 'write' | 'edit' | 'rewrite') => void;
  user: UserProfile | null;
  isAuthenticated: boolean;
  onRequireAuth: () => void;
}

const EditorPage = ({
  context,
  setContext,
  mode,
  setMode,
  user,
  isAuthenticated,
  onRequireAuth,
}: EditorPageProps) => {
  const [suggestions, setSuggestions] = useState<SuggestResponse>({
    suggestions: [],
    rewrite: '',
    rewrites: [],
    explanation: '',
    detected_blank: false,
    original: '',
  });
  const [showHistory, setShowHistory] = useState(false);
  const [showFavorites, setShowFavorites] = useState(false);
  const [showDocs, setShowDocs] = useState(false);
  const [rewriteSignal, setRewriteSignal] = useState(0);
  const [selection, setSelection] = useState<SelectionPayload | null>(null);
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null);
  const editorRef = useRef<TipTapEditor | null>(null);

  useEffect(() => {
    const storedDocId = localStorage.getItem('active_document_id');
    if (storedDocId) {
      setActiveDocumentId(storedDocId);
    }
  }, []);

  useEffect(() => {
    const pendingWord = localStorage.getItem(PENDING_INSERT_KEY);
    if (!pendingWord) {
      return;
    }
    let attempts = 0;
    const timer = window.setInterval(() => {
      attempts += 1;
      if (editorRef.current) {
        editorRef.current.chain().focus().insertContent(`${pendingWord} `).run();
        localStorage.removeItem(PENDING_INSERT_KEY);
        window.clearInterval(timer);
      } else if (attempts > 30) {
        window.clearInterval(timer);
      }
    }, 200);
    return () => window.clearInterval(timer);
  }, []);

  const handleInsertWord = (word: string) => {
    if (editorRef.current) {
      editorRef.current.chain().focus().insertContent(`${word} `).run();
    }
  };

  const handleInsertRewrite = (rewrite: string) => {
    if (editorRef.current) {
      editorRef.current.chain().focus().insertContent(`${rewrite} `).run();
    }
  };

  const handleSave = () => {
    if (!isAuthenticated) {
      onRequireAuth();
      return;
    }
    const editor = editorRef.current;
    if (!editor) return;
    const content = editor.getHTML();
    const text = editor.getText().trim();
    const title = text ? text.split('\n')[0].slice(0, 60) : 'Untitled';
    void documentsAPI
      .saveDocument({
        id: activeDocumentId || undefined,
        title,
        contentHtml: content,
        contentText: text,
        context,
        mode,
      })
      .then((doc) => {
        setActiveDocumentId(doc.id);
        localStorage.setItem('active_document_id', doc.id);
      });
  };

  const handleRewriteRequest = () => {
    setRewriteSignal((prev) => prev + 1);
  };

  return (
    <div className="editor-page">
      <div className="editor-shell">
        <AppHeader
          activePage="editor"
          isAuthenticated={isAuthenticated}
          user={user}
          onRequireAuth={onRequireAuth}
        />

        <section className="tools-home-hero editor-hero">
          <h2>Write sharper, sound truer.</h2>
          <p>Draft ideas, polish clarity, and transform tone without losing your voice.</p>
        </section>

        <section className="editor-workspace-panel">
          <div className="app-main">
            <div className="editor-main-left">
              <section className="editor-mode-grid" aria-label="Editor modes">
                {MODE_CARDS.map((card) => (
                  <button
                    key={card.id}
                    type="button"
                    className={`editor-mode-card ${mode === card.id ? 'is-active' : ''}`}
                    onClick={() => setMode(card.id)}
                  >
                    <h2>{card.title}</h2>
                    <p>{card.subtitle}</p>
                  </button>
                ))}
              </section>

              <Editor
                editorRef={editorRef}
                context={context}
                mode={mode}
                onSuggestionsUpdate={setSuggestions}
                onSelectionChange={setSelection}
                onSave={handleSave}
                rewriteSignal={rewriteSignal}
                isAuthenticated={isAuthenticated}
                documentId={activeDocumentId}
              />
            </div>

            <SuggestionSidebar
              suggestions={suggestions.suggestions}
              rewrite={suggestions.rewrite}
              rewrites={suggestions.rewrites}
              explanation={suggestions.explanation}
              original={suggestions.original}
              detectedBlank={suggestions.detected_blank}
              onInsertWord={handleInsertWord}
              onInsertRewrite={handleInsertRewrite}
              mode={mode}
              onRequestRewrite={handleRewriteRequest}
              isAuthenticated={isAuthenticated}
              selection={selection}
              context={context}
              onContextChange={setContext}
            />
          </div>
        </section>
      </div>

      <HistoryPanel
        isOpen={showHistory}
        onClose={() => setShowHistory(false)}
        documentId={activeDocumentId}
        onRestoreSnapshot={(content) => {
          if (editorRef.current) {
            editorRef.current.commands.setContent(content);
          }
        }}
      />

      <DocumentsPanel
        isOpen={showDocs}
        onClose={() => setShowDocs(false)}
        onSelectDocument={(doc: DocumentEntry) => {
          setActiveDocumentId(doc.id);
          localStorage.setItem('active_document_id', doc.id);
          setContext(doc.context);
          setMode(doc.mode);
          if (editorRef.current) {
            editorRef.current.commands.setContent(doc.contentHtml);
          }
        }}
        onDeleteDocument={(id) => {
          void documentsAPI.deleteDocument(id);
          if (activeDocumentId === id) {
            setActiveDocumentId(null);
            localStorage.removeItem('active_document_id');
          }
        }}
      />

      <FavoritesPanel
        isOpen={showFavorites}
        onClose={() => setShowFavorites(false)}
        onInsertFavorite={(content) => {
          handleInsertWord(content);
        }}
      />
    </div>
  );
};

export default EditorPage;
