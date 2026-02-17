import { Editor, EditorContent, useEditor } from '@tiptap/react';
import Placeholder from '@tiptap/extension-placeholder';
import StarterKit from '@tiptap/starter-kit';
import React, { useEffect, useRef, useState } from 'react';
import { writingAPI } from '../services/api';
import { loadSnapshots, saveSnapshot } from '../services/sessionHistory';
import type { SelectionPayload, SuggestResponse } from '../types';

interface EditorProps {
  context: string;
  mode: 'write' | 'edit' | 'rewrite';
  onSuggestionsUpdate: (payload: SuggestResponse) => void;
  onSelectionChange?: (selection: SelectionPayload | null) => void;
  editorRef: React.MutableRefObject<Editor | null>;
  rewriteSignal: number;
  isAuthenticated: boolean;
  documentId: string | null;
}

const EditorSurface = ({
  context,
  mode,
  onSuggestionsUpdate,
  onSelectionChange,
  editorRef,
  rewriteSignal,
  isAuthenticated,
  documentId,
}: EditorProps) => {
  const [loading, setLoading] = useState(false);
  const [wordCount, setWordCount] = useState(0);
  const [draftText, setDraftText] = useState('');
  const [selection, setSelection] = useState<SelectionPayload | null>(null);
  const debounceRef = useRef<number | null>(null);
  const requestRef = useRef(0);
  const lastSnapshotRef = useRef('');
  const lastRewriteSignalRef = useRef(rewriteSignal);
  const draftTextRef = useRef('');

  const editor = useEditor({
    extensions: [
      StarterKit,
      Placeholder.configure({
        placeholder: 'Start writing here...',
      }),
    ],
    content: '',
    editorProps: {
      attributes: {
        class: 'editor focus:outline-none max-w-none',
      },
    },
    onUpdate: ({ editor }) => {
      const text = editor.getText();
      const words = text.trim().split(/\s+/).filter(Boolean);
      setWordCount(words.length);
      setDraftText(text);
      draftTextRef.current = text;
    },
    onSelectionUpdate: ({ editor }) => {
      const { from, to } = editor.state.selection;
      if (from === to) {
        setSelection(null);
        if (onSelectionChange) onSelectionChange(null);
        return;
      }
      const selectedText = editor.state.doc.textBetween(from, to, ' ').trim();
      if (!selectedText) {
        setSelection(null);
        if (onSelectionChange) onSelectionChange(null);
        return;
      }
      const payload = { text: selectedText, start: from, end: to };
      setSelection(payload);
      if (onSelectionChange) onSelectionChange(payload);
    },
  });

  // expose editor instance to parent via ref
  useEffect(() => {
    if (editorRef) {
      editorRef.current = editor;
    }
    return () => {
      if (editorRef) editorRef.current = null;
    };
  }, [editor, editorRef]);

  const fetchSuggestions = async (text: string, trigger: 'auto' | 'button') => {
    const requestId = ++requestRef.current;
    setLoading(true);
    try {
      const suggestionsData = await writingAPI.getSuggestions(
        text,
        context,
        mode,
        selection,
        trigger,
      );
      if (requestId !== requestRef.current) {
        return;
      }

      onSuggestionsUpdate({
        suggestions: suggestionsData.suggestions || [],
        rewrite: suggestionsData.rewrite || '',
        rewrites: suggestionsData.rewrites || [],
        explanation: suggestionsData.explanation || '',
        detected_blank: suggestionsData.detected_blank ?? false,
        original: text,
      });
    } catch (error) {
      console.error('Error fetching suggestions:', error);
    } finally {
      setLoading(false);
    }
  };

  const saveHistorySnapshot = async (force: boolean) => {
    if (!isAuthenticated) return;
    const trimmed = draftTextRef.current.trim();
    if (!trimmed) return;
    if (!force && trimmed === lastSnapshotRef.current) {
      return;
    }
    lastSnapshotRef.current = trimmed;
    const contentHtml = editor?.getHTML() || '';
    const contentText = editor?.getText() || '';
    saveSnapshot(documentId, {
      id: `snap_${Date.now()}`,
      ts: Date.now(),
      contentHtml,
      contentText,
      context,
      mode,
    });
  };

  useEffect(() => {
    const autosaveInterval = setInterval(() => {
      void saveHistorySnapshot(false);
    }, 60000);
    return () => clearInterval(autosaveInterval);
  }, [editor, mode, documentId, isAuthenticated]);

  useEffect(() => {
    if (!isAuthenticated) {
      lastSnapshotRef.current = '';
      return;
    }
    const existing = loadSnapshots(documentId);
    lastSnapshotRef.current = existing[0]?.contentText?.trim() || '';
  }, [documentId, isAuthenticated]);

  useEffect(() => {
    if (lastRewriteSignalRef.current === rewriteSignal) {
      return;
    }
    lastRewriteSignalRef.current = rewriteSignal;
    if (mode !== 'rewrite') {
      return;
    }
    const trimmed = draftText.trim();
    if (!trimmed || trimmed.length < 5) {
      return;
    }
    fetchSuggestions(trimmed, 'button');
  }, [rewriteSignal, draftText, mode, selection, context]);

  useEffect(() => {
    if (debounceRef.current) {
      window.clearTimeout(debounceRef.current);
    }
    const trimmed = draftText.trim();
    if (mode === 'rewrite') {
      return;
    }
    if (!trimmed || trimmed.length < 5) {
      onSuggestionsUpdate({
        suggestions: [],
        rewrite: '',
        rewrites: [],
        explanation: '',
        detected_blank: false,
        original: trimmed,
      });
      setLoading(false);
      return;
    }

    debounceRef.current = window.setTimeout(() => {
      fetchSuggestions(trimmed, 'auto');
    }, 800);

    return () => {
      if (debounceRef.current) {
        window.clearTimeout(debounceRef.current);
      }
    };
  }, [draftText, context, mode, selection, onSuggestionsUpdate]);

  useEffect(() => {
    if (mode === 'rewrite') {
      onSuggestionsUpdate({
        suggestions: [],
        rewrite: '',
        rewrites: [],
        explanation: '',
        detected_blank: false,
        original: draftText.trim(),
      });
    }
  }, [mode, draftText, onSuggestionsUpdate]);

  if (!editor) {
    return <div className="p-6 text-center text-sage">Loading editor...</div>;
  }

  return (
    <div className="editor-container">
      <div className="editor-card">
        <EditorContent
          editor={editor}
          className="editor-content-font"
          data-placeholder="Start writing here..."
        />
        {loading && (
          <div className="editor-status">Analyzing your writing...</div>
        )}
      </div>
      <div className="editor-footer">
        <span>{wordCount} words</span>
        <span className="editor-hint">
          {isAuthenticated
            ? 'Session autosave every 60s'
            : 'Not saved. Login to enable save and history.'}
        </span>
      </div>
    </div>
  );
};

export default EditorSurface;
