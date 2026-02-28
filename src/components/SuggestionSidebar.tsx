import React, { useMemo, useState } from 'react';
import { favoritesAPI, feedbackAPI } from '../services/api';
import type { FeedbackTask, SelectionPayload, SuggestionItem } from '../types';

interface SuggestionSidebarProps {
  suggestions?: SuggestionItem[];
  rewrite?: string;
  rewrites?: string[];
  explanation?: string;
  original?: string;
  detectedBlank?: boolean;
  onInsertWord: (word: string) => void;
  onInsertRewrite: (rewrite: string) => void;
  onRequestRewrite: () => void;
  mode: 'write' | 'edit' | 'rewrite';
  isAuthenticated: boolean;
  selection?: SelectionPayload | null;
  context?: string;
  onContextChange?: (value: string) => void;
}

const SuggestionSidebar = ({
  suggestions = [],
  rewrite = '',
  rewrites = [],
  explanation = '',
  original = '',
  detectedBlank = false,
  onInsertWord,
  onInsertRewrite,
  onRequestRewrite,
  mode,
  isAuthenticated,
  selection,
  context = 'neutral',
  onContextChange = () => {},
}: SuggestionSidebarProps) => {
  const [savedWords, setSavedWords] = useState<string[]>([]);
  const [savedRewrites, setSavedRewrites] = useState<string[]>([]);
  const [ratedWords, setRatedWords] = useState<Record<string, number>>({});
  const [ratedRewrites, setRatedRewrites] = useState<Record<string, number>>({});
  const contexts = [
    'neutral',
    'hopeful',
    'horror',
    'nostalgia',
    'academic',
    'romantic',
    'joyful',
    'melancholic',
    'mysterious',
    'formal',
  ];

  const topWord = suggestions[0]?.word || '';
  const selectionWord = selection?.text?.split(/\s+/)[0] || '';
  const sessionId =
    localStorage.getItem('wordcraft_feedback_session') ||
    (() => {
      const generated = `sess_${Math.random().toString(36).slice(2, 10)}`;
      localStorage.setItem('wordcraft_feedback_session', generated);
      return generated;
    })();

  const panelTitle = useMemo(() => {
    if (mode === 'edit') return 'Clarity & Grammar Fixes';
    if (mode === 'rewrite') return 'Rewrite Options (Click to Generate)';
    return 'Word Ideas (Tone + Context)';
  }, [mode]);

  const panelHelp = useMemo(() => {
    if (mode === 'edit') {
      return 'Improves clarity and grammar while keeping your meaning.';
    }
    if (mode === 'rewrite') {
      return 'Generates rewrite versions when you click.';
    }
    return 'Suggests words for blanks and highlights. No heavy rewriting.';
  }, [mode]);

  const emptyState = useMemo(() => {
    if (selectionWord) {
      return 'Suggestions are focused on the selected word.';
    }
    if (detectedBlank) {
      return 'Fill the blank first to unlock rewrite.';
    }
    if (mode === 'rewrite') {
      return 'Click “Generate Rewrite” to create rewrite options.';
    }
    return 'Write something in the editor to see suggestions.';
  }, [selectionWord, mode, detectedBlank]);

  const highlightWord = (text: string, word: string) => {
    if (!text || !word) return text;
    const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`\\b${escaped}\\b`, 'i');
    const match = text.match(regex);
    if (!match || match.index === undefined) return text;
    const start = match.index;
    const end = start + match[0].length;
    return (
      <>
        {text.slice(0, start)}
        <span className="rewrite-highlight">{text.slice(start, end)}</span>
        {text.slice(end)}
      </>
    );
  };

  const handleSaveWord = async (word: string) => {
    if (!isAuthenticated) {
      return;
    }
    try {
      await favoritesAPI.saveFavorite({
        word,
        source: 'suggest',
        type: 'suggestion',
        context,
      });
      setSavedWords([...savedWords, word]);
    } catch (error) {
      console.error('Error saving word:', error);
    }
  };

  const submitRating = async (
    task: FeedbackTask,
    candidate: string,
    rating: number,
    extra: {
      source?: string;
      pos?: string;
      model_score?: number;
      reason?: string;
      input_payload?: Record<string, unknown>;
      input_text?: string;
    } = {},
  ) => {
    try {
      await feedbackAPI.submitRating({
        task,
        candidate,
        rating,
        context,
        mode,
        source: extra.source || 'ui',
        session_id: sessionId,
        ...extra,
      });
    } catch (error) {
      console.error('Error submitting feedback rating:', error);
    }
  };

  const submitImplicitGood = (
    task: FeedbackTask,
    candidate: string,
    extra: {
      pos?: string;
      model_score?: number;
      reason?: string;
      input_payload?: Record<string, unknown>;
      input_text?: string;
    } = {},
  ) => {
    void submitRating(task, candidate, 4, {
      ...extra,
      source: extra.source || 'implicit_insert',
      reason:
        extra.reason || 'Implicit positive feedback from insert/accept action.',
    });
  };

  const renderRating = (
    value: string,
    activeRating: number | undefined,
    onRate: (rating: number) => void,
  ) => (
    <div className="rating-row" aria-label={`Rate ${value}`}>
      <span className="rating-label">Rate</span>
      {[1, 2, 3, 4, 5].map((rating) => (
        <button
          key={`${value}-${rating}`}
          type="button"
          className={`rating-chip ${activeRating === rating ? 'is-active' : ''}`}
          onClick={() => onRate(rating)}
        >
          {rating}
        </button>
      ))}
    </div>
  );

  const handleSaveRewrite = async (text: string) => {
    if (!isAuthenticated) {
      return;
    }
    try {
      await favoritesAPI.saveFavorite({
        word: text,
        source: 'suggest',
        type: 'rewrite',
        context,
      });
      setSavedRewrites([...savedRewrites, text]);
    } catch (error) {
      console.error('Error saving rewrite:', error);
    }
  };

  return (
    <aside className="right-panel">
      <div className="panel-card">
        <label className="tool-context-field panel-context-field">
          Tone / Context
          <select
            value={context}
            onChange={(event) => onContextChange(event.target.value)}
          >
            {contexts.map((ctx) => (
              <option key={ctx} value={ctx}>
                {ctx.charAt(0).toUpperCase() + ctx.slice(1)}
              </option>
            ))}
          </select>
        </label>

        <div className="panel-title">{panelTitle}</div>
        <p className="panel-mode-help">{panelHelp}</p>

        {suggestions.length === 0 && !rewrite && rewrites.length === 0 && (
          <p className="panel-empty">{emptyState}</p>
        )}

        {(rewrite || rewrites.length > 0 || mode === 'rewrite') && (
          <div className="rewrite-block">
            <div className="rewrite-title">Rewrite</div>
            {original && (
              <>
                <div className="rewrite-label">Original</div>
                <p className="rewrite-original">{original}</p>
              </>
            )}
            <div className="rewrite-label">
              {mode === 'rewrite' ? 'Rewrite Variants' : 'Suggestion'}
            </div>
            {rewrites.length > 0 ? (
              <div className="rewrite-variants">
                {rewrites.map((variant, index) => (
                  <div key={`${variant}-${index}`} className="rewrite-variant-card">
                    <div className="rewrite-variant-label">Variant {index + 1}</div>
                    <p className="rewrite-suggestion">
                      {highlightWord(variant, topWord)}
                    </p>
                    <div className="rewrite-actions">
                      <button
                        onClick={() => {
                          onInsertRewrite(variant);
                          submitImplicitGood('editor_rewrite', variant, {
                            reason: 'Implicit positive feedback from rewrite accept action.',
                            input_payload: {
                              original,
                              context,
                              mode,
                              selection: selection || null,
                            },
                            input_text: original,
                          });
                        }}
                        className="btn-accept"
                      >
                        Accept
                      </button>
                    <button
                      onClick={() => handleSaveRewrite(variant)}
                      className="btn-ghost"
                      disabled={!isAuthenticated}
                      title={
                        isAuthenticated
                          ? 'Save rewrite'
                          : 'Login to save your writing'
                      }
                    >
                      Save
                    </button>
                    </div>
                    {renderRating(
                      variant,
                      ratedRewrites[variant],
                      (rating) => {
                        setRatedRewrites((current) => ({ ...current, [variant]: rating }));
                        void submitRating('editor_rewrite', variant, rating, {
                          reason: 'Rewrite variant rating from sidebar.',
                          input_payload: {
                            original,
                            context,
                            mode,
                            selection: selection || null,
                          },
                          input_text: original,
                        });
                      },
                    )}
                  </div>
                ))}
              </div>
            ) : rewrite ? (
              <>
                <p className="rewrite-suggestion">
                  {highlightWord(rewrite, topWord)}
                </p>
                <div className="rewrite-actions">
                  <button
                    onClick={() => {
                      onInsertRewrite(rewrite);
                      submitImplicitGood('editor_rewrite', rewrite, {
                        reason: 'Implicit positive feedback from rewrite accept action.',
                        input_payload: {
                          original,
                          context,
                          mode,
                          selection: selection || null,
                        },
                        input_text: original,
                      });
                    }}
                    className="btn-accept"
                  >
                    Accept
                  </button>
                  <button
                    onClick={() => handleSaveRewrite(rewrite)}
                    className="btn-ghost"
                    disabled={!isAuthenticated}
                    title={
                      isAuthenticated ? 'Save rewrite' : 'Login to save your writing'
                    }
                  >
                    Save
                  </button>
                </div>
                {renderRating(
                  rewrite,
                  ratedRewrites[rewrite],
                  (rating) => {
                    setRatedRewrites((current) => ({ ...current, [rewrite]: rating }));
                    void submitRating('editor_rewrite', rewrite, rating, {
                      reason: 'Single rewrite rating from sidebar.',
                      input_payload: {
                        original,
                        context,
                        mode,
                        selection: selection || null,
                      },
                      input_text: original,
                    });
                  },
                )}
              </>
            ) : (
              <div className="rewrite-actions">
                <button onClick={onRequestRewrite} className="btn-accept btn-large">
                  Generate Rewrite (3 versions)
                </button>
              </div>
            )}
          </div>
        )}

        {mode !== 'rewrite' &&
          suggestions.map((item, idx) => (
            <div
              key={`${item.word}-${idx}`}
              className={`suggestion-card ${
                mode === 'edit'
                  ? 'suggestion-card-edit'
                  : 'suggestion-card-draft'
              }`}
            >
              <div className="suggestion-row">
                <div className="suggestion-word">{item.word}</div>
                <div className="pos-tag">
                  {mode === 'edit' ? 'FIX' : item.pos || 'ADJ'}
                </div>
              </div>
              {mode === 'edit' && (
                <div className="suggestion-issue">
                  Issue detected → Fix suggestion
                </div>
              )}
              <div className="suggestion-example">
                {mode === 'edit'
                  ? item.note || 'Improves clarity and correctness.'
                  : item.note || `"${item.word}" adds a precise, evocative tone.`}
              </div>
              <div className="suggestion-actions">
                <button
                  onClick={() => {
                    onInsertWord(item.word);
                    submitImplicitGood('editor_suggestion', item.word, {
                      pos: item.pos,
                      model_score: item.score,
                      reason: item.note || 'Inserted into editor.',
                      input_payload: {
                        context,
                        mode,
                        selection: selection || null,
                        detected_blank: detectedBlank,
                      },
                      input_text: original,
                    });
                  }}
                  className="btn-accept"
                >
                  Insert
                </button>
                <button
                  onClick={() => handleSaveWord(item.word)}
                  className="btn-ghost"
                  disabled={!isAuthenticated}
                  title={
                    isAuthenticated ? 'Save word' : 'Login to save your writing'
                  }
                >
                  Save
                </button>
              </div>
              {renderRating(
                item.word,
                ratedWords[item.word],
                (rating) => {
                  setRatedWords((current) => ({ ...current, [item.word]: rating }));
                  void submitRating('editor_suggestion', item.word, rating, {
                    pos: item.pos,
                    model_score: item.score,
                    reason: item.note,
                    input_payload: {
                      context,
                      mode,
                      selection: selection || null,
                      detected_blank: detectedBlank,
                    },
                    input_text: original,
                  });
                },
              )}
            </div>
          ))}

        {explanation && (
          <div className="explanation-card">
            <div className="rewrite-label">Why this works</div>
            <div className="explanation-text">{explanation}</div>
          </div>
        )}

      </div>
    </aside>
  );
};

export default SuggestionSidebar;
