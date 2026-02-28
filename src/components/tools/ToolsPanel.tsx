import React, { useEffect, useMemo, useState } from 'react';
import { constraintsAPI, favoritesAPI, feedbackAPI, lexicalAPI, onewordAPI } from '../../services/api';
import type {
  ConstraintRelation,
  ConstraintResult,
  FeedbackTask,
  LexicalResultDetail,
  LexicalTask,
  OneWordResult,
  SelectionPayload,
} from '../../types';

interface ToolsPanelProps {
  isOpen?: boolean;
  onClose?: () => void;
  selection?: SelectionPayload | null;
  isAuthenticated?: boolean;
  context?: string;
  onContextChange?: (value: string) => void;
  embedded?: boolean;
  activeTool?: ToolTab;
  onActiveToolChange?: (tool: ToolTab) => void;
}

export type ToolTab = 'synonyms' | 'rhymes' | 'smart_match' | 'one_word';

const CONTEXTS = [
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

const TASK_TAGS: Record<LexicalTask, string> = {
  synonyms: 'SYN',
  antonyms: 'ANT',
  rhymes: 'RHYME',
  homonyms: 'HOMOPHONE',
};

const TOOL_META: Record<ToolTab, { title: string; help: string }> = {
  one_word: {
    title: 'One-Word Substitution',
    help: 'Describe the word you want. We will suggest ranked single-word options.',
  },
  smart_match: {
    title: 'Smart Match',
    help: 'Combine rhyme and synonym or antonym constraints in one query.',
  },
  synonyms: {
    title: 'Synonyms & Antonyms',
    help: 'Enter a word, then choose lexical contrast direction.',
  },
  rhymes: {
    title: 'Rhyme & Homonym',
    help: 'Find phonetic matches and practical homophone variants.',
  },
};

const formatLabel = (value: string) =>
  value ? value.charAt(0).toUpperCase() + value.slice(1) : 'Neutral';

const fallbackCopy = (value: string) => {
  const textArea = document.createElement('textarea');
  textArea.value = value;
  textArea.style.position = 'fixed';
  textArea.style.opacity = '0';
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();
  document.execCommand('copy');
  document.body.removeChild(textArea);
};

const ToolsPanel = ({
  isOpen = false,
  onClose = () => {},
  selection,
  isAuthenticated = false,
  context = 'neutral',
  onContextChange,
  embedded = false,
  activeTool: controlledTool,
  onActiveToolChange,
}: ToolsPanelProps) => {
  const [internalActiveTool, setInternalActiveTool] = useState<ToolTab>(
    controlledTool ?? 'one_word',
  );
  const [copiedWord, setCopiedWord] = useState('');
  const [ratings, setRatings] = useState<Record<string, number>>({});

  const [contrastTask, setContrastTask] = useState<'synonyms' | 'antonyms'>('synonyms');
  const [soundTask, setSoundTask] = useState<'rhymes' | 'homonyms'>('rhymes');
  const [lexicalInput, setLexicalInput] = useState('');
  const [toolResults, setToolResults] = useState<string[]>([]);
  const [toolResultDetails, setToolResultDetails] = useState<Record<string, LexicalResultDetail>>(
    {},
  );
  const [loadingLexical, setLoadingLexical] = useState(false);
  const [lexicalSearched, setLexicalSearched] = useState(false);
  const [lexicalError, setLexicalError] = useState('');

  const [rhymeWith, setRhymeWith] = useState('');
  const [meaningOf, setMeaningOf] = useState('');
  const [relation, setRelation] = useState<ConstraintRelation>('synonym');
  const [smartResults, setSmartResults] = useState<ConstraintResult[]>([]);
  const [smartNote, setSmartNote] = useState('');
  const [smartLoading, setSmartLoading] = useState(false);
  const [smartSearched, setSmartSearched] = useState(false);

  const [oneWordQuery, setOneWordQuery] = useState('');
  const [oneWordResults, setOneWordResults] = useState<OneWordResult[]>([]);
  const [oneWordNote, setOneWordNote] = useState('');
  const [oneWordLoading, setOneWordLoading] = useState(false);
  const [oneWordSearched, setOneWordSearched] = useState(false);

  const selectionWord = selection?.text?.split(/\s+/)[0] || '';
  const sessionId =
    localStorage.getItem('wordcraft_feedback_session') ||
    (() => {
      const generated = `sess_${Math.random().toString(36).slice(2, 10)}`;
      localStorage.setItem('wordcraft_feedback_session', generated);
      return generated;
    })();
  const activeTool = controlledTool ?? internalActiveTool;
  const isVisible = embedded || isOpen;

  const activeLexicalTask = useMemo<LexicalTask | null>(() => {
    if (activeTool === 'synonyms') return contrastTask;
    if (activeTool === 'rhymes') return soundTask;
    return null;
  }, [activeTool, contrastTask, soundTask]);

  const activeLexicalWord = lexicalInput.trim() || selectionWord;
  const toolMeta = TOOL_META[activeTool];

  const submitImplicitCopy = async (
    task: FeedbackTask,
    candidate: string,
    options: {
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
        rating: 4,
        context,
        source: 'implicit_copy',
        session_id: sessionId,
        reason: options.reason || 'Implicit positive feedback from copy action.',
        ...options,
      });
    } catch (error) {
      console.error('Error submitting implicit copy feedback:', error);
    }
  };

  useEffect(() => {
    if (controlledTool) {
      setInternalActiveTool(controlledTool);
    }
  }, [controlledTool]);

  useEffect(() => {
    if (!isVisible) return;
    setToolResults([]);
    setToolResultDetails({});
    setLexicalSearched(false);
    setLexicalError('');
    setSmartResults([]);
    setSmartSearched(false);
    setSmartNote('');
    setOneWordResults([]);
    setOneWordSearched(false);
    setOneWordNote('');
  }, [activeTool, isVisible]);

  const handleToolSelect = (task: ToolTab) => {
    if (!controlledTool) {
      setInternalActiveTool(task);
    }
    onActiveToolChange?.(task);
  };

  const handleCopyWord = async (word: string) => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(word);
      } else {
        fallbackCopy(word);
      }
      setCopiedWord(word);
      window.setTimeout(() => setCopiedWord((current) => (current === word ? '' : current)), 1200);
    } catch (error) {
      fallbackCopy(word);
      setCopiedWord(word);
      window.setTimeout(() => setCopiedWord((current) => (current === word ? '' : current)), 1200);
    }
    if (activeTool === 'one_word') {
      const detail = oneWordResults.find((item) => item.word === word);
      void submitImplicitCopy('oneword', word, {
        model_score: detail?.score,
        reason: detail?.reason || 'Copied one-word substitution result.',
        input_payload: {
          query: oneWordQuery.trim(),
          context,
        },
        input_text: oneWordQuery.trim(),
      });
      return;
    }
    if (activeTool === 'smart_match') {
      const detail = smartResults.find((item) => item.word === word);
      void submitImplicitCopy('constraints', word, {
        model_score: detail?.score,
        reason: detail?.reason || 'Copied smart match result.',
        input_payload: {
          rhyme_with: rhymeWith.trim(),
          relation,
          meaning_of: meaningOf.trim(),
          context,
        },
        input_text: `${rhymeWith.trim()}|${relation}|${meaningOf.trim()}`,
      });
      return;
    }
    if (activeLexicalTask) {
      const detail = toolResultDetails[word];
      void submitImplicitCopy('lexical', word, {
        pos: detail?.pos || undefined,
        model_score: detail?.score,
        reason: detail?.reason || 'Copied lexical result.',
        input_payload: {
          word: activeLexicalWord,
          lexical_task: activeLexicalTask,
          context,
        },
        input_text: activeLexicalWord,
      });
    }
  };

  const runLexicalSearch = async () => {
    setLexicalSearched(true);
    setLexicalError('');
    if (!activeLexicalTask || !activeLexicalWord) {
      setToolResults([]);
      setToolResultDetails({});
      return;
    }
    setLoadingLexical(true);
    try {
      const response = await lexicalAPI.getResults(activeLexicalWord, activeLexicalTask, context);
      setToolResults(response.results || []);
      const detailsMap: Record<string, LexicalResultDetail> = {};
      for (const detail of response.details || []) {
        detailsMap[detail.word] = detail;
      }
      setToolResultDetails(detailsMap);
    } catch (error) {
      setToolResults([]);
      setToolResultDetails({});
      setLexicalError('Unable to fetch lexical suggestions right now.');
    } finally {
      setLoadingLexical(false);
    }
  };

  const handleSmartMatch = async () => {
    setSmartSearched(true);
    if (!rhymeWith.trim() || !meaningOf.trim()) {
      setSmartResults([]);
      setSmartNote('Enter both fields to run Smart Match.');
      return;
    }
    setSmartLoading(true);
    try {
      const response = await constraintsAPI.getMatches({
        rhyme_with: rhymeWith.trim(),
        relation,
        meaning_of: meaningOf.trim(),
        context,
        limit: 10,
      });
      setSmartResults(response.results || []);
      setSmartNote(response.notes || '');
    } catch (error) {
      setSmartResults([]);
      setSmartNote('Unable to fetch smart match results.');
    } finally {
      setSmartLoading(false);
    }
  };

  const handleFindOneWord = async () => {
    setOneWordSearched(true);
    if (!oneWordQuery.trim()) {
      setOneWordResults([]);
      setOneWordNote('Describe the word you want to find.');
      return;
    }
    setOneWordLoading(true);
    try {
      const response = await onewordAPI.getResults({
        query: oneWordQuery.trim(),
        context,
        limit: 10,
      });
      setOneWordResults(response.results || []);
      setOneWordNote(response.note || '');
    } catch (error) {
      setOneWordResults([]);
      setOneWordNote('Unable to fetch one-word substitutions.');
    } finally {
      setOneWordLoading(false);
    }
  };

  const handleSaveLexical = async (word: string) => {
    if (!isAuthenticated || !activeLexicalTask) return;
    try {
      await favoritesAPI.saveFavorite({
        word,
        source: 'lexical',
        type: activeLexicalTask,
        context,
        related_to: activeLexicalWord,
      });
    } catch (error) {
      console.error('Error saving lexical word:', error);
    }
  };

  const handleSaveSmart = async (word: string) => {
    if (!isAuthenticated) return;
    try {
      await favoritesAPI.saveFavorite({
        word,
        source: 'constraints',
        type: 'smart_match',
        context,
        related_to: `${relation}:${meaningOf.trim()}|rhyme:${rhymeWith.trim()}`,
      });
    } catch (error) {
      console.error('Error saving smart match word:', error);
    }
  };

  const handleSaveOneWord = async (word: string) => {
    if (!isAuthenticated) return;
    try {
      await favoritesAPI.saveFavorite({
        word,
        source: 'oneword',
        type: 'oneword',
        context,
        related_to: oneWordQuery.trim(),
      });
    } catch (error) {
      console.error('Error saving one-word result:', error);
    }
  };

  const submitRating = async (
    task: FeedbackTask,
    candidate: string,
    rating: number,
    options: {
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
        source: 'tools_ui',
        session_id: sessionId,
        ...options,
      });
    } catch (error) {
      console.error('Error submitting tools feedback rating:', error);
    }
  };

  const renderRating = (
    key: string,
    label: string,
    activeRating: number | undefined,
    onRate: (rating: number) => void,
  ) => (
    <div className="rating-row" aria-label={`Rate ${label}`}>
      <span className="rating-label">Rate</span>
      {[1, 2, 3, 4, 5].map((rating) => (
        <button
          key={`${key}-${rating}`}
          type="button"
          className={`rating-chip ${activeRating === rating ? 'is-active' : ''}`}
          onClick={() => onRate(rating)}
        >
          {rating}
        </button>
      ))}
    </div>
  );

  const renderActions = (
    word: string,
    ratingKey: string,
    onSave: (value: string) => void,
    onRate: (rating: number) => void,
  ) => (
    <>
      <div className="tool-result-actions">
        <button type="button" className="btn-accept" onClick={() => handleCopyWord(word)}>
          {copiedWord === word ? 'Copied' : 'Copy'}
        </button>
        {isAuthenticated && (
          <button type="button" className="btn-ghost" onClick={() => onSave(word)}>
            Save
          </button>
        )}
      </div>
      {renderRating(ratingKey, word, ratings[ratingKey], onRate)}
    </>
  );

  const panelBody = (
    <>
      {!embedded && (
        <div className="panel-header">
          <h2 className="panel-title">Tools</h2>
          <button onClick={onClose} className="panel-close" type="button">
            ×
          </button>
        </div>
      )}

      <div className="panel-body">
        {!embedded && (
          <div className="tool-inline-nav">
            {(['one_word', 'smart_match', 'synonyms', 'rhymes'] as ToolTab[]).map((task) => (
              <button
                key={task}
                type="button"
                className={`tool-chip ${activeTool === task ? 'is-active' : ''}`}
                onClick={() => handleToolSelect(task)}
              >
                {TOOL_META[task].title}
              </button>
            ))}
          </div>
        )}

        <div className="tool-panel-head">
          <div>
            <h2 className="tool-panel-title">{toolMeta.title}</h2>
            <p className="tool-panel-help">{toolMeta.help}</p>
          </div>
          {onContextChange && (
            <label className="tool-context-field">
              Tone / Context
              <select value={context} onChange={(event) => onContextChange(event.target.value)}>
                {CONTEXTS.map((ctx) => (
                  <option key={ctx} value={ctx}>
                    {formatLabel(ctx)}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>

        {activeTool === 'synonyms' || activeTool === 'rhymes' ? (
          <div className="tool-workspace">
            <div className="tool-toggle-row">
              {activeTool === 'synonyms' ? (
                <>
                  <button
                    type="button"
                    className={`tool-toggle ${contrastTask === 'synonyms' ? 'is-active' : ''}`}
                    onClick={() => setContrastTask('synonyms')}
                  >
                    Synonyms
                  </button>
                  <button
                    type="button"
                    className={`tool-toggle ${contrastTask === 'antonyms' ? 'is-active' : ''}`}
                    onClick={() => setContrastTask('antonyms')}
                  >
                    Antonyms
                  </button>
                </>
              ) : (
                <>
                  <button
                    type="button"
                    className={`tool-toggle ${soundTask === 'rhymes' ? 'is-active' : ''}`}
                    onClick={() => setSoundTask('rhymes')}
                  >
                    Rhymes
                  </button>
                  <button
                    type="button"
                    className={`tool-toggle ${soundTask === 'homonyms' ? 'is-active' : ''}`}
                    onClick={() => setSoundTask('homonyms')}
                  >
                    Homophones
                  </button>
                </>
              )}
            </div>

            <div className="tool-hero-input">
              <input
                value={lexicalInput}
                onChange={(event) => setLexicalInput(event.target.value)}
                placeholder={
                  activeTool === 'synonyms'
                    ? 'Enter a word to explore lexical contrast...'
                    : 'Enter a word to find rhymes or homophones...'
                }
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    void runLexicalSearch();
                  }
                }}
              />
            </div>

            <div className="tool-action-row">
              <button type="button" className="btn-accept" onClick={() => void runLexicalSearch()}>
                {contrastTask === 'antonyms'
                  ? 'Get antonyms'
                  : soundTask === 'homonyms'
                    ? 'Get homophones'
                    : activeTool === 'rhymes'
                      ? 'Get rhymes'
                      : 'Get synonyms'}
              </button>
            </div>

            <div className="tool-results-list">
              {loadingLexical ? (
                <p className="panel-muted">Looking up results...</p>
              ) : !lexicalSearched ? (
                <p className="panel-muted">Type a word to see results.</p>
              ) : lexicalError ? (
                <p className="panel-muted">{lexicalError}</p>
              ) : toolResults.length === 0 ? (
                <p className="panel-muted">No results found for this query.</p>
              ) : (
                toolResults.map((result) => {
                  const detail = toolResultDetails[result];
                  return (
                    <article key={result} className="tool-result-card">
                      <div className="tool-result-head">
                        <h3 className="tool-result-word">{result}</h3>
                        <span className="tool-result-tag">
                          {detail?.pos?.toUpperCase() || (activeLexicalTask ? TASK_TAGS[activeLexicalTask] : 'LEX')}
                        </span>
                      </div>
                      <p className="tool-result-reason">
                        {detail?.reason || 'Semantically related lexical match.'}
                      </p>
                      {renderActions(
                        result,
                        `lexical:${activeLexicalTask}:${result}`,
                        handleSaveLexical,
                        (rating) => {
                          const key = `lexical:${activeLexicalTask}:${result}`;
                          setRatings((current) => ({ ...current, [key]: rating }));
                          void submitRating('lexical', result, rating, {
                            pos: detail?.pos || undefined,
                            model_score: detail?.score,
                            reason: detail?.reason,
                            input_payload: {
                              word: activeLexicalWord,
                              lexical_task: activeLexicalTask,
                              context,
                            },
                            input_text: activeLexicalWord,
                          });
                        },
                      )}
                    </article>
                  );
                })
              )}
            </div>
          </div>
        ) : null}

        {activeTool === 'smart_match' ? (
          <div className="tool-workspace">
            <div className="tool-input-grid">
              <label className="tool-field">
                Rhyme with
                <input
                  value={rhymeWith}
                  onChange={(event) => setRhymeWith(event.target.value)}
                  placeholder="e.g., night"
                />
              </label>
              <label className="tool-field">
                Relation
                <select
                  value={relation}
                  onChange={(event) => setRelation(event.target.value as ConstraintRelation)}
                >
                  <option value="synonym">Synonym</option>
                  <option value="antonym">Antonym</option>
                </select>
              </label>
              <label className="tool-field tool-field-wide">
                Meaning target
                <input
                  value={meaningOf}
                  onChange={(event) => setMeaningOf(event.target.value)}
                  placeholder="e.g., bright"
                />
              </label>
            </div>

            <div className="tool-action-row">
              <button type="button" className="btn-accept" onClick={() => void handleSmartMatch()}>
                Find matches
              </button>
            </div>

            <div className="tool-results-list">
              {smartLoading ? (
                <p className="panel-muted">Finding smart matches...</p>
              ) : !smartSearched ? (
                <p className="panel-muted">Type a word to see results.</p>
              ) : smartResults.length === 0 ? (
                <p className="panel-muted">{smartNote || 'No compatible matches found yet.'}</p>
              ) : (
                <>
                  {smartNote && <p className="panel-muted">{smartNote}</p>}
                  {smartResults.map((result) => (
                    <article key={result.word} className="tool-result-card">
                      <div className="tool-result-head">
                        <h3 className="tool-result-word">{result.word}</h3>
                        <span className="tool-result-tag">SMART</span>
                      </div>
                      <p className="tool-result-reason">{result.reason}</p>
                      {renderActions(
                        result.word,
                        `constraints:${rhymeWith}:${relation}:${meaningOf}:${result.word}`,
                        handleSaveSmart,
                        (rating) => {
                          const key = `constraints:${rhymeWith}:${relation}:${meaningOf}:${result.word}`;
                          setRatings((current) => ({ ...current, [key]: rating }));
                          void submitRating('constraints', result.word, rating, {
                            model_score: result.score,
                            reason: result.reason,
                            input_payload: {
                              rhyme_with: rhymeWith.trim(),
                              relation,
                              meaning_of: meaningOf.trim(),
                              context,
                            },
                            input_text: `${rhymeWith.trim()}|${relation}|${meaningOf.trim()}`,
                          });
                        },
                      )}
                    </article>
                  ))}
                </>
              )}
            </div>
          </div>
        ) : null}

        {activeTool === 'one_word' ? (
          <div className="tool-workspace">
            <div className="tool-input-grid">
              <label className="tool-field tool-field-wide">
                Describe the word you want
                <input
                  value={oneWordQuery}
                  onChange={(event) => setOneWordQuery(event.target.value)}
                  placeholder="e.g., self obsessed, fear of heights..."
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      void handleFindOneWord();
                    }
                  }}
                />
              </label>
            </div>

            <div className="tool-action-row">
              <button type="button" className="btn-accept" onClick={() => void handleFindOneWord()}>
                Find one word
              </button>
            </div>

            <div className="tool-results-list">
              {oneWordLoading ? (
                <p className="panel-muted">Finding one-word substitutions...</p>
              ) : !oneWordSearched ? (
                <p className="panel-muted">Type a word to see results.</p>
              ) : oneWordResults.length === 0 ? (
                <p className="panel-muted">
                  {oneWordNote || 'No one-word substitutions found for that description.'}
                </p>
              ) : (
                <>
                  {oneWordNote && <p className="panel-muted">{oneWordNote}</p>}
                  {oneWordResults.map((result) => (
                    <article key={result.word} className="tool-result-card">
                      <div className="tool-result-head">
                        <h3 className="tool-result-word">{result.word}</h3>
                        <span className="tool-result-tag">ONE-WORD</span>
                      </div>
                      <p className="tool-result-reason">{result.meaning || result.reason}</p>
                      {renderActions(
                        result.word,
                        `oneword:${oneWordQuery}:${result.word}`,
                        handleSaveOneWord,
                        (rating) => {
                          const key = `oneword:${oneWordQuery}:${result.word}`;
                          setRatings((current) => ({ ...current, [key]: rating }));
                          void submitRating('oneword', result.word, rating, {
                            model_score: result.score,
                            reason: result.reason,
                            input_payload: {
                              query: oneWordQuery.trim(),
                              context,
                            },
                            input_text: oneWordQuery.trim(),
                          });
                        },
                      )}
                    </article>
                  ))}
                </>
              )}
            </div>
          </div>
        ) : null}

        {!isAuthenticated && (
          <p className="panel-muted tool-auth-hint">Login to save words to your vocabulary.</p>
        )}
      </div>
    </>
  );

  if (embedded) {
    return <section className="tools-home-panel">{panelBody}</section>;
  }
  if (!isOpen) {
    return null;
  }
  return (
    <div className="overlay is-open" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        {panelBody}
      </div>
    </div>
  );
};

export default ToolsPanel;
