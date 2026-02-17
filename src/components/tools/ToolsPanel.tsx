import React, { useEffect, useState } from 'react';
import { constraintsAPI, favoritesAPI, lexicalAPI, onewordAPI } from '../../services/api';
import type {
  ConstraintRelation,
  ConstraintResult,
  LexicalTask,
  LexicalResultDetail,
  OneWordResult,
  SelectionPayload,
} from '../../types';

interface ToolsPanelProps {
  isOpen?: boolean;
  onClose?: () => void;
  selection?: SelectionPayload | null;
  onInsertWord: (word: string) => void;
  isAuthenticated?: boolean;
  context?: string;
  embedded?: boolean;
  activeTool?: ToolTab;
  onActiveToolChange?: (tool: ToolTab) => void;
}

export type ToolTab = LexicalTask | 'smart_match' | 'one_word';

const LEXICAL_TASKS: LexicalTask[] = ['synonyms', 'antonyms', 'homonyms', 'rhymes'];
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

const TOOL_LABELS: Record<ToolTab, string> = {
  synonyms: 'Synonyms',
  antonyms: 'Antonyms',
  homonyms: 'Homonyms',
  rhymes: 'Rhymes',
  smart_match: 'Smart Match',
  one_word: 'One-Word',
};

const isLexicalTool = (task: ToolTab): task is LexicalTask => {
  return LEXICAL_TASKS.includes(task as LexicalTask);
};

const formatLabel = (value: string) =>
  value ? value.charAt(0).toUpperCase() + value.slice(1) : 'Neutral';

const ToolsPanel = ({
  isOpen = false,
  onClose = () => {},
  selection,
  onInsertWord,
  isAuthenticated = false,
  context = 'neutral',
  embedded = false,
  activeTool: controlledTool,
  onActiveToolChange,
}: ToolsPanelProps) => {
  const [internalActiveTool, setInternalActiveTool] = useState<ToolTab>(
    controlledTool ?? 'one_word',
  );
  const [toolWord, setToolWord] = useState('');
  const [toolResults, setToolResults] = useState<string[]>([]);
  const [toolResultDetails, setToolResultDetails] = useState<Record<string, LexicalResultDetail>>({});
  const [loading, setLoading] = useState(false);

  const [rhymeWith, setRhymeWith] = useState('');
  const [meaningOf, setMeaningOf] = useState('');
  const [relation, setRelation] = useState<ConstraintRelation>('synonym');
  const [smartResults, setSmartResults] = useState<ConstraintResult[]>([]);
  const [smartNote, setSmartNote] = useState<string>('');
  const [smartLoading, setSmartLoading] = useState(false);

  const [oneWordQuery, setOneWordQuery] = useState('');
  const [oneWordContext, setOneWordContext] = useState('global');
  const [oneWordResults, setOneWordResults] = useState<OneWordResult[]>([]);
  const [oneWordNote, setOneWordNote] = useState('');
  const [oneWordLoading, setOneWordLoading] = useState(false);

  const selectionWord = selection?.text?.split(/\s+/)[0] || '';
  const activeWord = toolWord.trim() || selectionWord;
  const selectedOneWordContext = oneWordContext === 'global' ? context : oneWordContext;
  const toolTasks: ToolTab[] = [...LEXICAL_TASKS, 'smart_match', 'one_word'];
  const activeTool = controlledTool ?? internalActiveTool;
  const isVisible = embedded || isOpen;

  useEffect(() => {
    if (!isVisible) {
      return;
    }
    setToolResults([]);
    setToolResultDetails({});
    setSmartResults([]);
    setSmartNote('');
    setOneWordResults([]);
    setOneWordNote('');
  }, [isVisible]);

  useEffect(() => {
    if (controlledTool) {
      setInternalActiveTool(controlledTool);
    }
  }, [controlledTool]);

  useEffect(() => {
    let isActive = true;
    const run = async () => {
      if (!isVisible || !isLexicalTool(activeTool)) {
        return;
      }
      if (!activeWord) {
        setToolResults([]);
        return;
      }
      setLoading(true);
      try {
        const response = await lexicalAPI.getResults(activeWord, activeTool, context);
        if (isActive) {
          setToolResults(response.results || []);
          const detailsMap: Record<string, LexicalResultDetail> = {};
          for (const item of response.details || []) {
            detailsMap[item.word] = item;
          }
          setToolResultDetails(detailsMap);
        }
      } catch (error) {
        if (isActive) {
          setToolResults([]);
          setToolResultDetails({});
        }
      } finally {
        if (isActive) {
          setLoading(false);
        }
      }
    };
    void run();
    return () => {
      isActive = false;
    };
  }, [activeTool, activeWord, isVisible]);

  const handleSmartMatch = async () => {
    if (!rhymeWith.trim() || !meaningOf.trim()) {
      setSmartResults([]);
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
    if (!oneWordQuery.trim()) {
      setOneWordResults([]);
      setOneWordNote('Describe the word you want to find.');
      return;
    }
    setOneWordLoading(true);
    try {
      const response = await onewordAPI.getResults({
        query: oneWordQuery.trim(),
        context: selectedOneWordContext,
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

  const handleSaveLexical = async (word: string) => {
    if (!isAuthenticated || !isLexicalTool(activeTool)) return;
    try {
      await favoritesAPI.saveFavorite({
        word,
        source: 'lexical',
        type: activeTool,
        context,
        related_to: activeWord,
      });
    } catch (error) {
      console.error('Error saving lexical word:', error);
    }
  };

  const handleSaveOneWord = async (word: string) => {
    if (!isAuthenticated) return;
    try {
      await favoritesAPI.saveFavorite({
        word,
        source: 'oneword',
        type: 'oneword',
        context: selectedOneWordContext,
        related_to: oneWordQuery.trim(),
      });
    } catch (error) {
      console.error('Error saving one-word result:', error);
    }
  };

  const resetResults = () => {
    setToolResults([]);
    setToolResultDetails({});
    setSmartResults([]);
    setSmartNote('');
    setOneWordResults([]);
    setOneWordNote('');
    setLoading(false);
  };

  const handleToolSelect = (task: ToolTab) => {
    if (!controlledTool) {
      setInternalActiveTool(task);
    }
    if (onActiveToolChange) {
      onActiveToolChange(task);
    }
    resetResults();
  };

  const panelBody = (
    <>
      <div className="panel-header">
        <h2 className="panel-title">Tools</h2>
        {!embedded && (
          <button onClick={onClose} className="panel-close" type="button">
            ×
          </button>
        )}
      </div>

      <div className="panel-body">
        <div className="tools-section">
          <div className="tools-header">Lexical Tools</div>
          <div className="tools-chips">
            {toolTasks.map((task) => (
              <button
                key={task}
                className={`tool-chip ${activeTool === task ? 'is-active' : ''}`}
                onClick={() => handleToolSelect(task)}
              >
                {TOOL_LABELS[task]}
              </button>
            ))}
          </div>

          {isLexicalTool(activeTool) && (
            <>
              <div className="tools-input">
                <input
                  value={toolWord}
                  onChange={(event) => setToolWord(event.target.value)}
                  placeholder={
                    selectionWord ? `Using selection: ${selectionWord}` : 'Type a word'
                  }
                />
              </div>
              <div className="tools-results">
                {loading ? (
                  <p className="panel-muted">Fetching {activeTool}…</p>
                ) : !activeWord ? (
                  <p className="panel-muted">Select or type a word to use tools.</p>
                ) : toolResults.length === 0 ? (
                  <p className="panel-muted">No results found.</p>
                ) : (
                  <div className="tools-result-grid">
                    {toolResults.map((result) => (
                      <div key={result} className="tools-result-row">
                        <div className="tools-result-detail">
                          <button
                            onClick={() => onInsertWord(result)}
                            className="tools-result-chip"
                          >
                            {result}
                          </button>
                          {toolResultDetails[result]?.reason && (
                            <p className="tools-result-note">
                              {toolResultDetails[result]?.reason}
                            </p>
                          )}
                        </div>
                        <button
                          type="button"
                          className="btn-ghost"
                          disabled={!isAuthenticated}
                          title={
                            isAuthenticated ? 'Save word' : 'Login to save your writing'
                          }
                          onClick={() => handleSaveLexical(result)}
                        >
                          Save
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}

          {activeTool === 'smart_match' && (
            <div className="smart-match">
              <div className="smart-row">
                <label>Rhyme with</label>
                <input
                  value={rhymeWith}
                  onChange={(event) => setRhymeWith(event.target.value)}
                  placeholder="night"
                />
              </div>
              <div className="smart-row">
                <label>Relation</label>
                <select
                  value={relation}
                  onChange={(event) => setRelation(event.target.value as ConstraintRelation)}
                >
                  <option value="synonym">Synonym</option>
                  <option value="antonym">Antonym</option>
                </select>
              </div>
              <div className="smart-row">
                <label>Meaning of</label>
                <input
                  value={meaningOf}
                  onChange={(event) => setMeaningOf(event.target.value)}
                  placeholder="bright"
                />
              </div>
              <button type="button" className="btn-accept" onClick={handleSmartMatch}>
                Find words
              </button>

              <div className="tools-results">
                {smartLoading ? (
                  <p className="panel-muted">Finding smart matches…</p>
                ) : smartResults.length === 0 ? (
                  <p className="panel-muted">{smartNote || 'Enter both words to see results.'}</p>
                ) : (
                  <>
                    {smartNote && <p className="panel-muted">{smartNote}</p>}
                    <div className="smart-results">
                      {smartResults.map((result) => (
                        <div key={result.word} className="smart-result-card">
                          <div className="smart-result-header">
                            <span className="smart-word">{result.word}</span>
                            <span className="smart-score">{result.score.toFixed(2)}</span>
                          </div>
                          <p className="smart-reason">{result.reason}</p>
                          <div className="smart-actions">
                            <button
                              type="button"
                              className="btn-accept"
                              onClick={() => onInsertWord(result.word)}
                            >
                              Insert
                            </button>
                            <button
                              type="button"
                              className="btn-ghost"
                              disabled={!isAuthenticated}
                              title={
                                isAuthenticated ? 'Save word' : 'Login to save your writing'
                              }
                              onClick={() => handleSaveSmart(result.word)}
                            >
                              Save
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {activeTool === 'one_word' && (
            <div className="smart-match">
              <div className="smart-row">
                <label>Describe the word you want</label>
                <input
                  value={oneWordQuery}
                  onChange={(event) => setOneWordQuery(event.target.value)}
                  placeholder="a person who loves themselves too much"
                />
              </div>

              <div className="smart-row">
                <label>Tone / context (optional)</label>
                <select
                  value={oneWordContext}
                  onChange={(event) => setOneWordContext(event.target.value)}
                >
                  <option value="global">Use global ({formatLabel(context)})</option>
                  {CONTEXTS.map((ctx) => (
                    <option key={ctx} value={ctx}>
                      {formatLabel(ctx)}
                    </option>
                  ))}
                </select>
              </div>

              <button type="button" className="btn-accept" onClick={handleFindOneWord}>
                Find one word
              </button>

              <div className="tools-results">
                {oneWordLoading ? (
                  <p className="panel-muted">Finding one-word substitutions…</p>
                ) : oneWordResults.length === 0 ? (
                  <p className="panel-muted">
                    {oneWordNote || 'Describe a phrase to generate one-word results.'}
                  </p>
                ) : (
                  <>
                    {oneWordNote && <p className="panel-muted">{oneWordNote}</p>}
                    <div className="smart-results">
                      {oneWordResults.map((result) => (
                        <div key={result.word} className="smart-result-card">
                          <div className="smart-result-header">
                            <span className="smart-word">{result.word}</span>
                            <span className="smart-score">{result.score.toFixed(2)}</span>
                          </div>
                          <p className="smart-reason">{result.meaning || result.reason}</p>
                          <div className="smart-actions">
                            <button
                              type="button"
                              className="btn-accept"
                              onClick={() => onInsertWord(result.word)}
                            >
                              Insert
                            </button>
                            <button
                              type="button"
                              className="btn-ghost"
                              disabled={!isAuthenticated}
                              title={
                                isAuthenticated ? 'Save word' : 'Login to save your writing'
                              }
                              onClick={() => handleSaveOneWord(result.word)}
                            >
                              Save
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
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
