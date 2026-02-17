import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ToolsPanel, { type ToolTab } from '../components/tools/ToolsPanel';
import type { UserProfile } from '../types';

interface ToolsHomeProps {
  context: string;
  setContext: (value: string) => void;
  isAuthenticated: boolean;
  user: UserProfile | null;
  onRequireAuth: () => void;
  onLogout: () => void;
}

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

const TOOL_CARDS: Array<{
  id: ToolTab;
  title: string;
  subtitle: string;
  description: string;
}> = [
  {
    id: 'one_word',
    title: 'One-Word Substitution',
    subtitle: 'Describe it, get precise picks',
    description: 'Turns phrase-level descriptions into ranked single-word options.',
  },
  {
    id: 'smart_match',
    title: 'Smart Match',
    subtitle: 'Multi-constraint retrieval',
    description: 'Blend rhyme + semantic relation in one pass for high-precision outputs.',
  },
  {
    id: 'synonyms',
    title: 'Synonyms & Antonyms',
    subtitle: 'Fast lexical contrast',
    description: 'Swap tone or direction quickly with paired lexical alternatives.',
  },
  {
    id: 'rhymes',
    title: 'Rhyme & Homonym',
    subtitle: 'Sound-level creativity',
    description: 'Find phonetic matches and homonym variants for craft-heavy writing.',
  },
];

const formatLabel = (value: string) =>
  value ? value.charAt(0).toUpperCase() + value.slice(1) : 'Neutral';

const ToolsHome = ({
  context,
  setContext,
  isAuthenticated,
  user,
  onRequireAuth,
  onLogout,
}: ToolsHomeProps) => {
  const navigate = useNavigate();
  const [activeTool, setActiveTool] = useState<ToolTab>('one_word');

  const handleInsertFromHome = (word: string) => {
    localStorage.setItem('wordcraft_pending_insert', word);
    navigate('/editor');
  };

  return (
    <div className="tools-home-shell">
      <header className="tools-home-header">
        <div className="tools-home-copy">
          <p className="tools-home-kicker">WordCraft USP</p>
          <h1>Tools-first writing intelligence.</h1>
          <p>
            Run one-word substitutions, smart multi-constraint matches, and lexical
            searches before you enter deep writing mode.
          </p>
        </div>
        <div className="tools-home-actions">
          <label className="tools-home-context">
            Tone
            <select value={context} onChange={(event) => setContext(event.target.value)}>
              {CONTEXTS.map((ctx) => (
                <option key={ctx} value={ctx}>
                  {formatLabel(ctx)}
                </option>
              ))}
            </select>
          </label>
          <button type="button" className="btn-outline" onClick={() => navigate('/editor')}>
            Start Writing
          </button>
          {isAuthenticated ? (
            <button type="button" className="btn-outline" onClick={onLogout}>
              Logout {user?.username ? `(${user.username})` : ''}
            </button>
          ) : (
            <button type="button" className="btn-outline" onClick={onRequireAuth}>
              Login / Register
            </button>
          )}
        </div>
      </header>

      <section className="tool-home-grid">
        {TOOL_CARDS.map((card) => (
          <button
            key={card.id}
            type="button"
            className={`tool-home-card ${activeTool === card.id ? 'is-active' : ''}`}
            onClick={() => setActiveTool(card.id)}
          >
            <h2>{card.title}</h2>
            <p className="tool-home-subtitle">{card.subtitle}</p>
            <p className="tool-home-description">{card.description}</p>
          </button>
        ))}
      </section>

      <section className="tool-home-panel-shell">
        <ToolsPanel
          embedded
          activeTool={activeTool}
          onActiveToolChange={setActiveTool}
          onInsertWord={handleInsertFromHome}
          isAuthenticated={isAuthenticated}
          context={context}
        />
      </section>

      <div className="tools-home-footer">
        <span>Insert from Tools opens `/editor` and places the word in your draft.</span>
        <button type="button" className="btn-accept" onClick={() => navigate('/editor')}>
          Start Writing
        </button>
      </div>
    </div>
  );
};

export default ToolsHome;
