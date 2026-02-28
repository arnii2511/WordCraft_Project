import React, { useState } from 'react';
import ToolsPanel, { type ToolTab } from '../components/tools/ToolsPanel';
import AppHeader from '../components/AppHeader';
import type { UserProfile } from '../types';

interface ToolsHomeProps {
  context: string;
  setContext: (value: string) => void;
  isAuthenticated: boolean;
  user: UserProfile | null;
  onRequireAuth: () => void;
}

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

const ToolsHome = ({
  context,
  setContext,
  isAuthenticated,
  user,
  onRequireAuth,
}: ToolsHomeProps) => {
  const [activeTool, setActiveTool] = useState<ToolTab>('one_word');

  return (
    <div className="tools-home-shell">
      <AppHeader
        activePage="tools"
        isAuthenticated={isAuthenticated}
        user={user}
        onRequireAuth={onRequireAuth}
      />

      <section className="tools-home-hero">
        <h2>Find the exact word, fast.</h2>
        <p>
          One-word substitutions, smart matches, rhymes, and lexical contrasts.
        </p>
      </section>

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
          isAuthenticated={isAuthenticated}
          context={context}
          onContextChange={setContext}
        />
      </section>

      <section className="tools-home-how">
        <div className="tools-home-how-item">Understands context</div>
        <div className="tools-home-how-item">Matches grammar</div>
        <div className="tools-home-how-item">Explains suggestions</div>
      </section>
    </div>
  );
};

export default ToolsHome;
