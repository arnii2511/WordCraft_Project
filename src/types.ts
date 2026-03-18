export type SuggestionPos = 'NOUN' | 'VERB' | 'ADJ' | 'ADV' | 'X';
export type VocabularyPreference = 'balanced' | 'advanced';

export interface SuggestionItem {
  word: string;
  score?: number;
  pos?: SuggestionPos;
  note?: string;
  source?: string;
  ml_score?: number;
  behavior_score?: number;
  confidence?: number;
  score_breakdown?: Record<string, unknown>;
}

export interface SuggestResponse {
  suggestions: SuggestionItem[];
  rewrite?: string;
  rewrites?: string[];
  explanation?: string;
  detected_blank?: boolean;
  original?: string;
}

export interface SelectionPayload {
  text: string;
  start: number;
  end: number;
}

export type LexicalTask = 'synonyms' | 'antonyms' | 'homonyms' | 'rhymes';

export interface LexicalResponse {
  word: string;
  task: LexicalTask;
  results: string[];
  details?: LexicalResultDetail[];
}

export interface LexicalResultDetail {
  word: string;
  score: number;
  pos?: string | null;
  reason: string;
  source?: string;
  ml_score?: number;
  behavior_score?: number;
  confidence?: number;
  score_breakdown?: Record<string, unknown>;
}

export type ConstraintRelation = 'synonym' | 'antonym';

export interface ConstraintResult {
  word: string;
  score: number;
  rhyme: boolean;
  relation_match: boolean;
  reason: string;
  source?: string;
  ml_score?: number;
  behavior_score?: number;
  confidence?: number;
  score_breakdown?: Record<string, unknown>;
}

export interface ConstraintResponse {
  results: ConstraintResult[];
  notes?: string | null;
}

export interface OneWordResult {
  word: string;
  score: number;
  reason: string;
  meaning?: string | null;
  source?: string;
  ml_score?: number;
  behavior_score?: number;
  confidence?: number;
  score_breakdown?: Record<string, unknown>;
}

export interface OneWordResponse {
  query: string;
  results: OneWordResult[];
  note?: string | null;
}

export interface UserProfile {
  id: string;
  email: string;
  username: string;
  phone?: string;
  bio?: string;
  interests?: string;
  created_at?: string | null;
}

export interface AuthResponse {
  token: string;
  user: UserProfile;
}

export interface HistoryEntry {
  query: string;
  timestamp: string;
  response?: SuggestResponse;
}

export interface FavoriteEntry {
  id: string;
  word: string;
  source: string;
  type: string;
  context?: string;
  related_to?: string;
  created_at?: string | null;
}

export interface DocumentEntry {
  id: string;
  title: string;
  contentHtml: string;
  contentText: string;
  context: string;
  mode: 'write' | 'edit' | 'rewrite';
  createdAt?: string | null;
  updatedAt?: string | null;
}

export type FeedbackTask =
  | 'editor_suggestion'
  | 'editor_rewrite'
  | 'lexical'
  | 'constraints'
  | 'oneword';

export interface FeedbackPayload {
  task: FeedbackTask;
  candidate: string;
  rating: number;
  context?: string;
  mode?: 'write' | 'edit' | 'rewrite';
  input_payload?: Record<string, unknown>;
  source?: string;
  pos?: string;
  model_score?: number;
  reason?: string;
  session_id?: string;
  input_text?: string;
  vocabulary_preference?: VocabularyPreference;
}

export interface ImplicitFeedbackPayload {
  task: FeedbackTask;
  candidate: string;
  action: 'inserted' | 'copied' | 'favorited';
  context?: string;
  mode?: 'write' | 'edit' | 'rewrite';
  input_payload?: Record<string, unknown>;
  pos?: string;
  model_score?: number;
  reason?: string;
  session_id?: string;
  input_text?: string;
  vocabulary_preference?: VocabularyPreference;
}

export interface FeedbackResponse {
  id: string;
  task: string;
  candidate: string;
  rating: number;
  quality: 'bad' | 'average' | 'good';
  label: number;
  message: string;
}
