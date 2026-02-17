export type SuggestionPos = 'NOUN' | 'VERB' | 'ADJ' | 'ADV' | 'X';

export interface SuggestionItem {
  word: string;
  score?: number;
  pos?: SuggestionPos;
  note?: string;
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
}

export type ConstraintRelation = 'synonym' | 'antonym';

export interface ConstraintResult {
  word: string;
  score: number;
  rhyme: boolean;
  relation_match: boolean;
  reason: string;
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
