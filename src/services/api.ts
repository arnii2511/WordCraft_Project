import axios from 'axios';
import type {
  AuthResponse,
  ConstraintResponse,
  DocumentEntry,
  FeedbackPayload,
  FeedbackResponse,
  FavoriteEntry,
  LexicalTask,
  LexicalResponse,
  OneWordResponse,
  SuggestResponse,
  UserProfile,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const publicApi = axios.create({
  baseURL: API_BASE_URL,
});

const protectedApi = axios.create({
  baseURL: API_BASE_URL,
});

protectedApi.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

const storeProfile = (profile: UserProfile) => {
  localStorage.setItem('user_profile', JSON.stringify(profile));
  localStorage.setItem('user_email', profile.email);
};

const clearProfile = () => {
  localStorage.removeItem('auth_token');
  localStorage.removeItem('user_profile');
  localStorage.removeItem('user_email');
};

// Authentication
export const authAPI = {
  login: async (email: string, password: string): Promise<AuthResponse> => {
    const response = await publicApi.post<AuthResponse>('/auth/login', { email, password });
    localStorage.setItem('auth_token', response.data.token);
    storeProfile(response.data.user);
    return response.data;
  },

  register: async (payload: Record<string, string>): Promise<AuthResponse> => {
    const response = await publicApi.post<AuthResponse>('/auth/register', payload);
    return response.data;
  },

  getMe: async (): Promise<UserProfile> => {
    const response = await protectedApi.get<UserProfile>('/auth/me');
    storeProfile(response.data);
    return response.data;
  },

  updateMe: async (payload: {
    email?: string;
    username?: string;
    phone?: string;
    bio?: string;
    interests?: string;
  }): Promise<UserProfile> => {
    const response = await protectedApi.put<UserProfile>('/auth/me', payload);
    storeProfile(response.data);
    return response.data;
  },

  changePassword: async (payload: {
    current_password: string;
    new_password: string;
  }): Promise<{ message: string }> => {
    const response = await protectedApi.post<{ message: string }>(
      '/auth/change-password',
      payload,
    );
    return response.data;
  },

  logout: () => {
    clearProfile();
  },

  getProfile: (): UserProfile | null => {
    const stored = localStorage.getItem('user_profile');
    if (!stored) return null;
    try {
      return JSON.parse(stored) as UserProfile;
    } catch {
      return null;
    }
  },

  isAuthenticated: () => {
    return !!localStorage.getItem('auth_token');
  },
};

// Writing Assistance API
export const writingAPI = {
  getSuggestions: async (
    sentence: string,
    context: string,
    mode: 'write' | 'edit' | 'rewrite',
    selection: { text: string; start: number; end: number } | null,
    trigger: 'auto' | 'button',
  ): Promise<SuggestResponse> => {
    const response = await publicApi.post<SuggestResponse>('/suggest', {
      sentence,
      context,
      mode,
      selection,
      trigger,
    });
    return response.data;
  },
};

// Lexical Tools API
export const lexicalAPI = {
  getResults: async (
    word: string,
    task: LexicalTask,
    context?: string,
  ): Promise<LexicalResponse> => {
    const response = await publicApi.post<LexicalResponse>('/lexical', {
      word,
      task,
      context,
    });
    return response.data;
  },
};

export const constraintsAPI = {
  getMatches: async (payload: {
    rhyme_with: string;
    relation: 'synonym' | 'antonym';
    meaning_of: string;
    context?: string;
    limit?: number;
  }): Promise<ConstraintResponse> => {
    const response = await publicApi.post<ConstraintResponse>('/constraints', payload);
    return response.data;
  },
};

export const onewordAPI = {
  getResults: async (payload: {
    query: string;
    context?: string;
    limit?: number;
  }): Promise<OneWordResponse> => {
    const response = await publicApi.post<OneWordResponse>('/oneword', payload);
    return response.data;
  },
};

// Favorites API
export const favoritesAPI = {
  saveFavorite: async (payload: {
    word: string;
    source?: string;
    type?: string;
    context?: string;
    related_to?: string;
  }): Promise<FavoriteEntry> => {
    const response = await protectedApi.post<FavoriteEntry>('/saved-words', payload);
    return response.data;
  },

  getFavorites: async (): Promise<FavoriteEntry[]> => {
    const response = await protectedApi.get<FavoriteEntry[]>('/saved-words');
    return response.data;
  },

  deleteFavorite: async (id: number | string) => {
    await protectedApi.delete(`/saved-words/${id}`);
  },
};

// Documents API
export const documentsAPI = {
  getDocuments: async (): Promise<DocumentEntry[]> => {
    const response = await protectedApi.get('/documents');
    return (response.data || []).map((doc: any) => ({
      id: doc.id,
      title: doc.title,
      contentHtml: doc.content_html || '',
      contentText: doc.content_text || '',
      context: doc.context,
      mode: doc.mode,
      createdAt: doc.created_at,
      updatedAt: doc.updated_at,
    }));
  },

  saveDocument: async (
    payload: {
      id?: string;
      title: string;
      contentHtml: string;
      contentText: string;
      context: string;
      mode: 'write' | 'edit' | 'rewrite';
    },
  ): Promise<DocumentEntry> => {
    if (payload.id) {
      const response = await protectedApi.put(`/documents/${payload.id}`, {
        title: payload.title,
        content_html: payload.contentHtml,
        content_text: payload.contentText,
        context: payload.context,
        mode: payload.mode,
      });
      const doc = response.data;
      return {
        id: doc.id,
        title: doc.title,
        contentHtml: doc.content_html || '',
        contentText: doc.content_text || '',
        context: doc.context,
        mode: doc.mode,
        createdAt: doc.created_at,
        updatedAt: doc.updated_at,
      };
    }
    const response = await protectedApi.post('/documents', {
      title: payload.title,
      content_html: payload.contentHtml,
      content_text: payload.contentText,
      context: payload.context,
      mode: payload.mode,
    });
    const doc = response.data;
    return {
      id: doc.id,
      title: doc.title,
      contentHtml: doc.content_html || '',
      contentText: doc.content_text || '',
      context: doc.context,
      mode: doc.mode,
      createdAt: doc.created_at,
      updatedAt: doc.updated_at,
    };
  },

  deleteDocument: async (id: string) => {
    await protectedApi.delete(`/documents/${id}`);
  },
};

export const feedbackAPI = {
  submitRating: async (payload: FeedbackPayload): Promise<FeedbackResponse> => {
    // Feedback endpoint accepts optional user; allow guest telemetry too.
    const response = await publicApi.post<FeedbackResponse>('/feedback', payload);
    return response.data;
  },
};

export default {
  authAPI,
  writingAPI,
  lexicalAPI,
  constraintsAPI,
  onewordAPI,
  documentsAPI,
  favoritesAPI,
  feedbackAPI,
};
