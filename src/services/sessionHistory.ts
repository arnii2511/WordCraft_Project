export interface SessionSnapshot {
  id: string;
  ts: number;
  contentText: string;
  contentHtml: string;
  context?: string;
  mode?: string;
}

const HISTORY_PREFIX = 'wordcraft_history_';

const getHistoryKey = (documentId?: string | null) =>
  `${HISTORY_PREFIX}${documentId || 'guest'}`;

const loadSnapshots = (documentId?: string | null): SessionSnapshot[] => {
  const key = getHistoryKey(documentId);
  const raw = sessionStorage.getItem(key);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed as SessionSnapshot[];
    }
    return [];
  } catch {
    return [];
  }
};

const saveSnapshot = (
  documentId: string | null,
  snapshot: SessionSnapshot,
  maxItems = 50,
) => {
  const key = getHistoryKey(documentId);
  const history = loadSnapshots(documentId);
  history.unshift(snapshot);
  sessionStorage.setItem(key, JSON.stringify(history.slice(0, maxItems)));
};

const deleteSnapshot = (documentId: string | null, snapshotId: string) => {
  const key = getHistoryKey(documentId);
  const history = loadSnapshots(documentId);
  const filtered = history.filter((item) => item.id !== snapshotId);
  sessionStorage.setItem(key, JSON.stringify(filtered));
};

const clearSnapshots = (documentId?: string | null) => {
  const key = getHistoryKey(documentId);
  sessionStorage.removeItem(key);
};

const clearAllSnapshots = () => {
  const keysToRemove: string[] = [];
  for (let i = 0; i < sessionStorage.length; i += 1) {
    const key = sessionStorage.key(i);
    if (key && key.startsWith(HISTORY_PREFIX)) {
      keysToRemove.push(key);
    }
  }
  keysToRemove.forEach((key) => sessionStorage.removeItem(key));
};

export {
  clearAllSnapshots,
  clearSnapshots,
  deleteSnapshot,
  getHistoryKey,
  loadSnapshots,
  saveSnapshot,
};
