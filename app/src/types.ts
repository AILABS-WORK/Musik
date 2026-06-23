export interface Track {
  id: number;
  name: string;
  path: string;
  fmt: string | null;
  duration: number | null;
  genre: string | null;
  /** Major genre and subgenre split out (genre is the combined "Major / Sub" label). */
  major?: string | null;
  subgenre?: string | null;
  confidence: number | null;
  assignment_status: string | null;
  status: string;
  bpm: number | null;
  music_key: string | null;
  energy: number | null;
}

export interface Genre {
  id: number;
  name: string;
  parent_id: number | null;
  level: string;
  source: string;
  has_centroid: number;
}

export interface SimilarItem {
  track_id: number;
  name: string;
  score: number;
}

export interface Health {
  status: string;
  model: string;
  db: string;
  library: string | null;
  tracks: number;
  genres: number;
}

export interface Progress {
  running: boolean;
  done: number;
  total: number;
  last: string;
  error: string | null;
}

export interface AppConfig {
  library_root: string | null;
  db_path: string;
  active_model: string;
  organize_root: string | null;
  organize_mode: string;
  confidence_threshold: number;
}
