// Hand-written for now. Phase 6 replaces these with types generated from the
// backend's OpenAPI schema (libs/api-types); the shape is intentionally identical.

export type JobStatus = 'queued' | 'processing' | 'done' | 'failed';
export type MediaType = 'image' | 'video';

export interface Detection {
  id: number;
  plate_text: string;
  confidence: number;
  bbox: [number, number, number, number];
  frame_index: number;
  timestamp_ms: number | null;
  crop_url: string | null;
}

export interface Job {
  id: string;
  status: JobStatus;
  media_type: MediaType;
  source_filename: string;
  media_url: string | null;
  error: string;
  frame_count: number | null;
  annotated_frame_urls: string[];
  detections: Detection[];
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface Paginated<T> {
  count: number;
  next: string | null;
  previous: string | null;
  results: T[];
}

export const isTerminal = (status: JobStatus) => status === 'done' || status === 'failed';
