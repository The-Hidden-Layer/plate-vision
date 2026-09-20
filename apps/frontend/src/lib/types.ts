/**
 * Re-exports the generated API types so components import from one place.
 * The source of truth is the backend's OpenAPI schema; regenerate with
 * `pnpm nx run api-types:generate`.
 */

import type { JobStatus } from '@org/api-types';

export type {
  Detection,
  Job,
  JobStatus,
  MediaType,
  PaginatedJobList,
} from '@org/api-types';

/** A job in a terminal state can no longer change, so polling can stop. */
export const isTerminal = (status: JobStatus) => status === 'done' || status === 'failed';
