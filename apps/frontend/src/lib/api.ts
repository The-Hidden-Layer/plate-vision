import type { Job, Paginated } from './types';

/** Pull a readable message out of a DRF error body. */
async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === 'string') return body.detail;
    // DRF field errors: { file: ["..."] }
    const first = Object.values(body ?? {}).flat()[0];
    if (typeof first === 'string') return first;
  } catch {
    // fall through to the status line
  }
  return `Request failed (HTTP ${response.status})`;
}

export async function uploadMedia(file: File, signal?: AbortSignal): Promise<Job> {
  const form = new FormData();
  form.append('file', file);

  const response = await fetch('/api/jobs', { method: 'POST', body: form, signal });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function fetchJob(id: string, signal?: AbortSignal): Promise<Job> {
  const response = await fetch(`/api/jobs/${id}`, { signal, cache: 'no-store' });
  if (response.status === 404) throw new Error('That job does not exist.');
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function fetchJobs(signal?: AbortSignal): Promise<Paginated<Job>> {
  const response = await fetch('/api/jobs', { signal, cache: 'no-store' });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}
