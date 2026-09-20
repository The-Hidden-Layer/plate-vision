import type { JobStatus } from '../lib/types';

const STYLES: Record<JobStatus, string> = {
  queued:
    'bg-neutral-100 text-neutral-700 ring-neutral-300 dark:bg-neutral-800 dark:text-neutral-300 dark:ring-neutral-700',
  processing:
    'bg-amber-50 text-amber-800 ring-amber-300 dark:bg-amber-950 dark:text-amber-200 dark:ring-amber-800',
  done: 'bg-emerald-50 text-emerald-800 ring-emerald-300 dark:bg-emerald-950 dark:text-emerald-200 dark:ring-emerald-800',
  failed:
    'bg-red-50 text-red-800 ring-red-300 dark:bg-red-950 dark:text-red-200 dark:ring-red-800',
};

const LABELS: Record<JobStatus, string> = {
  queued: 'Queued',
  processing: 'Processing',
  done: 'Done',
  failed: 'Failed',
};

export function StatusPill({ status }: { status: JobStatus }) {
  const busy = status === 'queued' || status === 'processing';
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${STYLES[status]}`}
      aria-live="polite"
    >
      <span
        className={`h-1.5 w-1.5 rounded-full bg-current ${busy ? 'animate-pulse' : ''}`}
        aria-hidden
      />
      {LABELS[status]}
    </span>
  );
}
