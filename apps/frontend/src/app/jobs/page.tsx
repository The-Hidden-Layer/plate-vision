'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

import { StatusPill } from '../../components/StatusPill';
import { fetchJobs } from '../../lib/api';
import { recognizedDetections } from '../../lib/detections';
import { formatRelative } from '../../lib/format';
import { isTerminal, type Job } from '../../lib/types';

const REFRESH_INTERVAL_MS = 5000;

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let cancelled = false;

    async function load() {
      try {
        const page = await fetchJobs(controller.signal);
        if (cancelled) return;
        setJobs(page.results);
        setError(null);
        // Keep refreshing only while something can still change.
        if (page.results.some((job) => !isTerminal(job.status))) {
          timer = setTimeout(load, REFRESH_INTERVAL_MS);
        }
      } catch (cause) {
        if (cancelled || controller.signal.aborted) return;
        setError(cause instanceof Error ? cause.message : 'Could not load jobs.');
      }
    }

    load();
    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  return (
    <div>
      <h1 className="text-xl font-semibold tracking-tight">Recent jobs</h1>

      {error && (
        <p role="alert" className="mt-6 text-sm text-red-700 dark:text-red-300">
          {error}
        </p>
      )}

      {!jobs && !error && <p className="mt-6 text-sm text-neutral-500">Loading…</p>}

      {jobs?.length === 0 && (
        <div className="mt-6 rounded-lg border border-dashed border-neutral-300 p-10 text-center dark:border-neutral-700">
          <p className="text-sm text-neutral-500">Nothing has been processed yet.</p>
          <Link href="/" className="mt-3 inline-block text-sm underline underline-offset-2">
            Upload a file
          </Link>
        </div>
      )}

      {jobs && jobs.length > 0 && (
        <ul className="mt-6 divide-y divide-neutral-200 rounded-lg border border-neutral-200 dark:divide-neutral-800 dark:border-neutral-800">
          {jobs.map((job) => (
            <li key={job.id}>
              <Link
                href={`/jobs/${job.id}`}
                className="flex items-center justify-between gap-4 px-4 py-3.5 transition hover:bg-neutral-50 dark:hover:bg-neutral-900/50"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{job.source_filename}</p>
                  <p className="mt-0.5 text-xs text-neutral-500">
                    {job.media_type} · {formatRelative(job.created_at)}
                    {job.status === 'done' && ` · ${recognizedDetections(job.detections).length} recognized plates`}
                  </p>
                </div>
                <StatusPill status={job.status} />
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
