'use client';

import Link from 'next/link';
import { use, useEffect, useState } from 'react';

import { DetectionTable } from '../../../components/DetectionTable';
import { FrameGallery } from '../../../components/FrameGallery';
import { StatusPill } from '../../../components/StatusPill';
import { VideoCoverage } from '../../../components/VideoCoverage';
import { VideoResults } from '../../../components/VideoResults';
import { fetchJob } from '../../../lib/api';
import { recognizedDetections } from '../../../lib/detections';
import { formatDuration } from '../../../lib/format';
import { isTerminal, type Job } from '../../../lib/types';

const POLL_INTERVAL_MS = 2000;

export default function JobPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let cancelled = false;

    async function poll() {
      try {
        const next = await fetchJob(id, controller.signal);
        if (cancelled) return;
        setJob(next);
        setError(null);
        // Stop polling once the job can no longer change.
        if (!isTerminal(next.status)) timer = setTimeout(poll, POLL_INTERVAL_MS);
      } catch (cause) {
        if (cancelled || controller.signal.aborted) return;
        setError(cause instanceof Error ? cause.message : 'Could not load this job.');
        timer = setTimeout(poll, POLL_INTERVAL_MS);
      }
    }

    poll();
    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, [id]);

  if (error && !job) {
    return (
      <Empty>
        <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        <Link href="/" className="mt-4 inline-block text-sm underline underline-offset-2">
          Upload something else
        </Link>
      </Empty>
    );
  }

  if (!job) {
    return <Empty><p className="text-sm text-neutral-500">Loading job…</p></Empty>;
  }

  const duration = formatDuration(job.started_at, job.finished_at);
  const busy = !isTerminal(job.status);
  const recognized = recognizedDetections(job.detections);

  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="truncate text-xl font-semibold tracking-tight">{job.source_filename}</h1>
          <p className="mt-1 text-sm text-neutral-500">
            {job.media_type}
            {job.media_type === 'video' && job.frame_count ? ` · ${job.frame_count} source frames` : ''}
            {duration ? ` · processed in ${duration}` : ''}
          </p>
          {job.media_type === 'video' && job.status === 'done' && (
            <VideoCoverage analysis={job.video_analysis} />
          )}
        </div>
        <StatusPill status={job.status} />
      </header>

      {busy && (
        <div className="flex items-center gap-3 rounded-lg border border-neutral-200 px-4 py-4 text-sm dark:border-neutral-800">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-700 dark:border-neutral-700 dark:border-t-neutral-200" />
          <span className="text-neutral-600 dark:text-neutral-400">
            {job.status === 'queued'
              ? 'Waiting for a worker…'
              : 'Running detection and recognition…'}
          </span>
        </div>
      )}

      {job.status === 'failed' && (
        <div
          role="alert"
          className="rounded-lg border border-red-300 bg-red-50 px-4 py-4 dark:border-red-900 dark:bg-red-950/50"
        >
          <p className="text-sm font-medium text-red-900 dark:text-red-200">
            Processing failed
          </p>
          <pre className="mt-2 overflow-x-auto whitespace-pre-wrap break-words font-mono text-xs text-red-800 dark:text-red-300">
            {job.error || 'No further detail was reported.'}
          </pre>
        </div>
      )}

      {error && job && (
        <p className="text-xs text-amber-700 dark:text-amber-400">
          Lost contact with the server; retrying… ({error})
        </p>
      )}

      {job.media_type === 'video' && (job.media_url ? (
        <VideoResults
          key={job.id}
          src={job.media_url}
          detections={job.detections}
          sampleFps={job.video_analysis?.sample_fps}
          status={job.status}
        />
      ) : (
        <p className="text-sm text-neutral-500">The original video is unavailable for playback.</p>
      ))}

      {job.status === 'done' && (
        <>
          {job.media_type === 'video' ? (
            <details className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
              <summary className="cursor-pointer text-sm font-semibold">
                All recognized plates ({recognized.length})
              </summary>
              <div className="mt-4"><DetectionTable detections={recognized} /></div>
            </details>
          ) : <section>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-neutral-500">
              Recognized plates{' '}
              <span className="font-normal normal-case">
                ({recognized.length})
              </span>
            </h2>
            <DetectionTable detections={recognized} />
          </section>}

          {job.media_type === 'image' && job.annotated_frame_urls.length > 0 && (
            <section>
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-neutral-500">
                Annotated frames{' '}
                <span className="font-normal normal-case">
                  ({job.annotated_frame_urls.length})
                </span>
              </h2>
              <FrameGallery urls={job.annotated_frame_urls} />
            </section>
          )}
        </>
      )}

      <footer className="border-t border-neutral-200 pt-5 text-sm dark:border-neutral-800">
        <Link href="/" className="underline underline-offset-2">
          Upload another file
        </Link>
      </footer>
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="py-20 text-center">{children}</div>;
}
