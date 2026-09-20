'use client';

import { useRef, useState } from 'react';
import { useRouter } from 'next/navigation';

import { uploadMedia } from '../lib/api';
import { formatBytes } from '../lib/format';

const MAX_UPLOAD_MB = Number(process.env.NEXT_PUBLIC_MAX_UPLOAD_MB ?? '200');
const ACCEPT = 'image/*,video/*';

/** Mirrors the server-side check so the user hears about it before uploading. */
function localValidationError(file: File): string | null {
  const isMedia = file.type.startsWith('image/') || file.type.startsWith('video/');
  const hasMediaExtension = /\.(jpe?g|png|webp|bmp|mp4|mov|avi|mkv|webm)$/i.test(file.name);
  if (!isMedia && !hasMediaExtension) {
    return 'Choose an image or a video file.';
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return `That file is ${formatBytes(file.size)}. The limit is ${MAX_UPLOAD_MB} MB.`;
  }
  return null;
}

export function UploadForm() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  function select(next: File | null) {
    setError(next ? localValidationError(next) : null);
    setFile(next);
  }

  async function submit() {
    if (!file || uploading) return;
    const invalid = localValidationError(file);
    if (invalid) return setError(invalid);

    setUploading(true);
    setError(null);
    try {
      const job = await uploadMedia(file);
      router.push(`/jobs/${job.id}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Upload failed.');
      setUploading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          select(e.dataTransfer.files?.[0] ?? null);
        }}
        className={`rounded-xl border-2 border-dashed p-10 text-center transition ${
          dragging
            ? 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/30'
            : 'border-neutral-300 dark:border-neutral-700'
        }`}
      >
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          Drag an image or video here
        </p>
        <p className="mt-1 text-xs text-neutral-500">or</p>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="mt-3 rounded-lg border border-neutral-300 px-4 py-2 text-sm font-medium transition hover:bg-neutral-100 dark:border-neutral-700 dark:hover:bg-neutral-800"
        >
          Choose a file
        </button>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="sr-only"
          onChange={(e) => select(e.target.files?.[0] ?? null)}
        />
        <p className="mt-4 text-xs text-neutral-500">
          JPEG, PNG, WebP, BMP · MP4, MOV, AVI, MKV, WebM · up to {MAX_UPLOAD_MB} MB
        </p>
      </div>

      {file && (
        <div className="flex items-center justify-between gap-3 rounded-lg border border-neutral-200 px-4 py-3 text-sm dark:border-neutral-800">
          <div className="min-w-0">
            <p className="truncate font-medium">{file.name}</p>
            <p className="text-xs text-neutral-500">
              {formatBytes(file.size)}
              {file.type ? ` · ${file.type}` : ''}
            </p>
          </div>
          <button
            type="button"
            onClick={() => select(null)}
            disabled={uploading}
            className="shrink-0 text-xs text-neutral-500 underline underline-offset-2 hover:text-neutral-800 disabled:opacity-40 dark:hover:text-neutral-200"
          >
            Remove
          </button>
        </div>
      )}

      {error && (
        <p
          role="alert"
          className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200"
        >
          {error}
        </p>
      )}

      <button
        type="button"
        onClick={submit}
        disabled={!file || uploading || Boolean(error)}
        className="rounded-lg bg-neutral-900 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-neutral-700 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-white dark:text-neutral-900 dark:hover:bg-neutral-200"
      >
        {uploading ? 'Uploading…' : 'Detect plates'}
      </button>
    </div>
  );
}
