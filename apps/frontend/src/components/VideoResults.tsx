'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

import { buildPlateTimeline, plateCueAt, type PlateCue } from '../lib/video-detections';
import { formatTimestamp } from '../lib/format';
import type { Detection, JobStatus } from '../lib/types';
import { DetectionTable } from './DetectionTable';
import { PlateText } from './PlateText';

type Props = {
  src: string;
  detections: Detection[];
  sampleFps?: number;
  status: JobStatus;
};

export function VideoResults({ src, detections, sampleFps, status }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [activeCue, setActiveCue] = useState<PlateCue>();
  const [playbackError, setPlaybackError] = useState(false);
  const timeline = useMemo(
    () => buildPlateTimeline(detections, sampleFps),
    [detections, sampleFps],
  );

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let callback: number | undefined;
    let animation: number | undefined;
    const update = (seconds = video.currentTime) => {
      setActiveCue(status === 'done' ? plateCueAt(timeline, seconds * 1000) : undefined);
    };
    const stop = () => {
      if (callback !== undefined) video.cancelVideoFrameCallback(callback);
      if (animation !== undefined) cancelAnimationFrame(animation);
      callback = animation = undefined;
    };
    const tick = () => {
      if (video.paused || video.ended) return;
      if (typeof video.requestVideoFrameCallback === 'function') {
        callback = video.requestVideoFrameCallback((_now, metadata) => {
          update(metadata.mediaTime);
          tick();
        });
      } else {
        animation = requestAnimationFrame(() => {
          update();
          tick();
        });
      }
    };
    const sync = () => update();
    const play = () => { stop(); update(); tick(); };
    const pause = () => { stop(); update(); };
    video.addEventListener('play', play);
    video.addEventListener('pause', pause);
    video.addEventListener('ended', pause);
    video.addEventListener('seeking', sync);
    video.addEventListener('seeked', sync);
    video.addEventListener('timeupdate', sync);
    video.addEventListener('loadedmetadata', sync);
    update();
    tick();
    return () => {
      stop();
      video.removeEventListener('play', play);
      video.removeEventListener('pause', pause);
      video.removeEventListener('ended', pause);
      video.removeEventListener('seeking', sync);
      video.removeEventListener('seeked', sync);
      video.removeEventListener('timeupdate', sync);
      video.removeEventListener('loadedmetadata', sync);
    };
  }, [timeline, status]);

  const current = playbackError ? [] : activeCue?.detections ?? [];

  return (
    <section className="flex flex-col gap-4" aria-label="Video and synchronized plates">
      <div>
        <h2 className="text-sm font-semibold uppercase tracking-wide text-neutral-500">Video playback</h2>
        <p className="mt-1 text-sm text-neutral-500">
          Play or seek to see readable plates at that moment. Labels follow the analyzed frames.
        </p>
      </div>
      <div
        className="relative mx-auto w-full overflow-hidden rounded-lg bg-black"
        style={size.height > 0 ? { maxWidth: `${65 * size.width / size.height}vh` } : undefined}
      >
        <video
          ref={videoRef}
          src={src}
          controls
          playsInline
          preload="metadata"
          className="block h-auto w-full"
          aria-label="Uploaded video"
          onLoadedMetadata={(event) => {
            setSize({ width: event.currentTarget.videoWidth, height: event.currentTarget.videoHeight });
            setPlaybackError(false);
          }}
          onError={() => setPlaybackError(true)}
        />
        {size.width > 0 && size.height > 0 && (
          <div className="pointer-events-none absolute inset-0" aria-hidden="true">
            {current.map((detection) => {
              const [x1, y1, x2, y2] = detection.bbox;
              return (
                <div
                  key={detection.id}
                  className="absolute rounded-sm border-2 border-emerald-400"
                  style={{
                    left: `${100 * x1 / size.width}%`,
                    top: `${100 * y1 / size.height}%`,
                    width: `${100 * (x2 - x1) / size.width}%`,
                    height: `${100 * (y2 - y1) / size.height}%`,
                  }}
                >
                  <span className="absolute left-0 top-0 whitespace-nowrap rounded-sm bg-emerald-950/90 px-1.5 py-0.5 font-mono text-xs font-semibold text-white">
                    <PlateText text={detection.plate_text} />
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
      {playbackError && (
        <p role="alert" className="text-sm text-amber-700 dark:text-amber-400">
          This video could not be played. Try uploading an MP4 with H.264 video, or{' '}
          <a href={src} download className="underline">download the original video</a>.
        </p>
      )}
      <div>
        <h3 className="mb-3 text-sm font-semibold text-neutral-600 dark:text-neutral-300">
          Readable plates at this moment ({current.length})
          {current.length > 0 && activeCue && (
            <span className="ml-2 font-normal text-neutral-500">Observed at {formatTimestamp(activeCue.start)}</span>
          )}
        </h3>
        {status !== 'done' ? (
          <p className="text-sm text-neutral-500">
            {status === 'failed' ? 'Plate analysis failed for this video.' : 'Plate results will appear here when analysis finishes.'}
          </p>
        ) : current.length > 0 ? (
          <DetectionTable detections={current} />
        ) : (
          <p className="rounded-lg border border-dashed border-neutral-300 p-6 text-center text-sm text-neutral-500 dark:border-neutral-700">
            {playbackError ? 'Playback is unavailable. All recognized plates are listed below.' : 'No readable plates at this moment.'}
          </p>
        )}
      </div>
    </section>
  );
}
