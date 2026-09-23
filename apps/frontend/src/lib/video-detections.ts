import { recognizedDetections } from './detections';
import type { Detection } from './types';

export type PlateCue = { start: number; end: number; detections: Detection[] };

export function buildPlateTimeline(detections: Detection[], sampleFps?: number): PlateCue[] {
  const frames = new Map<number, Detection[]>();
  for (const detection of detections) {
    const timestamp = detection.timestamp_ms;
    if (timestamp === null || !Number.isFinite(timestamp) || timestamp < 0) continue;
    const frame = frames.get(timestamp) ?? [];
    frame.push(detection);
    frames.set(timestamp, frame);
  }
  const times = [...frames.keys()].sort((a, b) => a - b);
  // Never carry an old plate across gaps in the sampled video. Older jobs
  // lack a sampling rate; show each known observation for at most 500 ms.
  const interval = sampleFps && Number.isFinite(sampleFps) && sampleFps > 0
    ? 1000 / sampleFps
    : 500;
  return times.map((start, index) => ({
    start,
    end: Math.min(start + interval, times[index + 1] ?? Infinity),
    detections: recognizedDetections(frames.get(start) ?? []),
  }));
}

export function plateCueAt(timeline: PlateCue[], timeMs: number): PlateCue | undefined {
  let low = 0;
  let high = timeline.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (timeline[mid].start <= timeMs) low = mid + 1;
    else high = mid;
  }
  const cue = timeline[low - 1];
  return cue && timeMs < cue.end ? cue : undefined;
}
