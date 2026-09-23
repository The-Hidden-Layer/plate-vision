import type { Job } from '../lib/types';

export function VideoCoverage({ analysis }: { analysis: Job['video_analysis'] }) {
  return (
    <p className="mt-1 text-sm text-neutral-500">
      {analysis
        ? `${analysis.sampled_frame_count} frames analyzed · ${analysis.sample_fps} fps target`
        : 'Analysis coverage was not recorded for this older job.'}
    </p>
  );
}
