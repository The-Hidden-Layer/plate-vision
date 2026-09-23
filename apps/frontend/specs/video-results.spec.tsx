import { act, fireEvent, render, within } from '@testing-library/react';
import React from 'react';

import { VideoResults } from '../src/components/VideoResults';
import { buildPlateTimeline, plateCueAt } from '../src/lib/video-detections';
import type { Detection } from '../src/lib/types';

const plate = (id: number, timestamp_ms: number | null, plate_text = '22ب41763'): Detection => ({
  id, timestamp_ms, plate_text, confidence: 0.95,
  frame_index: id * 15, bbox: [100, 50, 300, 100], crop_url: `/media/${id}.jpg`,
});

describe('Video plate timeline', () => {
  it('groups simultaneous plates, sorts observations, and excludes unreadable or untimed results', () => {
    const timeline = buildPlateTimeline([
      plate(4, 2000), plate(1, 1000), plate(2, 1000, '12الف34567'),
      plate(3, 1000, '  '), plate(5, null),
    ], 2);
    expect(plateCueAt(timeline, 999)).toBeUndefined();
    expect(plateCueAt(timeline, 1000)?.detections.map((d) => d.id)).toEqual([1, 2]);
    expect(plateCueAt(timeline, 1499)?.detections).toHaveLength(2);
    expect(plateCueAt(timeline, 1500)).toBeUndefined();
    expect(plateCueAt(timeline, 2000)?.detections[0].id).toBe(4);
  });

  it('clears a readable plate at the next unreadable observation and across long gaps', () => {
    const timeline = buildPlateTimeline([plate(1, 0), plate(2, 250, ''), plate(3, 10000)], 2);
    expect(plateCueAt(timeline, 249)?.detections).toHaveLength(1);
    expect(plateCueAt(timeline, 250)?.detections).toHaveLength(0);
    expect(plateCueAt(timeline, 5000)).toBeUndefined();
  });

  it('uses the sampling interval and a bounded fallback for old jobs', () => {
    expect(plateCueAt(buildPlateTimeline([plate(1, 0)], 4), 250)).toBeUndefined();
    expect(plateCueAt(buildPlateTimeline([plate(1, 0)]), 500)).toBeUndefined();
  });
});

describe('Video results', () => {
  const detections = [plate(1, 1000), plate(2, 1000, ''), plate(3, 2000, '12الف34567')];

  it('updates readable rows and scaled overlays when playing and seeking in either direction', () => {
    const view = render(<VideoResults src="/media/video.mp4" detections={detections} sampleFps={2} status="done" />);
    const video = view.getByLabelText('Uploaded video') as HTMLVideoElement;
    Object.defineProperties(video, {
      videoWidth: { value: 1000 }, videoHeight: { value: 500 },
    });
    fireEvent.loadedMetadata(video);
    expect(view.queryByRole('table')).toBeNull();

    video.currentTime = 1.1;
    fireEvent.timeUpdate(video);
    expect(within(view.getByRole('table')).getByLabelText('22ب41763')).toBeTruthy();
    expect(view.queryByAltText('Plate crop unreadable')).toBeNull();
    expect(within(view.getByRole('table')).getAllByRole('row')).toHaveLength(2);
    const overlay = view.container.querySelector('[aria-hidden="true"] > div') as HTMLElement;
    expect(overlay.style.left).toBe('10%');
    expect(overlay.style.top).toBe('10%');
    expect(overlay.style.width).toBe('20%');

    video.currentTime = 2.1;
    fireEvent.seeking(video);
    expect(within(view.getByRole('table')).getByLabelText('12الف34567')).toBeTruthy();
    video.currentTime = 1.2;
    fireEvent.seeked(video);
    expect(within(view.getByRole('table')).getByLabelText('22ب41763')).toBeTruthy();
    video.currentTime = 1.6;
    fireEvent.timeUpdate(video);
    expect(view.queryByRole('table')).toBeNull();
    expect(view.container.querySelector('[aria-hidden="true"] > div')).toBeNull();
  });

  it('uses decoded video frame timestamps and cancels frame updates on unmount', () => {
    const view = render(<VideoResults src="/media/video.mp4" detections={detections} sampleFps={2} status="done" />);
    const video = view.getByLabelText('Uploaded video') as HTMLVideoElement;
    let onFrame: VideoFrameRequestCallback | undefined;
    video.requestVideoFrameCallback = jest.fn((callback) => { onFrame = callback; return 1; });
    video.cancelVideoFrameCallback = jest.fn();
    Object.defineProperty(video, 'paused', { value: false });
    fireEvent.play(video);
    act(() => onFrame?.(0, { mediaTime: 1.1 } as VideoFrameCallbackMetadata));
    expect(view.getByRole('table')).toBeTruthy();
    act(() => onFrame?.(0, { mediaTime: 1.8 } as VideoFrameCallbackMetadata));
    expect(view.queryByRole('table')).toBeNull();
    view.unmount();
    expect(video.cancelVideoFrameCallback).toHaveBeenCalled();
  });

  it('adds results at the current playback position when analysis completes', () => {
    const view = render(<VideoResults src="/media/video.mp4" detections={[]} status="processing" />);
    const video = view.getByLabelText('Uploaded video') as HTMLVideoElement;
    video.currentTime = 1.1;
    fireEvent.timeUpdate(video);
    expect(view.getByText(/when analysis finishes/)).toBeTruthy();
    view.rerender(<VideoResults src="/media/video.mp4" detections={detections} sampleFps={2} status="done" />);
    expect(view.getByRole('table')).toBeTruthy();
  });

  it('shows a useful playback error and clears overlays', () => {
    const view = render(<VideoResults src="/media/video.avi" detections={detections} status="done" />);
    const video = view.getByLabelText('Uploaded video') as HTMLVideoElement;
    video.currentTime = 1.1;
    fireEvent.timeUpdate(video);
    fireEvent.error(video);
    expect(view.getByRole('alert').textContent).toContain('could not be played');
    expect(view.getByRole('link', { name: /download the original/ }).getAttribute('href')).toBe('/media/video.avi');
    expect(view.queryByRole('table')).toBeNull();
  });
});
