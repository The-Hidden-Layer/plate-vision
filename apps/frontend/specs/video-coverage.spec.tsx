import { render } from '@testing-library/react';
import React from 'react';

import { VideoCoverage } from '../src/components/VideoCoverage';

describe('Video coverage', () => {
  it('shows actual analyzed frames and the configured sampling rate', () => {
    const { getByText } = render(
      <VideoCoverage analysis={{ sampled_frame_count: 481, sample_fps: 2 }} />,
    );
    expect(getByText('481 frames analyzed · 2 fps target')).toBeTruthy();
  });

  it('does not imply full coverage for older eight-frame jobs', () => {
    const { getByText } = render(<VideoCoverage analysis={null} />);
    expect(getByText(/coverage was not recorded/)).toBeTruthy();
  });
});
