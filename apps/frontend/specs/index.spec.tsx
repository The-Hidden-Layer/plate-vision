import { render } from '@testing-library/react';
import React from 'react';

import Page from '../src/app/page';

jest.mock('next/navigation', () => ({ useRouter: () => ({ push: jest.fn() }) }));

describe('Upload page', () => {
  it('renders the dropzone', () => {
    const { getByText } = render(<Page />);
    expect(getByText(/Drag an image or video here/i)).toBeTruthy();
  });
});
