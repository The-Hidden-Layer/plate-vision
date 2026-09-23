describe('upload proxy limits', () => {
  const originalLimit = process.env.MAX_UPLOAD_MB;
  const originalPublicLimit = process.env.NEXT_PUBLIC_MAX_UPLOAD_MB;

  afterEach(() => {
    if (originalLimit === undefined) delete process.env.MAX_UPLOAD_MB;
    else process.env.MAX_UPLOAD_MB = originalLimit;
    if (originalPublicLimit === undefined) delete process.env.NEXT_PUBLIC_MAX_UPLOAD_MB;
    else process.env.NEXT_PUBLIC_MAX_UPLOAD_MB = originalPublicLimit;
    jest.resetModules();
  });

  it('forwards the advertised maximum file size plus multipart overhead', () => {
    delete process.env.MAX_UPLOAD_MB;
    process.env.NEXT_PUBLIC_MAX_UPLOAD_MB = '200';
    jest.resetModules();
    const config = require('../next.config');
    expect(config.experimental.proxyClientMaxBodySize).toBeGreaterThan(200 * 1024 * 1024);
    expect(config.experimental.proxyTimeout).toBeGreaterThanOrEqual(120_000);
  });

  it('tracks a configured backend limit instead of silently truncating to 10 MB', () => {
    process.env.MAX_UPLOAD_MB = '64';
    process.env.NEXT_PUBLIC_MAX_UPLOAD_MB = '64';
    jest.resetModules();
    const config = require('../next.config');
    expect(config.experimental.proxyClientMaxBodySize).toBe(65 * 1024 * 1024);
  });
});
