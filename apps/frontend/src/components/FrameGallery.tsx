export function FrameGallery({ urls }: { urls: string[] }) {
  if (urls.length === 0) return null;

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {urls.map((url) => (
        <a
          key={url}
          href={url}
          target="_blank"
          rel="noreferrer"
          className="group overflow-hidden rounded-lg border border-neutral-200 transition hover:border-neutral-400 dark:border-neutral-800 dark:hover:border-neutral-600"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={url}
            alt="Annotated frame with detected plates outlined"
            className="aspect-video w-full bg-neutral-100 object-contain dark:bg-neutral-900"
            loading="lazy"
          />
        </a>
      ))}
    </div>
  );
}
