import { UploadForm } from '../components/UploadForm';

export default function UploadPage() {
  return (
    <div className="mx-auto max-w-xl">
      <h1 className="text-2xl font-semibold tracking-tight">Detect license plates</h1>
      <p className="mt-2 mb-8 text-sm text-neutral-600 dark:text-neutral-400">
        Upload an image or a video. It is queued for processing and you can watch the
        result appear.
      </p>
      <UploadForm />
    </div>
  );
}
