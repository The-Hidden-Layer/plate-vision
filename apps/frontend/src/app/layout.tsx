import Link from 'next/link';
import './global.css';

export const metadata = {
  title: 'plate-vision',
  description: 'License plate detection and recognition demo',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-white text-neutral-900 antialiased dark:bg-neutral-950 dark:text-neutral-100">
        <header className="border-b border-neutral-200 dark:border-neutral-800">
          <nav className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
            <Link href="/" className="font-semibold tracking-tight">
              plate<span className="text-emerald-600 dark:text-emerald-400">vision</span>
            </Link>
            <div className="flex items-center gap-5 text-sm">
              <Link href="/" className="text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100">
                Upload
              </Link>
              <Link href="/jobs" className="text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100">
                Recent jobs
              </Link>
            </div>
          </nav>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-10">{children}</main>
      </body>
    </html>
  );
}
