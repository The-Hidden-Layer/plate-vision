import type { Detection } from '../lib/types';
import { formatTimestamp } from '../lib/format';

function Confidence({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2">
      <div
        className="h-1.5 w-16 overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800"
        role="img"
        aria-label={`${pct} percent confidence`}
      >
        <div className="h-full rounded-full bg-emerald-500" style={{ width: `${pct}%` }} />
      </div>
      <span className="tabular-nums text-neutral-600 dark:text-neutral-400">{pct}%</span>
    </div>
  );
}

export function DetectionTable({ detections }: { detections: Detection[] }) {
  if (detections.length === 0) {
    return (
      <p className="rounded-lg border border-dashed border-neutral-300 p-6 text-center text-sm text-neutral-500 dark:border-neutral-700">
        No plates were found in this media.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-neutral-200 dark:border-neutral-800">
      <table className="w-full text-left text-sm">
        <thead className="bg-neutral-50 text-xs uppercase tracking-wide text-neutral-500 dark:bg-neutral-900 dark:text-neutral-400">
          <tr>
            <th scope="col" className="px-4 py-2.5 font-medium">Plate</th>
            <th scope="col" className="px-4 py-2.5 font-medium">Confidence</th>
            <th scope="col" className="px-4 py-2.5 font-medium">Frame</th>
            <th scope="col" className="px-4 py-2.5 font-medium">Time</th>
            <th scope="col" className="px-4 py-2.5 font-medium">Crop</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-neutral-200 dark:divide-neutral-800">
          {detections.map((d) => {
            const time = formatTimestamp(d.timestamp_ms);
            return (
              <tr key={d.id} className="hover:bg-neutral-50 dark:hover:bg-neutral-900/50">
                <td className="px-4 py-2.5 font-mono font-medium">
                  {d.plate_text || <span className="text-neutral-400">unreadable</span>}
                </td>
                <td className="px-4 py-2.5"><Confidence value={d.confidence} /></td>
                <td className="px-4 py-2.5 tabular-nums text-neutral-600 dark:text-neutral-400">
                  {d.frame_index}
                </td>
                <td className="px-4 py-2.5 tabular-nums text-neutral-600 dark:text-neutral-400">
                  {time ?? <span className="text-neutral-400">—</span>}
                </td>
                <td className="px-4 py-2">
                  {d.crop_url ? (
                    <a href={d.crop_url} target="_blank" rel="noreferrer">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={d.crop_url}
                        alt={`Plate crop ${d.plate_text || 'unreadable'}`}
                        className="h-8 rounded border border-neutral-200 bg-neutral-100 object-contain dark:border-neutral-700 dark:bg-neutral-800"
                      />
                    </a>
                  ) : (
                    <span className="text-neutral-400">—</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
