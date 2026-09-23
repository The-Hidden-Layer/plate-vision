import type { Detection } from './types';

export function recognizedDetections(detections: Detection[]): Detection[] {
  return detections.filter((detection) => detection.plate_text.trim().length > 0);
}
