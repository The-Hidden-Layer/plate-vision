/**
 * Public type surface for the plate-vision API.
 *
 * `schema.ts` is generated — do not edit it. Refresh both it and the OpenAPI
 * document it comes from with:
 *
 *     pnpm nx run api-types:generate
 *
 * This file is types only, so importing it emits no runtime code.
 */

import type { components } from './schema';

export type Job = components['schemas']['Job'];
export type Detection = components['schemas']['Detection'];
export type PaginatedJobList = components['schemas']['PaginatedJobList'];
export type JobStatus = components['schemas']['StatusEnum'];
export type MediaType = components['schemas']['MediaTypeEnum'];

export type { components, paths } from './schema';
