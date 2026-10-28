/*
 * Adaptive nprobe — Dynamically adjust IVF probe count
 * 
 * Based on coarse quantizer distance distribution: when the nearest centroid
 * is much closer than the nprobe-th centroid (large distance gap), reduce nprobe
 * without significant recall loss.
 *
 * Reference: Faiss IVF indexing — nlist clusters organized by k-means centroids.
 * Adjacent cluster IDs tend to have nearby centroids; a clear gap in distances
 * means the query is well-localized and fewer clusters need searching.
 */
#include <stdint.h>
#include <stddef.h>

/* Default: if dist[0]/dist[k] < threshold, shrink nprobe to k */
#define ADAPTIVE_DEFAULT_THRESHOLD 3.0f
#define ADAPTIVE_DEFAULT_MIN_NPROBE 8

/**
 * adaptive_nprobe — Compute optimal nprobe from centroid distance distribution
 *
 * @dists:        Array of nprobe_max centroid distances (sorted ascending)
 * @nprobe_max:   Original nprobe (e.g., 32)
 * @threshold:    Distance ratio threshold (default 3.0). Lower = more conservative.
 * @min_nprobe:   Floor value for nprobe (default 8). Never go below this.
 *
 * Returns: adjusted nprobe between min_nprobe and nprobe_max.
 *
 * Algorithm:
 *   For k from min_nprobe to nprobe_max:
 *     if dist[0] / dist[k] < threshold:
 *       return k  (distance gap found — query is well-localized to first k clusters)
 *   return nprobe_max  (no clear gap — use full nprobe)
 */
int adaptive_nprobe(const float *dists, int nprobe_max,
                     float threshold, int min_nprobe)
{
    if (!dists || nprobe_max < 1) return min_nprobe;
    if (min_nprobe < 1) min_nprobe = 1;
    if (min_nprobe >= nprobe_max) return nprobe_max;

    float first_dist = dists[0];
    if (first_dist <= 0.0f) return nprobe_max;  /* exact match — search all */

    for (int k = min_nprobe; k < nprobe_max; k++) {
        if (first_dist / dists[k] < threshold) {
            return k;
        }
    }
    return nprobe_max;
}

/**
 * adaptive_nprobe_default — Convenience wrapper with default parameters
 */
int adaptive_nprobe_default(const float *dists, int nprobe_max)
{
    return adaptive_nprobe(dists, nprobe_max,
                           ADAPTIVE_DEFAULT_THRESHOLD,
                           ADAPTIVE_DEFAULT_MIN_NPROBE);
}
