/**
 * uniform_grid.hpp
 *
 * Uniform-grid traversability graph: places vertices on a regular grid,
 * keeps only cells with sufficient EDT clearance, and connects pairs within
 * `visibility_range_px` that pass a full Bresenham line-of-sight check.
 *
 * Depends on brushfire_voronoi.hpp for the shared type aliases (OccGrid,
 * FloatGrid, NavGraph) and the mergeCloseNodes helper.
 */

#pragma once

#include "brushfire_voronoi.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Bresenham line-of-sight check
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Returns true iff every pixel along the line from (r0,c0) to (r1,c1) is
 * obstacle-free AND has EDT >= robot_radius (robot body fits through).
 */
static bool lineOfSight(
    const OccGrid&   occ,
    const FloatGrid& dist_map,
    int r0, int c0, int r1, int c1,
    float robot_radius)
{
    const int rows = static_cast<int>(occ.rows());
    const int cols = static_cast<int>(occ.cols());

    const int dr = std::abs(r1 - r0);
    const int dc = std::abs(c1 - c0);
    const int sr = (r0 < r1) ? 1 : -1;
    const int sc = (c0 < c1) ? 1 : -1;
    int err = dr - dc;

    int r = r0, c = c0;
    while (true) {
        if (r < 0 || r >= rows || c < 0 || c >= cols)  return false;
        if (occ(r, c) > 0)                              return false;
        if (dist_map(r, c) < robot_radius)              return false;
        if (r == r1 && c == c1) break;

        const int e2 = 2 * err;
        if (e2 > -dc) { err -= dc; r += sr; }
        if (e2 <  dr) { err += dr; c += sc; }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform-grid graph builder
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Build a traversability graph by:
 *  1. Placing vertices on a regular pixel grid (step = grid_step_px),
 *     keeping only cells that are free and have EDT >= robot_radius.
 *  2. Connecting every pair of vertices within `visibility_range_px` that
 *     passes the full line-of-sight + clearance check.
 */
static NavGraph buildUniformGraph(
    const OccGrid&   occ,
    const FloatGrid& dist_map,
    float            robot_radius,
    float            grid_step_px,
    float            visibility_range_px)
{
    const int rows   = static_cast<int>(occ.rows());
    const int cols   = static_cast<int>(occ.cols());
    const int step   = std::max(1, static_cast<int>(std::round(grid_step_px)));
    const float vis_sq = visibility_range_px * visibility_range_px;

    NavGraph graph;

    // ── Sample vertices ───────────────────────────────────────────────────────
    for (int r = 0; r < rows; r += step) {
        for (int c = 0; c < cols; c += step) {
            if (occ(r, c) > 0)                continue; // obstacle
            if (dist_map(r, c) < robot_radius) continue; // too close to wall
            graph.vertices.emplace_back(static_cast<float>(c),
                                        static_cast<float>(r));
        }
    }

    // ── Build edges via pairwise line-of-sight ────────────────────────────────
    const int nv = static_cast<int>(graph.vertices.size());
    for (int i = 0; i < nv; ++i) {
        const float xi = graph.vertices[i].first;  // col
        const float yi = graph.vertices[i].second; // row
        for (int j = i + 1; j < nv; ++j) {
            const float dx = xi - graph.vertices[j].first;
            const float dy = yi - graph.vertices[j].second;
            if (dx*dx + dy*dy > vis_sq) continue; // outside visibility range

            if (lineOfSight(occ, dist_map,
                            static_cast<int>(std::round(yi)),
                            static_cast<int>(std::round(xi)),
                            static_cast<int>(std::round(graph.vertices[j].second)),
                            static_cast<int>(std::round(graph.vertices[j].first)),
                            robot_radius))
            {
                graph.edges.emplace_back(i, j);
            }
        }
    }
    return graph;
}
