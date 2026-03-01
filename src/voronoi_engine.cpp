/**
 * voronoi_engine.cpp
 *
 * C++17 traversability-graph engine with pybind11 Python bindings.
 *
 * Two algorithms are supported, selected at run-time via the `algo` parameter:
 *
 * 1. "voronoi" (default)
 *    Pipeline
 *    a. Accept a 2-D uint8 occupancy grid (0 = free, >0 = obstacle).
 *    b. Compute the exact Euclidean Distance Transform (EDT) using multi-source
 *       Dijkstra; each cell records its nearest obstacle source (Voronoi diagram).
 *    c. Extract the Voronoi skeleton: free cells ≥ robot_radius from obstacles
 *       that border a cell with a different nearest-obstacle source.
 *    d. Thin the skeleton to a 1-pixel medial axis (Zhang–Suen).
 *    e. Build a navigation graph: junctions/endpoints become vertices,
 *       degree-2 chains become edges.
 *    f. Merge vertices closer than node_solution_px (Union-Find).
 *
 * 2. "uniform"
 *    a. Compute EDT (same as above, needed for clearance checks).
 *    b. Place vertices on a regular grid spaced grid_step_px apart;
 *       discard any sample whose EDT < robot_radius.
 *    c. For every pair of vertices within visibility_range_px:
 *       perform a Bresenham line-of-sight check that verifies every cell
 *       along the segment has EDT ≥ robot_radius (robot body fits through).
 *    d. Merge vertices closer than node_solution_px (Union-Find).
 *
 * External deps: Eigen (header-only), pybind11 (header-only), C++ stdlib.
 * Compiler: g++/clang, -std=c++17.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases
// ─────────────────────────────────────────────────────────────────────────────
using OccGrid = Eigen::Matrix<uint8_t,  Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ByteGrid = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using IntGrid  = Eigen::Matrix<int,     Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FloatGrid = Eigen::Matrix<float,  Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ─────────────────────────────────────────────────────────────────────────────
// 8-connected neighbourhood offsets
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int DR8[8] = {-1, -1, -1,  0,  0,  1,  1,  1};
static constexpr int DC8[8] = {-1,  0,  1, -1,  1, -1,  0,  1};

// ─────────────────────────────────────────────────────────────────────────────
// Step 1 – Exact EDT + Voronoi source map
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Multi-source Dijkstra that propagates from every obstacle pixel.
 * Each cell records:
 *   dist_map  – Euclidean distance to its nearest obstacle.
 *   src_r/c   – row/column of that nearest obstacle (the Voronoi source).
 */
static void computeEDT(
    const OccGrid&  occ,
    FloatGrid&      dist_map,
    IntGrid&        src_r_map,
    IntGrid&        src_c_map)
{
    const int rows = static_cast<int>(occ.rows());
    const int cols = static_cast<int>(occ.cols());

    dist_map  = FloatGrid::Constant(rows, cols, std::numeric_limits<float>::infinity());
    src_r_map = IntGrid::Constant(rows, cols, -1);
    src_c_map = IntGrid::Constant(rows, cols, -1);

    // Priority queue: (distance, row, col, source_row, source_col)
    using Entry = std::tuple<float, int, int, int, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;

    // Seed all obstacle cells (distance 0, source = self)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (occ(r, c) > 0) {
                dist_map(r, c)  = 0.0f;
                src_r_map(r, c) = r;
                src_c_map(r, c) = c;
                pq.emplace(0.0f, r, c, r, c);
            }
        }
    }

    while (!pq.empty()) {
        auto [d, r, c, sr, sc] = pq.top();
        pq.pop();

        // Stale entry
        if (d > dist_map(r, c) + 1e-6f) continue;

        for (int k = 0; k < 8; ++k) {
            const int nr = r + DR8[k];
            const int nc = c + DC8[k];
            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
            if (occ(nr, nc) > 0) continue; // don't overwrite obstacle cells with "free" distances

            const float nd = std::sqrt(
                static_cast<float>((nr - sr) * (nr - sr) + (nc - sc) * (nc - sc)));

            if (nd < dist_map(nr, nc) - 1e-6f) {
                dist_map(nr, nc)  = nd;
                src_r_map(nr, nc) = sr;
                src_c_map(nr, nc) = sc;
                pq.emplace(nd, nr, nc, sr, sc);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2 – Voronoi skeleton extraction
// ─────────────────────────────────────────────────────────────────────────────
/**
 * A free cell is on the skeleton if:
 *  • EDT >= robot_radius  (sufficient clearance for the robot), AND
 *  • At least one 8-neighbour (also free & EDT >= robot_radius) has a
 *    *different* nearest obstacle source.
 */
static ByteGrid extractSkeleton(
    const OccGrid&   occ,
    const FloatGrid& dist_map,
    const IntGrid&   src_r_map,
    const IntGrid&   src_c_map,
    float            robot_radius)
{
    const int rows = static_cast<int>(occ.rows());
    const int cols = static_cast<int>(occ.cols());
    ByteGrid skeleton = ByteGrid::Zero(rows, cols);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (occ(r, c) > 0)               continue;
            if (dist_map(r, c) < robot_radius) continue;

            const int my_sr = src_r_map(r, c);
            const int my_sc = src_c_map(r, c);

            for (int k = 0; k < 8; ++k) {
                const int nr = r + DR8[k];
                const int nc = c + DC8[k];
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                if (occ(nr, nc) > 0)               continue;
                if (dist_map(nr, nc) < robot_radius) continue;

                if (src_r_map(nr, nc) != my_sr || src_c_map(nr, nc) != my_sc) {
                    skeleton(r, c) = 1;
                    break;
                }
            }
        }
    }
    return skeleton;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3 – Zhang–Suen thinning
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Classic Zhang–Suen (1984) parallel skeletonisation.
 * Reduces a binary image to a ~1-pixel-wide medial axis.
 */
static void zhangSuenThinning(ByteGrid& img)
{
    const int rows = static_cast<int>(img.rows());
    const int cols = static_cast<int>(img.cols());

    // Named accessor that returns 0 for out-of-bounds
    auto g = [&](int r, int c) -> int {
        if (r < 0 || r >= rows || c < 0 || c >= cols) return 0;
        return img(r, c) ? 1 : 0;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<std::pair<int,int>> to_delete;
            for (int r = 1; r < rows - 1; ++r) {
                for (int c = 1; c < cols - 1; ++c) {
                    if (!img(r, c)) continue;

                    // Clockwise neighbourhood: P2..P9 (using standard ZS labelling)
                    const int p2 = g(r-1, c),   p3 = g(r-1, c+1);
                    const int p4 = g(r,   c+1),  p5 = g(r+1, c+1);
                    const int p6 = g(r+1, c),    p7 = g(r+1, c-1);
                    const int p8 = g(r,   c-1),  p9 = g(r-1, c-1);

                    // B(P1): number of ON neighbours
                    const int B = p2+p3+p4+p5+p6+p7+p8+p9;
                    if (B < 2 || B > 6) continue;

                    // A(P1): number of 0→1 transitions in cyclic order
                    const int A = (p2==0&&p3==1) + (p3==0&&p4==1) + (p4==0&&p5==1)
                                + (p5==0&&p6==1) + (p6==0&&p7==1) + (p7==0&&p8==1)
                                + (p8==0&&p9==1) + (p9==0&&p2==1);
                    if (A != 1) continue;

                    if (pass == 0) {
                        if (p2 * p4 * p6 != 0) continue;
                        if (p4 * p6 * p8 != 0) continue;
                    } else {
                        if (p2 * p4 * p8 != 0) continue;
                        if (p2 * p6 * p8 != 0) continue;
                    }
                    to_delete.emplace_back(r, c);
                }
            }
            for (auto [r, c] : to_delete) {
                img(r, c) = 0;
                changed   = true;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4 – Graph extraction
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Returns the 8-connected skeleton degree of cell (r,c).
 */
static int skelDegree(const ByteGrid& skel, int r, int c)
{
    const int rows = static_cast<int>(skel.rows());
    const int cols = static_cast<int>(skel.cols());
    int cnt = 0;
    for (int k = 0; k < 8; ++k) {
        const int nr = r + DR8[k], nc = c + DC8[k];
        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && skel(nr, nc))
            ++cnt;
    }
    return cnt;
}

struct NavGraph {
    std::vector<std::pair<float,float>> vertices; // (col=x, row=y) in pixel coords
    std::vector<std::pair<int,int>>     edges;    // pairs of vertex indices
};

/**
 * Algorithm
 * ---------
 *  a) Label every skeleton cell whose degree ≠ 2 as a *graph node*
 *     (degree > 2 → junction; degree == 1 → endpoint; degree == 0 → isolated).
 *  b) Handle the degenerate all-degree-2 (pure-loop) case by injecting a
 *     synthetic node at the highest-EDT point.
 *  c) Remove node cells from a copy of the skeleton, then flood-fill the
 *     remaining edge cells into connected components.
 *  d) Each component touches exactly two node cells → one edge.
 *  e) Also directly link *adjacent* node cells (no intervening edge cells).
 */
static NavGraph buildGraph(const ByteGrid& skel, const FloatGrid& dist_map)
{
    const int rows = static_cast<int>(skel.rows());
    const int cols = static_cast<int>(skel.cols());

    NavGraph graph;
    IntGrid  node_id = IntGrid::Constant(rows, cols, -1);

    // ── (a) Identify node cells ──────────────────────────────────────────────
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!skel(r, c)) continue;
            if (skelDegree(skel, r, c) != 2) {
                node_id(r, c) = static_cast<int>(graph.vertices.size());
                graph.vertices.emplace_back(static_cast<float>(c),
                                            static_cast<float>(r));
            }
        }
    }

    // ── (b) Pure-loop fallback ───────────────────────────────────────────────
    if (graph.vertices.empty()) {
        float best_d = -1.0f;
        int   best_r = -1, best_c = -1;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                if (skel(r, c) && dist_map(r, c) > best_d) {
                    best_d = dist_map(r, c);
                    best_r = r;
                    best_c = c;
                }
        if (best_r >= 0) {
            node_id(best_r, best_c) = 0;
            graph.vertices.emplace_back(static_cast<float>(best_c),
                                        static_cast<float>(best_r));
        }
        return graph; // loop with one synthetic vertex, no edges needed
    }

    // ── (c) Connected components of edge-skeleton cells ──────────────────────
    ByteGrid edge_skel = skel;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            if (node_id(r, c) >= 0)
                edge_skel(r, c) = 0;

    IntGrid comp = IntGrid::Constant(rows, cols, -1);
    int     n_comp = 0;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!edge_skel(r, c) || comp(r, c) >= 0) continue;
            const int cid = n_comp++;
            std::queue<std::pair<int,int>> q;
            q.emplace(r, c);
            comp(r, c) = cid;
            while (!q.empty()) {
                auto [cr, cc] = q.front(); q.pop();
                for (int k = 0; k < 8; ++k) {
                    const int nr = cr + DR8[k], nc = cc + DC8[k];
                    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                    if (!edge_skel(nr, nc) || comp(nr, nc) >= 0)       continue;
                    comp(nr, nc) = cid;
                    q.emplace(nr, nc);
                }
            }
        }
    }

    // ── (d) Map components → pairs of node cells ─────────────────────────────
    // comp_ends[cid] = {first_node_id, second_node_id}
    std::vector<std::pair<int,int>> comp_ends(n_comp, {-1, -1});

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!edge_skel(r, c)) continue;
            const int cid = comp(r, c);
            for (int k = 0; k < 8; ++k) {
                const int nr = r + DR8[k], nc = c + DC8[k];
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                const int nid = node_id(nr, nc);
                if (nid < 0) continue;
                if      (comp_ends[cid].first  < 0)                    comp_ends[cid].first  = nid;
                else if (comp_ends[cid].second < 0 &&
                         comp_ends[cid].first  != nid)                  comp_ends[cid].second = nid;
            }
        }
    }

    // Deduplicated edge set
    std::unordered_set<long long> added;
    auto edge_key = [](int a, int b) -> long long {
        if (a > b) std::swap(a, b);
        return static_cast<long long>(a) * 1'000'000LL + b;
    };

    for (auto& [j1, j2] : comp_ends) {
        if (j1 >= 0 && j2 >= 0) {
            const auto k = edge_key(j1, j2);
            if (!added.count(k)) {
                added.insert(k);
                graph.edges.emplace_back(j1, j2);
            }
        }
    }

    // ── (e) Directly adjacent node cells ─────────────────────────────────────
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (node_id(r, c) < 0) continue;
            const int nid_a = node_id(r, c);
            for (int k = 0; k < 8; ++k) {
                const int nr = r + DR8[k], nc = c + DC8[k];
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                const int nid_b = node_id(nr, nc);
                if (nid_b < 0 || nid_b == nid_a) continue;
                const auto key = edge_key(nid_a, nid_b);
                if (!added.count(key)) {
                    added.insert(key);
                    graph.edges.emplace_back(nid_a, nid_b);
                }
            }
        }
    }

    return graph;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5 – Node merging
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Merge any two graph vertices that are within `merge_dist_px` pixels of each
 * other into a single representative vertex (the one with the highest EDT
 * clearance is kept).  Edges are remapped; self-loops and duplicates are
 * removed.
 *
 * Uses a simple disjoint-set (Union-Find) structure – no external libraries.
 */
static NavGraph mergeCloseNodes(
    const NavGraph&  g,
    float            merge_dist_px,
    const FloatGrid& dist_map)
{
    const int n = static_cast<int>(g.vertices.size());
    if (n == 0 || merge_dist_px <= 0.0f) return g;

    // ── Union-Find ────────────────────────────────────────────────────────────
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);

    std::function<int(int)> find = [&](int x) -> int {
        return parent[x] == x ? x : (parent[x] = find(parent[x]));
    };
    auto unite = [&](int a, int b) {
        a = find(a); b = find(b);
        if (a != b) parent[a] = b;
    };

    const float dist_sq = merge_dist_px * merge_dist_px;
    for (int i = 0; i < n; ++i) {
        const float xi = g.vertices[i].first;
        const float yi = g.vertices[i].second;
        for (int j = i + 1; j < n; ++j) {
            const float dx = xi - g.vertices[j].first;
            const float dy = yi - g.vertices[j].second;
            if (dx*dx + dy*dy <= dist_sq)
                unite(i, j);
        }
    }

    // ── Pick best representative per group (highest EDT clearance) ────────────
    std::unordered_map<int, int>   group_best;     // root → best vertex index
    std::unordered_map<int, float> group_best_edt; // root → best EDT value

    for (int i = 0; i < n; ++i) {
        const int   root = find(i);
        const int   r    = static_cast<int>(std::round(g.vertices[i].second));
        const int   c    = static_cast<int>(std::round(g.vertices[i].first));
        const float edt  =
            (r >= 0 && r < static_cast<int>(dist_map.rows()) &&
             c >= 0 && c < static_cast<int>(dist_map.cols()))
            ? dist_map(r, c) : 0.0f;

        if (!group_best.count(root) || edt > group_best_edt[root]) {
            group_best[root]     = i;
            group_best_edt[root] = edt;
        }
    }

    // ── Build merged vertex list + old→new index map ──────────────────────────
    NavGraph out;
    std::unordered_map<int, int> old_to_new; // old index → new index

    for (auto& [root, best_idx] : group_best) {
        const int new_id = static_cast<int>(out.vertices.size());
        out.vertices.push_back(g.vertices[best_idx]);
        // Tag every member of this group with the new id
        for (int i = 0; i < n; ++i)
            if (find(i) == root)
                old_to_new[i] = new_id;
    }

    // ── Remap edges; discard self-loops and duplicates ────────────────────────
    std::unordered_set<long long> added;
    auto edge_key_m = [](int a, int b) -> long long {
        if (a > b) std::swap(a, b);
        return static_cast<long long>(a) * 1'000'000LL + b;
    };

    for (auto& [i, j] : g.edges) {
        const int ni = old_to_new.at(i);
        const int nj = old_to_new.at(j);
        if (ni == nj) continue;
        const long long k = edge_key_m(ni, nj);
        if (!added.count(k)) {
            added.insert(k);
            out.edges.emplace_back(ni, nj);
        }
    }

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Uniform grid sampling + line-of-sight graph
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Bresenham line-of-sight check.
 * Returns true iff every pixel along the line from (r0,c0) to (r1,c1)
 * is obstacle-free AND has EDT >= robot_radius (robot body fits through).
 */
static bool lineOfSight(
    const OccGrid&   occ,
    const FloatGrid& dist_map,
    int r0, int c0, int r1, int c1,
    float robot_radius)
{
    const int rows = static_cast<int>(occ.rows());
    const int cols = static_cast<int>(occ.cols());

    int dr = std::abs(r1 - r0);
    int dc = std::abs(c1 - c0);
    int sr = (r0 < r1) ? 1 : -1;
    int sc = (c0 < c1) ? 1 : -1;
    int err = dr - dc;

    int r = r0, c = c0;
    while (true) {
        if (r < 0 || r >= rows || c < 0 || c >= cols)  return false;
        if (occ(r, c) > 0)                              return false;
        if (dist_map(r, c) < robot_radius)              return false;
        if (r == r1 && c == c1) break;

        int e2 = 2 * err;
        if (e2 > -dc) { err -= dc; r += sr; }
        if (e2 <  dr) { err += dr; c += sc; }
    }
    return true;
}

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
    float            L,
    float            visibility_range_px)
{
    const int rows = static_cast<int>(occ.rows());
    const int cols = static_cast<int>(occ.cols());
    const int step = std::max(1, static_cast<int>(std::round(grid_step_px)));
    const float vis_sq = visibility_range_px * visibility_range_px;

    NavGraph graph;

    // ── Sample vertices ─────────────────────────────────────────────────────────
    for (int r = 0; r < rows; r += step) {
        for (int c = 0; c < cols; c += step) {
            if (occ(r, c) > 0)               continue;  // obstacle
            if (dist_map(r, c) < robot_radius) continue;  // too close to wall
            graph.vertices.emplace_back(static_cast<float>(c),
                                        static_cast<float>(r));
        }
    }

    // ── Build edges via pairwise line-of-sight ───────────────────────────────
    const int nv = static_cast<int>(graph.vertices.size());
    for (int i = 0; i < nv; ++i) {
        const float xi = graph.vertices[i].first;   // col
        const float yi = graph.vertices[i].second;  // row
        for (int j = i + 1; j < nv; ++j) {
            const float dx = xi - graph.vertices[j].first;
            const float dy = yi - graph.vertices[j].second;
            if (dx*dx + dy*dy > vis_sq) continue;   // outside visibility range

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

// ─────────────────────────────────────────────────────────────────────────────
// Public pybind11 entry point
// ─────────────────────────────────────────────────────────────────────────────
/**
 * compute_graph(occ_array, robot_radius, node_solution_px, algo,
 *               grid_step_px, visibility_range_px) -> dict
 */
static py::dict compute_graph(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> occ_array,
    float       robot_radius,
    float       node_solution_px,
    std::string algo,
    float       grid_step_px,
    float       visibility_range_px)
{
    // Normalise algo string
    std::transform(algo.begin(), algo.end(), algo.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    if (algo != "voronoi" && algo != "uniform")
        throw std::runtime_error(
            "Unknown algo '" + algo + "'. Must be 'voronoi' or 'uniform'.");

    const auto buf = occ_array.request();
    if (buf.ndim != 2)
        throw std::runtime_error("occ_array must be a 2-D numpy array");

    const int rows = static_cast<int>(buf.shape[0]);
    const int cols = static_cast<int>(buf.shape[1]);
    const auto* ptr = static_cast<const uint8_t*>(buf.ptr);

    // Zero-copy Eigen view
    const OccGrid occ = Eigen::Map<const OccGrid>(ptr, rows, cols);

    // EDT is needed by both algorithms
    FloatGrid dist_map;
    IntGrid   src_r_map, src_c_map;
    computeEDT(occ, dist_map, src_r_map, src_c_map);

    // ── Algorithm branch ──────────────────────────────────────────────────────
    ByteGrid skeleton = ByteGrid::Zero(rows, cols);  // empty for uniform
    NavGraph  nav;

    if (algo == "voronoi") {
        skeleton = extractSkeleton(occ, dist_map, src_r_map, src_c_map, robot_radius);
        zhangSuenThinning(skeleton);
        nav = buildGraph(skeleton, dist_map);
    } else {
        nav = buildUniformGraph(occ, dist_map, robot_radius,
                                grid_step_px, visibility_range_px);
    }

    nav = mergeCloseNodes(nav, node_solution_px, dist_map);

    // ── Package results ───────────────────────────────────────────────────────
    auto skel_arr = py::array_t<uint8_t>({rows, cols});
    Eigen::Map<ByteGrid>(
        static_cast<uint8_t*>(skel_arr.request().ptr), rows, cols) = skeleton;

    auto dist_arr = py::array_t<float>({rows, cols});
    Eigen::Map<FloatGrid>(
        static_cast<float*>(dist_arr.request().ptr), rows, cols) = dist_map;

    py::dict result;
    result["vertices"] = nav.vertices;
    result["edges"]    = nav.edges;
    result["skeleton"] = skel_arr;
    result["dist_map"] = dist_arr;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(voronoi_engine, m)
{
    m.doc() = R"doc(
voronoi_engine – C++17 traversability-graph planner
====================================================
Builds a navigation graph from a 2-D occupancy grid using one of two
algorithms, selected at run-time:

  "voronoi"  – Voronoi-corridor skeleton (medial-axis + junction graph).
  "uniform"  – Uniform grid sampling with pairwise line-of-sight edges.

Functions
---------
compute_graph(occ_array, robot_radius, node_solution_px,
              algo, grid_step_px, visibility_range_px) -> dict
)doc";

    m.def(
        "compute_graph",
        &compute_graph,
        py::arg("occ_array"),
        py::arg("robot_radius"),
        py::arg("node_solution_px"),
        py::arg("algo")               = "voronoi",
        py::arg("grid_step_px")       = 20.0f,
        py::arg("visibility_range_px") = 200.0f,
        R"doc(
Compute a traversability navigation graph from a binary occupancy grid.

Parameters
----------
occ_array           : numpy.ndarray[uint8], shape (H, W)
                      0 = free, non-zero = obstacle.
robot_radius        : float
                      Minimum clearance in pixels.  Vertices and path cells
                      closer to any obstacle than this value are discarded.
node_solution_px    : float
                      Merge distance in pixels.  Vertices closer than this
                      are collapsed into one (highest-EDT representative).
                      Pass 0.0 to disable.
algo                : str, default "voronoi"
                      "voronoi" – Voronoi skeleton + junction graph.
                      "uniform" – grid sampling + line-of-sight edges.
grid_step_px        : float, default 20.0
                      [uniform] pixel spacing between sampled vertices.
visibility_range_px : float, default 200.0
                      [uniform] max pixel distance within which a LOS check
                      is attempted between two vertices.

Returns
-------
dict
    "vertices"  : list of (float x, float y) tuples – pixel coords (col, row).
    "edges"     : list of (int i, int j) tuples – vertex index pairs.
    "skeleton"  : numpy.ndarray[uint8] (H, W) – Voronoi skeleton (zeros for uniform).
    "dist_map"  : numpy.ndarray[float32] (H, W) – Euclidean distance transform.
)doc");
}
