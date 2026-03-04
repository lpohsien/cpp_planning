/**
 * brushfire_voronoi.hpp
 *
 * Dynamic brushfire algorithm for Voronoi diagram maintenance, based on:
 *
 *   B. Lau, C. Sprunk, W. Burgard,
 *   "Efficient Grid-Based Spatial Representations for Robot Navigation
 *    in Dynamic Environments", Robotics and Autonomous Systems, 2013.
 *   http://ais.informatik.uni-freiburg.de/publications/papers/lau13ras.pdf
 *
 * Overview
 * --------
 * Each free cell v stores:
 *   dist  – Euclidean distance to its nearest obstacle.
 *   obst  – Flat linear index (row*cols + col) of that obstacle, or INVALID.
 *
 * A cell is on the Voronoi diagram (isVoronoi) when it has at least one
 * 8-connected free neighbour whose nearest-obstacle source differs.
 *
 * The class supports:
 *   init()             – static initialisation from a full occupancy grid
 *                        (multi-source brushfire from all obstacle cells).
 *   addObstacles()     – dynamic: new obstacle cells added; re-propagate.
 *   removeObstacles()  – dynamic: obstacle cells removed; re-propagate.
 *   extractSkeleton()  – return the raw Voronoi skeleton mask.
 *   buildGraph()       – Zhang–Suen thinning + junction-graph extraction.
 *   mergeCloseNodes()  – collapse nearby vertices (Union-Find).
 */

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Shared type aliases (also used by uniform_grid.hpp and map_engine.cpp)
// ─────────────────────────────────────────────────────────────────────────────
using OccGrid   = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ByteGrid  = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using IntGrid   = Eigen::Matrix<int,     Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FloatGrid = Eigen::Matrix<float,   Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ─────────────────────────────────────────────────────────────────────────────
// 8-connected neighbourhood offsets
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int DR8[8] = {-1, -1, -1,  0,  0,  1,  1,  1};
static constexpr int DC8[8] = {-1,  0,  1, -1,  1, -1,  0,  1};

// ─────────────────────────────────────────────────────────────────────────────
// Navigation graph types
// ─────────────────────────────────────────────────────────────────────────────
struct NavGraph {
    std::vector<std::pair<float, float>> vertices; // (col=x, row=y) pixel coords
    std::vector<std::pair<int,   int  >> edges;    // vertex index pairs
};

// ─────────────────────────────────────────────────────────────────────────────
// BrushfireVoronoi – dynamic Voronoi via the Lau et al. brushfire algorithm
// ─────────────────────────────────────────────────────────────────────────────
class BrushfireVoronoi {
public:
    static constexpr int INVALID = -1;

    // ── Initialise from a full occupancy grid ─────────────────────────────────
    /**
     * Runs the static brushfire (multi-source BFS from all obstacle cells)
     * to initialise dist_ and obst_ for every cell.
     *
     * occ          – 0 = free, >0 = obstacle.
     * robot_radius – minimum clearance (pixels) required for Voronoi cells.
     */
    void init(const OccGrid& occ, float robot_radius)
    {
        rows_         = static_cast<int>(occ.rows());
        cols_         = static_cast<int>(occ.cols());
        robot_radius_ = robot_radius;
        occ_          = occ;

        dist_ = FloatGrid::Constant(rows_, cols_,
                                    std::numeric_limits<float>::infinity());
        obst_ = IntGrid::Constant(rows_, cols_, INVALID);

        // Priority queue: (distance, flat_index, obst_flat_index)
        using Entry = std::tuple<float, int, int>;
        std::priority_queue<Entry,
                            std::vector<Entry>,
                            std::greater<Entry>> pq;

        // Seed all obstacle cells: distance = 0, obst = self
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                if (occ_(r, c) > 0) {
                    const int idx = r * cols_ + c;
                    dist_(r, c) = 0.0f;
                    obst_(r, c) = idx;
                    pq.emplace(0.0f, idx, idx);
                }
            }
        }

        propagate_(pq);
    }

    // ── Dynamic update: add new obstacle cells ────────────────────────────────
    /**
     * Marks each cell in `new_obstacles` as occupied and propagates the
     * shorter distances outward (Algorithm 1 in Lau et al.).
     *
     * new_obstacles – list of (row, col) pairs to add.
     */
    void addObstacles(const std::vector<std::pair<int,int>>& new_obstacles)
    {
        using Entry = std::tuple<float, int, int>;
        std::priority_queue<Entry,
                            std::vector<Entry>,
                            std::greater<Entry>> pq;

        for (auto [r, c] : new_obstacles) {
            if (r < 0 || r >= rows_ || c < 0 || c >= cols_) continue;
            occ_(r, c)  = 255;
            const int idx = r * cols_ + c;
            dist_(r, c) = 0.0f;
            obst_(r, c) = idx;
            pq.emplace(0.0f, idx, idx);
        }

        propagate_(pq);
    }

    // ── Dynamic update: remove obstacle cells ─────────────────────────────────
    /**
     * Marks each cell in `rem_obstacles` as free, invalidates the obst/dist
     * of every cell that depended on them, then re-seeds from their surviving
     * neighbours and re-propagates (Algorithm 2 in Lau et al.).
     *
     * rem_obstacles – list of (row, col) pairs to remove.
     */
    void removeObstacles(const std::vector<std::pair<int,int>>& rem_obstacles)
    {
        // Collect all cells whose obst points to any removed obstacle
        std::unordered_set<int> removed_set;
        for (auto [r, c] : rem_obstacles) {
            if (r < 0 || r >= rows_ || c < 0 || c >= cols_) continue;
            occ_(r, c) = 0;
            removed_set.insert(r * cols_ + c);
        }

        // Find all cells that referenced a removed obstacle
        std::vector<int> affected;
        for (int r = 0; r < rows_; ++r) {
            for (int c = 0; c < cols_; ++c) {
                if (removed_set.count(obst_(r, c))) {
                    affected.push_back(r * cols_ + c);
                }
            }
        }

        // Invalidate all affected cells
        for (int idx : affected) {
            const int r = idx / cols_, c = idx % cols_;
            dist_(r, c) = std::numeric_limits<float>::infinity();
            obst_(r, c) = INVALID;
        }

        // Re-seed from neighbours of affected cells that still have valid obst
        using Entry = std::tuple<float, int, int>;
        std::priority_queue<Entry,
                            std::vector<Entry>,
                            std::greater<Entry>> pq;

        for (int idx : affected) {
            const int r = idx / cols_, c = idx % cols_;
            for (int k = 0; k < 8; ++k) {
                const int nr = r + DR8[k];
                const int nc = c + DC8[k];
                if (nr < 0 || nr >= rows_ || nc < 0 || nc >= cols_) continue;
                const int nobst = obst_(nr, nc);
                if (nobst == INVALID) continue;
                // Compute actual Euclidean distance from this cell to the
                // surviving obstacle
                const int obst_r = nobst / cols_, oc = nobst % cols_;
                const float nd = std::sqrt(
                    static_cast<float>((nr - obst_r) * (nr - obst_r) +
                                       (nc - oc)  * (nc - oc)));
                if (nd < dist_(nr, nc) + 1e-6f) {
                    pq.emplace(dist_(nr, nc), nr * cols_ + nc, nobst);
                }
            }
        }

        propagate_(pq);
    }

    // ── Accessors ─────────────────────────────────────────────────────────────
    const FloatGrid& distMap()  const { return dist_; }
    const IntGrid&   obstMap()  const { return obst_; }
    int              rows()     const { return rows_; }
    int              cols()     const { return cols_; }

    // ── isVoronoi (Algorithm 2, Lau et al.) ──────────────────────────────────
    /**
     * Returns true iff cell (r, c) is on the Voronoi diagram:
     *   1. The cell is free and has EDT >= robot_radius.
     *   2. At least one 8-neighbour (also free and EDT >= robot_radius)
     *      has a different nearest-obstacle source.
     */
    bool isVoronoi(int r, int c) const
    {
        if (occ_(r, c) > 0)                return false;
        if (dist_(r, c) < robot_radius_)   return false;
        const int my_obst = obst_(r, c);
        if (my_obst == INVALID)            return false;

        for (int k = 0; k < 8; ++k) {
            const int nr = r + DR8[k];
            const int nc = c + DC8[k];
            if (nr < 0 || nr >= rows_ || nc < 0 || nc >= cols_) continue;
            if (occ_(nr, nc) > 0)               continue;
            if (dist_(nr, nc) < robot_radius_)  continue;
            if (obst_(nr, nc) == INVALID)        continue;
            if (obst_(nr, nc) != my_obst)        return true;
        }
        return false;
    }

    // ── Extract raw Voronoi skeleton ──────────────────────────────────────────
    /**
     * Returns a binary ByteGrid where 1 = Voronoi cell (isVoronoi == true).
     */
    ByteGrid extractSkeleton() const
    {
        ByteGrid skeleton = ByteGrid::Zero(rows_, cols_);
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < cols_; ++c)
                if (isVoronoi(r, c))
                    skeleton(r, c) = 1;
        return skeleton;
    }

    // ── Zhang–Suen thinning ───────────────────────────────────────────────────
    /**
     * Classic Zhang–Suen (1984) parallel skeletonisation.
     * Reduces a binary image to a ~1-pixel-wide medial axis in-place.
     */
    static void zhangSuenThinning(ByteGrid& img)
    {
        const int rows = static_cast<int>(img.rows());
        const int cols = static_cast<int>(img.cols());

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

                        const int p2 = g(r-1, c),   p3 = g(r-1, c+1);
                        const int p4 = g(r,   c+1),  p5 = g(r+1, c+1);
                        const int p6 = g(r+1, c),    p7 = g(r+1, c-1);
                        const int p8 = g(r,   c-1),  p9 = g(r-1, c-1);

                        const int B = p2+p3+p4+p5+p6+p7+p8+p9;
                        if (B < 2 || B > 6) continue;

                        const int A = (p2==0&&p3==1) + (p3==0&&p4==1)
                                    + (p4==0&&p5==1) + (p5==0&&p6==1)
                                    + (p6==0&&p7==1) + (p7==0&&p8==1)
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

    // ── Build navigation graph from thinned skeleton ──────────────────────────
    /**
     * Nodes = skeleton cells whose 8-connected degree ≠ 2 (junctions or
     * endpoints).  Degree-2 chains collapse into edges.  A synthetic node is
     * injected when only a pure loop exists.
     */
    static NavGraph buildGraph(const ByteGrid& skel, const FloatGrid& dist_map)
    {
        const int rows = static_cast<int>(skel.rows());
        const int cols = static_cast<int>(skel.cols());

        NavGraph  graph;
        IntGrid   node_id = IntGrid::Constant(rows, cols, -1);

        // ── (a) Identify node cells ──────────────────────────────────────────
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (!skel(r, c)) continue;
                if (skelDegree_(skel, r, c) != 2) {
                    node_id(r, c) = static_cast<int>(graph.vertices.size());
                    graph.vertices.emplace_back(static_cast<float>(c),
                                                static_cast<float>(r));
                }
            }
        }

        // ── (b) Pure-loop fallback ───────────────────────────────────────────
        if (graph.vertices.empty()) {
            float best_d = -1.0f;
            int   best_r = -1, best_c = -1;
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                    if (skel(r, c) && dist_map(r, c) > best_d) {
                        best_d = dist_map(r, c);
                        best_r = r; best_c = c;
                    }
            if (best_r >= 0) {
                node_id(best_r, best_c) = 0;
                graph.vertices.emplace_back(static_cast<float>(best_c),
                                            static_cast<float>(best_r));
            }
            return graph;
        }

        // ── (c) Connected components of non-node skeleton cells ──────────────
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

        // ── (d) Map components → pairs of node cells ─────────────────────────
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
                    if      (comp_ends[cid].first  < 0)               comp_ends[cid].first  = nid;
                    else if (comp_ends[cid].second < 0 &&
                             comp_ends[cid].first  != nid)             comp_ends[cid].second = nid;
                }
            }
        }

        std::unordered_set<long long> added;
        auto edge_key = [](int a, int b) -> long long {
            if (a > b) std::swap(a, b);
            return static_cast<long long>(a) * 1'000'000LL + b;
        };

        for (auto& [j1, j2] : comp_ends) {
            if (j1 >= 0 && j2 >= 0) {
                const auto k = edge_key(j1, j2);
                if (!added.count(k)) { added.insert(k); graph.edges.emplace_back(j1, j2); }
            }
        }

        // ── (e) Directly adjacent node cells ─────────────────────────────────
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
                    if (!added.count(key)) { added.insert(key); graph.edges.emplace_back(nid_a, nid_b); }
                }
            }
        }

        return graph;
    }

    // ── Merge close nodes ─────────────────────────────────────────────────────
    /**
     * Merge vertices within `merge_dist_px` pixels into a single representative
     * (the one with highest EDT clearance).  Uses Union-Find internally.
     */
    static NavGraph mergeCloseNodes(
        const NavGraph&  g,
        float            merge_dist_px,
        const FloatGrid& dist_map)
    {
        const int n = static_cast<int>(g.vertices.size());
        if (n == 0 || merge_dist_px <= 0.0f) return g;

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
                if (dx*dx + dy*dy <= dist_sq) unite(i, j);
            }
        }

        std::unordered_map<int, int>   group_best;
        std::unordered_map<int, float> group_best_edt;

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

        NavGraph out;
        std::unordered_map<int, int> old_to_new;

        for (auto& [root, best_idx] : group_best) {
            const int new_id = static_cast<int>(out.vertices.size());
            out.vertices.push_back(g.vertices[best_idx]);
            for (int i = 0; i < n; ++i)
                if (find(i) == root)
                    old_to_new[i] = new_id;
        }

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
            if (!added.count(k)) { added.insert(k); out.edges.emplace_back(ni, nj); }
        }

        return out;
    }

private:
    int       rows_         = 0;
    int       cols_         = 0;
    float     robot_radius_ = 0.0f;
    OccGrid   occ_;
    FloatGrid dist_;
    IntGrid   obst_;   // flat index (r*cols+c) of nearest obstacle, or INVALID

    // ── Internal: skeleton-cell degree ───────────────────────────────────────
    static int skelDegree_(const ByteGrid& skel, int r, int c)
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

    // ── Internal: brushfire propagation (shared by init + dynamic updates) ───
    /**
     * Drains a priority queue of (distance, cell_flat_idx, obst_flat_idx)
     * entries, updating dist_ and obst_ for each cell that can be improved.
     *
     * This implements the brushfire wavefront expansion described in
     * Algorithm 1 of Lau et al.
     */
    void propagate_(
        std::priority_queue<
            std::tuple<float, int, int>,
            std::vector<std::tuple<float, int, int>>,
            std::greater<std::tuple<float, int, int>>>& pq)
    {
        while (!pq.empty()) {
            auto [d, vidx, oidx] = pq.top();
            pq.pop();

            const int r = vidx / cols_;
            const int c = vidx % cols_;

            // Stale entry: a shorter path was already committed
            if (d > dist_(r, c) + 1e-6f) continue;

            const int   obst_r = oidx / cols_;
            const int   oc  = oidx % cols_;

            for (int k = 0; k < 8; ++k) {
                const int nr = r + DR8[k];
                const int nc = c + DC8[k];
                if (nr < 0 || nr >= rows_ || nc < 0 || nc >= cols_) continue;
                if (occ_(nr, nc) > 0) continue; // obstacle cells keep dist=0

                // Euclidean distance from neighbour to this obstacle source
                const float nd = std::sqrt(
                    static_cast<float>((nr - obst_r) * (nr - obst_r) +
                                       (nc - oc)  * (nc - oc)));

                if (nd < dist_(nr, nc) - 1e-6f) {
                    dist_(nr, nc) = nd;
                    obst_(nr, nc) = oidx;
                    pq.emplace(nd, nr * cols_ + nc, oidx);
                }
            }
        }
    }
};
