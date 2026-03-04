/**
 * map_engine.cpp
 *
 * C++17 traversability-graph dispatcher with pybind11 Python bindings.
 *
 * Based on config `algo`, routes to one of two algorithms:
 *
 * 1. "voronoi" (default)
 *    Computes the Euclidean Distance Transform (EDT) via the dynamic brushfire
 *    algorithm (Lau et al., RAS 2013), extracts the Voronoi skeleton, thins it
 *    with Zhang–Suen, and builds a navigation graph from junctions / endpoints.
 *    The BrushfireVoronoi class supports incremental updates when obstacles
 *    change, avoiding a full recompute (see brushfire_voronoi.hpp).
 *
 * 2. "uniform"
 *    Places vertices on a regular pixel grid, keeps only cells with
 *    sufficient EDT clearance, and connects pairs within a visibility range
 *    using a Bresenham line-of-sight check (see uniform_grid.hpp).
 *
 * Both algorithms share:
 *   • Vertex-merging (Union-Find) to collapse nearby graph nodes.
 *   • The same pybind11 output dict: vertices, edges, skeleton, dist_map.
 *
 * Python module name: map_engine
 * Build: see build.sh
 *
 * References
 * ----------
 * B. Lau, C. Sprunk, W. Burgard,
 * "Efficient Grid-Based Spatial Representations for Robot Navigation
 *  in Dynamic Environments", RAS 2013.
 * http://ais.informatik.uni-freiburg.de/publications/papers/lau13ras.pdf
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "brushfire_voronoi.hpp"
#include "uniform_grid.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// compute_graph – stateless dispatcher (main pybind11 entry point)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * compute_graph(occ_array, robot_radius, node_solution_px, algo,
 *               grid_step_px, visibility_range_px) -> dict
 *
 * Accepts a 2-D uint8 numpy array (0 = free, >0 = obstacle) and returns a
 * dict with keys: "vertices", "edges", "skeleton", "dist_map".
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

    // Zero-copy Eigen view of the occupancy grid
    const OccGrid occ = Eigen::Map<const OccGrid>(ptr, rows, cols);

    // ── Algorithm branch ──────────────────────────────────────────────────────
    ByteGrid skeleton = ByteGrid::Zero(rows, cols);
    NavGraph  nav;
    FloatGrid dist_map;

    if (algo == "voronoi") {
        // Dynamic brushfire (Lau et al. 2013)
        BrushfireVoronoi bf;
        bf.init(occ, robot_radius);

        dist_map = bf.distMap();
        skeleton = bf.extractSkeleton();
        BrushfireVoronoi::zhangSuenThinning(skeleton);
        nav = BrushfireVoronoi::buildGraph(skeleton, dist_map);
    } else {
        // Uniform grid sampling + line-of-sight
        BrushfireVoronoi bf;
        bf.init(occ, robot_radius);
        dist_map = bf.distMap();

        nav = buildUniformGraph(occ, dist_map, robot_radius,
                                grid_step_px, visibility_range_px);
    }

    nav = BrushfireVoronoi::mergeCloseNodes(nav, node_solution_px, dist_map);

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
// VoronoiEngine – stateful pybind11 class for dynamic map updates
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Exposes the dynamic brushfire Voronoi engine as a Python class.
 * Supports incremental obstacle add/remove without full recomputation.
 *
 * Usage (Python):
 *   import map_engine, numpy as np
 *   eng = map_engine.VoronoiEngine(occ_array, robot_radius=5.0)
 *   eng.add_obstacles([[10, 20], [11, 20]])
 *   eng.remove_obstacles([[10, 20]])
 *   result = eng.get_graph(node_solution_px=3.0)
 */
class VoronoiEngine {
public:
    /**
     * Initialise from a full occupancy grid.
     * occ_array    – uint8 numpy array (H, W), 0 = free, >0 = obstacle.
     * robot_radius – minimum clearance in pixels.
     */
    explicit VoronoiEngine(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> occ_array,
        float robot_radius)
    {
        const auto buf = occ_array.request();
        if (buf.ndim != 2)
            throw std::runtime_error("occ_array must be a 2-D numpy array");
        const int rows = static_cast<int>(buf.shape[0]);
        const int cols = static_cast<int>(buf.shape[1]);
        const auto* ptr = static_cast<const uint8_t*>(buf.ptr);
        const OccGrid occ = Eigen::Map<const OccGrid>(ptr, rows, cols);
        bf_.init(occ, robot_radius);
    }

    /** Add obstacle cells.  cells – list of [row, col] pairs. */
    void addObstacles(const std::vector<std::pair<int,int>>& cells)
    {
        bf_.addObstacles(cells);
    }

    /** Remove obstacle cells.  cells – list of [row, col] pairs. */
    void removeObstacles(const std::vector<std::pair<int,int>>& cells)
    {
        bf_.removeObstacles(cells);
    }

    /**
     * Extract the navigation graph from the current EDT / Voronoi state.
     * node_solution_px – merge threshold in pixels (0 = disabled).
     * Returns the same dict as compute_graph.
     */
    py::dict getGraph(float node_solution_px) const
    {
        const int rows = bf_.rows();
        const int cols = bf_.cols();

        ByteGrid skeleton = bf_.extractSkeleton();
        BrushfireVoronoi::zhangSuenThinning(skeleton);

        const FloatGrid& dist_map = bf_.distMap();
        NavGraph nav = BrushfireVoronoi::buildGraph(skeleton, dist_map);
        nav = BrushfireVoronoi::mergeCloseNodes(nav, node_solution_px, dist_map);

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

    /** Return the current EDT dist_map as a numpy array. */
    py::array_t<float> distMap() const
    {
        const int rows = bf_.rows();
        const int cols = bf_.cols();
        auto arr = py::array_t<float>({rows, cols});
        Eigen::Map<FloatGrid>(
            static_cast<float*>(arr.request().ptr), rows, cols) = bf_.distMap();
        return arr;
    }

private:
    BrushfireVoronoi bf_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(map_engine, m)
{
    m.doc() = R"doc(
map_engine – C++17 traversability-graph dispatcher
====================================================
Builds a navigation graph from a 2-D occupancy grid using one of two
algorithms, selected at run-time:

  "voronoi"  – Dynamic brushfire Voronoi-corridor skeleton (Lau et al. 2013).
  "uniform"  – Uniform grid sampling with pairwise line-of-sight edges.

Functions
---------
compute_graph(occ_array, robot_radius, node_solution_px,
              algo, grid_step_px, visibility_range_px) -> dict

Classes
-------
VoronoiEngine – stateful engine supporting dynamic obstacle add/remove.
)doc";

    // ── compute_graph ─────────────────────────────────────────────────────────
    m.def(
        "compute_graph",
        &compute_graph,
        py::arg("occ_array"),
        py::arg("robot_radius"),
        py::arg("node_solution_px"),
        py::arg("algo")                = "voronoi",
        py::arg("grid_step_px")        = 20.0f,
        py::arg("visibility_range_px") = 200.0f,
        R"doc(
Compute a traversability navigation graph from a binary occupancy grid.

Parameters
----------
occ_array           : numpy.ndarray[uint8], shape (H, W)
                      0 = free, non-zero = obstacle.
robot_radius        : float
                      Minimum clearance in pixels.
node_solution_px    : float
                      Merge distance in pixels (0 = disabled).
algo                : str, default "voronoi"
                      "voronoi" – dynamic brushfire Voronoi skeleton.
                      "uniform" – grid sampling + line-of-sight edges.
grid_step_px        : float, default 20.0
                      [uniform] pixel spacing between sampled vertices.
visibility_range_px : float, default 200.0
                      [uniform] max pixel distance for LOS check.

Returns
-------
dict
    "vertices"  : list of (float x, float y) – pixel coords (col, row).
    "edges"     : list of (int i, int j)     – vertex index pairs.
    "skeleton"  : numpy.ndarray[uint8] (H, W) – Voronoi skeleton.
    "dist_map"  : numpy.ndarray[float32] (H, W) – EDT in pixels.
)doc");

    // ── VoronoiEngine ─────────────────────────────────────────────────────────
    py::class_<VoronoiEngine>(m, "VoronoiEngine", R"doc(
Stateful dynamic-brushfire Voronoi engine.

Supports incremental add/remove of obstacle cells without full recomputation,
following the algorithm of Lau et al. (RAS 2013).
)doc")
        .def(py::init<
                 py::array_t<uint8_t, py::array::c_style | py::array::forcecast>,
                 float>(),
             py::arg("occ_array"),
             py::arg("robot_radius"),
             "Initialise the engine from a full occupancy grid.")
        .def("add_obstacles",    &VoronoiEngine::addObstacles,
             py::arg("cells"),
             "Add obstacle cells (list of [row, col] pairs).")
        .def("remove_obstacles", &VoronoiEngine::removeObstacles,
             py::arg("cells"),
             "Remove obstacle cells (list of [row, col] pairs).")
        .def("get_graph",        &VoronoiEngine::getGraph,
             py::arg("node_solution_px") = 0.0f,
             "Return the navigation graph as a dict (same format as compute_graph).")
        .def("dist_map",         &VoronoiEngine::distMap,
             "Return the current EDT dist_map as a float32 numpy array.");
}
