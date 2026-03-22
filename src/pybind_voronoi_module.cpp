#include "VoronoiGraphBuilder.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "clipper2/clipper.triangulation.h"

#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kClipperPrecision = 6;

Clipper2Lib::PointD tupleToPoint2D(const std::vector<double>& xy)
{
    if (xy.size() < 2) {
        throw std::runtime_error("Expected at least two values for a 2D point");
    }
    return Clipper2Lib::PointD(xy[0], xy[1]);
}

Eigen::Vector3d tupleToPoint3D(const std::vector<double>& xyz)
{
    if (xyz.size() < 2) {
        throw std::runtime_error("Expected at least two values for a point");
    }
    const double z = xyz.size() >= 3 ? xyz[2] : 0.0;
    return Eigen::Vector3d(xyz[0], xyz[1], z);
}

Clipper2Lib::PathsD parsePolygons(const py::iterable& polygons)
{
    Clipper2Lib::PathsD out;
    for (const py::handle& poly_obj : polygons) {
        Clipper2Lib::PathD poly;
        for (const py::handle& pt_obj : poly_obj.cast<py::iterable>()) {
            poly.push_back(tupleToPoint2D(pt_obj.cast<std::vector<double>>()));
        }
        if (!poly.empty()) {
            out.push_back(std::move(poly));
        }
    }
    return out;
}

std::vector<Eigen::Vector3d> parsePoints3D(const py::iterable& points)
{
    std::vector<Eigen::Vector3d> out;
    for (const py::handle& point_obj : points) {
        out.push_back(tupleToPoint3D(point_obj.cast<std::vector<double>>()));
    }
    return out;
}

std::vector<std::pair<Eigen::Vector3d, std::vector<Eigen::Vector3d>>> parseEdgesOfInterest(const py::iterable& edges)
{
    std::vector<std::pair<Eigen::Vector3d, std::vector<Eigen::Vector3d>>> out;
    for (const py::handle& edge_group_obj : edges) {
        const py::dict edge_group = edge_group_obj.cast<py::dict>();
        if (!edge_group.contains("anchor") || !edge_group.contains("endpoints")) {
            throw std::runtime_error("Each edge group must include keys: anchor, endpoints");
        }

        const Eigen::Vector3d anchor = tupleToPoint3D(edge_group["anchor"].cast<std::vector<double>>());
        const std::vector<Eigen::Vector3d> endpoints = parsePoints3D(edge_group["endpoints"].cast<py::iterable>());
        out.push_back({anchor, endpoints});
    }
    return out;
}

std::vector<std::array<double, 6>> buildDelaunayTriangles(
    const Clipper2Lib::PathsD& boundaries,
    const Clipper2Lib::PathsD& no_go_zones)
{
    std::vector<std::array<double, 6>> out;

    Clipper2Lib::PathsD merged_boundaries =
        Clipper2Lib::Union(boundaries, Clipper2Lib::FillRule::NonZero, kClipperPrecision);
    Clipper2Lib::PathsD merged_no_go =
        Clipper2Lib::Union(no_go_zones, Clipper2Lib::FillRule::NonZero, kClipperPrecision);

    Clipper2Lib::PathsD navigable = merged_no_go.empty()
        ? merged_boundaries
        : Clipper2Lib::Difference(
            merged_boundaries,
            merged_no_go,
            Clipper2Lib::FillRule::NonZero,
            kClipperPrecision);

    Clipper2Lib::PathsD triangles;
    const auto tri_result = Clipper2Lib::Triangulate(navigable, kClipperPrecision, triangles, true);
    if (tri_result != Clipper2Lib::TriangulateResult::success) {
        return out;
    }

    out.reserve(triangles.size());
    for (const auto& tri : triangles) {
        if (tri.size() < 3) continue;
        out.push_back({tri[0].x, tri[0].y, tri[1].x, tri[1].y, tri[2].x, tri[2].y});
    }

    return out;
}

py::dict buildVoronoiGraphPy(
    const py::iterable& boundaries,
    const py::iterable& no_go_zones,
    const py::iterable& points_of_interest,
    const py::iterable& edges_of_interest,
    bool connect_using_centroids,
    bool connect_using_midpoints,
    size_t min_connections,
    double min_distance_between_vertices)
{
    const Clipper2Lib::PathsD cxx_boundaries = parsePolygons(boundaries);
    const Clipper2Lib::PathsD cxx_no_go = parsePolygons(no_go_zones);
    const std::vector<Eigen::Vector3d> cxx_poi = parsePoints3D(points_of_interest);
    const auto cxx_edges = parseEdgesOfInterest(edges_of_interest);
    const auto delaunay_triangles = buildDelaunayTriangles(cxx_boundaries, cxx_no_go);

    VoronoiGraph graph = buildVoronoiGraph(
        cxx_boundaries,
        cxx_no_go,
        cxx_poi,
        cxx_edges,
        connect_using_centroids,
        connect_using_midpoints,
        min_connections,
        min_distance_between_vertices);

    std::vector<std::array<double, 3>> vertices;
    vertices.reserve(graph.vertices.size());
    for (const auto& vertex : graph.vertices) {
        vertices.push_back({vertex.x(), vertex.y(), vertex.z()});
    }

    std::vector<std::array<size_t, 2>> edges;
    for (size_t i = 0; i < graph.tdjacencyList.size(); ++i) {
        for (const auto& [neighbor_idx, _metadata] : graph.tdjacencyList[i]) {
            if (i < neighbor_idx) {
                edges.push_back({i, neighbor_idx});
            }
        }
    }

    py::dict result;
    result["vertices"] = vertices;
    result["edges"] = edges;
    result["delaunay_triangles"] = delaunay_triangles;
    return result;
}

}  // namespace

PYBIND11_MODULE(map_engine_voronoi, m)
{
    m.doc() = "Pybind wrapper for buildVoronoiGraph";

    m.def(
        "build_voronoi_graph",
        &buildVoronoiGraphPy,
        py::arg("boundaries"),
        py::arg("no_go_zones"),
        py::arg("points_of_interest"),
        py::arg("edges_of_interest"),
        py::arg("connect_using_centroids") = true,
        py::arg("connect_using_midpoints") = false,
        py::arg("min_connections") = 2,
        py::arg("min_distance_between_vertices") = 0.0);
}
