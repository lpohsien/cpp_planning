#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include "clipper2/clipper.h"
#include "clipper2/clipper.triangulation.h"

struct VoronoiGraph
{
    std::vector<Eigen::Vector3d> vertices;
    std::vector<std::map<size_t, std::vector<size_t>>> tdjacencyList;
};

namespace {

constexpr int kClipperPrecision = 6;
struct EdgeKey {
    int64_t ax;
    int64_t ay;
    int64_t bx;
    int64_t by;

    bool operator<(const EdgeKey& other) const
    {
        if (ax != other.ax) return ax < other.ax;
        if (ay != other.ay) return ay < other.ay;
        if (bx != other.bx) return bx < other.bx;
        return by < other.by;
    }
};

struct TriangleData {
    std::array<Clipper2Lib::PointD, 3> pts;
    size_t center_idx;
};

class DisjointSet {
public:
    explicit DisjointSet(size_t n) : parent_(n), rank_(n, 0)
    {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    size_t find(size_t x)
    {
        if (parent_[x] != x) parent_[x] = find(parent_[x]);
        return parent_[x];
    }

    void unite(size_t a, size_t b)
    {
        size_t ra = find(a);
        size_t rb = find(b);
        if (ra == rb) return;
        if (rank_[ra] < rank_[rb]) std::swap(ra, rb);
        parent_[rb] = ra;
        if (rank_[ra] == rank_[rb]) ++rank_[ra];
    }

private:
    std::vector<size_t> parent_;
    std::vector<unsigned> rank_;
};

Eigen::Vector3d toEigen(const Clipper2Lib::PointD& p)
{
    return Eigen::Vector3d(p.x, p.y, 0.0);
}

Clipper2Lib::PointD toPointD(const Eigen::Vector3d& p)
{
    return Clipper2Lib::PointD(p.x(), p.y());
}

int64_t quantize(double value)
{
    const double scale = std::pow(10.0, static_cast<double>(kClipperPrecision));
    return static_cast<int64_t>(std::llround(value * scale));
}

EdgeKey makeEdgeKey(const Clipper2Lib::PointD& a, const Clipper2Lib::PointD& b)
{
    EdgeKey key{quantize(a.x), quantize(a.y), quantize(b.x), quantize(b.y)};
    const bool swapped = (key.ax > key.bx) || (key.ax == key.bx && key.ay > key.by);
    if (swapped) {
        std::swap(key.ax, key.bx);
        std::swap(key.ay, key.by);
    }
    return key;
}

bool pointInsidePath(const Clipper2Lib::PointD& p, const Clipper2Lib::PathD& poly)
{
    const auto res = Clipper2Lib::PointInPolygon(p, poly);
    return res == Clipper2Lib::PointInPolygonResult::IsInside ||
           res == Clipper2Lib::PointInPolygonResult::IsOn;
}

bool isInsideAny(const Clipper2Lib::PointD& p, const Clipper2Lib::PathsD& polys)
{
    for (const auto& poly : polys) {
        if (poly.size() >= 3 && pointInsidePath(p, poly)) {
            return true;
        }
    }
    return false;
}

bool isInsideNavigable(
    const Clipper2Lib::PointD& p,
    const Clipper2Lib::PathsD& boundaries,
    const Clipper2Lib::PathsD& no_go)
{
    if (!isInsideAny(p, boundaries)) return false;
    if (isInsideAny(p, no_go)) return false;
    return true;
}

double cross2d(const Clipper2Lib::PointD& a, const Clipper2Lib::PointD& b, const Clipper2Lib::PointD& c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool onSegment(const Clipper2Lib::PointD& a, const Clipper2Lib::PointD& b, const Clipper2Lib::PointD& p)
{
    if (std::abs(cross2d(a, b, p)) > 1e-9) return false;
    const double min_x = std::min(a.x, b.x) - 1e-9;
    const double max_x = std::max(a.x, b.x) + 1e-9;
    const double min_y = std::min(a.y, b.y) - 1e-9;
    const double max_y = std::max(a.y, b.y) + 1e-9;
    return p.x >= min_x && p.x <= max_x && p.y >= min_y && p.y <= max_y;
}

bool segmentsIntersect(
    const Clipper2Lib::PointD& a,
    const Clipper2Lib::PointD& b,
    const Clipper2Lib::PointD& c,
    const Clipper2Lib::PointD& d)
{
    const double c1 = cross2d(a, b, c);
    const double c2 = cross2d(a, b, d);
    const double c3 = cross2d(c, d, a);
    const double c4 = cross2d(c, d, b);

    if (((c1 > 0.0 && c2 < 0.0) || (c1 < 0.0 && c2 > 0.0)) &&
        ((c3 > 0.0 && c4 < 0.0) || (c3 < 0.0 && c4 > 0.0))) {
        return true;
    }

    return onSegment(a, b, c) || onSegment(a, b, d) || onSegment(c, d, a) || onSegment(c, d, b);
}

bool segmentIntersectsPolygons(
    const Clipper2Lib::PointD& a,
    const Clipper2Lib::PointD& b,
    const Clipper2Lib::PathsD& polys)
{
    for (const auto& poly : polys) {
        const size_t n = poly.size();
        if (n < 2) continue;
        for (size_t i = 0; i < n; ++i) {
            const auto& p0 = poly[i];
            const auto& p1 = poly[(i + 1) % n];
            if (segmentsIntersect(a, b, p0, p1)) return true;
        }
    }
    return false;
}

bool isSegmentNavigable(
    const Clipper2Lib::PointD& a,
    const Clipper2Lib::PointD& b,
    const Clipper2Lib::PathsD& boundaries,
    const Clipper2Lib::PathsD& no_go)
{
    const Clipper2Lib::PointD midpoint((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
    if (!isInsideNavigable(a, boundaries, no_go)) return false;
    if (!isInsideNavigable(b, boundaries, no_go)) return false;
    if (!isInsideNavigable(midpoint, boundaries, no_go)) return false;

    if (segmentIntersectsPolygons(a, b, no_go)) return false;
    return true;
}

bool circumcenter(
    const Clipper2Lib::PointD& a,
    const Clipper2Lib::PointD& b,
    const Clipper2Lib::PointD& c,
    Clipper2Lib::PointD& out)
{
    const double ax = a.x;
    const double ay = a.y;
    const double bx = b.x;
    const double by = b.y;
    const double cx = c.x;
    const double cy = c.y;

    const double d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (std::abs(d) < 1e-12) return false;

    const double a2 = ax * ax + ay * ay;
    const double b2 = bx * bx + by * by;
    const double c2 = cx * cx + cy * cy;

    out.x = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    out.y = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    return true;
}

size_t addVertex(VoronoiGraph& graph, const Eigen::Vector3d& v)
{
    graph.vertices.push_back(v);
    graph.tdjacencyList.emplace_back();
    return graph.vertices.size() - 1;
}

void addUndirectedEdge(VoronoiGraph& graph, size_t a, size_t b)
{
    if (a == b) return;
    if (a >= graph.tdjacencyList.size() || b >= graph.tdjacencyList.size()) return;

    auto& ab = graph.tdjacencyList[a][b];
    auto& ba = graph.tdjacencyList[b][a];
    if (ab.empty()) ab.push_back(b);
    if (ba.empty()) ba.push_back(a);
}

void connectToNearestVisible(
    VoronoiGraph& graph,
    size_t source_idx,
    size_t min_connections,
    const Clipper2Lib::PathsD& boundaries,
    const Clipper2Lib::PathsD& no_go)
{
    if (graph.vertices.empty() || source_idx >= graph.vertices.size()) return;

    struct Candidate {
        size_t idx;
        double dist2;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(graph.vertices.size());

    const Eigen::Vector3d src = graph.vertices[source_idx];
    for (size_t i = 0; i < graph.vertices.size(); ++i) {
        if (i == source_idx) continue;
        const double d2 = (graph.vertices[i] - src).squaredNorm();
        candidates.push_back({i, d2});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return lhs.dist2 < rhs.dist2;
    });

    size_t connected = 0;
    for (const auto& c : candidates) {
        if (connected >= min_connections) break;
        const auto a = toPointD(graph.vertices[source_idx]);
        const auto b = toPointD(graph.vertices[c.idx]);
        if (!isSegmentNavigable(a, b, boundaries, no_go)) continue;
        addUndirectedEdge(graph, source_idx, c.idx);
        ++connected;
    }
}

void mergeCloseVertices(
    VoronoiGraph& graph,
    double min_dist,
    const Clipper2Lib::PathsD& boundaries,
    const Clipper2Lib::PathsD& no_go)
{
    if (min_dist <= 0.0 || graph.vertices.size() < 2) return;

    const double min_dist2 = min_dist * min_dist;
    const size_t n = graph.vertices.size();
    DisjointSet dsu(n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if ((graph.vertices[i] - graph.vertices[j]).squaredNorm() > min_dist2) continue;
            const auto a = toPointD(graph.vertices[i]);
            const auto b = toPointD(graph.vertices[j]);
            if (isSegmentNavigable(a, b, boundaries, no_go)) dsu.unite(i, j);
        }
    }

    std::map<size_t, size_t> root_to_new_idx;
    std::vector<Eigen::Vector3d> sums;
    std::vector<size_t> counts;

    for (size_t i = 0; i < n; ++i) {
        const size_t root = dsu.find(i);
        auto it = root_to_new_idx.find(root);
        if (it == root_to_new_idx.end()) {
            const size_t new_idx = sums.size();
            root_to_new_idx[root] = new_idx;
            sums.push_back(graph.vertices[i]);
            counts.push_back(1);
        } else {
            sums[it->second] += graph.vertices[i];
            ++counts[it->second];
        }
    }

    VoronoiGraph merged;
    merged.vertices.resize(sums.size());
    merged.tdjacencyList.resize(sums.size());
    for (size_t i = 0; i < sums.size(); ++i) {
        merged.vertices[i] = sums[i] / static_cast<double>(counts[i]);
    }

    for (size_t i = 0; i < n; ++i) {
        for (const auto& [j, _meta] : graph.tdjacencyList[i]) {
            if (i >= j) continue;
            const size_t ni = root_to_new_idx[dsu.find(i)];
            const size_t nj = root_to_new_idx[dsu.find(j)];
            addUndirectedEdge(merged, ni, nj);
        }
    }

    graph = std::move(merged);
}

}  // namespace

VoronoiGraph buildVoronoiGraph(
    Clipper2Lib::PathsD aBoundaries,
    Clipper2Lib::PathsD aNoGoZones,
    std::vector<Eigen::Vector3d> aPointsOfInterest,
    std::vector<std::pair<Eigen::Vector3d, std::vector<Eigen::Vector3d>>> aEdgesOfInterest,
    bool aConnectUsingCentroids,
    bool aConnectUsingMidpoints,
    size_t aMinConnections,
    double aMinDistanceBetweenVertices)
{
    VoronoiGraph graph;

    if (aBoundaries.empty()) return graph;

    aBoundaries = Clipper2Lib::Union(aBoundaries, Clipper2Lib::FillRule::NonZero, kClipperPrecision);
    aNoGoZones = Clipper2Lib::Union(aNoGoZones, Clipper2Lib::FillRule::NonZero, kClipperPrecision);
    
    Clipper2Lib::PathsD navigable = aNoGoZones.empty()
        ? aBoundaries
        : Clipper2Lib::Difference(aBoundaries, aNoGoZones, Clipper2Lib::FillRule::NonZero, kClipperPrecision);

    if (navigable.empty()) return graph;

    Clipper2Lib::PathsD triangles;
    const auto tri_result = Clipper2Lib::Triangulate(navigable, kClipperPrecision, triangles, true);
    if (tri_result != Clipper2Lib::TriangulateResult::success) {
        return graph;
    }

    std::vector<TriangleData> tri_data;
    tri_data.reserve(triangles.size());
    std::map<EdgeKey, std::vector<size_t>> edge_to_tris;

    for (const auto& tri : triangles) {
        if (tri.size() < 3) continue;
        std::array<Clipper2Lib::PointD, 3> pts{tri[0], tri[1], tri[2]};

        Clipper2Lib::PointD center;
        bool center_ok = circumcenter(pts[0], pts[1], pts[2], center) &&
                         isInsideNavigable(center, aBoundaries, aNoGoZones);
        if (!center_ok) {
            if (!aConnectUsingCentroids) continue;
            center = Clipper2Lib::PointD(
                (pts[0].x + pts[1].x + pts[2].x) / 3.0,
                (pts[0].y + pts[1].y + pts[2].y) / 3.0);
            if (!isInsideNavigable(center, aBoundaries, aNoGoZones)) continue;
        }

        const size_t center_idx = addVertex(graph, toEigen(center));
        const size_t tri_idx = tri_data.size();
        tri_data.push_back({pts, center_idx});

        for (size_t e = 0; e < 3; ++e) {
            const auto& a = pts[e];
            const auto& b = pts[(e + 1) % 3];
            edge_to_tris[makeEdgeKey(a, b)].push_back(tri_idx);

            if (aConnectUsingMidpoints) {
                const Clipper2Lib::PointD mid((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
                if (!isInsideNavigable(mid, aBoundaries, aNoGoZones)) continue;
                const size_t mid_idx = addVertex(graph, toEigen(mid));
                if (isSegmentNavigable(mid, center, aBoundaries, aNoGoZones)) {
                    addUndirectedEdge(graph, center_idx, mid_idx);
                }
            }
        }
    }

    for (const auto& [edge_key, owners] : edge_to_tris) {
        (void)edge_key;
        if (owners.size() < 2) continue;
        for (size_t i = 0; i < owners.size(); ++i) {
            for (size_t j = i + 1; j < owners.size(); ++j) {
                const size_t a = tri_data[owners[i]].center_idx;
                const size_t b = tri_data[owners[j]].center_idx;
                const auto pa = toPointD(graph.vertices[a]);
                const auto pb = toPointD(graph.vertices[b]);
                if (isSegmentNavigable(pa, pb, aBoundaries, aNoGoZones)) {
                    addUndirectedEdge(graph, a, b);
                }
            }
        }
    }

    const size_t connect_count = std::max<size_t>(aMinConnections, 1);

    for (const auto& poi : aPointsOfInterest) {
        const auto p = toPointD(poi);
        if (!isInsideNavigable(p, aBoundaries, aNoGoZones)) continue;
        const size_t idx = addVertex(graph, poi);
        connectToNearestVisible(graph, idx, connect_count, aBoundaries, aNoGoZones);
    }

    for (const auto& edge_group : aEdgesOfInterest) {
        const auto anchor_p = toPointD(edge_group.first);
        if (!isInsideNavigable(anchor_p, aBoundaries, aNoGoZones)) continue;

        const size_t anchor_idx = addVertex(graph, edge_group.first);
        connectToNearestVisible(graph, anchor_idx, connect_count, aBoundaries, aNoGoZones);

        for (const auto& endpoint : edge_group.second) {
            const auto end_p = toPointD(endpoint);
            if (!isInsideNavigable(end_p, aBoundaries, aNoGoZones)) continue;

            const size_t end_idx = addVertex(graph, endpoint);
            connectToNearestVisible(graph, end_idx, connect_count, aBoundaries, aNoGoZones);
            if (isSegmentNavigable(anchor_p, end_p, aBoundaries, aNoGoZones)) {
                addUndirectedEdge(graph, anchor_idx, end_idx);
            }
        }
    }

    mergeCloseVertices(graph, aMinDistanceBetweenVertices, aBoundaries, aNoGoZones);
    return graph;
}