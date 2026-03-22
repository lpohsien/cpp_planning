#pragma once

#include <Eigen/Dense>

#include <map>
#include <utility>
#include <vector>

#include "clipper2/clipper.h"

struct VoronoiGraph
{
    std::vector<Eigen::Vector3d> vertices;
    std::vector<std::map<size_t, std::vector<size_t>>> tdjacencyList;
};

VoronoiGraph buildVoronoiGraph(
    Clipper2Lib::PathsD aBoundaries,
    Clipper2Lib::PathsD aNoGoZones,
    std::vector<Eigen::Vector3d> aPointsOfInterest,
    std::vector<std::pair<Eigen::Vector3d, std::vector<Eigen::Vector3d>>> aEdgesOfInterest,
    bool aConnectUsingCenter = true,
    bool aConnectUsingMidpoints = false,
    size_t aMinConnections = 2,
    double aMinDistanceBetweenVertices = 0.0);
