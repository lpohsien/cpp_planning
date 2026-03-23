#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from voronoi_tests.voronoi_csv_io import (
    plot_scene,
    read_edges_of_interest_csv,
    read_points_csv,
    read_polygons,
    write_delaunay_triangles_csv,
    write_graph_edges_csv,
    write_graph_nodes_csv,
)

import map_engine_voronoi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Voronoi graph builder from CSV inputs")
    parser.add_argument("--boundary-files", nargs="+", required=True)
    parser.add_argument("--no-go-files", nargs="*", default=[])
    parser.add_argument("--poi-file", required=True)
    parser.add_argument("--edges-file", required=True)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "output" / "voronoi_h_room"))
    parser.add_argument("--connect-using-center", action="store_true", default=True)
    parser.add_argument("--connect-using-midpoints", action="store_true", default=False)
    parser.add_argument("--min-connections", type=int, default=2)
    parser.add_argument("--min-distance-between-vertices", type=float, default=0.2)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    boundaries = read_polygons(args.boundary_files)
    no_go_zones = read_polygons(args.no_go_files)
    poi = read_points_csv(args.poi_file)
    edges_of_interest = read_edges_of_interest_csv(args.edges_file)

    graph = map_engine_voronoi.build_voronoi_graph(
        boundaries=boundaries,
        no_go_zones=no_go_zones,
        points_of_interest=poi,
        edges_of_interest=edges_of_interest,
        connect_using_center=args.connect_using_center,
        connect_using_midpoints=args.connect_using_midpoints,
        min_connections=args.min_connections,
        min_distance_between_vertices=args.min_distance_between_vertices,
    )

    vertices = [list(v) for v in graph["vertices"]]
    edges = [list(e) for e in graph["edges"]]
    delaunay_triangles = [list(t) for t in graph.get("delaunay_triangles", [])]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_csv = output_dir / "voronoi_nodes.csv"
    edges_csv = output_dir / "voronoi_edges.csv"
    triangles_csv = output_dir / "delaunay_triangles.csv"
    image_png = output_dir / "voronoi_overlay.png"

    write_graph_nodes_csv(nodes_csv, vertices)
    write_graph_edges_csv(edges_csv, edges)
    write_delaunay_triangles_csv(triangles_csv, delaunay_triangles)

    plot_scene(
        boundaries=boundaries,
        no_go_zones=no_go_zones,
        vertices=vertices,
        edges=edges,
        delaunay_triangles=delaunay_triangles,
        points_of_interest=poi,
        title="Voronoi Graph From CSV Inputs",
        output_image_path=image_png,
        show=args.show,
    )

    print(f"Wrote graph nodes: {nodes_csv}")
    print(f"Wrote graph edges: {edges_csv}")
    print(f"Wrote Delaunay triangles: {triangles_csv}")
    print(f"Wrote visualization: {image_png}")


if __name__ == "__main__":
    main()
