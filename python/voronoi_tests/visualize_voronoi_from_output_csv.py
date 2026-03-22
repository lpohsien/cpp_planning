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
    read_delaunay_triangles_csv,
    read_graph_edges_csv,
    read_graph_nodes_csv,
    read_polygons,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-visualize an existing Voronoi graph from output CSV files"
    )
    parser.add_argument("--boundary-files", nargs="+", required=True)
    parser.add_argument("--no-go-files", nargs="*", default=[])
    parser.add_argument("--nodes-csv", required=True)
    parser.add_argument("--edges-csv", required=True)
    parser.add_argument("--triangles-csv", default=None)
    parser.add_argument("--output-image", default="output/voronoi_h_room/voronoi_overlay_replay.png")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    boundaries = read_polygons(args.boundary_files)
    no_go_zones = read_polygons(args.no_go_files)
    vertices = read_graph_nodes_csv(args.nodes_csv)
    edges = read_graph_edges_csv(args.edges_csv)
    delaunay_triangles = read_delaunay_triangles_csv(args.triangles_csv) if args.triangles_csv else []

    output_image = Path(args.output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)

    plot_scene(
        boundaries=boundaries,
        no_go_zones=no_go_zones,
        vertices=vertices,
        edges=edges,
        delaunay_triangles=delaunay_triangles,
        points_of_interest=None,
        title="Voronoi Graph Replay From Output CSV",
        output_image_path=output_image,
        show=args.show,
    )

    print(f"Wrote visualization: {output_image}")


if __name__ == "__main__":
    main()
