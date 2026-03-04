"""
main.py
-------
Entry point for the Voronoi navigation-graph planner.

Usage
-----
    uv run python/main.py [--config CONFIG] [--input_image IMAGE]
                          [--robot_radius R] [--overlay_buffer BOOL]

All CLI flags are optional overrides of config.xml values.

Pipeline
--------
  1. Load config.xml
  2. Load and binarise the occupancy-map image  (image_parser)
  3. Run the C++ map engine                     (image_parser → map_engine.so)
  4. Save the navigation graph to two CSV files (main)
       • <prefix>_vertices.csv  —  id, x_px, y_px, x_m, y_m
       • <prefix>_edges.csv     —  from_id, to_id
  5. Render and save the annotated figure       (visualizer)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Make sure the package root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from python.config_loader import PlannerConfig, load_config
from python.image_parser  import parse_and_plan
from python.visualizer    import render


# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Voronoi navigation-graph planner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",        default="config.xml",
                   help="Path to the XML configuration file.")
    p.add_argument("--input_image",   default=None,
                   help="Override config input_image path.")
    p.add_argument("--robot_radius",  type=float, default=None,
                   help="Override robot_radius (pixels).")
    p.add_argument("--overlay_buffer", type=lambda s: s.lower() in ("true","1","yes"),
                   default=None,
                   help="Override overlay_buffer (true/false).")
    p.add_argument("--no_display",    action="store_true",
                   help="Skip plt.show(); only save the figure.")
    return p.parse_args()


def _apply_overrides(cfg: PlannerConfig, args: argparse.Namespace) -> PlannerConfig:
    """Return a new PlannerConfig with CLI overrides applied."""
    kw = cfg.__dict__.copy()
    if args.input_image   is not None: kw["input_image"]   = args.input_image
    if args.robot_radius  is not None: kw["robot_radius"]  = args.robot_radius
    if args.overlay_buffer is not None: kw["overlay_buffer"] = args.overlay_buffer
    return PlannerConfig(**kw)


def _save_graph_files(graph_result: dict, cfg: PlannerConfig) -> None:
    """
    Write the navigation graph to two CSV files:
      <prefix>_vertices.csv  —  one row per vertex: id, x_px, y_px, x_m, y_m
      <prefix>_edges.csv     —  one row per edge  : from_id, to_id
    """
    prefix   = cfg.output_graph_prefix
    scale    = cfg.map_scale
    vertices = graph_result["vertices"]
    edges    = graph_result["edges"]

    # ── vertices CSV ───────────────────────────────────────────────────────────
    vert_path = Path(f"{prefix}_vertices.csv")
    with vert_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x_px", "y_px", "x_m", "y_m"])
        for i, (x_px, y_px) in enumerate(vertices):
            writer.writerow([i,
                             f"{x_px:.4f}", f"{y_px:.4f}",
                             f"{x_px * scale:.6f}", f"{y_px * scale:.6f}"])

    # ── edges CSV ────────────────────────────────────────────────────────────────
    edge_path = Path(f"{prefix}_edges.csv")
    with edge_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["from_id", "to_id"])
        for src, dst in edges:
            writer.writerow([src, dst])

    print(f"[main] Vertices saved → {vert_path.resolve()}  ({len(vertices)} rows)")
    print(f"[main] Edges    saved → {edge_path.resolve()}  ({len(edges)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    # 1. Config
    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args)
    print(f"[main] Config   : {Path(args.config).resolve()}")
    print(f"[main] Input    : {cfg.input_image}")
    print(f"[main] Algo     : {cfg.traversability_sampling_algo}")
    print(f"[main] RobotR   : {cfg.robot_radius:.1f} px  "
          f"= {cfg.robot_radius * cfg.map_scale:.3f} m")
    print(f"[main] NodeMerge: {cfg.node_solution:.3f} m  "
          f"= {cfg.node_solution / cfg.map_scale:.1f} px")
    if cfg.traversability_sampling_algo == "uniform":
        print(f"[main] GridStep : {cfg.uniform_grid_resolution:.2f} m  "
              f"= {cfg.uniform_grid_resolution / cfg.map_scale:.1f} px")
        print(f"[main] VisRange : {cfg.visibility_range:.2f} m  "
              f"= {cfg.visibility_range / cfg.map_scale:.1f} px")

    # 2 + 3. Image parse → C++ engine
    t0 = time.perf_counter()
    graph_result, grey = parse_and_plan(cfg)
    dt = time.perf_counter() - t0

    verts = graph_result["vertices"]
    edges = graph_result["edges"]
    print(f"[main] Engine   : {len(verts)} vertices, {len(edges)} edges  "
          f"({dt*1000:.1f} ms)")

    # 4. Save graph CSV files
    _save_graph_files(graph_result, cfg)

    # 5. Render + save figure
    fig = render(graph_result, grey, cfg)

    if not args.no_display:
        import matplotlib.pyplot as plt
        plt.show()

    print("[main] Done.")


if __name__ == "__main__":
    main()
