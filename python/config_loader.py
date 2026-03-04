"""
config_loader.py
----------------
Parses config.xml and exposes a typed, immutable dataclass.

All distance values remain in **pixels** unless noted; the `map_scale`
field carries the metres-per-pixel conversion factor.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlannerConfig:
    # ── I/O paths ─────────────────────────────────────────────────────────────
    input_image: str
    output_graph_prefix: str
    output_image: str

    # ── Map metadata ──────────────────────────────────────────────────────────
    map_scale: float           # metres per pixel
    obstacle_threshold: int    # greyscale cutoff [0-255]; darker ⟹ obstacle

    # ── Robot / planner parameters ────────────────────────────────────────────
    robot_radius: float   # minimum clearance in **metres**
    node_solution: float  # maximum distance in **metres** under which two graph
                          # nodes are considered identical and merged into one

    # ── Algorithm selection ───────────────────────────────────────────────────
    traversability_sampling_algo: str  # "voronoi" | "uniform"

    # ── Uniform-sampling parameters (used when algo == "uniform") ───────────
    uniform_grid_resolution: float  # spacing between sampled vertices in **metres**
    visibility_range: float         # max edge length in **metres** for LOS checks

    # ── Visualisation toggles ─────────────────────────────────────────────────
    overlay_graph: bool         # draw vertices + edges
    overlay_buffer: bool        # draw EDT heat-map (clearance field)
    overlay_skeleton: bool      # draw raw Voronoi skeleton

    # ── Output quality ────────────────────────────────────────────────────────
    output_dpi: int


def load_config(xml_path: str | Path = "config.xml") -> PlannerConfig:
    """Parse *xml_path* and return a :class:`PlannerConfig`.

    Unit handling
    -------------
    Both ``robot_radius`` and ``node_solution`` are stored as-is in **metres**.
    Conversion to pixels happens at the C++ engine call site in
    ``image_parser.py``.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _get(tag: str, default: str = "") -> str:
        el = root.find(tag)
        return el.text.strip() if (el is not None and el.text) else default

    def _get_first(*tags: str, default: str = "") -> str:
        """Return the text of the first matching tag."""
        for tag in tags:
            el = root.find(tag)
            if el is not None and el.text:
                return el.text.strip()
        return default

    def _bool(tag: str, default: bool = False) -> bool:
        return _get(tag, str(default)).lower() in ("true", "1", "yes")

    map_scale = float(_get("map_scale", "0.05"))

    def _metres_to_px(raw: float) -> float:
        """Convert metres→pixels when raw / map_scale > 1 (metric units)."""
        if map_scale > 0 and (raw / map_scale) > 1.0:
            return raw / map_scale
        return raw  # already in pixels

    raw_robot_radius  = float(_get("robot_radius", "0.5"))
    # node_solution is always in metres; kept as-is, no pixel conversion here.
    raw_node_solution = float(_get_first(
        "node_solution", "node_separation_distance", default="1.0"))

    algo = _get("traversability_sampling_algo", "voronoi").strip().lower()
    if algo not in ("voronoi", "uniform"):
        raise ValueError(
            f"Unknown traversability_sampling_algo '{algo}'. "
            "Must be 'voronoi' or 'uniform'.")

    return PlannerConfig(
        input_image       = _get("input_image", "map.png"),
        output_graph_prefix = _get("output_graph_prefix", "nav_graph"),
        output_image      = _get("output_image", "nav_output.png"),
        map_scale         = map_scale,
        obstacle_threshold= int(_get("obstacle_threshold", "128")),
        robot_radius      = raw_robot_radius,   # stored in metres
        node_solution     = raw_node_solution,   # stored in metres
        traversability_sampling_algo = algo,
        uniform_grid_resolution = float(_get("uniform_grid_resolution", "2.0")),
        visibility_range        = float(_get("visibility_range", "10.0")),
        overlay_graph     = _bool("overlay_graph", True),
        overlay_buffer    = _bool("overlay_buffer", True),
        overlay_skeleton  = _bool("overlay_skeleton", True),
        output_dpi        = int(_get("output_dpi", "150")),
    )
