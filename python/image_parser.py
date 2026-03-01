"""
image_parser.py
---------------
Loads an occupancy-map image with PIL, converts it to a binary uint8
numpy array, and delegates path-planning to the C++ Voronoi engine.

Convention
----------
* Bright pixels (greyscale >= obstacle_threshold) → **free**   (0 in occ grid)
* Dark  pixels (greyscale <  obstacle_threshold) → **obstacle** (255 in occ grid)

This matches the common ROS occupancy-map convention where white = free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config_loader import PlannerConfig


def load_occupancy_map(cfg: PlannerConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the image specified in *cfg* and convert it to a binary occupancy
    grid.

    Returns
    -------
    occ_grid : np.ndarray, dtype=uint8, shape=(H, W)
        0 = free cell, 255 = obstacle cell.
    grey     : np.ndarray, dtype=uint8, shape=(H, W)
        Raw greyscale version of the input image (for visualisation).
    """
    img_path = Path(cfg.input_image)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path.resolve()}")

    # Convert to greyscale (L = 8-bit luminance)
    img  = Image.open(img_path).convert("L")
    grey = np.asarray(img, dtype=np.uint8)            # shape (H, W)

    # Threshold: pixels *darker* than the threshold are obstacles
    threshold = np.uint8(cfg.obstacle_threshold)
    occ_grid  = np.where(grey < threshold,
                         np.uint8(255),
                         np.uint8(0)).astype(np.uint8)

    return occ_grid, grey


def run_voronoi_engine(
    occ_grid : np.ndarray,
    cfg      : PlannerConfig,
) -> dict[str, Any]:
    """
    Pass the occupancy grid to the C++ Voronoi engine and return its output.

    The engine is imported as a compiled pybind11 extension (`voronoi_engine`).
    The extension must have been built (via `build.sh`) before calling this
    function.

    Parameters
    ----------
    occ_grid : np.ndarray uint8 (H, W)  — 0=free, 255=obstacle.
    cfg      : PlannerConfig

    Returns
    -------
    dict with keys:
        "vertices"  : list[(float x, float y)]  pixel coords (col, row)
        "edges"     : list[(int i, int j)]       index pairs
        "skeleton"  : np.ndarray uint8 (H, W)   Voronoi skeleton mask (zeros for uniform)
        "dist_map"  : np.ndarray float32 (H, W) EDT in pixels
    """
    try:
        import voronoi_engine  # compiled .so, placed next to this script
    except ImportError as exc:
        raise ImportError(
            "Could not import the 'voronoi_engine' C++ extension.\n"
            "Run  ./build.sh  first to compile it."
        ) from exc

    # Ensure C-contiguous uint8
    occ_c = np.ascontiguousarray(occ_grid, dtype=np.uint8)

    scale = cfg.map_scale if cfg.map_scale > 0 else 1.0
    result = voronoi_engine.compute_graph(
        occ_c,
        float(cfg.robot_radius),
        # node_solution is in metres; convert to pixels for the C++ engine
        float(cfg.node_solution / scale),
        cfg.traversability_sampling_algo,
        # uniform_grid_resolution: metres → pixels
        float(cfg.uniform_grid_resolution / scale),
        # visibility_range: metres → pixels
        float(cfg.visibility_range / scale),
    )
    return result


def parse_and_plan(cfg: PlannerConfig) -> tuple[dict[str, Any], np.ndarray]:
    """
    Convenience wrapper: load image → run engine → return (graph_dict, grey).

    Returns
    -------
    graph_result : dict  (see :func:`run_voronoi_engine`)
    grey         : np.ndarray uint8 (H, W)  greyscale source image
    """
    occ_grid, grey = load_occupancy_map(cfg)
    graph_result   = run_voronoi_engine(occ_grid, cfg)
    return graph_result, grey
