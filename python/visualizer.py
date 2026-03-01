"""
visualizer.py
-------------
Renders the Voronoi navigation graph (and optional overlays) using matplotlib.

Overlay layers (all optional, controlled via config.xml):
  overlay_buffer   – EDT heat-map showing robot clearance.
  overlay_skeleton – The raw Voronoi skeleton (before graph extraction).
  overlay_graph    – Navigation-graph edges and vertices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np

from .config_loader import PlannerConfig


# ─── colour scheme ────────────────────────────────────────────────────────────
_EDGE_COLOUR    = "#00BFFF"   # deep-sky blue
_VERTEX_COLOUR  = "#FF4500"   # orange-red
_SKELETON_COLOUR = "#39FF14"  # neon green


def render(
    graph_result : dict[str, Any],
    grey         : np.ndarray,
    cfg          : PlannerConfig,
    save_path    : str | Path | None = None,
) -> plt.Figure:
    """
    Produce an annotated figure and optionally save it.

    Parameters
    ----------
    graph_result : dict returned by the C++ engine / image_parser.
    grey         : uint8 greyscale source image  (H, W).
    cfg          : PlannerConfig.
    save_path    : If given, figure is saved here; overrides cfg.output_image.

    Returns
    -------
    matplotlib Figure (caller is responsible for plt.show() / plt.close()).
    """
    vertices : list[tuple[float, float]] = graph_result["vertices"]
    edges    : list[tuple[int,   int  ]] = graph_result["edges"]
    skeleton : np.ndarray               = graph_result["skeleton"]   # uint8 H×W
    dist_map : np.ndarray               = graph_result["dist_map"]   # float32 H×W

    H, W = grey.shape

    fig, ax = plt.subplots(figsize=(W / 100 + 1.4, H / 100 + 1.0), dpi=cfg.output_dpi)
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.10)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # ── Base image ─────────────────────────────────────────────────────────────
    ax.imshow(grey, cmap="gray", vmin=0, vmax=255, origin="upper",
              extent=[0, W, H, 0])

    # ── Buffer / clearance heat-map ────────────────────────────────────────────
    if cfg.overlay_buffer:
        # Mask obstacle cells so they appear transparent
        obstacle_mask = grey < cfg.obstacle_threshold
        dist_display  = dist_map.astype(float)
        dist_display[obstacle_mask] = np.nan

        # Clip display range to [robot_radius, 3*robot_radius] for contrast
        lo = max(cfg.robot_radius, 1.0)
        hi = cfg.robot_radius * 3.0
        dist_display = np.clip(dist_display, lo, hi)

        # Convert to a masked array so NaN cells become transparent
        masked = np.ma.masked_invalid(dist_display)

        cmap_buf = plt.cm.plasma.copy()
        cmap_buf.set_bad(alpha=0.0)       # NaN → fully transparent
        ax.imshow(masked, cmap=cmap_buf, alpha=0.40, origin="upper",
                  extent=[0, W, H, 0],
                  vmin=lo, vmax=hi,
                  interpolation="nearest")

    # ── Voronoi skeleton ───────────────────────────────────────────────────────
    if cfg.overlay_skeleton:
        skel_rgba = np.zeros((H, W, 4), dtype=np.float32)
        skel_mask = skeleton > 0
        r_ch, g_ch, b_ch = mcolors.to_rgb(_SKELETON_COLOUR)
        skel_rgba[skel_mask, 0] = r_ch
        skel_rgba[skel_mask, 1] = g_ch
        skel_rgba[skel_mask, 2] = b_ch
        skel_rgba[skel_mask, 3] = 0.7   # 70% opacity
        ax.imshow(skel_rgba, origin="upper", extent=[0, W, H, 0],
                  interpolation="nearest")

    # ── Navigation graph ───────────────────────────────────────────────────────
    if cfg.overlay_graph:
        if edges:
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]

            for i, j in edges:
                ax.plot(
                    [xs[i], xs[j]],
                    [ys[i], ys[j]],
                    color=_EDGE_COLOUR,
                    linewidth=1.5,
                    solid_capstyle="round",
                    zorder=3,
                )

        if vertices:
            vx = [v[0] for v in vertices]
            vy = [v[1] for v in vertices]
            ax.scatter(vx, vy,
                       s=20,
                       c=_VERTEX_COLOUR,
                       marker="o",
                       linewidths=0.5,
                       edgecolors="white",
                       zorder=4)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_handles = []
    if cfg.overlay_buffer:
        legend_handles.append(
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=plt.cm.plasma(0.6), markersize=8,
                       label="Clearance (EDT)"))
    if cfg.overlay_skeleton:
        legend_handles.append(
            plt.Line2D([0], [0], color=_SKELETON_COLOUR, linewidth=2,
                       label="Voronoi skeleton"))
    if cfg.overlay_graph and edges:
        legend_handles.append(
            plt.Line2D([0], [0], color=_EDGE_COLOUR, linewidth=2,
                       label=f"Nav edges ({len(edges)})"))
    if cfg.overlay_graph and vertices:
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_VERTEX_COLOUR, markersize=8,
                       label=f"Nav nodes ({len(vertices)})"))

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=7, framealpha=0.7)

    # ── Axis labels and ticks ───────────────────────────────────────────────────
    scale = cfg.map_scale  # metres per pixel

    def _fmt_x(val: float, _pos: object) -> str:
        """Show column index and its metric equivalent."""
        return f"{val:.0f} px\n{val * scale:.1f} m" if scale > 0 else f"{val:.0f}"

    def _fmt_y(val: float, _pos: object) -> str:
        """Show row index and its metric equivalent."""
        return f"{val:.0f} px\n{val * scale:.1f} m" if scale > 0 else f"{val:.0f}"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_x))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_y))

    # Limit tick density so the labels don't overlap
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))

    ax.set_xlabel("X  (column / East)", color="white", fontsize=8, labelpad=4)
    ax.set_ylabel("Y  (row / South)",   color="white", fontsize=8, labelpad=4)

    # Style the tick labels and spines for a dark background
    ax.tick_params(colors="white", labelsize=6, which="both", length=3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")

    # ── Title / metadata ───────────────────────────────────────────────────────
    ax.set_title(
        f"Voronoi Nav Graph  |  {W}×{H} px  |  "
        f"scale {cfg.map_scale:.4f} m/px  |  "
        f"robot_r={cfg.robot_radius:.1f} px  |  "
        f"{len(vertices)} nodes  {len(edges)} edges",
        fontsize=7, pad=4, color="white",
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    out = Path(save_path) if save_path else Path(cfg.output_image)
    fig.savefig(out, dpi=cfg.output_dpi, bbox_inches="tight",
                facecolor="black")
    print(f"[visualizer] Saved → {out.resolve()}")

    return fig
