from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def read_polygon_csv(path: str | Path) -> list[list[float]]:
    polygon: list[list[float]] = []
    with _as_path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"vertex_x", "vertex_y", "point_type"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must include headers: vertex_x, vertex_y, point_type")

        for row in reader:
            polygon.append([float(row["vertex_x"]), float(row["vertex_y"])])
    return polygon


def read_polygons(paths: Iterable[str | Path]) -> list[list[list[float]]]:
    return [read_polygon_csv(path) for path in paths]


def read_points_csv(path: str | Path) -> list[list[float]]:
    points: list[list[float]] = []
    with _as_path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no headers")

        x_key = "poi_x" if "poi_x" in reader.fieldnames else "vertex_x"
        y_key = "poi_y" if "poi_y" in reader.fieldnames else "vertex_y"
        z_key = "poi_z" if "poi_z" in reader.fieldnames else "vertex_z"

        if x_key not in reader.fieldnames or y_key not in reader.fieldnames:
            raise ValueError(f"{path} must include x/y fields")

        for row in reader:
            z_value = float(row[z_key]) if z_key in row and row[z_key] != "" else 0.0
            points.append([float(row[x_key]), float(row[y_key]), z_value])
    return points


def read_edges_of_interest_csv(path: str | Path) -> list[dict[str, list[list[float]] | list[float]]]:
    grouped: dict[tuple[float, float, float], list[list[float]]] = defaultdict(list)
    with _as_path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "anchor_x",
            "anchor_y",
            "anchor_z",
            "endpoint_x",
            "endpoint_y",
            "endpoint_z",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{path} must include: anchor_x, anchor_y, anchor_z, endpoint_x, endpoint_y, endpoint_z"
            )

        for row in reader:
            anchor = (float(row["anchor_x"]), float(row["anchor_y"]), float(row["anchor_z"]))
            endpoint = [float(row["endpoint_x"]), float(row["endpoint_y"]), float(row["endpoint_z"])]
            grouped[anchor].append(endpoint)

    out: list[dict[str, list[list[float]] | list[float]]] = []
    for anchor, endpoints in grouped.items():
        out.append({"anchor": [anchor[0], anchor[1], anchor[2]], "endpoints": endpoints})
    return out


def write_graph_nodes_csv(path: str | Path, vertices: list[list[float]]) -> None:
    with _as_path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vertex_x", "vertex_y", "vertex_z"])
        writer.writerows(vertices)


def write_graph_edges_csv(path: str | Path, edges: list[list[int]]) -> None:
    with _as_path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src_idx", "dst_idx"])
        writer.writerows(edges)


def write_delaunay_triangles_csv(path: str | Path, triangles: list[list[float]]) -> None:
    with _as_path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["v0_x", "v0_y", "v1_x", "v1_y", "v2_x", "v2_y"])
        writer.writerows(triangles)


def read_graph_nodes_csv(path: str | Path) -> list[list[float]]:
    points: list[list[float]] = []
    with _as_path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"vertex_x", "vertex_y", "vertex_z"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must include headers: vertex_x, vertex_y, vertex_z")

        for row in reader:
            points.append([float(row["vertex_x"]), float(row["vertex_y"]), float(row["vertex_z"])])
    return points


def read_graph_edges_csv(path: str | Path) -> list[list[int]]:
    edges: list[list[int]] = []
    with _as_path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"src_idx", "dst_idx"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must include headers: src_idx, dst_idx")

        for row in reader:
            edges.append([int(row["src_idx"]), int(row["dst_idx"])])
    return edges


def read_delaunay_triangles_csv(path: str | Path) -> list[list[float]]:
    triangles: list[list[float]] = []
    with _as_path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"v0_x", "v0_y", "v1_x", "v1_y", "v2_x", "v2_y"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must include headers: v0_x, v0_y, v1_x, v1_y, v2_x, v2_y")

        for row in reader:
            triangles.append(
                [
                    float(row["v0_x"]),
                    float(row["v0_y"]),
                    float(row["v1_x"]),
                    float(row["v1_y"]),
                    float(row["v2_x"]),
                    float(row["v2_y"]),
                ]
            )
    return triangles


def _plot_polygon(ax: Axes, polygon: list[list[float]], color: str, label: str, alpha: float) -> None:
    if not polygon:
        return
    xs = [p[0] for p in polygon] + [polygon[0][0]]
    ys = [p[1] for p in polygon] + [polygon[0][1]]
    ax.fill(xs, ys, color=color, alpha=alpha, label=label)
    ax.plot(xs, ys, color=color, linewidth=1.5)


def plot_scene(
    boundaries: list[list[list[float]]],
    no_go_zones: list[list[list[float]]],
    vertices: list[list[float]],
    edges: list[list[int]],
    delaunay_triangles: list[list[float]] | None = None,
    points_of_interest: list[list[float]] | None = None,
    title: str = "Voronoi Graph",
    output_image_path: str | Path | None = None,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, poly in enumerate(boundaries):
        _plot_polygon(ax, poly, color="#8ecae6", label="Boundary" if i == 0 else "", alpha=0.25)

    for i, poly in enumerate(no_go_zones):
        _plot_polygon(ax, poly, color="#e63946", label="No-go zone" if i == 0 else "", alpha=0.45)

    if delaunay_triangles:
        triangle_label_drawn = False
        for tri in delaunay_triangles:
            x0, y0, x1, y1, x2, y2 = tri
            xs = [x0, x1, x2, x0]
            ys = [y0, y1, y2, y0]
            label = "Delaunay triangles" if not triangle_label_drawn else ""
            ax.plot(xs, ys, color="#6c757d", linewidth=0.8, alpha=0.45, zorder=2, label=label)
            triangle_label_drawn = True

    if vertices:
        vx = [v[0] for v in vertices]
        vy = [v[1] for v in vertices]
        ax.scatter(vx, vy, s=14, c="#1d3557", label="Graph nodes", zorder=4)

    for src_idx, dst_idx in edges:
        x0, y0 = vertices[src_idx][0], vertices[src_idx][1]
        x1, y1 = vertices[dst_idx][0], vertices[dst_idx][1]
        ax.plot([x0, x1], [y0, y1], color="#264653", linewidth=1.0, alpha=0.9, zorder=3)

    if points_of_interest:
        px = [p[0] for p in points_of_interest]
        py = [p[1] for p in points_of_interest]
        ax.scatter(px, py, s=50, marker="*", c="#ffb703", edgecolors="black", label="POI", zorder=6)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    compact = [(h, label) for h, label in zip(handles, labels) if label]
    if compact:
        ax.legend(
            [c[0] for c in compact],
            [c[1] for c in compact],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
        fig.subplots_adjust(right=0.78)

    if output_image_path is not None:
        img_path = _as_path(output_image_path)
        img_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(img_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
