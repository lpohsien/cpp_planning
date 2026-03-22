# cpp_planning

## Clipper2 setup

This project vendors Clipper2 as a git submodule at:

- `third_party/Clipper2`

Initialize it after cloning:

```bash
git submodule update --init --recursive
```

The build script uses Clipper2 headers and sources from:

- `third_party/Clipper2/CPP/Clipper2Lib/include`
- `third_party/Clipper2/CPP/Clipper2Lib/src`

## Build

```bash
./build.sh --clean
./build.sh
```

`build.sh` will:

1. Sync Python dependencies with `uv sync`
2. Compile the `map_engine` pybind11 module with Eigen + Clipper2
3. Run a smoke test for existing `voronoi` and `uniform` algorithms

## Voronoi graph builder

The C++ Clipper2-based implementation is in:

- `src/VoronoiGraphBuilder.cpp`

It performs constrained triangulation with `Clipper2Lib::Triangulate` and builds a Voronoi-style adjacency graph from triangle centers.

## Voronoi CSV Test Runner

This repository now includes an end-to-end CSV-driven test harness for the Voronoi graph builder.

### Input fixtures

Sample H-shaped room inputs are under:

- `input/voronoi_h_room/boundary_h_room.csv`
- `input/voronoi_h_room/no_go_left_square.csv`
- `input/voronoi_h_room/points_of_interest.csv`
- `input/voronoi_h_room/edges_of_interest.csv`

Polygon CSV files use headers:

- `vertex_x,vertex_y,point_type`

For this test setup, `point_type` is always `1`.

### Build pybind module

```bash
python python/build_map_engine_voronoi.py
```

This builds `map_engine_voronoi` into the `python/` directory.

Optional import smoke-check:

```bash
python -c "import sys; sys.path.insert(0, 'python'); import map_engine_voronoi; print('map_engine_voronoi import OK')"
```

### Run test and generate output CSV + plot

```bash
python python/voronoi_tests/run_voronoi_csv_test.py \
	--boundary-files input/voronoi_h_room/boundary_h_room.csv \
	--no-go-files input/voronoi_h_room/no_go_left_square.csv \
	--poi-file input/voronoi_h_room/points_of_interest.csv \
	--edges-file input/voronoi_h_room/edges_of_interest.csv
```

Outputs are written to `output/voronoi_h_room/`:

- `voronoi_nodes.csv` with headers `vertex_x,vertex_y,vertex_z`
- `voronoi_edges.csv` with headers `src_idx,dst_idx`
- `delaunay_triangles.csv` with headers `v0_x,v0_y,v1_x,v1_y,v2_x,v2_y`
- `voronoi_overlay.png`

### Replay visualization from output CSV

```bash
python python/voronoi_tests/visualize_voronoi_from_output_csv.py \
	--boundary-files input/voronoi_h_room/boundary_h_room.csv \
	--no-go-files input/voronoi_h_room/no_go_left_square.csv \
	--nodes-csv output/voronoi_h_room/voronoi_nodes.csv \
	--edges-csv output/voronoi_h_room/voronoi_edges.csv \
	--triangles-csv output/voronoi_h_room/delaunay_triangles.csv
```
