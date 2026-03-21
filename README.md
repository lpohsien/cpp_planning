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
