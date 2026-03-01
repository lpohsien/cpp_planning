#!/usr/bin/env bash
# ============================================================================
# build.sh – compile the voronoi_engine pybind11 extension
# ============================================================================
# Requirements
#   • uv        – Python environment/package manager
#   • g++ or clang++ with C++17 support
#   • Eigen3 headers in ./eigen/
#
# Usage
#   ./build.sh           – install deps + compile
#   ./build.sh --clean   – remove build artefacts
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Clean ────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--clean" ]]; then
    echo "[build] Removing compiled extension files …"
    rm -f voronoi_engine*.so voronoi_engine*.pyd
    echo "[build] Done."
    exit 0
fi

# ── 1. Install / sync Python dependencies via uv ─────────────────────────────
echo "[build] Syncing Python environment (uv) …"
uv sync --quiet

# ── 2. Locate tools ───────────────────────────────────────────────────────────
# Prefer clang++ on macOS, g++ on Linux
if command -v clang++ &>/dev/null; then
    CXX="clang++"
elif command -v g++ &>/dev/null; then
    CXX="g++"
else
    echo "[build] ERROR: neither clang++ nor g++ found on PATH." >&2
    exit 1
fi
echo "[build] Compiler : $CXX  ($($CXX --version | head -1))"

# ── 3. Gather include paths via the uv-managed Python ────────────────────────
PYTHON="uv run python"

PYTHON_INCLUDE=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND11_INCLUDE=$($PYTHON -c "import pybind11; print(pybind11.get_include())")
EIGEN_INCLUDE="$SCRIPT_DIR/eigen"
EXT_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

echo "[build] Python include : $PYTHON_INCLUDE"
echo "[build] pybind11 include: $PYBIND11_INCLUDE"
echo "[build] Eigen include  : $EIGEN_INCLUDE"
echo "[build] Extension suffix: $EXT_SUFFIX"

OUTPUT="$SCRIPT_DIR/voronoi_engine${EXT_SUFFIX}"

# ── 4. Platform-specific linker flags ────────────────────────────────────────
OS="$(uname -s)"
LDFLAGS=""
if [[ "$OS" == "Darwin" ]]; then
    # macOS: -undefined dynamic_lookup avoids linking against libpythonX.Y directly
    LDFLAGS="-undefined dynamic_lookup"
else
    # Linux: link against libpythonX.Y
    LDFLAGS="$($PYTHON -c "
import sysconfig, os
libdir = sysconfig.get_config_var('LIBDIR') or ''
ver    = sysconfig.get_config_var('LDVERSION') or sysconfig.get_config_var('py_version_short')
lib    = os.path.join(libdir, f'libpython{ver}.so')
if os.path.exists(lib):
    print(f'-L{libdir} -lpython{ver}')
else:
    print(f'-L{libdir} -lpython{ver}')
")"
fi

# ── 5. Compile ────────────────────────────────────────────────────────────────
echo "[build] Compiling $OUTPUT …"

$CXX \
    -O2 \
    -shared \
    -fPIC \
    -std=c++17 \
    -Wall \
    -Wextra \
    -Wno-unused-parameter \
    -I"$PYTHON_INCLUDE" \
    -I"$PYBIND11_INCLUDE" \
    -I"$EIGEN_INCLUDE" \
    src/voronoi_engine.cpp \
    $LDFLAGS \
    -o "$OUTPUT"

echo "[build] ✓  Built: $(basename "$OUTPUT")"

# ── 6. Quick smoke-test ───────────────────────────────────────────────────────
echo "[build] Smoke-testing import …"
$PYTHON -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('$OUTPUT').parent))
import voronoi_engine, numpy as np
# 20×20 grid: border = obstacle, interior = free
occ = np.zeros((20, 20), dtype=np.uint8)
occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 255
# Test voronoi mode
r = voronoi_engine.compute_graph(occ, 1.0, 0.0, 'voronoi')
print(f'  voronoi: vertices={len(r[\"vertices\"])}  edges={len(r[\"edges\"])}')
# Test uniform mode
r = voronoi_engine.compute_graph(occ, 1.0, 0.0, 'uniform', 3.0, 50.0)
print(f'  uniform: vertices={len(r[\"vertices\"])}  edges={len(r[\"edges\"])}')
print('  Import OK.')
"

echo ""
echo "[build] All done. Run the planner with:"
echo "        uv run python/main.py"
