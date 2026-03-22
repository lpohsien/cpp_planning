#!/usr/bin/env python3
"""Builds the map_engine_voronoi pybind extension in-place under python/."""

from __future__ import annotations

import platform
import shlex
import subprocess
import sysconfig
from pathlib import Path

import pybind11


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "python"
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise RuntimeError("Could not determine Python extension suffix")

    output_path = output_dir / f"map_engine_voronoi{ext_suffix}"

    include_dirs = [
        Path(pybind11.get_include()),
        Path(pybind11.get_include(user=True)),
        Path(sysconfig.get_paths()["include"]),
        repo_root / "src",
        repo_root / "third_party" / "eigen",
        repo_root / "third_party" / "Clipper2" / "CPP" / "Clipper2Lib" / "include",
    ]

    plat_include = sysconfig.get_paths().get("platinclude")
    if plat_include:
        include_dirs.append(Path(plat_include))

    clipper_src = repo_root / "third_party" / "Clipper2" / "CPP" / "Clipper2Lib" / "src"
    sources = [
        repo_root / "src" / "pybind_voronoi_module.cpp",
        repo_root / "src" / "VoronoiGraphBuilder.cpp",
        clipper_src / "clipper.engine.cpp",
        clipper_src / "clipper.offset.cpp",
        clipper_src / "clipper.rectclip.cpp",
        clipper_src / "clipper.triangulation.cpp",
    ]

    cmd = [
        "c++",
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++17",
        "-fPIC",
    ]

    if platform.system() == "Darwin":
        cmd.extend(["-undefined", "dynamic_lookup"])

    cmd.extend(f"-I{inc}" for inc in include_dirs)
    cmd.extend(str(src) for src in sources)
    cmd.extend(["-o", str(output_path)])

    print("Compiling extension with command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=repo_root)
    print(f"Built: {output_path}")


if __name__ == "__main__":
    main()
