#!/usr/bin/env python3

import sys
import os
import inspect
import trimesh
import coacd
import numpy as np
import pymeshlab
from pathlib import Path
import tempfile
from typing import Optional, Dict, Any
from dataclasses import dataclass

from data.transforms.io_xfm import load_mesh
from util.path import ensure_directory


@dataclass
class Config:
    # COCAD
    max_concavity: float = 0.05
    max_convex_hull: int = 16
    # coacd >= 1.0 uses a mode string. `preprocess` is retained only so old
    # serialized Hydra configs still load; new runs should set the mode.
    preprocess: Optional[bool] = None
    preprocess_mode: str = 'auto'
    resolution: int = 2048
    mcts_max_depth: int = 4
    mcts_iterations: int = 512
    mcts_nodes: int = 32

    # MeshLab
    simplify: bool = True
    max_face_count: int = 512
    min_face_count: int = 32
    num_proc: int = 16
    verbose: bool = False


def apply_coacd(cfg: Config, f_in: str,
                f_out: Optional[str] = None,
                d_out: Optional[str] = None,
                aux: Optional[Dict[str, Any]] = None):
    # if not cfg.verbose:
    #    sys.stdout = open(os.devnull, 'w')
    #    sys.stderr = open(os.devnull, 'w')

    # f_out = str(ensure_directory(d_out) / Path(f_in).name)
    p_in = Path(f_in)

    if f_out is None:
        assert (d_out is not None)
        f_out = str(ensure_directory(d_out) /
                    p_in.parent.parent.with_suffix(p_in.suffix).name)

    if Path(f_out).is_file():
        return

    # mesh = trimesh.load(f_in, file_type='obj')
    mesh = load_mesh(f_in, as_mesh=True,
                     file_type='obj')

    imesh = coacd.Mesh()
    imesh.vertices = mesh.vertices
    imesh.indices = mesh.faces
    kwargs = dict(
        threshold=cfg.max_concavity,
        max_convex_hull=cfg.max_convex_hull,
        resolution=cfg.resolution,
        mcts_max_depth=cfg.mcts_max_depth,
        mcts_iterations=cfg.mcts_iterations,
        mcts_nodes=cfg.mcts_nodes,
    )
    try:
        parameters = inspect.signature(coacd.run_coacd).parameters
    except (TypeError, ValueError):
        parameters = {}
    if 'preprocess_mode' in parameters:
        # Preserve the semantics of old serialized configs that only contain
        # preprocess=false while supporting the mode-based coacd >= 1.0 API.
        kwargs['preprocess_mode'] = (
            cfg.preprocess_mode
            if cfg.preprocess is None
            else ('auto' if cfg.preprocess else 'off')
        )
    elif 'preprocess' in parameters:
        kwargs['preprocess'] = (
            cfg.preprocess
            if cfg.preprocess is not None
            else cfg.preprocess_mode != 'off'
        )
    parts = coacd.run_coacd(imesh, **kwargs)

    def as_trimesh(part):
        if hasattr(part, 'vertices') and hasattr(part, 'indices'):
            vertices, faces = part.vertices, part.indices
        else:
            vertices, faces = part
        return trimesh.Trimesh(
            np.asarray(vertices),
            np.asarray(faces).reshape((-1, 3)),
            process=False,
        )

    mesh_parts = [as_trimesh(part) for part in parts]
    if not mesh_parts:
        raise RuntimeError(f'CoACD produced no convex parts for {f_in}')

    if aux is not None:
        aux['num_part'] = len(mesh_parts)

    if cfg.simplify:
        with tempfile.TemporaryDirectory() as tmpdir:
            sources = []
            for i, p in enumerate(mesh_parts):
                # Export each part.
                dst_obj = F'{tmpdir}/p{i:03d}-pre.obj'
                p.export(dst_obj)

                # simplify part with meshlab
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(dst_obj)
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=max(cfg.min_face_count,
                                      cfg.max_face_count // len(mesh_parts)),
                    preserveboundary=True,
                    preservenormal=True,
                    preservetopology=True)
                out_file = F'{tmpdir}/p{i:03d}-post.obj'
                ms.save_current_mesh(out_file)
                sources.append(out_file)

            scene = trimesh.Scene()
            for f in sources:
                p = trimesh.load(f, file_type='obj')
                scene.add_geometry(p)
            scene.export(f_out)

    else:
        scene = trimesh.Scene()
        for p in mesh_parts:
            scene.add_geometry(p)
        scene.export(f_out)
