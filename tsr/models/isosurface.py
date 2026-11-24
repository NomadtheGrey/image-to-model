from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
def _as_numpy(volume):
    if hasattr(volume, "detach"):
        return volume.detach().cpu().numpy()
    if isinstance(volume, np.ndarray):
        return volume
    return np.asarray(volume)


def _pick_marching_cubes():
    """Choose a working marching-cubes implementation, by capability.

    The check must happen here, at import time: the PyPI package named
    "mcubes" is a different library (class-based API) from PyMCubes, whose
    module is also imported as "mcubes" and which owns the
    mcubes.marching_cubes() function this code was written against. An
    import-only check accepts the wrong package silently and the missing
    function only surfaces mid-generation, disguised as an AttributeError
    from the retry in forward(). Capability — the attribute exists — is
    the only honest test. scikit-image is the floor: it ships with rembg,
    is pure NumPy, and works everywhere this app runs.
    """
    try:
        from torchmcubes import marching_cubes as gpu_marching_cubes

        return gpu_marching_cubes
    except ImportError:
        pass

    try:
        import mcubes

        pycubes_fn = getattr(mcubes, "marching_cubes", None)
    except ImportError:
        pycubes_fn = None
    if callable(pycubes_fn):
        def pycubes_marching_cubes(volume, isovalue=0.0):
            verts, faces = pycubes_fn(_as_numpy(volume), isovalue)
            return (
                torch.from_numpy(verts.astype(np.float32)),
                torch.from_numpy(faces.astype(np.int64)),
            )

        return pycubes_marching_cubes

    from skimage.measure import marching_cubes as ski_marching_cubes

    def skimage_marching_cubes(volume, isovalue=0.0):
        verts, faces, _, _ = ski_marching_cubes(_as_numpy(volume), level=isovalue)
        return (
            torch.from_numpy(verts.astype(np.float32)),
            torch.from_numpy(faces.astype(np.int64)),
        )

    return skimage_marching_cubes


marching_cubes = _pick_marching_cubes()


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func: Callable = marching_cubes
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        # One plain call: _pick_marching_cubes already guaranteed a working
        # implementation, so there is nothing to retry. The old
        # catch-AttributeError-and-retry printed a torchmcubes/CUDA message
        # for failures that had nothing to do with either, then re-ran the
        # same broken call and let the real error escape under the lie.
        v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
        v_pos = v_pos[..., [2, 1, 0]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
