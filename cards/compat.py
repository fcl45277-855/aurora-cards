"""Two small helpers used by the harness when preparing an Aurora batch for a model running in reduced precision.

Vendored (re-implemented) from the authors' earlier evaluation code so that the release depends only on the public
Aurora package.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import torch


def preserve_coordinate_precision(Batch: Any, Metadata: Any) -> None:
    """Keep latitude/longitude in FP32 when Aurora casts a Batch to BF16.

    Aurora's Batch.type casts every tensor, including longitude; BF16 rounds the last ERA5 longitude to 360, which
    violates Aurora's coordinate check. This process-local override casts only the feature tensors.
    """
    def type_keep_coordinates(self: Any, dtype: torch.dtype) -> Any:
        return Batch(
            surf_vars={k: v.type(dtype) for k, v in self.surf_vars.items()},
            static_vars={k: v.type(dtype) for k, v in self.static_vars.items()},
            atmos_vars={k: v.type(dtype) for k, v in self.atmos_vars.items()},
            metadata=Metadata(lat=self.metadata.lat, lon=self.metadata.lon, atmos_levels=self.metadata.atmos_levels,
                              time=self.metadata.time, rollout_step=self.metadata.rollout_step),
        )
    Batch.type = type_keep_coordinates


def prepare(model: torch.nn.Module, batch: Any) -> Any:
    """Apply the model's batch transform, crop to the patch grid and move tensors to the model's device and dtype,
    keeping latitude/longitude in FP32."""
    prepared = model.batch_transform_hook(batch).crop(model.patch_size)
    parameter = next(model.parameters())
    metadata = dataclasses.replace(
        prepared.metadata,
        lat=prepared.metadata.lat.to(device=parameter.device, dtype=torch.float32),
        lon=prepared.metadata.lon.to(device=parameter.device, dtype=torch.float32),
    )
    return type(prepared)(
        static_vars={k: v.to(device=parameter.device, dtype=parameter.dtype) for k, v in prepared.static_vars.items()},
        surf_vars={k: v.to(device=parameter.device, dtype=parameter.dtype) for k, v in prepared.surf_vars.items()},
        atmos_vars={k: v.to(device=parameter.device, dtype=parameter.dtype) for k, v in prepared.atmos_vars.items()},
        metadata=metadata,
    )
