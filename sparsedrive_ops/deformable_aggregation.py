import os

import torch
from torch.autograd.function import Function, once_differentiable
from torch.utils.cpp_extension import load


_EXT = None
_EXT_LOAD_ERROR = None


def _load_ext():
    global _EXT, _EXT_LOAD_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_LOAD_ERROR is not None:
        return None
    if not torch.cuda.is_available():
        _EXT_LOAD_ERROR = RuntimeError("CUDA is not available")
        return None

    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(this_dir, "src")
    sources = [
        os.path.join(src_dir, "deformable_aggregation.cpp"),
        os.path.join(src_dir, "deformable_aggregation_cuda.cu"),
    ]
    try:
        _EXT = load(
            name="morai_sparsedrive_deformable_aggregation_ext",
            sources=sources,
            extra_cuda_cflags=[
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
            verbose=os.getenv("MORAI_DAF_VERBOSE", "0") == "1",
        )
    except Exception as exc:  # pragma: no cover - depends on local CUDA toolchain.
        _EXT_LOAD_ERROR = exc
        _EXT = None
    return _EXT


def is_deformable_aggregation_available():
    return _load_ext() is not None


class DeformableAggregationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        ext = _load_ext()
        if ext is None:
            raise RuntimeError(f"deformable aggregation CUDA op unavailable: {_EXT_LOAD_ERROR}")

        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = ext.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        ext = _load_ext()
        if ext is None:
            raise RuntimeError(f"deformable aggregation CUDA op unavailable: {_EXT_LOAD_ERROR}")

        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        ext.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )


def deformable_aggregation_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    return DeformableAggregationFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )


def feature_maps_format(feature_maps):
    """
    Convert multi-camera, multi-level FPN tensors to the official SparseDrive
    CUDA op layout.

    input : list of [B, num_cams, C, H, W]
    output:
      col_feats         [B, sum(num_cams * H_l * W_l), C]
      spatial_shape     [num_cams, num_levels, 2]
      scale_start_index [num_cams, num_levels]
    """
    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []
    col_feats = []

    for feat in feature_maps:
        spatial_shape.append(feat.shape[-2:])
        col_feats.append(torch.reshape(feat, (bs, num_cams, feat.shape[2], -1)))

    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    spatial_shape = [spatial_shape] * num_cams
    spatial_shape = torch.tensor(
        spatial_shape,
        dtype=torch.int64,
        device=col_feats.device,
    )
    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    scale_start_index = torch.cat(
        [torch.tensor([0], device=scale_start_index.device, dtype=scale_start_index.dtype),
         scale_start_index[:-1]]
    )
    scale_start_index = scale_start_index.reshape(num_cams, -1)
    return col_feats, spatial_shape, scale_start_index
