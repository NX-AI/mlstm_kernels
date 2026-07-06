#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import functools

import numpy as np
import torch

def torch2triton_dtype(dtype):
    import triton.language as tl

    _torch_to_triton_dtype = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    return _torch_to_triton_dtype[dtype]


def dtype2str(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.float64:
        return "fp64"
    elif dtype == torch.bfloat16:
        return "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(
            ctx,
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{
                k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
                for k, v in kwargs.items()
            },
        )

    return wrapper


def contiguous_noctx(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{
                k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
                for k, v in kwargs.items()
            },
        )

    return wrapper

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(dtype=torch.float64).numpy()


def tensor_or_none(x):
    return x if x is None else torch.tensor(x)


def int_or_none(x):
    return x if x is None else int(x)
