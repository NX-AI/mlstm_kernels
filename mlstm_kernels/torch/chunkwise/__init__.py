#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from .native import mlstm_chunkwise__native_autograd, mlstm_chunkwise__native_custbw

registry = {
    "native_autograd": mlstm_chunkwise__native_autograd,
    "native_custbw": mlstm_chunkwise__native_custbw,
}

try:
    from .triton_limit_chunk import mlstm_chunkwise__limit_chunk
    from .triton_xl_chunk import mlstm_chunkwise__xl_chunk
    from .triton_xl_chunk_siging import mlstm_siging_chunkwise__xl_chunk

    registry.update({
        "triton_limit_chunk": mlstm_chunkwise__limit_chunk,
        "triton_xl_chunk": mlstm_chunkwise__xl_chunk,
        "triton_xl_chunk_siging": mlstm_siging_chunkwise__xl_chunk,
    })
except ModuleNotFoundError:
    # triton unavailable (CPU-only / macOS)
    pass
