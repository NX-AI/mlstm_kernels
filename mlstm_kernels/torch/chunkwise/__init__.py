from .native import mlstm_chunkwise__native_autograd, mlstm_chunkwise__native_custbw
from .triton_limit_chunk import mlstm_chunkwise__limit_chunk
from .triton_xl_chunk import mlstm_chunkwise__xl_chunk

registry = {
    "native_autograd": mlstm_chunkwise__native_autograd,
    "native_custbw": mlstm_chunkwise__native_custbw,
    "triton_limit_chunk": mlstm_chunkwise__limit_chunk,
    "triton_xl_chunk": mlstm_chunkwise__xl_chunk,
}