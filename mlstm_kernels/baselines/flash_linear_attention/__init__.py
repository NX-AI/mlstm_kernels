
from .simple_gla import chunk_simple_gla as simple_gla_triton
from .gla import chunk_gla as gla_triton
from .gla import fused_chunk_gla as fused_gla_triton
from .gla import fused_recurrent_gla as fused_recurrent_gla_triton


registry = {"triton_simple_gla": simple_gla_triton, 
            "triton_gla": gla_triton,
            "triton_fused_gla": fused_gla_triton,
            "triton_fused_recurrent_gla": fused_recurrent_gla_triton}