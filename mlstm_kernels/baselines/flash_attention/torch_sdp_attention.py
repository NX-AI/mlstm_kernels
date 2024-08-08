import torch

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

def attention_causal_pt_fa2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return scaled_dot_product_attention(query, key, value, scale=scale)
    
def attention_causal_pt_cudnn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return scaled_dot_product_attention(query, key, value, scale=scale)
    
def attention_causal_pt_math(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.MATH):
        return scaled_dot_product_attention(query, key, value, scale=scale)
    
def attention_causal_pt_efficient(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return scaled_dot_product_attention(query, key, value, scale=scale)