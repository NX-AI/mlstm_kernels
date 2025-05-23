#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
from torch import nn

from . import (
    get_mlstm_kernel,
    get_mlstm_sequence_kernel,
    get_mlstm_step_kernel,
)
from .kernel_wrappers import (
    wrap_chunkwise__arbitrary_sequence_length,
    wrap_chunkwise__pad_zeros,
)

ChunkwiseKernelType = Literal[
    "chunkwise--native_autograd",
    "chunkwise--native_custbw",
    "chunkwise--triton_limit_chunk",
    "chunkwise--triton_xl_chunk",
    "chunkwise--triton_xl_chunk_siging",
    "parallel--native_autograd",
    "parallel--native_custbw",
    "parallel--native_stablef_autograd",
    "parallel--native_stablef_custbw",
    "parallel--triton_limit_headdim",
    "parallel--native_siging_autograd",
    "parallel--native_siging_custbw",
]
SequenceKernelType = Literal["native_sequence__native", "native_sequence__triton"]
StepKernelType = Literal["native", "triton"]

DtypeType = Literal["float32", "bfloat16", "float16"]

BackendModeType = Literal["train", "train_with_padding", "inference"]


@dataclass
class mLSTMBackendConfig:
    chunkwise_kernel: ChunkwiseKernelType = "chunkwise--native_autograd"
    """The chunkwise kernel to use for chunkwise parallel processing of the sequence.
    This kernel is used for training.
    Also supports fully parallel (i.e. quadratic) backends for comparison.
    """
    sequence_kernel: SequenceKernelType = "native_sequence__native"
    """The sequence kernel to use for processing sequneces step-by-step.
    Used only for parts of the prefill sequence in inference mode.
    """
    step_kernel: StepKernelType = "native"
    """The step kernel to use for processing a single step.
    Used for generation in inference mode.
    """
    mode: BackendModeType = "train"
    """The mode of operation for the backend. Determines how the `forward` method behaves.
    """
    chunk_size: int = 64
    """The chunk size of the chunkwise kernel.
    If the mode is 'train_with_padding', this is the inputs are padded to multiples of this size.
    """
    return_last_states: bool = True
    """Whether to return the last states of the sequence in training mode.
    Inference mode always returns the last states.
    """
    autocast_kernel_dtype: DtypeType = "bfloat16"
    """The dtype to use for autocast behavior in the kernel.
    If autocast is enabled all inputs are cast to this dtype before the kernel is called.
    """
    eps: float = 1e-6
    """Epsilon value for numerical stability in the kernel."""
    inference_state_dtype: DtypeType = "float32"
    """The dtype to use for the state tensors in inference mode."""
    normalize_siging: bool = True
    """Whether to normalize the combination matrix C in siging kernels.
    This is only relevant for chunkwise siging kernels (mLSTM with sigmoid input gate).
    For other kernels (i.e. mLSTM with exponential input gate) it will be ignored.
    """

    def __post_init__(self):
        if self.return_last_states and "parallel" in self.chunkwise_kernel:
            raise ValueError(
                "return_last_states=True is not supported with parallel kernels."
            )
        if self.return_last_states and self.mode == "train_with_padding":
            raise ValueError(
                "return_last_states=True is not supported with train_with_padding mode."
            )


class mLSTMBackend(nn.Module):
    """mLSTM Backend Module for PyTorch.

    This module wraps the mLSTM kernels and provides a high-level interface for training and inference.
    """

    config_class = mLSTMBackendConfig

    def __init__(self, config: mLSTMBackendConfig):
        super().__init__()
        self.config = config
        self.chunkwise_kernel_fn = get_mlstm_kernel(config.chunkwise_kernel)
        self.sequence_kernel_fn = get_mlstm_sequence_kernel(config.sequence_kernel)
        self.step_kernel_fn = get_mlstm_step_kernel(config.step_kernel)

        self._inference_fn = partial(
            wrap_chunkwise__arbitrary_sequence_length,
            mlstm_chunkwise_kernel=self.chunkwise_kernel_fn,
            mlstm_sequence_kernel=partial(
                self.sequence_kernel_fn,
                dtype_state=getattr(torch, config.inference_state_dtype),
            ),
            mlstm_step_kernel=partial(
                self.step_kernel_fn,
                dtype_state=getattr(torch, config.inference_state_dtype),
            ),
            chunk_size=config.chunk_size,
            eps=config.eps,
            autocast_kernel_dtype=getattr(torch, config.autocast_kernel_dtype),
            return_last_states=True,
            dtype_state=config.inference_state_dtype,
        )

        train_kernel_fn = partial(
            self.chunkwise_kernel_fn,
            autocast_kernel_dtype=getattr(torch, config.autocast_kernel_dtype),
            eps=config.eps,
            chunk_size=config.chunk_size,
        )
        if "with_padding" in config.mode:
            train_kernel_fn = partial(
                wrap_chunkwise__pad_zeros, mlstm_chunkwise_kernel=train_kernel_fn
            )
        self._train_fn = train_kernel_fn

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        i: torch.Tensor,
        f: torch.Tensor,
        c_initial: torch.Tensor = None,
        n_initial: torch.Tensor = None,
        m_initial: torch.Tensor = None,
        return_last_states: bool = None,
        mode: Literal["train", "inference"] = None,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        """Forward pass of the mLSTM backend.

        Depending on the configured mode, this method will call the appropriate kernel function.

        Args:
            q: The query tensor of shape (B, NH, S, DHQK).
            k: The key tensor of shape (B, NH, S, DHQK).
            v: The value tensor of shape (B, NH, S, DHHV).
            i: The input gate preactivation tensor of shape (B, NH, S).
            f: The forget gate preactivation tensor of shape (B, NH, S).
            c_initial: The initial cell state tensor of shape (B, NH, DHQK, DHHV).
                                                Defaults to None.
            n_initial: The initial hidden state tensor of shape (B, NH, DHQK). Defaults to None.
            m_initial: The initial memory tensor of shape (B, NH, 1). Defaults to None.
            return_last_states: Whether to return the last states of the sequence. Defaults to None.
                                                If None, the value from the config is used.

        Returns:
            hidden states of shape (B, NH, S, DHHV)
            hidden states and last states the last states are the cell state c (B, NH, DHQK, DHHV),
            the normalizer state n (B, NH, DHQK), and the max state m (B, NH, 1)
        """
        if mode is None:
            mode = self.config.mode

        if "train" in mode:
            if return_last_states is None:
                return_last_states = self.config.return_last_states

            if self.config.mode == "train_with_padding":
                assert not return_last_states, "return_last_states=True is not supported with train_with_padding mode."

            train_fn_args = {
                "q": q,
                "k": k,
                "v": v,
                "i": i,
                "f": f,
                "c_initial": c_initial,
                "n_initial": n_initial,
                "m_initial": m_initial,
                "return_last_states": return_last_states,
            }
            if "siging" in self.config.chunkwise_kernel:
                train_fn_args["normalize"] = self.config.normalize_siging

            return self._train_fn(
                **train_fn_args,
            )

        elif "inference" in mode:
            # inference mode always returns the last states
            return self._inference_fn(
                q=q,
                k=k,
                v=v,
                i=i,
                f=f,
                c_initial=c_initial,
                n_initial=n_initial,
                m_initial=m_initial,
            )
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

    def extra_repr(self) -> str:
        return f"{self.config}"
