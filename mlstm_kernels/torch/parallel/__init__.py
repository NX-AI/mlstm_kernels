#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from .native import mlstm_parallel__native_autograd, mlstm_parallel__native_custbw
from .native_siging import (
    mlstm_siging_parallel__native_autograd,
    mlstm_siging_parallel__native_custbw,
)
from .native_stablef import (
    mlstm_parallel__native_stablef_autograd,
    mlstm_parallel__native_stablef_custbw,
)
from .triton_limit_headdim import mlstm_parallel__limit_headdim

registry = {
    "native_autograd": mlstm_parallel__native_autograd,
    "native_custbw": mlstm_parallel__native_custbw,
    "native_stablef_autograd": mlstm_parallel__native_stablef_autograd,
    "native_stablef_custbw": mlstm_parallel__native_stablef_custbw,
    "triton_limit_headdim": mlstm_parallel__limit_headdim,
    "native_siging_autograd": mlstm_siging_parallel__native_autograd,
    "native_siging_custbw": mlstm_siging_parallel__native_custbw,
}