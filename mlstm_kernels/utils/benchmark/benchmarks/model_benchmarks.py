import contextlib
import logging
import typing
from dataclasses import dataclass
from pprint import pprint
from typing import Any

import torch

from ..cuda_graphs import compile_with_cuda_graphs
from ..param_handling import ModelSpec
from .interface import BenchmarkFnContextManagerCfgType, ModelBenchmarkInterface

LOGGER = logging.getLogger(__name__)


@dataclass
class mLSTMSimpleModelBenchmark(ModelBenchmarkInterface):
    model_name: str = "mLSTM"

    embedding_dim: int = 256
    num_heads: int = 4
    num_blocks: int = 2
    vocab_size: int = 1000
    # feedforward
    ffn_proj_factor: float = 2.6667
    ffn_round_up_to_multiple_of: int = 64
    # mlstm
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    # mlstm backend
    chunkwise_kernel: str = "chunkwise--triton_xl_chunk"
    sequence_kernel: str = "native_sequence__native"
    step_kernel: str = "native"
    mode: str = "inference"
    chunk_size: int = 128

    inference_state_dtype: str = "float32"
    autocast_kernel_dtype: str = "bfloat16"

    # benchmark
    # amp does not work with torch compile
    amp_enabled: bool = False
    amp_dtype: str = "bfloat16"
    weight_dtype: str = "float32"

    use_torch_compile_model: bool = True
    use_torch_compile_generate: bool = False
    use_cuda_graphs_model: bool = False
    use_cuda_graphs_generate: bool = False
    cuda_graph_warmups: int = 3

    batch_size: int = 1
    prefill_length: int = 128
    generation_length: int = 1

    def setup_model(self) -> None:
        from mlstm_simple.model import mLSTM, mLSTMConfig

        mlstm_config = mLSTMConfig(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            vocab_size=self.vocab_size,
            ffn_proj_factor=self.ffn_proj_factor,
            ffn_round_up_to_multiple_of=self.ffn_round_up_to_multiple_of,
            qk_dim_factor=self.qk_dim_factor,
            v_dim_factor=self.v_dim_factor,
            chunkwise_kernel=self.chunkwise_kernel,
            sequence_kernel=self.sequence_kernel,
            step_kernel=self.step_kernel,
            mode=self.mode,
            chunk_size=self.chunk_size,
            inference_state_dtype=self.inference_state_dtype,
            autocast_kernel_dtype=self.autocast_kernel_dtype,
            return_last_states=True,
        )

        self.model = mLSTM(mlstm_config).to(
            dtype=getattr(torch, self.weight_dtype), device=torch.device(self.device)
        )

        LOGGER.info(f"Setting up model: {self.model}")
        LOGGER.info(f"Model config: {pprint(mlstm_config)}")

        def count_num_params(model):
            return sum(p.numel() for p in model.parameters())

        LOGGER.info(f"Model number of parameters: {count_num_params(self.model)}")

        if self.use_torch_compile_model:
            # Note: reduce-overhead gives torch.compile error (inplace copy: copy_ shows up in log, but unclear what the reason is)
            self.model = torch.compile(
                self.model, dynamic=False, fullgraph=True, mode="default"
            )

    def setup_benchmark(self) -> None:
        if self.model is None:
            self.setup_model()

        from mlstm_simple.generate import generate_tokens

        # setup prefill inputs
        if self.prefill_length > 0:
            self.prefill_tokens = torch.randint(
                low=0, high=self.vocab_size, size=(self.batch_size, self.prefill_length)
            ).to(device=torch.device(self.device))
        else:
            self.prefill_tokens = None
        pf_shape = (
            self.prefill_tokens.shape if self.prefill_tokens is not None else None
        )
        LOGGER.info(
            f"Prefill tokens shape: {pf_shape}, Generating {self.generation_length} tokens."
        )

        assert self.benchmark_fn_context_manager in typing.get_args(
            BenchmarkFnContextManagerCfgType
        ), (
            f"Invalid benchmark_fn_context_manager: {self.benchmark_fn_context_manager},",
            f" expected one of {typing.get_args(BenchmarkFnContextManagerCfgType)}",
        )

        if self.benchmark_fn_context_manager == "none":
            benchmark_fn_context_manager = contextlib.nullcontext
        elif self.benchmark_fn_context_manager == "no_grad":
            benchmark_fn_context_manager = torch.no_grad
        elif self.benchmark_fn_context_manager == "inference_mode":
            benchmark_fn_context_manager = torch.inference_mode


        self.generate_fn = generate_tokens
        if self.use_torch_compile_generate:
            self.generate_fn = torch.compile(
                self.generate_fn, dynamic=False, fullgraph=False, mode="reduce-overhead"
            )

        self.generated_tokens = torch.empty(
            (self.batch_size, self.generation_length + 1), dtype=torch.long
        ).to(device=torch.device(self.device))

        # setup generation function
        if not self.use_cuda_graphs_model:
            def llm_forward(tokens, state):
                return self.model(
                    x=tokens,
                    state=state,
                )
        else:
            import jax
            LOGGER.info("Setting up model with CUDA graphs on forward function.")
            input_tokens = self.generated_tokens.new_empty((self.batch_size, 1))
            # Infer state shape.
            _, state = self.model(x=input_tokens, state=None)
            input_state = jax.tree.map(lambda x: torch.empty_like(x), state)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(self.cuda_graph_warmups):
                    logits, output_state = self.model(x=input_tokens, state=input_state)
                s.synchronize()
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            torch.cuda.current_stream().wait_stream(s)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits, output_state = self.model(x=input_tokens, state=input_state)
            
            def llm_forward(tokens, state):
                input_tokens.copy_(tokens)
                if state is None:
                    jax.tree.map(lambda x: x.zero_(), input_state)
                else:
                    jax.tree.map(lambda x, y: x.copy_(y), input_state, state)
                graph.replay()
                return logits, output_state
            

        def benchmark_fn():
            with torch.autocast(
                device_type=torch.device(self.device).type, enabled=self.amp_enabled
            ):
                with benchmark_fn_context_manager():
                    generated_tokens, state = self.generate_fn(
                        llm_forward=llm_forward,
                        prefill_tokens=self.prefill_tokens,
                        max_length=self.generation_length,
                        batch_size_no_prefill=self.batch_size,
                        generated_tokens=self.generated_tokens,
                        device=self.device,
                    )
                    if generated_tokens is not None:
                        # +1 since in generate there is an beginning of sequence token added always
                        assert (
                            tuple(generated_tokens.shape)
                            == (self.batch_size, self.generation_length + 1)
                        ), f"Generated tokens shape: {tuple(generated_tokens.shape)}, expected {(self.batch_size, self.generation_length+1)}"

        if self.use_cuda_graphs_generate:
            LOGGER.info("Setting up benchmark with CUDA graphs on benchmark function.")
            graph = compile_with_cuda_graphs(benchmark_fn, warmups=self.cuda_graph_warmups)
            self.benchmark_fn = lambda: graph.replay()
        else:
            self.benchmark_fn = benchmark_fn

    def available_kernels(self) -> list[str]:
        return ["mlstm_simple"]


def create_mlstm_model_benchmark(
    model_spec: ModelSpec, param_dict: dict[str, Any]
) -> mLSTMSimpleModelBenchmark:
    mlstm_model_benchmark = mLSTMSimpleModelBenchmark()

    mlstm_model_benchmark.model_name = model_spec.model_name
    mlstm_model_benchmark.amp_enabled = model_spec.amp_enabled
    mlstm_model_benchmark.amp_dtype = model_spec.amp_dtype
    mlstm_model_benchmark.weight_dtype = model_spec.weight_dtype
    mlstm_model_benchmark.use_torch_compile_model = model_spec.use_torch_compile_model

    mlstm_model_benchmark.set_params(param_dict)

    return mlstm_model_benchmark
