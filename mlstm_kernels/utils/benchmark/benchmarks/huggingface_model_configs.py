from transformers.models.falcon_mamba import FalconMambaConfig
from transformers.models.llama import LlamaConfig
from transformers.models.mamba2 import Mamba2Config
from transformers.models.mistral import MistralConfig
from transformers.models.xlstm import xLSTMConfig
from transformers.models.zamba import ZambaConfig

ministral8b_config = {
    "architectures": ["MistralForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "max_position_embeddings": 32768,
    "model_type": "mistral",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_theta": 100000000.0,
    "sliding_window": 32768,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.46.0.dev0",
    "use_cache": True,
    "vocab_size": 131072,
}

codestral_mamba_config = {
    "_name_or_path": "/raid/pablo/codestral-hf-good/",
    "architectures": ["Mamba2ForCausalLM"],
    "bos_token_id": 0,
    "chunk_size": 256,
    "conv_kernel": 4,
    "eos_token_id": 0,
    "expand": 2,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.1,
    "intermediate_size": 8192,
    "layer_norm_epsilon": 1e-05,
    "model_type": "mamba2",
    "n_groups": 8,
    "norm_before_gate": True,
    "num_heads": 128,
    "num_hidden_layers": 64,
    "pad_token_id": 0,
    "rescale_prenorm_residual": False,
    "residual_in_fp32": True,
    "rms_norm": True,
    "state_size": 128,
    "tie_word_embeddings": False,
    "time_step_floor": 0.0001,
    "time_step_init_scheme": "random",
    "time_step_limit": (0.0, float("inf")),
    "time_step_max": 0.1,
    "time_step_min": 0.001,
    "time_step_rank": 256,
    "time_step_scale": 1.0,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.44.0.dev0",
    "use_bias": False,
    "use_cache": True,
    "use_conv_bias": True,
    "vocab_size": 32768,
}

llama_2_config = {
    "_name_or_path": "meta-llama/Llama-2-7b-hf",
    "architectures": ["LlamaForCausalLM"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "transformers_version": "4.31.0.dev0",
    "use_cache": True,
    "vocab_size": 32000,
}

llama_3_1_config = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.43.0.dev0",
    "use_cache": True,
    "vocab_size": 128256,
}

falcon_mamba_config = {
    "_name_or_path": "./",
    "architectures": ["FalconMambaForCausalLM"],
    "bos_token_id": 0,
    "conv_kernel": 4,
    "eos_token_id": 11,
    "expand": 16,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.1,
    "intermediate_size": 8192,
    "layer_norm_epsilon": 1e-05,
    "model_type": "falcon_mamba",
    "num_hidden_layers": 64,
    "pad_token_id": 11,
    "rescale_prenorm_residual": False,
    "residual_in_fp32": True,
    "state_size": 16,
    "tie_word_embeddings": False,
    "time_step_floor": 0.0001,
    "time_step_init_scheme": "random",
    "time_step_max": 0.1,
    "time_step_min": 0.001,
    "time_step_rank": 256,
    "time_step_scale": 1.0,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.43.0.dev0",
    "use_bias": False,
    "use_cache": True,
    "use_conv_bias": True,
    "vocab_size": 65024,
}

zamba_config = {
    "add_bias_linear": False,
    "architectures": ["Zamba2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "conv_dimension": 4,
    "eos_token_id": 2,
    "expansion_factor": 2,
    "ffn_hidden_size": 14336,
    "ft_lora": False,
    "gated_linear_unit": True,
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "kv_channels": 112,
    "layers_block_type": [
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "mamba",
        "hybrid",
        "mamba",
        "mamba",
        "mamba",
    ],
    "lora_rank": 128,
    "mamba_headdim": 64,
    "mamba_ngroups": 2,
    "max_position_embeddings": 4096,
    "model_type": "zamba2",
    "num_attention_heads": 32,
    "num_hidden_layers": 81,
    "num_key_value_heads": 32,
    "num_logits_to_keep": 1,
    "num_mem_blocks": 2,
    "num_query_groups": 32,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000,
    "sliding_window": None,
    "state_size": 64,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.43.0.dev0",
    "use_cache": True,
    "use_mamba_kernels": True,
    "use_mem_rope": True,
    "use_shared_attention_lora": False,
    "use_shared_block_lora": True,
    "vocab_size": 32000,
}


hf_model_registry = {
    "llama2": (LlamaConfig, llama_2_config),
    "llama3": (LlamaConfig, llama_3_1_config),
    "ministral8b": (MistralConfig, ministral8b_config),
    "codestral_mamba": (Mamba2Config, codestral_mamba_config),
    "falcon_mamba": (FalconMambaConfig, falcon_mamba_config),
    "zamba2": (ZambaConfig, zamba_config),
    "xlstm": (xLSTMConfig, {}),
}