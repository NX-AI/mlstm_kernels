#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""Contains code to produce the paper figures for the xLSTM 7B paper"""

import copy
import pickle
from pathlib import Path
from typing import Union

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from git import Optional
from matplotlib import pyplot as plt
from plot_results_for_paper import create_runtime_bar_plot

model_colors = {
    "mlstm_simple": mpl.colormaps["tab10"].colors[0],
    "xlstm": mpl.colormaps["tab10"].colors[1],
    "llama2": mpl.colormaps["tab10"].colors[2],
    "llama3": mpl.colormaps["tab10"].colors[3],
    "ministral8b": mpl.colormaps["tab10"].colors[5],
    "codestral_mamba": mpl.colormaps["tab10"].colors[6],
    "falcon_mamba": mpl.colormaps["tab10"].colors[4],
    "zamba2": mpl.colormaps["tab10"].colors[7],
}

xlstm_colors = {
    "llama3": "#165b89ff",
    "llama2": "#80a8b3ff",
    # "xLSTM": "#cc4391ff",
    "xlstm": "#861657ff",
    "xlstm_longctx": "#cc4391ff",
    "codestral_mamba": "#d08814ff",
    "falcon_mamba": "#ffd449ff",
    "RWKV4": "#145815ff",
    "rwkv6": "#548c2f",
}

model_labels = {
    "mlstm_simple": "mLSTM simple",
    "xlstm": "xLSTM 7B",
    "Default": "Default",
    "xlstm_longctx": "xLSTM 7B LCTX",
    "LCTX": "LCTX",
    "llama2": "Llama 2 7B",
    "llama3": "Llama 3 8B",
    "ministral8b": "Ministral8B",
    "codestral_mamba": "CodestralMamba 7B",
    "falcon_mamba": "FalconMamba 7B",
    "zamba2": "Zamba2",
    "rwkv5": "RWKV-5 7B",
    "rwkv6": "RWKV-6 7B",
}


kernel_colors = {
    "torch_flashattn": "#165b89",
    "torch_cudnn": "#439ebd",
    "flashattn3": "#80a8b3",
    "mamba2_noconv": "#d08814",
    "mamba2": "#d08814",
    "mamba": "#ffd449",
    "chunk_simple_gla": "#4fb72e",
    "fused_chunk_gla": "#548c2f",
    "chunk_gla": "#548c2f",
    "mlstmexp_triton_limit_chunk": "#f0acb9",
    "mlstmexp_torch_native": "#e52e66",
    "mlstmexp_triton_xl_chunk": "#e52e66",
    "mlstmsig_triton_xl_chunk": "#9a3c73",
}

linestyle_mapping = {
    "__tcm__": {"linestyle": "--", "label": ""},
}

style_dict = {
    "mlstm_simple": {
        "color": model_colors["mlstm_simple"],
        "label": model_labels["mlstm_simple"],
    },
    "xlstm": {"color": xlstm_colors["xlstm"], "label": model_labels["xlstm"]},
    "Default": {"color": xlstm_colors["xlstm"], "label": model_labels["Default"]},
    "xlstm_longctx": {
        "color": xlstm_colors["xlstm_longctx"],
        "label": model_labels["xlstm_longctx"],
    },
    "LCTX": {"color": xlstm_colors["xlstm_longctx"], "label": model_labels["LCTX"]},
    "llama2": {"color": xlstm_colors["llama2"], "label": model_labels["llama2"]},
    "llama3": {"color": xlstm_colors["llama3"], "label": model_labels["llama3"]},
    "ministral8b": {
        "color": model_colors["ministral8b"],
        "label": model_labels["ministral8b"],
    },
    "codestral_mamba": {
        "color": xlstm_colors["codestral_mamba"],
        "label": model_labels["codestral_mamba"],
    },
    "falcon_mamba": {
        "color": xlstm_colors["falcon_mamba"],
        "label": model_labels["falcon_mamba"],
    },
    "zamba2": {"color": model_colors["zamba2"], "label": model_labels["zamba2"]},
    "rwkv5": {"color": xlstm_colors["RWKV4"], "label": model_labels["rwkv5"]},
    "rwkv6": {"color": xlstm_colors["rwkv6"], "label": model_labels["rwkv6"]},
    "No Softcap": {"color": "#165b89", "label": "No Softcap"},
    "With Softcap": {"color": "#80a8b3", "label": "With Softcap"},
    "Exponential 76k Steps": {"color": "#165b89", "label": "Exponential 76k Steps"},
    "Exponential 150k Steps": {"color": "#80a8b3", "label": "Exponential 150k Steps"},
    "Cosine 76k Steps": {"color": "#d08814", "label": "Cosine 76k Steps"},
    "Cosine 150k Steps": {"color": "#ffd449", "label": "Cosine 150k Steps"},
    "Pre-Up Projection, RMSNorm-RMSNorm": {
        "color": "#165b89",
        "label": "Pre-Up Projection, RMSNorm-RMSNorm",
    },
    "Pre-Up Projection, RMSNorm-LayerNorm": {
        "color": "#80a8b3",
        "label": "Pre-Up Projection, RMSNorm-LayerNorm",
    },
    "Pre-Up Projection, LayerNorm-LayerNorm": {
        "color": "#d08814",
        "label": "Pre-Up Projection, LayerNorm-LayerNorm",
    },
    "Post-Up Projection, RMSNorm-LayerNorm": {
        "color": "#851656",
        "label": "Post-Up Projection, RMSNorm-LayerNorm",
    },
    "BiasInit 0": {"color": "#165b89", "label": "BiasInit 0"},
    "BiasInit -2": {"color": "#80a8b3", "label": "BiasInit -2"},
    "BiasInit -5": {"color": "#d08814", "label": "BiasInit -5"},
    # "BiasInit -10": {"color": "#ffd449", "label": "BiasInit -10"},
    "BiasInit -10": {"color": xlstm_colors["xlstm"], "label": "BiasInit -10"},
}


def select_columns(df, selected_columns, keep_col_regex: str):
    """Select columns from a DataFrame and keep columns matching a regex."""
    keep_col = df.filter(regex=keep_col_regex)
    selected_df = df[selected_columns.values()]
    selected_df.columns = selected_columns.keys()
    selected_df = pd.concat([keep_col, selected_df], axis=1)
    return selected_df


def read_ttf_data():
    with open("notebooks/plots_7B_model_benchmark/ttft_raw_data.p", "rb") as f:
        raw_data = pickle.load(f)

    ttft_1_df = raw_data["ttft_1"]
    ttft_100_df = raw_data["ttft_100"]

    selected_columns = {
        "llama3": "R--llama3__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False",
        "llama2": "R--llama2__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False",
        "falcon_mamba": "R--falcon_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False",
        "codestral_mamba": "R--codestral_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False",
        "xlstm": "R--xlstm__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False_isd-bfloat16_ed-4096_nh-8_nb-32_vs-50304_wm-fused_ck-chunkwise--triton_xl_chunk_sk-native_sequence__triton_step_fused_sk-triton_fused_cs-128_akd-bfloat16",
    }

    ttft_1_plot_df = select_columns(
        ttft_1_df, selected_columns, keep_col_regex=".*prefill.*"
    )

    ttft_100_plot_df = select_columns(
        ttft_100_df, selected_columns, keep_col_regex=".*prefill.*"
    )

    return ttft_1_plot_df, ttft_100_plot_df


def read_ttf_vllm_data():
    with open("notebooks/plots_7B_model_benchmark/ttft_raw_data_vllm.p", "rb") as f:
        raw_data = pickle.load(f)

    ttft_1_df = raw_data["ttft_1"]
    ttft_100_df = raw_data["ttft_100"]

    selected_columns = {
        "llama3": "R--llama3__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm",
        "llama2": "R--llama2__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm",
        "falcon_mamba": "R--falcon_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm",
        "codestral_mamba": "R--codestral_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm",
        "xlstm": "R--xlstm__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False_isd-bfloat16_ed-4096_nh-8_nb-32_vs-50304_wm-fused_ck-chunkwise--triton_xl_chunk_sk-native_sequence__triton_step_fused_sk-triton_fused_cs-128_akd-bfloat16",
    }

    ttft_1_plot_df = select_columns(
        ttft_1_df, selected_columns, keep_col_regex=".*prefill.*"
    )

    ttft_100_plot_df = select_columns(
        ttft_100_df, selected_columns, keep_col_regex=".*prefill.*"
    )

    return ttft_1_plot_df, ttft_100_plot_df


def read_gen_data():
    with open("notebooks/plots_7B_model_benchmark/gen_time_mem_data.p", "rb") as f:
        raw_data = pickle.load(f)

    gen_mem_df = raw_data["gen_mem_gb"]
    gen_time_df = raw_data["gen_time_seconds"]

    selected_columns_runtime = {
        "llama3": "R--llama3__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False",
        "llama2": "R--llama2__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False",
        "falcon_mamba": "R--falcon_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True",
        "codestral_mamba": "R--codestral_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True",
        "xlstm": "R--xlstm__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True_isd-bfloat16_ed-4096_nh-8_nb-32_vs-50304_wm-fused_ck-chunkwise--triton_xl_chunk_sk-native_sequence__triton_step_fused_sk-triton_fused_cs-128_akd-bfloat16",
    }
    selected_columns_memory = {
        "llama2": "M--llama2__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False",
        "llama3": "M--llama3__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False",
        "falcon_mamba": "M--falcon_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True",
        "codestral_mamba": "M--codestral_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True",
        "xlstm": "M--xlstm__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True_isd-bfloat16_ed-4096_nh-8_nb-32_vs-50304_wm-fused_ck-chunkwise--triton_xl_chunk_sk-native_sequence__triton_step_fused_sk-triton_fused_cs-128_akd-bfloat16",
    }

    gen_time_plot_df = select_columns(
        gen_time_df, selected_columns_runtime, keep_col_regex=".*generation.*"
    )

    gen_mem_plot_df = select_columns(
        gen_mem_df, selected_columns_memory, keep_col_regex=".*generation.*"
    )

    return gen_time_plot_df, gen_mem_plot_df


def read_gen_data2():
    # with open("notebooks/plots_7B_model_benchmark/gen_time_mem_data.p", "rb") as f:
    #     raw_data = pickle.load(f)

    with open("notebooks/plots_7B_model_benchmark/gen_time_mem_data_vllm.p", "rb") as f:
        raw_data_vllm = pickle.load(f)

    # gen_mem_df = raw_data["gen_mem_gb"]
    # gen_time_df = raw_data["gen_time_seconds"]
    gen_time_vllm_df = raw_data_vllm["gen_time_seconds"]

    selected_columns_runtime = {
        "llama3": "R--llama3__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False",
        "llama2": "R--llama2__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False",
        "falcon_mamba": "R--falcon_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True",
        "codestral_mamba": "R--codestral_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True",
        "xlstm": "R--xlstm__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True_isd-bfloat16_ed-4096_nh-8_nb-32_vs-50304_wm-fused_ck-chunkwise--triton_xl_chunk_sk-native_sequence__triton_step_fused_sk-triton_fused_cs-128_akd-bfloat16",
    }

    selected_columns_runtime_vllm = {
        "llama3": "R--llama3__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm",
        "llama2": "R--llama2__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm",
        "falcon_mamba": "R--falcon_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm",
        "codestral_mamba": "R--codestral_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm",
        "xlstm": "R--xlstm__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True_isd-bfloat16_ed-4096_nh-8_nb-32_vs-50304_wm-fused_ck-chunkwise--triton_xl_chunk_sk-native_sequence__triton_step_fused_sk-triton_fused_cs-128_akd-bfloat16",
    }

    # selected_columns_memory = {
    #     "llama2": "M--llama2__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False",
    #     "llama3": "M--llama3__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False",
    #     "falcon_mamba": "M--falcon_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True",
    #     "codestral_mamba": "M--codestral_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True",
    #     "xlstm": "M--xlstm__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-True_isd-bfloat16_ed-4096_nh-8_nb-32_vs-50304_wm-fused_ck-chunkwise--triton_xl_chunk_sk-native_sequence__triton_step_fused_sk-triton_fused_cs-128_akd-bfloat16",
    # }

    gen_time_plot_df = select_columns(
        gen_time_vllm_df, selected_columns_runtime, keep_col_regex=".*generation.*"
    )

    gen_time_vllm_plot_df = select_columns(
        gen_time_vllm_df, selected_columns_runtime_vllm, keep_col_regex=".*generation.*"
    )

    # gen_mem_plot_df = select_columns(
    #     gen_mem_df, selected_columns_memory, keep_col_regex=".*generation.*"
    # )

    return gen_time_plot_df, gen_time_vllm_plot_df  # , gen_mem_plot_df


def read_tokens_per_second_data():
    with open("notebooks/plots_7B_model_benchmark/ttft_raw_data.p", "rb") as f:
        raw_data = pickle.load(f)

    token_per_sec_df = raw_data["token_per_sec"]

    selected_columns = {
        "llama3": "R--llama3__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False",
        "llama2": "R--llama2__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False",
        "falcon_mamba": "R--falcon_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False",
        "codestral_mamba": "R--codestral_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False",
        "xlstm": "R--xlstm__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False_isd-bfloat16_ed-4096_nh-8_nb-32_vs-50304_wm-fused_ck-chunkwise--triton_xl_chunk_sk-native_sequence__triton_step_fused_sk-triton_fused_cs-128_akd-bfloat16",
    }
    token_per_sec_plot_df = select_columns(
        token_per_sec_df, selected_columns, keep_col_regex=".*prefill.*"
    )

    return token_per_sec_plot_df


def read_tokens_per_second_data_vllm():
    with open("notebooks/plots_7B_model_benchmark/ttft_raw_data_vllm.p", "rb") as f:
        raw_data = pickle.load(f)

    token_per_sec_df = raw_data["token_per_sec"]

    selected_columns = {
        "llama3": "R--llama3__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False",
        "llama2": "R--llama2__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False",
        "falcon_mamba": "R--falcon_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False",
        "codestral_mamba": "R--codestral_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False",
        "xlstm": "R--xlstm__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False_isd-bfloat16_ed-4096_nh-8_nb-32_vs-50304_wm-fused_ck-chunkwise--triton_xl_chunk_sk-native_sequence__triton_step_fused_sk-triton_fused_cs-128_akd-bfloat16",
    }

    selected_columns_vllm = {
        "llama3": "R--llama3__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm",
        "llama2": "R--llama2__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm",
        "falcon_mamba": "R--falcon_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm",
        "codestral_mamba": "R--codestral_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm",
        "xlstm": "R--xlstm__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False_isd-bfloat16_ed-4096_nh-8_nb-32_vs-50304_wm-fused_ck-chunkwise--triton_xl_chunk_sk-native_sequence__triton_step_fused_sk-triton_fused_cs-128_akd-bfloat16",
    }

    token_per_sec_plot_df = select_columns(
        token_per_sec_df, selected_columns, keep_col_regex=".*prefill.*"
    )

    token_per_sec_plot_vllm_df = select_columns(
        token_per_sec_df, selected_columns_vllm, keep_col_regex=".*prefill.*"
    )

    return token_per_sec_plot_df, token_per_sec_plot_vllm_df


def read_forward_throughput_data():
    with open("notebooks/plots_7B_model_benchmark/throughput_df.p", "rb") as f:
        plot_throughput_df = pickle.load(f)

    return plot_throughput_df


def read_forward_throughput_vllm_data():
    with open("notebooks/plots_7B_model_benchmark/throughput_vllm_df.p", "rb") as f:
        plot_throughput_df = pickle.load(f)

    return plot_throughput_df


def read_ruler_main_data():
    with open("notebooks/plots_7B_model_benchmark/avg_acc_ruler_main.pkl", "rb") as f:
        ruler_main_df = pickle.load(f)

    # Transform to pandas DataFrame
    ruler_main_df = pd.DataFrame(ruler_main_df)

    return ruler_main_df


def read_ruler_abl_data():
    with open("notebooks/plots_7B_model_benchmark/avg_acc_ruler_abl.pkl", "rb") as f:
        ruler_abl_df = pickle.load(f)

    # Transform to pandas DataFrame
    ruler_abl_df = pd.DataFrame(ruler_abl_df)

    return ruler_abl_df


def get_wandb_data(run_mapping: dict[str, str], key: str) -> pd.DataFrame:
    """
    Fetch a single metric from W&B runs and return aligned DataFrame.

    Args:
        run_mapping: Dictionary mapping run names to run paths (entity/project/run_id)
        key: Metric key to fetch from W&B

    Returns:
        DataFrame containing the metric for all runs, indexed by step
    """
    # Initialize W&B API
    api = wandb.Api()

    # Dictionary to collect series from each run
    run_series = {}

    # Collect data from each run
    for run_name, run_path in run_mapping.items():
        try:
            run = api.run(run_path)
            history_df = pd.DataFrame(run.scan_history(keys=[key, "_step"]))
            # history_df = run.history(keys=[key, '_step'], pandas=True)
            run_series[run_name] = pd.Series(
                history_df[key].values, index=history_df["_step"].values
            )
        except Exception as e:
            print(f"Error fetching data for run {run_name}: {str(e)}")
            continue

    # Combine all series into a DataFrame
    result_df = pd.DataFrame(run_series)

    return result_df


def prepare_lr_scheduler_data():
    # Define run mapping for this experiment
    run_mapping = {
        "exponential-76000steps": "xlstm/xlstm_jax/6umkgvtk",
        "exponential-150000steps": "xlstm/xlstm_jax/s10xk6rf",
        "cosine-76000steps": "xlstm/xlstm_jax/1zy4evam",
        "cosine-150000steps": "xlstm/xlstm_jax/bwgc7tr8",
    }

    # Define keys to fetch from W&B
    lr_key = "train/.optimizer/lr"
    ppl_key = "val/.dclm_perplexity"
    loss_key = "val/.dclm_loss"

    if not Path(
        "notebooks/plots_7B_model_benchmark/dumps/scheduler/lr_df.pkl"
    ).exists():
        dfs = get_wandb_data(run_mapping, keys=[lr_key, ppl_key, loss_key])

        # Access individual dataframes
        df_lr = dfs[lr_key]
        df_ppl = dfs[ppl_key]
        df_loss = dfs[loss_key]

        # Pickle dataframes
        df_lr.to_pickle("notebooks/plots_7B_model_benchmark/dumps/scheduler/lr_df.pkl")
        df_ppl.to_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/scheduler/ppl_df.pkl"
        )
        df_loss.to_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/scheduler/loss_df.pkl"
        )

    else:
        df_lr = pd.read_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/scheduler/lr_df.pkl"
        )
        df_ppl = pd.read_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/scheduler/ppl_df.pkl"
        )
        df_loss = pd.read_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/scheduler/loss_df.pkl"
        )

    return df_lr, df_ppl, df_loss


def prepare_softcap_data():
    # Get data from wandb api
    run_mapping = {
        "no softcap": "xlstm/xlstm_jax/9ow8tqnl",
        "with softcap": "xlstm/xlstm_jax/8egjdt0c",
    }

    # Define keys to fetch from W&B
    valid_loss_key = "val/.dclm_loss"
    grad_key = "train/.grad_norm_mean"

    if not Path(
        "notebooks/plots_7B_model_benchmark/dumps/softcap/valid_loss_df.pkl"
    ).exists():
        dfs = get_wandb_data(run_mapping, keys=[valid_loss_key, grad_key])

        # Access individual dataframes
        df_valid_loss = dfs[valid_loss_key]
        df_grad = dfs[grad_key]

        # Pickle dataframes
        df_valid_loss.to_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/softcap/valid_loss_df.pkl"
        )
        df_grad.to_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/softcap/grad_df.pkl"
        )
    else:
        df_valid_loss = pd.read_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/softcap/valid_loss_df.pkl"
        )
        df_grad = pd.read_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/softcap/grad_df.pkl"
        )

    return df_valid_loss, df_grad


def prepare_rmsnorm_data():
    run_mapping = {
        "Pre-Up Projection, RMSNorm-RMSNorm": "xlstm/xlstm_jax/hpnapw1i",
        "Pre-Up Projection, RMSNorm-LayerNorm": "xlstm/xlstm_jax/x21shgwg",
        "Pre-Up Projection, LayerNorm-LayerNorm": "xlstm/xlstm_jax/1r0wjk9v",
        "Post-Up Projection, RMSNorm-LayerNorm": "xlstm/xlstm_jax/exazu89h",
    }

    # Define keys to fetch from W&B
    valid_loss_key = "val/.dclm_loss"
    grad_key = "train/.grad_norm_mean"
    step_time_key = "train/.step_time"

    if not Path(
        "notebooks/plots_7B_model_benchmark/dumps/rmsnorm/valid_loss_df.pkl"
    ).exists():
        dfs = get_wandb_data(
            run_mapping, keys=[valid_loss_key, grad_key, step_time_key]
        )

        # Access individual dataframes
        df_valid_loss = dfs[valid_loss_key]
        df_grad = dfs[grad_key]
        df_steptime = dfs[step_time_key]

        # Pickle dataframes
        df_valid_loss.to_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/rmsnorm/valid_loss_df.pkl"
        )
        df_grad.to_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/rmsnorm/grad_df.pkl"
        )
        df_steptime.to_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/rmsnorm/steptime_df.pkl"
        )
    else:
        df_valid_loss = pd.read_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/rmsnorm/valid_loss_df.pkl"
        )
        df_grad = pd.read_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/rmsnorm/grad_df.pkl"
        )
        df_steptime = pd.read_pickle(
            "notebooks/plots_7B_model_benchmark/dumps/rmsnorm/steptime_df.pkl"
        )

    return df_valid_loss, df_grad, df_steptime


def prepare_bias_data():
    run_mapping = {
        "BiasInit 0": "xlstm/xlstm_jax/wfuc1fpq",
        "BiasInit -2": "xlstm/xlstm_jax/eowl52vx",
        "BiasInit -5": "xlstm/xlstm_jax/nxaspgyk",
        "BiasInit -10": "xlstm/xlstm_jax/2td17v6e",
    }

    # Define keys to fetch from W&B
    keys = ["val/.dclm_loss", "train/.grad_norm_max"]
    dfs = {}

    for key in keys:
        key_path = key.replace("/", "_").replace(".", "_")

        if not Path(
            f"notebooks/plots_7B_model_benchmark/dumps/bias/{key_path}.pkl"
        ).exists():
            dfs[key] = get_wandb_data(run_mapping, key=key)

            # Access individual dataframes
            df = dfs[key]

            # Pickle dataframes
            df.to_pickle(
                f"notebooks/plots_7B_model_benchmark/dumps/bias/{key_path}.pkl"
            )
        else:
            df = pd.read_pickle(
                f"notebooks/plots_7B_model_benchmark/dumps/bias/{key_path}.pkl"
            )
            dfs[key] = df

    return dfs


def savefig(fig, filename):
    dir = Path("./plots/")
    dir.mkdir(parents=True, exist_ok=True)
    for file_ending in ["pdf", "png", "svg"]:
        file = Path(f"./plots/plot_{filename}.{file_ending}")
        fig.savefig(file, dpi=300, bbox_inches="tight", pad_inches=0)


def plot_2col_legend(figsize):
    """Plots the legend for the 2 column paper figures"""
    # f, axes = plt.subplots(1, 2, figsize=figsize)
    # Create dummy lines

    fig, ax = plt.subplots()
    lines = [
        plt.Line2D([0], [0], color="red", label="Line 1"),
        plt.Line2D([0], [0], color="blue", label="Line 2"),
    ]

    # Create legend only
    plt.legend(handles=lines)

    # Remove the actual plot
    ax.set_axis_off()
    # After removing plot contents

    fig.tight_layout()  # Adjust layout
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())  # Get legend bounds
    fig.set_size_inches(bbox.width / fig.dpi, bbox.height / fig.dpi)  # Resize to legend

    savefig(fig=fig, filename="legend")


def plot_dfs(
    dfs: list,
    figsize: tuple,
    x_label: str,
    y_labels: list,
    x_axis_vals: pd.Series,
    x_ticks: Optional[list] = None,
    x_tick_labels: Optional[list] = None,
    x_scale: str | list[str] = "linear",  # Can be string or list
    y_scale: str | list[str] = "linear",
    x_lims: Optional[list] = None,
    y_lims: Optional[list] = None,
    legend_args: Optional[dict] = None,
    filename: Optional[str] = None,
    plot_kwargs: Optional[dict] = {"marker": "o", "linestyle": "-"},
    cmap_fixed_colors: int = 0,
    scientific_x: Optional[list[bool]] = None,
    scientific_y: Optional[list[bool]] = None,
    mask_nans: bool = False,
):
    f, axes = plt.subplots(1, len(dfs), figsize=figsize)
    if len(dfs) == 1:
        axes = [axes]
    style_dict_colname_mapping_exact = False

    # Convert x_scale and y_scale to lists if they're strings
    if isinstance(x_scale, str):
        x_scale = [x_scale] * len(dfs)
    if isinstance(y_scale, str):
        y_scale = [y_scale] * len(dfs)

    # Validate lengths
    if len(x_scale) != len(dfs) or len(y_scale) != len(dfs):
        raise ValueError("x_scale and y_scale must have same length as dfs")

    # Set default scientific notation settings if not provided
    if scientific_x is None:
        scientific_x = [False] * len(dfs)
    if scientific_y is None:
        scientific_y = [False] * len(dfs)

    # Validate lengths
    if len(scientific_x) != len(dfs) or len(scientific_y) != len(dfs):
        raise ValueError("scientific_x and scientific_y must have same length as dfs")

    # Create copper colormap colors
    copper_cmap = plt.cm.copper
    # Subtract the fixed colors of models that are in the style_dict
    n_columns_total = sum(len(df.columns) for df in dfs) - cmap_fixed_colors
    # Define range of colormap to use to avoid darkest colors
    cmap_min, cmap_max = 0.3, 1
    copper_colors = [
        copper_cmap(cmap_max - i * (cmap_max - cmap_min) / max(1, n_columns_total - 1))
        for i in range(n_columns_total)
    ]
    color_idx = 0  # Keep track of which color to use

    for df_idx, df in enumerate(dfs):
        ax = axes[df_idx]
        y_label = y_labels[df_idx]

        # Set scales using the individual settings for this df
        ax.set_xscale(x_scale[df_idx])
        ax.set_yscale(y_scale[df_idx])

        # Set scientific notation for x and y axis
        if x_scale[df_idx] == "linear":
            if scientific_x[df_idx]:
                ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
                ax.xaxis.get_offset_text().set_fontsize(
                    ax.xaxis.get_label().get_fontsize()
                )

        if y_scale[df_idx] == "linear":
            if scientific_y[df_idx]:
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.yaxis.get_offset_text().set_fontsize(
                    ax.yaxis.get_label().get_fontsize()
                )

        for col in df.columns:
            plot_kwargs_col = copy.deepcopy(plot_kwargs)

            # Apply styles from style_dict if col is in style_dict
            style_found = False
            if style_dict_colname_mapping_exact:
                if col in style_dict:
                    plot_kwargs_col.update(style_dict.get(col, {}))
                    style_found = True
            else:
                for col_key in style_dict.keys():
                    if col_key in col:
                        plot_kwargs_col.update(style_dict.get(col_key, {}))
                        style_found = True
                        break

            # If no style was found, use the next color in the copper colormap
            if not style_found and "color" not in plot_kwargs_col:
                plot_kwargs_col["color"] = copper_colors[color_idx]
                color_idx += 1

            if mask_nans:
                # Get data and create mask for non-NaN values
                y_data = df[col].values
                mask = ~np.isnan(y_data)
                x_data = x_axis_vals[df_idx][mask]
                y_data = y_data[mask]
            else:
                x_data = x_axis_vals[df_idx]
                y_data = df[col].values

            if "label" in plot_kwargs_col:
                ax.plot(x_data, y_data, **plot_kwargs_col)
            else:
                ax.plot(x_data, y_data, label=col, **plot_kwargs_col)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        if x_tick_labels is not None:
            ax.set_xticklabels(x_tick_labels)
        if x_lims is not None:
            ax.set_xlim(x_lims[df_idx])
        if y_lims is not None:
            ax.set_ylim(y_lims[df_idx])

        if df_idx == 0 and legend_args is not None:
            f.legend(**legend_args)
        ax.grid(alpha=0.2)

    f.tight_layout()

    if filename is not None:
        savefig(fig=f, filename=filename)
    else:
        plt.show()


def create_ttf_plot(figsize_2plots_per_col, legend_args):
    # Read in data
    ttf1_1_plot_df, ttf100_1_plot_df = read_ttf_data()

    # Specify the x-axis parameter and the y-axis labels
    x_axis_param = "prefill_length"
    y_labels = ["Time to First Token [s]", "Time to First 100 Tokens [s]"]

    # Collect x and y values
    x_axis_vals = []
    y_axis_val_dfs = []
    for df in [ttf1_1_plot_df, ttf100_1_plot_df]:
        x_axis_vals.append(df[f"P--{x_axis_param}"])

        exclude_regex = "P--.*|Unnamed.*"
        y_axis_val_df = df.drop(df.filter(regex=exclude_regex, axis=1).columns, axis=1)
        # Use seconds instead of milliseconds for TTF plots
        df = y_axis_val_df * 1e-3
        y_axis_val_dfs.append(df)

    # Plot the figures
    plot_dfs(
        dfs=y_axis_val_dfs,
        figsize=figsize_2plots_per_col,
        x_axis_vals=x_axis_vals,
        x_label="Prefill Length [Tokens]",
        y_labels=y_labels,
        y_lims=[[0.0, 1.0], [0.0, 4.0]],
        legend_args=legend_args,
        filename="ttfs",
    )


def create_ttf_vllm_plot(figsize_2plots_per_col, legend_args):
    # Read in data
    ttf1_1_plot_df, ttf100_1_plot_df = read_ttf_vllm_data()

    # Specify the x-axis parameter and the y-axis labels
    x_axis_param = "prefill_length"
    y_labels = ["Time to First Token [s]", "Time to First 100 Tokens [s]"]

    # Collect x and y values
    x_axis_vals = []
    y_axis_val_dfs = []
    for df in [ttf1_1_plot_df, ttf100_1_plot_df]:
        x_axis_vals.append(df[f"P--{x_axis_param}"])

        exclude_regex = "P--.*|Unnamed.*"
        y_axis_val_df = df.drop(df.filter(regex=exclude_regex, axis=1).columns, axis=1)
        # Use seconds instead of milliseconds for TTF plots
        df = y_axis_val_df * 1e-3
        y_axis_val_dfs.append(df)

    # Plot the figures
    plot_dfs(
        dfs=y_axis_val_dfs,
        figsize=figsize_2plots_per_col,
        x_axis_vals=x_axis_vals,
        x_label="Prefill Length [Tokens]",
        y_labels=y_labels,
        y_lims=[[0.0, 1.0], [0.0, 4.0]],
        legend_args=legend_args,
        filename="ttfs_vllm",
    )


def create_generation_time_and_memory_plot(figsize_2plots_per_col, legend_args):
    # Read in data
    gen_time_plot_df, gen_mem_plot_df = read_gen_data()

    # Specify the x-axis parameter and the y-axis labels
    x_axis_param = "generation_length"
    y_labels = ["Generation Time [s]", "Generation Memory [GB]"]

    # Collect x and y values
    x_axis_vals = []
    y_axis_val_dfs = []
    for df in [gen_time_plot_df, gen_mem_plot_df]:
        x_axis_vals.append(df[f"P--{x_axis_param}"])

        exclude_regex = "P--.*|Unnamed.*"
        y_axis_val_df = df.drop(df.filter(regex=exclude_regex, axis=1).columns, axis=1)
        y_axis_val_dfs.append(y_axis_val_df)

    # Plot the figures
    plot_dfs(
        dfs=y_axis_val_dfs,
        figsize=figsize_2plots_per_col,
        x_axis_vals=x_axis_vals,
        x_label="Generation Length [Tokens]",
        y_labels=y_labels,
        legend_args=legend_args,
        filename="generation",
    )


def create_generation_time_plot(figsize_2plot_per_col, legend_args):
    # Read in data
    gen_time_plot_df, gen_time_vllm_plot_df = read_gen_data2()

    # Specify the x-axis parameter and the y-axis labels
    x_axis_param = "generation_length"
    y_labels = ["Generation Time [s]", "Generation Time (VLLM) [s]"]

    # Collect x and y values
    x_axis_vals = []
    y_axis_val_dfs = []
    for df in [gen_time_plot_df, gen_time_vllm_plot_df]:  # , gen_mem_plot_df]:
        x_axis_vals.append(df[f"P--{x_axis_param}"])

        exclude_regex = "P--.*|Unnamed.*"
        y_axis_val_df = df.drop(df.filter(regex=exclude_regex, axis=1).columns, axis=1)
        y_axis_val_dfs.append(y_axis_val_df)

    # Plot the figures
    plot_dfs(
        dfs=y_axis_val_dfs,
        figsize=figsize_2plot_per_col,
        x_axis_vals=x_axis_vals,
        x_label="Generation Length [Tokens]",
        y_labels=y_labels,
        y_lims=[[0, 500], [0, 500]],
        legend_args=legend_args,
        filename="generationtime",
    )


def create_single_throughput_plot(figsize_1plot_per_col, legend_args):
    # Read in data
    token_per_sec_plot_df = read_tokens_per_second_data()

    # Specify the x-axis parameter and the y-axis labels
    x_axis_param = "prefill_length"
    y_labels = ["Tokens per Second"]

    # Collect x and y values
    x_axis_vals = []
    y_axis_val_dfs = []
    for df in [token_per_sec_plot_df]:
        x_axis_vals.append(df[f"P--{x_axis_param}"])

        exclude_regex = "P--.*|Unnamed.*"
        y_axis_val_df = df.drop(df.filter(regex=exclude_regex, axis=1).columns, axis=1)
        y_axis_val_dfs.append(y_axis_val_df)

    # Plot the figures
    plot_dfs(
        dfs=y_axis_val_dfs,
        figsize=figsize_1plot_per_col,
        x_axis_vals=x_axis_vals,
        x_label="Prefill Length [Tokens]",
        y_labels=y_labels,
        legend_args=legend_args,
        filename="tokens_per_sec",
    )


def create_throughput_vllm_plot(figsize_2plot_per_col, legend_args):
    # Read in data
    token_per_sec_plot_df, token_per_sec_plot_vllm_df = (
        read_tokens_per_second_data_vllm()
    )

    # Specify the x-axis parameter and the y-axis labels
    x_axis_param = "prefill_length"
    y_labels = ["Tokens per Second", "Tokens per Second (VLLM)"]

    # Collect x and y values
    x_axis_vals = []
    y_axis_val_dfs = []
    for df in [token_per_sec_plot_df, token_per_sec_plot_vllm_df]:
        x_axis_vals.append(df[f"P--{x_axis_param}"])

        exclude_regex = "P--.*|Unnamed.*"
        y_axis_val_df = df.drop(df.filter(regex=exclude_regex, axis=1).columns, axis=1)
        y_axis_val_dfs.append(y_axis_val_df)

    # Plot the figures
    plot_dfs(
        dfs=y_axis_val_dfs,
        figsize=figsize_2plot_per_col,
        x_axis_vals=x_axis_vals,
        x_label="Prefill Length [Tokens]",
        y_labels=y_labels,
        y_lims=[[0, 200], [0, 200]],
        legend_args=legend_args,
        filename="tokens_per_sec_vllm",
    )


def create_forward_throughput_barplot(
    figsize_1plot_per_col, legend_args, filename=None
):
    # Read in data
    forward_throughput_df = read_forward_throughput_data()

    # Switch columns such that the ordering of the models is the same as in the other plots.
    forward_throughput_df = forward_throughput_df[
        ["BS", "CTX", "llama3", "llama2", "falcon_mamba", "codestral_mamba", "xlstm"]
    ]

    f = create_runtime_bar_plot(
        data_df=forward_throughput_df,
        group_col_names=["BS", "CTX"],
        bar_label_font_size=8,
        style_dict=style_dict,
        figsize=figsize_1plot_per_col,
        x_label="Tokens",
        y_label="Tokens per Second",
        yticks=[10000, 20000, 30000, 40000, 50000],
        legend_args=legend_args,
    )

    f.tight_layout()

    if filename is not None:
        savefig(fig=f, filename=filename)
    else:
        plt.show()


def create_forward_throughput_vllm_barplot(
    figsize_1plot_per_col, legend_args, filename=None
):
    # Read in data
    forward_throughput_df = read_forward_throughput_vllm_data()
    forward_throughput_df = forward_throughput_df.rename(
        columns={
            "R--llama3__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm": "llama3",
            "R--codestral_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm": "codestral_mamba",
            "R--falcon_mamba__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm": "falcon_mamba",
            "R--llama2__ampdt-bfloat16__wdt-bfloat16__ucgg-False_ucgm-False_vllm": "llama2",
            "R--xlstm__tcm__ampdt-bfloat16__wdt-bfloat16__ucgg-True_ucgm-False_isd-bfloat16_ed-4096_nh-8_nb-32_vs-50304_wm-fused_ck-chunkwise--triton_xl_chunk_sk-native_sequence__triton_step_fused_sk-triton_fused_cs-128_akd-bfloat16": "xlstm",
            "P--batch_size": "BS",
            "P--prefill_length": "CTX",
        }
    )

    # Switch columns such that the ordering of the models is the same as in the other plots.
    forward_throughput_df = forward_throughput_df[
        ["BS", "CTX", "llama3", "llama2", "falcon_mamba", "codestral_mamba", "xlstm"]
    ]

    forward_throughput_df[
        ["llama3", "llama2", "falcon_mamba", "codestral_mamba", "xlstm"]
    ] = forward_throughput_df[
        ["llama3", "llama2", "falcon_mamba", "codestral_mamba", "xlstm"]
    ].round(0)

    f = create_runtime_bar_plot(
        data_df=forward_throughput_df,
        group_col_names=["BS", "CTX"],
        bar_label_font_size=8,
        style_dict=style_dict,
        figsize=figsize_1plot_per_col,
        x_label="Tokens",
        y_label="Tokens per Second",
        fmt=lambda x: f"{x:.0f}" if x > 0 else "NA",
        yticks=[10000, 20000, 30000, 40000, 50000],
        legend_args=legend_args,
    )

    f.tight_layout()

    if filename is not None:
        savefig(fig=f, filename=filename)
    else:
        plt.show()


def create_ruler_main_plot(figsize_1plot_per_col, legend_args):
    # Read in data
    ruler_main_df = read_ruler_main_data()

    # Specify the x-axis parameter and the y-axis labels
    y_labels = ["Average Accuracy [%]"]

    # Reorder columns to match the other plots
    ruler_main_df = ruler_main_df[
        [
            "llama3.1-8b",
            "llama-2-7b",
            "falcon-mamba-7b",
            "codestral-mamba-7b",
            "rwkv5-7b",
            "rwkv6-7b",
            "xLSTM-7b",
            "xLSTM-7b-longctx",
        ]
    ]

    # Rename columns to match other plots
    ruler_main_df.columns = [
        "llama3",
        "llama2",
        "falcon_mamba",
        "codestral_mamba",
        "rwkv5",
        "rwkv6",
        "xlstm",
        "xlstm_longctx",
    ]
    x_ticks = [4096, 8192, 16384, 32768, 65536, 131072]
    x_tick_labels = [4096, 8192, 16384, 32768, 65536, 131072]

    plot_dfs(
        dfs=[ruler_main_df],
        figsize=figsize_1plot_per_col,
        x_axis_vals=[ruler_main_df.index],
        x_label="Context Length [Tokens]",
        x_ticks=x_ticks,
        x_tick_labels=x_tick_labels,
        y_labels=y_labels,
        x_scale="log",
        y_lims=(0, 110),
        legend_args=legend_args,
        filename="ruler_main",
    )


def create_ruler_abl_plot(figsize_1plot_per_col, legend_args):
    # Read in data
    ruler_abl_df = read_ruler_abl_data()

    # Specify the x-axis parameter and the y-axis labels
    y_labels = ["Average Accuracy [%]"]

    x_ticks = [4096, 8192, 16384, 32768, 65536, 131072]
    x_tick_labels = [4096, 8192, 16384, 32768, 65536, 131072]

    x_ticks = [4096, 8192, 16384, 32768, 65536, 131072]
    x_tick_labels = [4096, 8192, 16384, 32768, 65536, 131072]

    # Rename columns to match other plots
    ruler_abl_df.columns = [
        "Default",
        "LCTX",
        "Short NH4",
        "Short NH8",
        "Short NH16",
        "Short NH32",
        "Short NH8, no IG",
        "Short NH8, fixed IG to 0",
    ]

    plot_dfs(
        dfs=[ruler_abl_df],
        figsize=figsize_1plot_per_col,
        x_axis_vals=[ruler_abl_df.index],
        x_label="Context Length [Tokens]",
        x_ticks=x_ticks,
        x_tick_labels=x_tick_labels,
        y_labels=y_labels,
        x_scale="log",
        y_lims=(0, 70),
        legend_args=legend_args,
        filename="ruler_abl",
        cmap_fixed_colors=2,
    )


def create_lr_scheduler_plot(figsize, legend_args):
    df_lr, df_ppl, df_loss = prepare_lr_scheduler_data()

    # Rename columns to match other plots with style
    df_lr.columns = [
        "Exponential 76k Steps",
        "Exponential 150k Steps",
        "Cosine 76k Steps",
        "Cosine 150k Steps",
    ]
    df_ppl.columns = [
        "Exponential 76k Steps",
        "Exponential 150k Steps",
        "Cosine 76k Steps",
        "Cosine 150k Steps",
    ]
    df_loss.columns = [
        "Exponential 76k Steps",
        "Exponential 150k Steps",
        "Cosine 76k Steps",
        "Cosine 150k Steps",
    ]

    plot_dfs(
        dfs=[df_lr, df_ppl],
        figsize=figsize,
        x_axis_vals=[df_lr.index, df_ppl.index],
        x_label="Steps",
        y_labels=["Learning Rate", "Perplexity"],
        y_lims=[(0, 0.00083), (11, 15)],
        legend_args=legend_args,
        scientific_x=[True, True],
        scientific_y=[True, False],
        plot_kwargs={"linestyle": "-"},
        filename="scheduler",
    )


def create_softcap_plot(figsize, legend_args):
    df_valid_loss, df_grad = prepare_softcap_data()

    # Rename columns to match other plots with style
    df_valid_loss.columns = ["No Softcap", "With Softcap"]
    df_grad.columns = ["No Softcap", "With Softcap"]

    plot_dfs(
        dfs=[df_valid_loss, df_grad],
        figsize=figsize,
        x_axis_vals=[df_valid_loss.index, df_grad.index],
        x_label="Steps",
        y_labels=["Validation Loss", "Gradient Norm"],
        y_lims=[(2.2, 2.7), (0.0, 0.25)],
        legend_args=legend_args,
        scientific_x=[True, True],
        scientific_y=[False, False],
        plot_kwargs={"linestyle": "-"},
        filename="softcap",
    )


def create_rmsnorm_plot(figsize, legend_args):
    df_loss, df_grad, df_steptime = prepare_rmsnorm_data()

    plot_dfs(
        dfs=[df_loss, df_grad, df_steptime],
        figsize=figsize,
        x_axis_vals=[df_loss.index, df_grad.index, df_steptime.index],
        x_label="Steps",
        y_labels=["Validation Loss", "Gradient Norm", "Step Time [s]"],
        x_lims=[(0, 1.5e4), (0, 1.5e4), (0, 1.5e4)],
        y_lims=[(2.2, 4), (0.0, 20), (0, 7)],
        y_scale=("linear", "log", "linear"),
        legend_args=legend_args,
        scientific_x=[True, True, True],
        scientific_y=[False, False, False],
        plot_kwargs={"linestyle": "-"},
        filename="rmsnorm",
    )


def create_bias_ablation_plot(figsize, legend_args):
    dfs = prepare_bias_data()
    dfs_list = list(dfs.values())

    plot_dfs(
        dfs=dfs_list,
        figsize=figsize,
        x_axis_vals=[dfs_list[0].index, dfs_list[1].index],
        x_label="Steps",
        y_labels=["Validation Loss", "Gradient Norm"],
        y_scale=["linear", "log"],
        y_lims=[(3.0, 3.6), (0.1, 100)],
        legend_args=legend_args,
        scientific_x=[True, True],
        scientific_y=[False, False],
        plot_kwargs={"linestyle": "-"},
        # mask_nans=True,
        filename="bias",
    )


def plot_paper_figures():
    switches_for_plots = {
        "ttf": True,  # False,
        "ttf_vllm": True,  # False,
        "gen": True,  # False,
        "gentime": True,
        "tps": True,  # False,
        "tps_all": True,  # False,
        "ft": True,  # False,
        "ft_all": True,  # False,
        "ruler_main": False,
        "ruler_abl": True,
        "scheduler": False,
        "softcap": False,
        "rmsnorm": False,
        "bias": False,
    }
    # 1. Produce the 2-plots-per-ICML column figures TTF1, TTF100 and generation time, generation memory.

    # The figsize for a 2-plot per column figure is smaller than the one for a single plot per column figure.
    figsize_2plots_per_col = (5.5, 5.5 / 2)

    # Define the legend arguments for the 2 plots per column figures
    legend_args = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.2),
        "ncol": 3,
        "frameon": False,
        "facecolor": "white",
    }

    if switches_for_plots["ttf"]:
        create_ttf_plot(figsize_2plots_per_col, legend_args)

    if switches_for_plots["ttf_vllm"]:
        create_ttf_vllm_plot(figsize_2plots_per_col, legend_args)

    # 2. Produce the generation time and generation memory plots
    if switches_for_plots["gen"]:
        create_generation_time_and_memory_plot(figsize_2plots_per_col, legend_args)

    if switches_for_plots["gentime"]:
        create_generation_time_plot(figsize_2plots_per_col, legend_args)

    # 3. Produce the single throughput plot.
    # This plot is a single plot per column figure,
    # so the aspect ratio is different. Double the height of the 2-plots-per-column figure and subtract 2 to make
    # space for the legend.
    figsize_1plot_per_col = (
        figsize_2plots_per_col[0],
        figsize_2plots_per_col[1] * 2 - 2,
    )

    # Define the legend arguments for the 2 plots per column figures
    legend_args = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.15),
        "ncol": 3,
        "frameon": False,
        "facecolor": "white",
    }

    if switches_for_plots["tps"]:
        create_single_throughput_plot(figsize_1plot_per_col, legend_args)

    if switches_for_plots["tps_all"]:
        create_throughput_vllm_plot(figsize_2plots_per_col, legend_args)

    # 4. Produce the forward throughput barplot.
    # This plot is also a single plot per column figure but the barplot needs more height to look good
    figsize_1plot_per_col = (
        figsize_2plots_per_col[0],
        figsize_2plots_per_col[1] * 2 - 1,
    )

    # The legend has to be adjusted to fit the plot.
    legend_args = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.3),
        "ncol": 3,
        "frameon": False,
        "facecolor": "white",
    }

    if switches_for_plots["ft"]:
        create_forward_throughput_barplot(
            figsize_1plot_per_col, legend_args, filename="forward_throughput"
        )

    if switches_for_plots["ft_all"]:
        create_forward_throughput_vllm_barplot(
            figsize_1plot_per_col, legend_args, filename="forward_throughput_all"
        )

    # 5. Produce RULER plots
    # Main RULER plot
    # These are again 1-plot-per-column figures, so the aspect ratio is the same as for the throughput plot.
    # However, we need more heigth for the large legend
    figsize_1plot_per_col = (
        figsize_2plots_per_col[0],
        figsize_2plots_per_col[1] * 2 - 2,
    )

    # Define the same legend arguments as for the throughput plot
    legend_args = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.2),
        "ncol": 3,
        "frameon": False,
        "facecolor": "white",
    }

    if switches_for_plots["ruler_main"]:
        create_ruler_main_plot(figsize_1plot_per_col, legend_args)

    # Ablation RULER plot. This plot has a larger legend. TODO: this is an appendix figure, so we have
    # 1 plot per page here
    legend_args = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.2),
        "ncol": 3,
        "frameon": False,
        "facecolor": "white",
    }

    if switches_for_plots["ruler_abl"]:
        create_ruler_abl_plot(figsize_1plot_per_col, legend_args)

    # 6. Procude the learning rate scheduler plots. These will be 2-plots-per-page figures, so the
    #    width is the same as a 1 plot-per-column figure
    figsize_2plots_per_page = (figsize_1plot_per_col[0] * 2, figsize_1plot_per_col[1])
    legend_args = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.1),
        "ncol": 2,
        "frameon": False,
        "facecolor": "white",
    }

    if switches_for_plots["scheduler"]:
        create_lr_scheduler_plot(figsize_2plots_per_page, legend_args)

    # 7. Softcapping ablation plots
    # These will be 2 plots per page figures
    legend_args = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.1),
        "ncol": 2,
        "frameon": False,
        "facecolor": "white",
    }

    if switches_for_plots["softcap"]:
        create_softcap_plot(figsize_2plots_per_page, legend_args)

    # 8. RMSNorm vs LayerNorm plot
    # This will a 2 plots per page figure
    legend_args = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.1),
        "ncol": 2,
        "frameon": False,
        "facecolor": "white",
    }

    if switches_for_plots["rmsnorm"]:
        create_rmsnorm_plot(figsize_2plots_per_page, legend_args)

    # 9. Bias ablation plot
    # This will be a 2 plots per page figure
    legend_args = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.1),
        "ncol": 2,
        "frameon": False,
        "facecolor": "white",
    }

    if switches_for_plots["bias"]:
        create_bias_ablation_plot(figsize_2plots_per_page, legend_args)


if __name__ == "__main__":
    plot_paper_figures()
