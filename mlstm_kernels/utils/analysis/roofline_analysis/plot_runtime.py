#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import pickle
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...plot.bar_plot import create_double_bar_plot
from .flops_mlstm import (
    count_flops_mlstmsig_chunkwise_parallel,
)
from .plot_config import get_plot_mpl_context
from .roofline_analysis_mlstm import (
    Acc_math_a100,
    Acc_math_b200,
    Acc_math_h100,
    Acc_math_v100,
    Acc_mem_a100,
    Acc_mem_b200,
    Acc_mem_h100,
    Acc_mem_v100,
    acc_math_projection,
    acc_mem_projection,
    get_arithmetic_intensity_mlstmsig,
    get_flop_optimal_chunk_size_mlstmsig,
    get_runtime_optimal_chunk_size_mlstmsig_intensity,
    get_theoretical_runtime_mlstmsig_math_mem_in_ms,
)


def plot_runtime(
    fns_calculate_theoretical_runtime: list[Callable],
    seq_len: int,
    runtime_df: pd.DataFrame = None,
    ax: Axes = None,
    num_points: int = 50,
    colors_theoretical_runtime: list[str] = "tab:blue",
    color_runtime_df: str = "#9a3c73",
    labels_theoretical_runtime: list[str] = None,
    add_chunk_size_optimal_runtime: bool = True,
    fn_calculate_optimal_chunk_sizes_for_intensities: Callable = None,
    fn_calculate_theoretical_runtime_for_optimal_chunk_sizes_and_intensities: Callable = None,
    show_title: bool = True,
    show_ylabel: bool = True,
    legend_args: dict = None,
    xscale: str = "log",
    yticks: np.ndarray = None,
    xticks: list = [32, 64, 256, 512, 1024, 2048, 4096],
    y_lim: tuple = None,
) -> Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if add_chunk_size_optimal_runtime:
        factor_causal_cs_opt_runtime = 1.0
        num_points_csr = 20 * num_points

        intensity_xdata_max = 100

        intensity_xdata = np.linspace(0, intensity_xdata_max, num_points_csr)
        chunk_sizes_optimal_fc10 = fn_calculate_optimal_chunk_sizes_for_intensities(
            Acc_intensity=acc_math_projection(intensity_xdata)
            / acc_mem_projection(intensity_xdata),
            factor_causal=factor_causal_cs_opt_runtime,
        )
        optimal_runtimes_fc10 = (
            fn_calculate_theoretical_runtime_for_optimal_chunk_sizes_and_intensities(
                chunk_size=chunk_sizes_optimal_fc10,
                factor_causal=factor_causal_cs_opt_runtime,
                acc_math=acc_math_projection(intensity_xdata),
                acc_mem=acc_mem_projection(intensity_xdata),
            )
        )
        factor_causal_cs_opt_runtime = 0.5
        intensity_xdata = np.linspace(0, intensity_xdata_max, num_points_csr)
        chunk_sizes_optimal_fc05 = fn_calculate_optimal_chunk_sizes_for_intensities(
            Acc_intensity=acc_math_projection(intensity_xdata)
            / acc_mem_projection(intensity_xdata),
            factor_causal=factor_causal_cs_opt_runtime,
        )
        optimal_runtimes_fc05 = (
            fn_calculate_theoretical_runtime_for_optimal_chunk_sizes_and_intensities(
                chunk_size=chunk_sizes_optimal_fc05,
                factor_causal=factor_causal_cs_opt_runtime,
                acc_math=acc_math_projection(intensity_xdata),
                acc_mem=acc_mem_projection(intensity_xdata),
            )
        )
        factor_causal_cs_opt_runtime = 0.5
        intensity_xdata = np.linspace(0, intensity_xdata_max, num_points_csr)
        chunk_sizes_optimal_fc05 = fn_calculate_optimal_chunk_sizes_for_intensities(
            Acc_intensity=acc_math_projection(intensity_xdata)
            / acc_mem_projection(intensity_xdata),
            factor_causal=factor_causal_cs_opt_runtime,
        )
        optimal_runtimes_fc05 = (
            fn_calculate_theoretical_runtime_for_optimal_chunk_sizes_and_intensities(
                chunk_size=chunk_sizes_optimal_fc05,
                factor_causal=factor_causal_cs_opt_runtime,
                acc_math=acc_math_projection(intensity_xdata),
                acc_mem=acc_mem_projection(intensity_xdata),
            )
        )
        factor_causal_cs_opt_runtime = 0.66
        intensity_xdata = np.linspace(0, intensity_xdata_max, num_points_csr)
        chunk_sizes_optimal_fc066 = fn_calculate_optimal_chunk_sizes_for_intensities(
            Acc_intensity=acc_math_projection(intensity_xdata)
            / acc_mem_projection(intensity_xdata),
            factor_causal=factor_causal_cs_opt_runtime,
        )
        optimal_runtimes_fc066 = (
            fn_calculate_theoretical_runtime_for_optimal_chunk_sizes_and_intensities(
                chunk_size=chunk_sizes_optimal_fc066,
                factor_causal=factor_causal_cs_opt_runtime,
                acc_math=acc_math_projection(intensity_xdata),
                acc_mem=acc_mem_projection(intensity_xdata),
            )
        )

        ax.plot(
            chunk_sizes_optimal_fc05,
            optimal_runtimes_fc05,
            label="chunk size optimal runtime",
            color="grey",
            linestyle=":",
        )

        ax.plot(
            chunk_sizes_optimal_fc10,
            optimal_runtimes_fc10,
            label="chunk size optimal runtime",
            color="grey",
            linestyle="--",
        )

        ax.plot(
            chunk_sizes_optimal_fc066,
            optimal_runtimes_fc066,
            label="chunk size optimal runtime",
            color="grey",
            linestyle="-",
        )

        # does not work.
        # xnew_runtimes = np.linspace(2, 14, num_points_csr)
        # ynew_chunksizes05 = np.interp(xnew_runtimes, optimal_runtimes_fc05, chunk_sizes_optimal_fc05)
        # ynew_chunksizes10 = np.interp(xnew_runtimes, optimal_runtimes_fc10, chunk_sizes_optimal_fc10)
        # # ax.fill_betweenx(xnew_runtimes, x1=ynew_runtimes05, x2=ynew_runtimes10, alpha=0.1, color="grey")
        # ax.plot(ynew_chunksizes05, xnew_runtimes, linestyle=":", color="red")

    # chunk_sizes = np.linspace(1, seq_len, num_points, dtype=int)
    chunk_sizes = np.logspace(5, np.log2(seq_len), num_points, base=2)

    for color, calculate_theoretical_runtime in zip(
        colors_theoretical_runtime, fns_calculate_theoretical_runtime
    ):
        runtime_fcausal05 = calculate_theoretical_runtime(
            chunk_size=chunk_sizes, factor_causal=0.5
        )
        runtime_fcausal1 = calculate_theoretical_runtime(
            chunk_size=chunk_sizes, factor_causal=1
        )
        runtime_fcausal066 = calculate_theoretical_runtime(
            chunk_size=chunk_sizes, factor_causal=0.66
        )

        ax.fill_between(
            chunk_sizes,
            runtime_fcausal05,
            runtime_fcausal1,
            alpha=0.1,
        )
        ax.plot(
            chunk_sizes,
            runtime_fcausal05,
            linestyle=":",
            color=color,
            label=r"$F_\text{causal}$=0.5",
        )
        ax.plot(
            chunk_sizes,
            runtime_fcausal1,
            linestyle="--",
            color=color,
            label=r"$F_\text{causal}$=1.0",
        )
        ax.plot(
            chunk_sizes,
            runtime_fcausal066,
            label=r"$F_\text{causal}$=0.66",
            color=color,
        )
    if runtime_df is not None:
        ax.plot(
            runtime_df["chunk_size"],
            runtime_df["runtime"],
            label="Measured Runtime",
            color=color_runtime_df,
            marker="s",
        )

    ax.set_xlabel(r"Chunk Size $L$ (logscale)")
    if show_ylabel:
        ax.set_ylabel(r"Runtime [ms]")

    if xscale is not None:
        ax.set_xscale(xscale, base=2)
    # max_exp = int(np.log2(head_dims_hv[-1]))
    # ax.set_xticks(np.logspace(0, max_exp, base=2, num=max_exp+1))
    # ax.set_xticks([16, 256, 512, 1024, 1536, 2048])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    if show_title:
        ax.set_title(None)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    if legend_args is not None:
        # Custom legend handles and labels
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="black", linestyle=":"),
            Line2D([0], [0], color="black", linestyle="-"),
            Line2D([0], [0], color="black", linestyle="--"),
        ]

        legend_labels = [
            r"$F_\text{causal}$=0.5",
            r"$F_\text{causal}$=0.66",
            r"$F_\text{causal}$=1.0",
        ]
        for color in colors_theoretical_runtime:
            custom_lines.append(Line2D([0], [0], color=color, lw=4))

        if labels_theoretical_runtime is not None:
            legend_labels += labels_theoretical_runtime

        custom_lines.append(Line2D([0], [0], color="grey", linestyle="-", lw=4))
        legend_labels.append("Chunk Size\nOptimal Runtime")

        custom_lines.append(Line2D([0], [0], color=color_runtime_df, marker="s"))
        legend_labels.append("mLSTMsig TFLA\nForward Measured")

        ax.legend(
            handles=custom_lines,
            labels=legend_labels,
            **legend_args,
        )

    ax.grid(alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ax.set_xscale("log", base=2)
    return ax.get_figure()


def load_runtime_df(file: Path) -> pd.DataFrame:
    with open(file, "rb") as f:
        all_results_dict = pickle.load(f)

    def create_memory_runtime_df(
        runtime_df: pd.DataFrame,
        memory_df: pd.DataFrame,
        kernel_name: str = "xl_chunk_siging",
        sequence_length: int = 8192,
    ) -> pd.DataFrame:
        selected_mlstm_runtime_fwbw_df = runtime_df.filter(
            regex=rf"sequence_length|.*{kernel_name}.*"
        )
        selected_mlstm_memory_fwbw_df = memory_df.filter(
            regex=rf"sequence_length|.*{kernel_name}.*"
        )

        selected_mlstm_runtime_ctx_fwbw_df = (
            selected_mlstm_runtime_fwbw_df[
                selected_mlstm_runtime_fwbw_df["sequence_length"] == sequence_length
            ]
            .drop(columns=["sequence_length"])
            .T
        )
        selected_mlstm_memory_ctx_fwbw_df = (
            selected_mlstm_memory_fwbw_df[
                selected_mlstm_memory_fwbw_df["sequence_length"] == sequence_length
            ]
            .drop(columns=["sequence_length"])
            .T
        )

        def extract_chunksize(specifier: str):
            return int(specifier.split("__")[-1].split("_")[0].split("-")[1])

        runtime_df = selected_mlstm_runtime_ctx_fwbw_df.rename(index=extract_chunksize)
        runtime_df.index.name = "chunk_size"
        runtime_df.columns = ["runtime"]
        runtime_df = runtime_df.reset_index()

        memory_df = selected_mlstm_memory_ctx_fwbw_df.rename(index=extract_chunksize)
        memory_df.index.name = "chunk_size"
        memory_df.columns = ["memory"]
        memory_df = memory_df / 1e9
        memory_df = memory_df.reset_index()

        memory_runtime_df = pd.concat([runtime_df, memory_df], axis=1)
        memory_runtime_df = memory_runtime_df.loc[
            :, ~memory_runtime_df.columns.duplicated()
        ]
        return memory_runtime_df

    runtime_memory_df = create_memory_runtime_df(
        runtime_df=all_results_dict["runtime"]["fw"],
        memory_df=all_results_dict["memory"]["fw"],
        sequence_length=8192,
        kernel_name="xl_chunk_siging",
    )
    return runtime_memory_df


def create_runtime_plot():
    # load the real measured data
    result_filename = "notebooks/plots_mlstm_kernel_benchmark_tfla_paper/mlstm_tfla_paper_consttoken_benchmark_results.p"
    file = Path(".") / "../.." / result_filename

    runtime_memory_df = load_runtime_df(file)

    ## parameters
    d_hv = 512
    p_qk = 0.5

    seq_len = 8192

    num_heads = 8
    batch_size = 8

    bytes_if = 2
    bytes_qkv = 2
    bytes_Cmn = 4

    factor_causal = 0.66
    ###

    calc_runtime_v100 = partial(
        get_theoretical_runtime_mlstmsig_math_mem_in_ms,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        d_hv=d_hv,
        p_qk=p_qk,
        factor_causal=factor_causal,
        acc_math=Acc_math_v100,
        acc_mem=Acc_mem_v100,
        bytes_if=bytes_if,
        bytes_qkv=bytes_qkv,
        bytes_Cmn=bytes_Cmn,
    )

    calc_runtime_a100 = partial(
        get_theoretical_runtime_mlstmsig_math_mem_in_ms,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        d_hv=d_hv,
        p_qk=p_qk,
        factor_causal=factor_causal,
        acc_math=Acc_math_a100,
        acc_mem=Acc_mem_a100,
        bytes_if=bytes_if,
        bytes_qkv=bytes_qkv,
        bytes_Cmn=bytes_Cmn,
    )

    calc_runtime_h100 = partial(
        get_theoretical_runtime_mlstmsig_math_mem_in_ms,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        d_hv=d_hv,
        p_qk=p_qk,
        factor_causal=factor_causal,
        acc_math=Acc_math_h100,
        acc_mem=Acc_mem_h100,
        bytes_if=bytes_if,
        bytes_qkv=bytes_qkv,
        bytes_Cmn=bytes_Cmn,
    )

    calc_runtime_b200 = partial(
        get_theoretical_runtime_mlstmsig_math_mem_in_ms,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        d_hv=d_hv,
        p_qk=p_qk,
        factor_causal=factor_causal,
        acc_math=Acc_math_b200,
        acc_mem=Acc_mem_b200,
        bytes_if=bytes_if,
        bytes_qkv=bytes_qkv,
        bytes_Cmn=bytes_Cmn,
    )

    calc_optimal_chunksize_for_intensities = partial(
        get_runtime_optimal_chunk_size_mlstmsig_intensity,
        d_hv=d_hv,
        p_qk=p_qk,
        bytes_Cmn=bytes_Cmn,
    )

    calc_runtime_opt_chunk_size_and_intensity = partial(
        get_theoretical_runtime_mlstmsig_math_mem_in_ms,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        d_hv=d_hv,
        p_qk=p_qk,
        bytes_Cmn=bytes_Cmn,
        bytes_if=bytes_if,
        bytes_qkv=bytes_qkv,
    )

    legend_kwargs = {
        "loc": "upper left",
        "ncol": 1,
        "bbox_to_anchor": (1.0, 1.1),
        "frameon": False,
        "facecolor": "white",
        # "alignment": "top",
        "labelspacing": 0.9,
    }

    with get_plot_mpl_context():
        fig = plot_runtime(
            fns_calculate_theoretical_runtime=[
                # calc_runtime_v100,
                calc_runtime_a100,
                calc_runtime_h100,
                calc_runtime_b200,
            ],
            colors_theoretical_runtime=[
                "tab:blue",
                "tab:orange",
                "tab:green",
            ],  # =["tab:purple", "tab:blue", "tab:orange", "tab:green"],
            seq_len=seq_len,
            runtime_df=runtime_memory_df,
            xticks=None,
            y_lim=[0, 15.0],
            legend_args=legend_kwargs,
            labels_theoretical_runtime=[
                # r"V100 Theoretical Runtime",  # Intensity$\approx$160 FLOP/byte",
                r"A100 Runtime",  # Intensity$\approx$160 FLOP/byte",
                r"H100 Runtime",  # Intensity$\approx$295 FLOP/byte",
                r"B200 Runtime",  # Intensity$\approx$292 FLOP/byte",
            ],
            fn_calculate_optimal_chunk_sizes_for_intensities=calc_optimal_chunksize_for_intensities,
            fn_calculate_theoretical_runtime_for_optimal_chunk_sizes_and_intensities=calc_runtime_opt_chunk_size_and_intensity,
        )
    return fig


def calculate_flops_and_flops_per_second_and_arithmetic_intensity_for_runtime_df(
    runtime_df: pd.DataFrame,
    fn_count_flops: Callable,
    fn_calculate_arithmetic_intensity: Callable,
    p_qk: float,
    d_hv: int,
    num_heads: int,
    factors_causal: list[float],
    seq_len: int,
    batch_size: int,
    bytes_if: int = 2,
    bytes_qkv: int = 2,
    bytes_Cmn: int = 4,
    chunk_size_col: str = "chunk_size",
    runtime_col: str = "runtime",
) -> pd.DataFrame:
    chunk_sizes = runtime_df[chunk_size_col]
    runtimes_ms = runtime_df[runtime_col]

    d_qk = p_qk * d_hv

    res_dict = {}
    res_dict["chunk_size"] = chunk_sizes
    res_dict["runtime_ms"] = runtimes_ms
    for fc in factors_causal:
        # 1) compute the flops for each chunk size
        flops_per_head_and_sequence = fn_count_flops(
            seq_len=seq_len,
            d_qk=d_qk,
            d_hv=d_hv,
            factor_causal=fc,
            chunk_size=chunk_sizes,
        )
        total_flops = batch_size * num_heads * flops_per_head_and_sequence

        res_dict[f"flops_fc{fc}"] = total_flops

        # 2) compute the flops per second for each chunk size
        flops_per_second = total_flops / (runtimes_ms / 1000)
        res_dict[f"flops_per_second_fc{fc}"] = flops_per_second

        # 3) compute the arithmetic intensity for each chunk size
        arithmetic_intensity = fn_calculate_arithmetic_intensity(
            chunk_size=chunk_sizes,
            d_hv=d_hv,
            p_qk=p_qk,
            factor_causal=fc,
            bytes_if=bytes_if,
            bytes_qkv=bytes_qkv,
            bytes_Cmn=bytes_Cmn,
        )
        res_dict[f"arithmetic_intensity_fc{fc}"] = arithmetic_intensity

    return pd.DataFrame(res_dict)


def create_runtime_vs_flops_per_second_or_flops_plot(
    which: Literal["flops", "flops_per_second"] = "flops",
):
    result_filename = "notebooks/plots_mlstm_kernel_benchmark_tfla_paper/mlstm_tfla_paper_consttoken_benchmark_results.p"
    file = Path(".") / "../.." / result_filename
    runtime_df = load_runtime_df(file)

    d_hv = 512
    p_qk = 0.5

    seq_len = 8192

    num_heads = 8
    batch_size = 8

    bytes_if = 2
    bytes_qkv = 2
    bytes_Cmn = 4

    factors_causal = [0.5, 0.66, 1.0]

    res_df = (
        calculate_flops_and_flops_per_second_and_arithmetic_intensity_for_runtime_df(
            runtime_df=runtime_df,
            fn_count_flops=count_flops_mlstmsig_chunkwise_parallel,
            fn_calculate_arithmetic_intensity=get_arithmetic_intensity_mlstmsig,
            p_qk=p_qk,
            d_hv=d_hv,
            num_heads=num_heads,
            factors_causal=factors_causal,
            seq_len=seq_len,
            batch_size=batch_size,
            bytes_if=bytes_if,
            bytes_qkv=bytes_qkv,
            bytes_Cmn=bytes_Cmn,
        )
    )

    if which == "flops":
        y_col = "flops_fc0.66"
        y_label = r"FLOPs ($F_{\text{causal}}$=0.66)"
        yaxis_label = "FLOPs"
        y_color = "tab:green"
        y_labelpad = None
    else:
        y_col = "flops_per_second_fc0.66"
        y_label = r"FLOPs/s ($F_{\text{causal}}$=0.66)"
        yaxis_label = "FLOPs/s"
        y_color = "#9a3c73"  # "tab:red"
        y_labelpad = -2

    with get_plot_mpl_context():
        fig = create_double_bar_plot(
            data=res_df,
            x_col="chunk_size",
            y_col_left="runtime_ms",
            y_col_right=y_col,
            left_label="Time [ms]",
            right_label=y_label,
            # title="Runtime and FLOPS/s for different chunk sizes",
            x_label="Chunk Size",
            left_color="tab:blue",
            right_color=y_color,
            figsize=(6, 2.5),
            bar_width=0.4,
            ax_grid="right",
            left_alpha=0.3,
            right_alpha=1.0,
            right_scilimits=(12, 12),
            # right_yerr=("flops_per_second_fc0.5", "flops_per_second_fc1.0"),
            capsize=5,
            # left_ylim=(0, 100),
            add_legend=True,
            right_labelpad=y_labelpad,
            right_yaxis_label=yaxis_label,
            bbox_to_anchor=(0.52, 1.15),
        )

    return fig
