#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .flops_mlstm import (
    count_flops_mlstmsig_chunkwise_parallel,
    count_flops_mlstmsig_parallel,
    count_flops_mlstmsig_recurrent,
)
from .plot_config import get_plot_mpl_context


def plot_mlstm_flops_for_all_formulations(
    ax: Axes,
    chunkwise_flop_fn: Callable,
    recurrent_flop_fn: Callable,
    parallel_flop_fn: Callable,
    seq_len: int,
    d_qk: int,
    d_hv: int,
    num_points: int = 30,
    color_recurrent: str = None,
    color_parallel: str = None,
    color_chunkwise: str = None,
    x_ticks: np.ndarray = None,
    add_causal_factors_parallel: bool = False,
    normalize_by_recurrent_flops: bool = True,
    legend_args: dict = None,
    show_y_label: bool = True,
    show_title: bool = True,
    ylim: tuple[float, float] = None,
) -> Figure:
    # calculate flops
    max_chunk_size_power_2 = np.log2(seq_len).astype(int)
    seq_len = np.repeat([seq_len], num_points).astype(float)
    chunk_sizes = np.logspace(0, max_chunk_size_power_2, num_points, base=2)

    flops_recurrent = recurrent_flop_fn(seq_len, d_qk, d_hv)
    flops_parallel_fcausal05 = parallel_flop_fn(seq_len, d_qk, d_hv, factor_causal=0.5)
    flops_parallel_fcausal066 = parallel_flop_fn(
        seq_len, d_qk, d_hv, factor_causal=0.66
    )
    flops_parallel_fcausal1 = parallel_flop_fn(seq_len, d_qk, d_hv, factor_causal=1.0)
    flops_chunkwise_fcausal05 = chunkwise_flop_fn(
        seq_len, d_qk, d_hv, factor_causal=0.5, chunk_size=chunk_sizes
    )
    flops_chunkwise_fcausal066 = chunkwise_flop_fn(
        seq_len, d_qk, d_hv, factor_causal=0.66, chunk_size=chunk_sizes
    )
    flops_chunkwise_fcausal1 = chunkwise_flop_fn(
        seq_len, d_qk, d_hv, factor_causal=1.0, chunk_size=chunk_sizes
    )

    if normalize_by_recurrent_flops:
        flops_parallel_fcausal05 /= flops_recurrent
        flops_parallel_fcausal066 /= flops_recurrent
        flops_parallel_fcausal1 /= flops_recurrent
        flops_chunkwise_fcausal05 /= flops_recurrent
        flops_chunkwise_fcausal066 /= flops_recurrent
        flops_chunkwise_fcausal1 /= flops_recurrent
        flops_recurrent /= flops_recurrent

    # plot recurrent
    ax.plot(chunk_sizes, flops_recurrent, label="Recurrent", color=color_recurrent)

    # plot parallel
    if add_causal_factors_parallel:
        ax.fill_between(
            chunk_sizes,
            flops_parallel_fcausal05,
            flops_parallel_fcausal1,
            color=color_parallel,
            alpha=0.1,
        )
        ax.plot(
            chunk_sizes, flops_parallel_fcausal05, color=color_parallel, linestyle=":"
        )
        ax.plot(
            chunk_sizes,
            flops_parallel_fcausal1,
            color=color_parallel,
            linestyle="--",
            label=("Parallel\n" + r"($F_\text{causal}$={0.5, 1.0})"),
        )
    ax.plot(
        chunk_sizes,
        flops_parallel_fcausal066,
        color=color_parallel,
        linestyle="-",
        label=("Parallel\n" + r"($F_\text{causal}$=0.66)"),
    )

    # plot chunkwise parallel
    ax.fill_between(
        chunk_sizes,
        flops_chunkwise_fcausal05,
        flops_chunkwise_fcausal1,
        color=color_chunkwise,
        alpha=0.1,
    )
    ax.plot(
        chunk_sizes, flops_chunkwise_fcausal05, color=color_chunkwise, linestyle=":"
    )
    ax.plot(
        chunk_sizes,
        flops_chunkwise_fcausal1,
        color=color_chunkwise,
        linestyle="--",
        label=("Chunkwise\n" + r"($F_\text{causal}$={0.5, 1.0})"),
    )
    ax.plot(
        chunk_sizes,
        flops_chunkwise_fcausal066,
        color=color_chunkwise,
        linestyle="-",
        label=("Chunkwise\n" + r"($F_\text{causal}$=0.66)"),
    )

    ax.set_xlabel(r"Chunk Size $L$ (logscale)")
    if show_y_label:
        if normalize_by_recurrent_flops:
            ax.set_ylabel("Ratio to Recurrent FLOPs")
        else:
            ax.set_ylabel("FLOPs")

    # Get handles and labels, then create a unique legend
    # handles, labels = ax.get_legend_handles_labels()
    # unique_labels = dict(zip(labels, handles))
    # ax.legend(unique_labels.values(), unique_labels.keys())
    if legend_args is not None:
        # Custom legend handles and labels
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="black", linestyle="-"),
            Line2D([0], [0], color="black", linestyle=":"),
            Line2D([0], [0], color="black", linestyle="--"),
        ]

        legend_labels = [
            r"$F_\text{causal}$=0.66",
            r"$F_\text{causal}$=0.5",
            r"$F_\text{causal}$=1.0",
        ]

        custom_lines.append(Line2D([0], [0], color=color_recurrent))
        legend_labels.append("Recurrent")
        for color in [color_parallel, color_chunkwise]:
            custom_lines.append(Line2D([0], [0], color=color, lw=4))

        legend_labels += ["Parallel", "Chunkwise"]
        ax.legend(
            handles=custom_lines,
            labels=legend_labels,
            **legend_args,
        )

        ax.legend(**legend_args)

    ax.set_xscale("log", base=2)
    # ax.minorticks_off()
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.set_yscale("log")
    ax.grid(alpha=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    if show_title:
        ax.set_title(rf"$d_{{qk}}$={d_qk}, $d_{{hv}}$={d_hv}", loc="center", pad=10)

    fig = ax.get_figure()

    return fig


def create_mlstm_flop_formulation_comparison_plot() -> Figure:
    fig_height = 4.3
    n_cols = 3
    # x_ticks = 2 ** np.arange(0, 14, 2)
    x_ticks = [1, 2, 4, 8, 16, 32, 128, 512, 2048, 8192]
    normalize_by_recurrent_flops: bool = False
    add_causal_factors_parallel: bool = True

    ylim = (0.0, 7.5e10)

    with get_plot_mpl_context():
        fig, ax = plt.subplots(
            1,
            n_cols,
            figsize=(1.1 * n_cols * fig_height, fig_height),
            sharey=True,
            sharex=True,
            gridspec_kw={"wspace": 0.07, "hspace": 0.0},
            # gridspec_kw={"width_ratios": [0.46, 0.54]},
        )
        fig = plot_mlstm_flops_for_all_formulations(
            ax=ax[0],
            chunkwise_flop_fn=count_flops_mlstmsig_chunkwise_parallel,
            recurrent_flop_fn=count_flops_mlstmsig_recurrent,
            parallel_flop_fn=count_flops_mlstmsig_parallel,
            seq_len=8192,
            d_qk=256,
            d_hv=512,
            num_points=50,
            color_recurrent="tab:blue",
            color_parallel="tab:orange",
            color_chunkwise="tab:green",
            x_ticks=x_ticks,
            normalize_by_recurrent_flops=normalize_by_recurrent_flops,
            add_causal_factors_parallel=add_causal_factors_parallel,
            show_y_label=True,
            ylim=ylim,
        )
        fig = plot_mlstm_flops_for_all_formulations(
            ax=ax[1],
            chunkwise_flop_fn=count_flops_mlstmsig_chunkwise_parallel,
            recurrent_flop_fn=count_flops_mlstmsig_recurrent,
            parallel_flop_fn=count_flops_mlstmsig_parallel,
            seq_len=8192,
            d_qk=128,
            d_hv=256,
            num_points=50,
            color_recurrent="tab:blue",
            color_parallel="tab:orange",
            color_chunkwise="tab:green",
            x_ticks=x_ticks,
            normalize_by_recurrent_flops=normalize_by_recurrent_flops,
            show_y_label=False,
            add_causal_factors_parallel=add_causal_factors_parallel,
        )
        fig = plot_mlstm_flops_for_all_formulations(
            ax=ax[2],
            chunkwise_flop_fn=count_flops_mlstmsig_chunkwise_parallel,
            recurrent_flop_fn=count_flops_mlstmsig_recurrent,
            parallel_flop_fn=count_flops_mlstmsig_parallel,
            seq_len=8192,
            d_qk=64,
            d_hv=128,
            num_points=50,
            color_recurrent="tab:blue",
            color_parallel="tab:orange",
            color_chunkwise="tab:green",
            x_ticks=x_ticks,
            normalize_by_recurrent_flops=normalize_by_recurrent_flops,
            show_y_label=False,
            add_causal_factors_parallel=add_causal_factors_parallel,
        )

        color_recurrent = "tab:blue"
        color_parallel = "tab:orange"
        color_chunkwise = "tab:green"
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

        custom_lines.append(Line2D([0], [0], color=color_recurrent))
        legend_labels.append("Recurrent")
        for color in [color_parallel, color_chunkwise]:
            custom_lines.append(Line2D([0], [0], color=color, lw=4))

        legend_labels += ["Parallel", "Chunkwise"]

        legend_kwargs = {
            "loc": "upper right",
            "ncol": 1,
            "bbox_to_anchor": (0.0, 0.9, 1.035, 0.0),
            "frameon": False,
            "facecolor": "white",
            # "alignment": "top",
            "labelspacing": 1.1,
        }
        fig.legend(handles=custom_lines, labels=legend_labels, **legend_kwargs)

    return fig


def create_mlstm_flop_formulation_comparison_plot_poster() -> Figure:
    fig_height = 4.5
    n_cols = 1
    # x_ticks = 2 ** np.arange(0, 14, 2)
    x_ticks = [1, 2, 4, 8, 16, 32, 128, 512, 2048, 8192]
    normalize_by_recurrent_flops: bool = False
    add_causal_factors_parallel: bool = True

    ylim = (0.0, 10e10)

    # with get_plot_mpl_context():
    fig, ax = plt.subplots(
        1,
        n_cols,
        figsize=(1.2 * n_cols * fig_height, fig_height),
        sharey=True,
        sharex=True,
        gridspec_kw={"wspace": 0.07, "hspace": 0.0},
        # gridspec_kw={"width_ratios": [0.46, 0.54]},
    )
    fig = plot_mlstm_flops_for_all_formulations(
        ax=ax,
        chunkwise_flop_fn=count_flops_mlstmsig_chunkwise_parallel,
        recurrent_flop_fn=count_flops_mlstmsig_recurrent,
        parallel_flop_fn=count_flops_mlstmsig_parallel,
        seq_len=8192,
        d_qk=256,
        d_hv=512,
        num_points=50,
        color_recurrent="tab:blue",
        color_parallel="tab:orange",
        color_chunkwise="tab:green",
        x_ticks=x_ticks,
        normalize_by_recurrent_flops=normalize_by_recurrent_flops,
        add_causal_factors_parallel=add_causal_factors_parallel,
        show_y_label=True,
        ylim=ylim,
    )

    color_recurrent = "tab:blue"
    color_parallel = "tab:orange"
    color_chunkwise = "tab:green"
    # Custom legend handles and labels
    from matplotlib.lines import Line2D

    # custom_lines = [
    #     Line2D([0], [0], color="black", linestyle=":"),
    #     Line2D([0], [0], color="black", linestyle="-"),
    #     Line2D([0], [0], color="black", linestyle="--"),
    # ]

    # legend_labels = [
    #     r"$F_\text{causal}$=0.5",
    #     r"$F_\text{causal}$=0.66",
    #     r"$F_\text{causal}$=1.0",
    # ]
    custom_lines = []
    legend_labels = []

    custom_lines.append(Line2D([0], [0], color=color_recurrent))
    legend_labels.append("Recurrent")
    for color in [color_parallel, color_chunkwise]:
        custom_lines.append(Line2D([0], [0], color=color, lw=4))

    legend_labels += ["Parallel", "Chunkwise"]

    legend_kwargs = {
        "loc": "upper left",
        "ncol": 1,
        "bbox_to_anchor": (0.92, 0.92),
        "frameon": False,
        "facecolor": "white",
        # "alignment": "top",
        "labelspacing": 1.1,
    }
    fig.legend(handles=custom_lines, labels=legend_labels, **legend_kwargs)

    return fig
