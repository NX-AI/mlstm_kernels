#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Callable
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .plot_config import get_plot_mpl_context
from .roofline_analysis_mlstm import (
    Acc_intensity_a100,
    Acc_intensity_h100,
    Acc_intensity_v100,
    get_arithmetic_intensity_mlstmsig,
)


def plot_arithmetic_intensity(
    fns_calculate_algorithmic_intensity: list[Callable],
    p_qk: float,
    ax: Axes = None,
    num_points: int = 50,
    colors: list[str] = ["tab:blue", "tab:orange"],
    labels: list[str] = None,
    acc_intensities: list[float] = None,
    acc_labels: list[str] = None,
    show_title: bool = True,
    title_suffix: str = None,
    show_ylabel: bool = True,
    legend_args: dict = None,
    yticks: np.ndarray = None,
    xticks: np.ndarray = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    alpha_factors_causal_05_10: float = 0.3,
) -> Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if acc_intensities is not None:
        for acc_intensity in acc_intensities:
            ax.axhline(y=acc_intensity, color="grey", linestyle="-", alpha=0.7)

    if acc_labels is not None:
        for acc_intensity, label in zip(acc_intensities, acc_labels):
            if acc_intensity == Acc_intensity_v100:
                y_val = acc_intensity - 15
            else:
                y_val = acc_intensity
            ax.text(
                6000,
                y_val,
                label,
                color="grey",
                ha="left",
                va="center",
                # backgroundcolor="white",
            )

    max_chunk_size = 4096
    # calculate flops
    max_chunk_size_power_2 = np.log2(max_chunk_size).astype(int)
    chunk_sizes = np.logspace(3, max_chunk_size_power_2, num_points, base=2)

    for calculate_algorithmic_intensity, color in zip(
        fns_calculate_algorithmic_intensity, colors
    ):
        algorithmic_intensity_fcausal05 = calculate_algorithmic_intensity(
            chunk_size=chunk_sizes, factor_causal=0.5
        )
        algorithmic_intensity_fcausal1 = calculate_algorithmic_intensity(
            chunk_size=chunk_sizes, factor_causal=1
        )
        algorithmic_intensity_fcausal066 = calculate_algorithmic_intensity(
            chunk_size=chunk_sizes, factor_causal=0.66
        )

        ax.fill_between(
            chunk_sizes,
            algorithmic_intensity_fcausal05,
            algorithmic_intensity_fcausal1,
            alpha=0.1,
            color=color,
        )
        ax.plot(
            chunk_sizes,
            algorithmic_intensity_fcausal05,
            linestyle=":",
            color=color,
            label=r"$F_\text{causal}$=0.5",
            alpha=alpha_factors_causal_05_10,
        )
        ax.plot(
            chunk_sizes,
            algorithmic_intensity_fcausal1,
            linestyle="--",
            color=color,
            label=r"$F_\text{causal}$=1.0",
            alpha=alpha_factors_causal_05_10,
        )
        ax.plot(
            chunk_sizes,
            algorithmic_intensity_fcausal066,
            label=r"$F_\text{causal}$=0.66",
            color=color,
            zorder=100,
        )

    ax.set_xlabel(r"Chunk Size $L$ (logscale)")
    if show_ylabel:
        ax.set_ylabel("Arithmetic Intensity [FLOP/byte]\n(logscale)")

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
        for color in colors:
            custom_lines.append(Line2D([0], [0], color=color, lw=4))

        if labels is not None:
            legend_labels += labels
        ax.legend(
            handles=custom_lines,
            labels=legend_labels,
            **legend_args,
        )

    # max_exp = int(np.log2(head_dims_hv[-1]))
    # ax.set_xticks(np.logspace(0, max_exp, base=2, num=max_exp+1))
    # ax.set_xticks([16, 256, 512, 1024, 1536, 2048])
    # ax.set_xticks([32, 128, 256, 512, 768, 1024])
    ax.set_xscale("log", base=2)
    # ax.minorticks_off()
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if xticks is not None:
        ax.set_xticks(xticks)

    ax.set_yscale("log")
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    if yticks is not None:
        ax.set_yticks(yticks)

    if show_title:
        title = rf"$d_{{qk}}={p_qk}d_{{hv}}$ ($p_{{qk}}$={p_qk})"
        if title_suffix is not None:
            title += f" {title_suffix}"
        ax.set_title(title)

    ax.grid(alpha=0.3, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # ax.set_xscale("log", base=2)
    return ax.get_figure()


def create_mlstm_arithmetic_intensity_plot() -> Figure:
    p_qk = 0.5
    bytes_if = 2
    bytes_qkv = 2

    acc_intensities = [Acc_intensity_v100, Acc_intensity_a100, Acc_intensity_h100]
    acc_labels = [r"V100$\approx$133", r"A100$\approx$160", r"H100$\approx$295"]

    alg_int_dhv512_fp32 = partial(
        get_arithmetic_intensity_mlstmsig,
        d_hv=512,
        p_qk=p_qk,
        bytes_if=bytes_if,
        bytes_qkv=bytes_qkv,
        bytes_Cmn=4,
    )
    alg_int_dhv128_fp32 = partial(
        get_arithmetic_intensity_mlstmsig,
        d_hv=128,
        p_qk=p_qk,
        bytes_if=bytes_if,
        bytes_qkv=bytes_qkv,
        bytes_Cmn=4,
    )

    alg_int_dhv512_bf16 = partial(
        get_arithmetic_intensity_mlstmsig,
        d_hv=512,
        p_qk=p_qk,
        bytes_if=bytes_if,
        bytes_qkv=bytes_qkv,
        bytes_Cmn=2,
    )
    alg_int_dhv128_bf16 = partial(
        get_arithmetic_intensity_mlstmsig,
        d_hv=128,
        p_qk=p_qk,
        bytes_if=bytes_if,
        bytes_qkv=bytes_qkv,
        bytes_Cmn=2,
    )

    n_cols = 2
    fig_height = 4

    colors = list(reversed(["tab:blue", "tab:orange"]))
    labels = [
        r"$d_{hv}=512$",
        r"$d_{hv}=128$",
    ]

    legend_kwargs = {
        "loc": "upper right",
        "ncol": 1,
        "bbox_to_anchor": (1.17, 0.9),
        "frameon": False,
        "facecolor": "white",
        # "alignment": "top",
        "labelspacing": 1.1,
    }
    xticks = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    xlim = (16, 5100)
    ylim = (6, 1600)

    with get_plot_mpl_context():
        fig, ax = plt.subplots(
            1,
            n_cols,
            figsize=(1.5 * n_cols * fig_height, fig_height),
            sharey=True,
            sharex=True,
            gridspec_kw={"wspace": 0.1, "hspace": 0.5},
            # gridspec_kw={"width_ratios": [0.46, 0.54]},
        )
        fig = plot_arithmetic_intensity(
            list(reversed([alg_int_dhv512_fp32, alg_int_dhv128_fp32])),
            p_qk,
            ax=ax[0],
            # yticks=np.arange(0, 950, 100),
            colors=colors,
            labels=labels,
            acc_intensities=acc_intensities,
            # acc_labels=acc_labels,
            show_title=True,
            title_suffix="| States: float32",
            xticks=xticks,
            xlim=xlim,
            ylim=ylim,
        )
        fig = plot_arithmetic_intensity(
            list(reversed([alg_int_dhv512_bf16, alg_int_dhv128_bf16])),
            p_qk,
            ax=ax[1],
            # yticks=np.arange(0, 950, 100),
            colors=colors,
            labels=labels,
            acc_intensities=acc_intensities,
            acc_labels=acc_labels,
            show_ylabel=False,
            show_title=True,
            title_suffix="| States: bfloat16",
            xticks=xticks,
            xlim=xlim,
            ylim=ylim,
        )
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
        for color in colors:
            custom_lines.append(Line2D([0], [0], color=color, lw=4))

        if labels is not None:
            legend_labels += labels

        fig.legend(
            handles=custom_lines,
            labels=legend_labels,
            **legend_kwargs,
        )

    return fig


def create_mlstm_arithmetic_intensity_plot_poster() -> Figure:
    p_qk = 0.5
    bytes_if = 2
    bytes_qkv = 2

    acc_intensities = [Acc_intensity_v100, Acc_intensity_a100, Acc_intensity_h100]
    acc_labels = [r"V100$\approx$133", r"A100$\approx$160", r"H100$\approx$295"]

    alg_int_dhv512_fp32 = partial(
        get_arithmetic_intensity_mlstmsig,
        d_hv=512,
        p_qk=p_qk,
        bytes_if=bytes_if,
        bytes_qkv=bytes_qkv,
        bytes_Cmn=4,
    )
    alg_int_dhv128_fp32 = partial(
        get_arithmetic_intensity_mlstmsig,
        d_hv=128,
        p_qk=p_qk,
        bytes_if=bytes_if,
        bytes_qkv=bytes_qkv,
        bytes_Cmn=4,
    )

    n_cols = 1
    fig_height = 4

    colors = list(reversed(["tab:blue", "tab:orange"]))
    labels = [
        r"$d_{hv}=512$",
        r"$d_{hv}=128$",
    ]

    legend_kwargs = {
        "loc": "upper left",
        "ncol": 1,
        "bbox_to_anchor": (1.17, 0.9),
        "frameon": False,
        "facecolor": "white",
        # "alignment": "top",
        "labelspacing": 1.1,
    }
    xticks = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    xlim = (16, 5100)
    ylim = (6, 1600)

    # with get_plot_mpl_context():
    fig, ax = plt.subplots(
        1,
        n_cols,
        figsize=(1.5 * n_cols * fig_height, fig_height),
        sharey=True,
        sharex=True,
        gridspec_kw={"wspace": 0.1, "hspace": 0.5},
        # gridspec_kw={"width_ratios": [0.46, 0.54]},
    )
    fig = plot_arithmetic_intensity(
        list(reversed([alg_int_dhv512_fp32, alg_int_dhv128_fp32])),
        p_qk,
        ax=ax,
        # yticks=np.arange(0, 950, 100),
        colors=colors,
        labels=labels,
        acc_intensities=acc_intensities,
        acc_labels=acc_labels,
        show_title=True,
        # title_suffix="| States: float32",
        xticks=xticks,
        xlim=xlim,
        ylim=ylim,
    )

    from matplotlib.lines import Line2D

    custom_lines = []
    legend_labels = []
    for color in colors:
        custom_lines.append(Line2D([0], [0], color=color, lw=4))

    if labels is not None:
        legend_labels += labels

    fig.legend(
        handles=custom_lines,
        labels=legend_labels,
        **legend_kwargs,
    )

    return fig
