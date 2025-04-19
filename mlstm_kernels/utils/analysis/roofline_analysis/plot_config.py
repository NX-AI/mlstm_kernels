#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

fontsize_delta = 4.5
FONTSIZE = 12 + fontsize_delta
SMALL_OFFSET = 2
FONTSIZE_SMALL = FONTSIZE - SMALL_OFFSET
FONTSIZE_TICKS = FONTSIZE_SMALL

MARKERSIZE = 6.0
LINEWIDTH = 2.0  # default 1.5

FIGSIZE = (2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54)
FIGSIZE_2COL = (4 * 0.7 * 12 * 1 / 2.54, 2 * 0.7 * 8 * 1 / 2.54)

GRIDSPEC_KWARGS = {"wspace": 0.115, "hspace": 0}


def get_plot_mpl_context():
    return mpl.rc_context(
        rc={
            "text.usetex": False,
            "font.size": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "legend.fontsize": FONTSIZE_SMALL,
            "xtick.labelsize": FONTSIZE_TICKS,
            "ytick.labelsize": FONTSIZE_TICKS,
            "axes.titlesize": FONTSIZE,
            "lines.markersize": MARKERSIZE,
            "lines.linewidth": LINEWIDTH,
        }
    )


def savefig(fig, filename: str):
    dir = Path("./plots/")
    dir.mkdir(parents=True, exist_ok=True)

    if filename is not None:
        for file_ending in ["png", "pdf", "svg"]:
            file = Path(f"./plots/plot_{filename}.{file_ending}")
            fig.savefig(file, dpi=300, bbox_inches="tight", pad_inches=-0.0020)
