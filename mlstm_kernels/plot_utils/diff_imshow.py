import torch
from matplotlib import pyplot as plt


def convert_to_diff_imarray(target: torch.Tensor, baseline: torch.Tensor = None):
    assert baseline.ndim == target.ndim
    if baseline.ndim > 2:
        baseline = baseline.clone()
        target = target.clone()
        # take only first element of each dimension
        while baseline.ndim > 2:
            baseline = baseline[0, ...]
        while target.ndim > 2:
            target = target[0, ...]

    if baseline is None:
        imarr = target.detach().float().abs().squeeze().cpu().numpy()
    else:
        imarr = (
            (target.detach().float() - baseline.detach().float())
            .abs()
            .squeeze()
            .cpu()
            .numpy()
        )
    if imarr.ndim < 2:
        imarr = imarr[:, None]
    return imarr


def plot_numerical_diffs(
    pt_fp32_baseline,
    cu_fp32,
    cu_bf16,
    cu_half,
    title,
    vmin=0.0,
    vmax=1e-2,
    figsize=(10, 6),
):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=figsize, ncols=3)

    pos1 = ax1.imshow(
        convert_to_diff_imarray(cu_fp32, pt_fp32_baseline),
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title("float32")
    fig.colorbar(pos1, ax=ax1)
    pos2 = ax2.imshow(
        convert_to_diff_imarray(cu_bf16, pt_fp32_baseline),
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title("bfloat16")
    fig.colorbar(pos2, ax=ax2)
    pos3 = ax3.imshow(
        convert_to_diff_imarray(cu_half, pt_fp32_baseline),
        vmin=vmin,
        vmax=vmax,
    )
    ax3.set_title("float16")
    fig.colorbar(pos3, ax=ax3)
    fig.suptitle(title)
    return fig


def plot_numerical_diffs_single(
    baseline, target=None, title="", vmin=0.0, vmax=1e-2, figsize=(10, 6)
):
    fig, ax1 = plt.subplots(figsize=figsize)
    pos1 = ax1.imshow(
        convert_to_diff_imarray(baseline, baseline=target),
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title(title)
    fig.colorbar(pos1, ax=ax1)
    return fig


def plot_numerical_diffs_per_batchhead(
    baseline,
    target=None,
    title="",
    vmin=0.0,
    vmax=1e-2,
    figsize=(10, 6),
    rtol: float = None,
    atol: float = None,
    max_num_batchhead_plots: int = -1, # -1 means all
):
    baseline = baseline.reshape(-1, baseline.shape[-2], baseline.shape[-1])
    if target is not None:
        target = target.reshape(-1, target.shape[-2], target.shape[-1])

    if max_num_batchhead_plots > 0:
        num_batchheads = min(max_num_batchhead_plots, baseline.shape[0])
    else:
        num_batchheads = baseline.shape[0]

    figs = []
    for i in range(num_batchheads):
        max_diff = (baseline[i, ...] - target[i, ...]).abs().max()
        title_i = f"BH({i}):{title}|max_diff:{max_diff}"
        if rtol is not None and atol is not None:
            allclose = torch.allclose(
                baseline[i, ...], target[i, ...], rtol=rtol, atol=atol
            )
            title_i += f"|allclose(atol={atol},rtol={rtol}):{allclose}"
        fig = plot_numerical_diffs_single(
            baseline=baseline[i, ...],
            target=target[i, ...],
            title=title_i,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
        )
        figs.append(fig)
    return figs