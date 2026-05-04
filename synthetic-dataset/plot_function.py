
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from inspect import signature
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import math, numpy as np
from typing import Union, Optional, Dict, Tuple
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from inspect import signature
import os
from datetime import datetime
import matplotlib.pyplot as plt

import os, re, numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
try:
    import wandb
except ImportError:
    wandb = None

def _slugify(text: str, max_len: int = 60) -> str:
    """
    Convert *text* to a filesystem-safe slug:
    - keep letters, digits, spaces, dash, underscore
    - collapse whitespace to single “_”
    - trim length (helps on Windows)
    """
    text = re.sub(r"[^\w\s\-]", "_", text)            # drop weird chars
    text = re.sub(r"\s+", "_", text).strip("_")        # spaces → _
    return text[:max_len] or "figure"                  # fallback

def save_fig(
    fig=None,
    filename: Union[str, None] = None,          # ← leave None to auto-name from title
    folder: str      = "plots",
    ext: str         = "png",
    dpi: int         = 300,
    tight: bool      = True,
    verbose: bool    = True,
):
    """
    Save *fig* to *folder/filename.ext*.

    If *filename* is **None**, we try in order:
      1. figure suptitle
      2. first axes title
      3. timestamp “figure_YYYYmmdd_HHMMSS”

    Usage
    -----
    >>> plt.plot(...)
    >>> plt.title("My awesome plot")      # or plt.suptitle(...)
    >>> save_fig()                        # → plots/My_awesome_plot.png
    """
    # -------- figure handle ------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    # -------- pick a base name --------------------------------------------
    if filename is None:
        # 1) suptitle
        title = fig._suptitle.get_text() if fig._suptitle else ""
        # 2) axes[0] title
        if not title and fig.axes:
            title = fig.axes[0].get_title()
        # 3) timestamp fallback
        if title:
            filename = _slugify(title)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{ts}"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"figure_{ts}_{np.random.randint(0,10)}"

    # ensure extension isn’t accidentally in *filename*
    filename = os.path.splitext(filename)[0]

    # -------- ensure directory exists -------------------------------------
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{filename}.{ext}")

    # -------- write file ---------------------------------------------------
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight" if tight else None,
    )

    if (wandb is not None and wandb.run is not None):
        # you can either log from the saved file:
        wandb.log({ f"fig/{filename}": wandb.Image(path) })

    #if verbose:
    #    print(f"✅  Saved figure → {path}")

    return path

def plot_ratio_heatmaps(true_grid, est_grid, edges, title_true="True", title_est="Estimated"):
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = ax[0].imshow(true_grid.numpy(), origin="lower", extent=extent,
                       interpolation="nearest", aspect="equal")
    ax[0].set_title(title_true); ax[0].set_xlabel(r"$x_1$"); ax[0].set_ylabel(r"$x_2$")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(est_grid.numpy() , origin="lower", extent=extent,
                       interpolation="nearest", aspect="equal")
    ax[1].set_title(title_est);  ax[1].set_xlabel(r"$x_1$"); ax[1].set_ylabel(r"$x_2$")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)

    plt.suptitle("Density-ratio heat maps (d = 2)")
    save_fig()
    plt.show()

def plot_ratio_single_grid(est_grid, edges, title="Estimated"):
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)

    im0 = ax.imshow(est_grid.numpy(), origin="lower", extent=extent,
                    interpolation="nearest", aspect="equal")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    fig.colorbar(im0, ax=ax, fraction=0.046)

    plt.title(title)
    save_fig()
    plt.show()

def ratio_vec_net_on_grid(model,pos, edges, device="cpu"):
    """
    Compute the ratio p(x)/q(x) on a grid of points using a trained model.
    The model is expected to take a tensor of shape (N, 2) as input and
    return a tensor of shape (N, 1) as output.
    masked_pos    : ℓ passed to models that expect a 'pos' argument
    """
    model.eval().to(device)

    # Cartesian product of grid indices → token pairs
    idx = torch.arange(len(edges), device=device) # [V, 1]
    id_pairs = torch.cat([idx.unsqueeze(1), idx.unsqueeze(1)], dim=1).long()   # [V, 2]
    t = torch.zeros(idx.shape[0], device=device)  # [V, 1]
    pos_in = torch.full((idx.shape[0],),pos,dtype=torch.long,device=device )
    id_pairs[:, pos] = model.token_emb.num_embeddings - 1  # mask token
    outputs = model(seq=id_pairs, pos=pos_in, t=t)  # [V, V]
    return outputs[:, :-1]  # Get rid of the last column (mask token)



def ratio_net_on_grid(model, edges, pos=0, device="cpu"):
    model.eval().to(device)

    # Cartesian product of grid indices → token pairs
    idx = torch.arange(len(edges), device=device)
    id_pairs = torch.cartesian_prod(idx, idx).long()   # [N, 2]

    with torch.no_grad():
        # Detect whether the forward() has 1 or 2 positional args
        n_inputs = len(signature(model.forward).parameters)
        if n_inputs == 1:           # RatioNet(seq)
            pred = model(id_pairs)
        elif n_inputs == 2:         # RatioNetAdaLN(seq, t)
            t = torch.zeros(id_pairs.shape[0], device=device)  # dummy time-step
            pred = model(id_pairs, t)
        elif n_inputs == 3:
            pred = ratio_vec_net_on_grid(model,pos, edges, device)
            return pred.cpu()
        # Squeeze trailing dim if present
        if pred.dim() > 1:
            pred = pred.squeeze(-1)

    return pred.cpu().reshape(len(edges), len(edges)).T

def true_ratio_grid(
    mu_S:    torch.Tensor,
    mu_T:    torch.Tensor,
    sigma2:  float = 0.1,
    grid_min: float = -1.5,
    grid_max: float =  1.5,
    step:     float = 0.01,
    device:   str   = "cpu",
    num_mixtures: int = 2,
):
    """
    Returns
    -------
    ratio_grid : [n_bins, n_bins]  torch.Tensor  (p(x)/q(x) on the grid)
    edges      : 1-D torch.Tensor  (grid centres, length = n_bins)
    """
    # ---- build grid --------------------------------------------------
    edges  = torch.arange(grid_min, grid_max + step * 0.5, step,
                          dtype=torch.float32, device=device)
    n_bins = edges.numel()

    xs, ys = torch.meshgrid(edges, edges, indexing="xy")          # [n_bins,n_bins]
    coords = torch.stack([xs.flatten(), ys.flatten()], dim=-1)    # [N,2]

    # ---- gaussian mixture densities ---------------------------------
    inv_2pi_sigma2 = 1.0 / (2 * math.pi * sigma2)

    def g(x, mu):
        return inv_2pi_sigma2 * torch.exp(-(x - mu).pow(2).sum(-1) / (2 * sigma2))

    if num_mixtures == 1:
        p = g(coords, mu_S.to(device))
        q = g(coords, mu_T.to(device))
    elif num_mixtures == 2:
        p = 0.5 * g(coords,  mu_S.to(device)) + 0.5 * g(coords, -mu_S.to(device))
        q = 0.5 * g(coords,  mu_T.to(device)) + 0.5 * g(coords, -mu_T.to(device))

    epsilon = 0
    ratio_grid = (q / (p + epsilon) + epsilon).reshape(n_bins, n_bins).cpu()  # [n_bins,n_bins]
    ratio_grid_log = torch.log(ratio_grid)               # [n_bins,n_bins]
    return ratio_grid, edges.cpu(), ratio_grid_log



def true_ratio_grid_3d(
    mu_S:     torch.Tensor,
    mu_T:     torch.Tensor,
    sigma2:   float   = 0.1,
    grid_min: float   = -1.5,
    grid_max: float   =  1.5,
    step:     float   =  0.01,
    device:   str     = "cpu",
):
    """
    Computes p(x)/q(x) on a uniform 3D grid for two 2‑component Gaussians.
    Returns:
      ratio_grid     : [B,B,B] torch.Tensor  with q/p at each grid point
      edges          : [B]       torch.Tensor  of grid centres
      ratio_grid_log : [B,B,B] torch.Tensor  of log(q/p)
    """
    # build 1D grid of centres
    edges  = torch.arange(grid_min, grid_max + 0.5*step, step,
                          dtype=torch.float32, device=device)
    B      = edges.numel()

    # form all grid coords in ℝ³
    xs, ys, zs = torch.meshgrid(edges, edges, edges, indexing="ij")
    coords     = torch.stack([xs.flatten(), ys.flatten(), zs.flatten()], dim=-1)

    # Gaussian density helper
    inv_2pi_sigma2 = 1.0 / ((2*math.pi*sigma2)**1.5)
    def g(x, mu):
        return inv_2pi_sigma2 * torch.exp(-((x - mu)**2).sum(-1) / (2*sigma2))

    # mixture densities p and q
    p = 0.5*g(coords,  mu_S.to(device)) + 0.5*g(coords, -mu_S.to(device))
    q = 0.5*g(coords,  mu_T.to(device)) + 0.5*g(coords, -mu_T.to(device))

    # reshape and move to CPU
    ratio = (q / (p + 1e-12)).reshape(B, B, B).cpu()
    return ratio, edges.cpu(), torch.log(ratio + 1e-12)


def plot_iid_gmm_3d(seqs, edges, title="empirical counts", limit_count=0):
    """
    seqs: [N,3] LongTensor of integer bin indices (built by build_iid_gmm_data)
    edges: 1D array-like of bin centers (or edges) of length B
    limit_count: caps any bin count at this value if > 0
    """
    d = seqs.size(1)
    if d != 3:
        raise ValueError("plot_iid_gmm_3d is only defined for 3‑D data (d=3).")

    B = len(edges)
    counts = torch.zeros(B, B, B, dtype=torch.long)
    discarded = 0

    for x_idx, y_idx, z_idx in seqs:
        if x_idx < 0 or x_idx >= B or y_idx < 0 or y_idx >= B or z_idx < 0 or z_idx >= B:
            discarded += 1
            continue
        counts[x_idx, y_idx, z_idx] += 1

    if discarded:
        print(f"discarded {discarded} out of {len(seqs)} samples due to containing a mask token")

    if limit_count > 0:
        counts = torch.clamp(counts, max=limit_count)

    # Convert to numpy
    counts_np = counts.numpy()
    edges_np = np.asarray(edges)

    # Find non‑zero bins
    ix, iy, iz = np.nonzero(counts_np)
    freqs    = counts_np[ix, iy, iz]

    # Map bin indices to coordinates
    x_coords = edges_np[ix]
    y_coords = edges_np[iy]
    z_coords = edges_np[iz]

    # Plot
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        x_coords, y_coords, z_coords,
        c=freqs,             # colour by count
        s=np.clip(freqs, 1, None),  # size by count (min size 1)
        cmap='viridis',
        depthshade=True
    )
    fig.colorbar(sc, label="count")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title(title)
    plt.tight_layout()
    save_fig()
    plt.show()

def plot_iid_gmm(seqs, edges, title="empirical counts", limit_count = 0):
    """seqs: [N,d] long tensor produced by build_iid_gmm_data"""
    d = seqs.size(1)
    if not d == 2:
        print("heat-map helper is 2-D only")
        return
    assert d == 2, "heat-map helper is 2-D only"
    img = torch.zeros(len(edges), len(edges))
    j = 0
    for x_idx, y_idx in seqs:
        if x_idx >= len(edges) or y_idx >= len(edges):
            j += 1
            continue
        img[x_idx, y_idx] += 1
    if limit_count > 0:
        img[img > limit_count] = limit_count
    if j > 0:
        print("discarded", j, "out of", len(seqs), "samples "", because there was still the Mask token")

    plt.figure(figsize=(5,4))
    plt.imshow(img.numpy(),
               origin="lower",
               extent=[edges[0], edges[-1], edges[0], edges[-1]],
               interpolation="nearest",
               aspect="equal")
    plt.colorbar(label="count")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.tight_layout()
    save_fig()
    plt.show()




def plot_iid_gmm_pair(src_seqs, tgt_seqs, edges,
                      titles=("Source p_X", "Target q_X"),
                      cmaps=("Blues", "Reds"),
                      limit_count=0, figsize=(10, 4)):
    """
    Side‑by‑side 2D histograms of (x₁,x₂) counts for source and target.

    Parameters
    ----------
    src_seqs   – LongTensor[N,2]: source index pairs
    tgt_seqs   – LongTensor[N,2]: target index pairs
    edges      – sequence of length M+1: bin edges along each axis
    titles     – 2‑tuple[str]: subplot titles
    cmaps      – 2‑tuple[colormap]: colormaps for each heat‑map
    limit_count– int: max bin count (0 = no cap)
    figsize    – 2‑tuple[float]: figure size in inches
    """
    def compute_img(seqs):
        img = torch.zeros(len(edges)-1, len(edges)-1)
        for x, y in seqs:
            if 0 <= x < img.size(0) and 0 <= y < img.size(1):
                img[x, y] += 1
        return torch.clamp(img, max=limit_count) if limit_count>0 else img

    plt.style.use('default')  # revert to white‐background style

    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        constrained_layout=True,
        facecolor='white'  # figure background
    )

    img_src, img_tgt = compute_img(src_seqs), compute_img(tgt_seqs)

    for ax, img, title, cmap in zip(axes, (img_src, img_tgt), titles, cmaps):
        ax.set_facecolor('white')  # axes background
        im = ax.imshow(
            img.numpy(),
            origin="lower",
            extent=[edges[0], edges[-1], edges[0], edges[-1]],
            interpolation="nearest",
            aspect="equal",
            cmap=cmap
        )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(title)

    fig.colorbar(im, ax=axes, label="count")
    save_fig()
    plt.show()

def plot_iid_gmm_points(seqs, edges,
                        title="IID GMM samples",
                        figsize=(5,4),
                        alpha=0.5, s=10, color="C0"):
    """
    Scatter‐plot of (x₁,x₂) samples from discrete indices.
    """
    assert seqs.size(1) == 2, "Only 2‑D supported"
    # compute bin‐centers
    idx = seqs.cpu().long()
    # keep only valid bins 0..len(centers)-1
    idx = idx[(idx[:, 0] >= 0) & (idx[:, 0] < len(edges)) &
                (idx[:, 1] >= 0) & (idx[:, 1] < len(edges))]
    num_mask_true = seqs.shape[0] - idx.shape[0]
    if num_mask_true > 0:
        print(f"Filtered out {num_mask_true} samples, because they contain the Mask token")
    x = edges[idx[:,0]]
    y = edges[idx[:,1]]

    plt.figure(figsize=figsize, facecolor="white")
    plt.scatter(x, y, s=s, alpha=alpha, c=color)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.xlim(edges[0], edges[-1])
    plt.ylim(edges[0], edges[-1])
    plt.tight_layout()
    save_fig()
    plt.show()

# ============================================================
# Decision‑boundary visualisation for a 2‑D domain classifier
# ============================================================

def plot_decision_boundary_classifier(
    model: torch.nn.Module,
    src: torch.Tensor,
    tgt: torch.Tensor,
    edges: Union[np.ndarray, torch.Tensor],
    *,
    steps: int = 400,
    device: Union[str, torch.device] = "cpu",
    cmap: str = "RdBu",
    alpha: float = 0.6,
    fig_size: tuple[int, int] = (6, 5),
):
    """
    Show p(domain = source | x) on a colour map plus the data clouds.

    Parameters
    ----------
    model   : trained classifier taking (B,2) tensors → σ‑probabilities.
    src     : (N,2) tensor with source samples (label 1 during training).
    tgt     : (M,2) tensor with target samples (label 0 during training).
    edges   : [xmin,xmax,ymin,ymax] array as returned by build_iid_gmm_data.
    steps   : grid resolution in each axis (larger ⇒ smoother).
    device  : CPU / GPU used for the forward pass.
    cmap    : matplotlib colormap for contourf.
    alpha   : opacity of scatter points.
    fig_size: size of the matplotlib figure.
    """
    model.eval().to(device)

    xmin, xmax, ymin, ymax = map(float, edges)
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, steps),
        np.linspace(ymin, ymax, steps),
    )
    grid = torch.from_numpy(
        np.c_[xx.ravel(), yy.ravel()]
    ).float().to(device)  # (steps²,2)

    with torch.no_grad():
        proba = model(grid).squeeze(-1).cpu().numpy()  # P(source|x)

    zz = proba.reshape(xx.shape)  # (steps,steps)

    # ---------- plotting ----------
    plt.figure(figsize=fig_size)
    # filled contours for decision surface (red≈target, blue≈source)
    plt.contourf(xx, yy, zz, levels=50, cmap=cmap, alpha=0.7)
    # decision boundary line (p=0.5)
    plt.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1)

    # data points
    plt.scatter(src[:, 0], src[:, 1], s=10, c="blue", alpha=alpha, label="source")
    plt.scatter(tgt[:, 0], tgt[:, 1], s=10, c="red", alpha=alpha, label="target")

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title("Classifier decision boundary")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    save_fig()
    plt.show()
    


def plot_distribution(edges: np.ndarray, probs: np.ndarray, title: str = "Distribution") -> None:
    """
    Plot a 1D or 2D discrete distribution defined on a grid.

    Parameters:
    - edges : array of grid points (length V)
    - probs : array of probabilities. Shape (V,) for 1D, (V,V) for 2D.
    - title : plot title.
    """
    # Validate dimensions
    if probs.ndim == 1:
        plt.figure()
        plt.plot(edges, probs, marker='o')
        plt.xlabel('x')
        plt.ylabel('Probability')
        plt.title(title)
        plt.grid(True)
        save_fig()
        plt.show()
    elif probs.ndim == 2:
        plt.figure()
        # Display as heatmap
        extent = [edges[0], edges[-1], edges[0], edges[-1]]
        plt.imshow(probs.T, origin='lower', extent=extent, aspect='auto')
        plt.colorbar(label='Probability')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(title)
        save_fig()
        plt.show()
    else:
        raise ValueError("plot_distribution only supports 1D or 2D distributions.")




def calc_log_ratio_discrete(
    src_probs: np.ndarray,
    tgt_probs: np.ndarray,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Compute elementwise log ratio log(tgt_probs / src_probs).
    Adds a small epsilon to avoid division by zero.

    Parameters:
    - src_probs: array of source probabilities (any shape)
    - tgt_probs: array of target probabilities (same shape)
    - eps: small constant for numerical stability

    Returns:
    - log_ratio: array of same shape, values = log((tgt+eps)/(src+eps))
    """
    if src_probs.shape != tgt_probs.shape:
        raise ValueError("src_probs and tgt_probs must have the same shape")
    src = src_probs + eps
    tgt = tgt_probs + eps
    return np.log(tgt / src)


def plot_log_ratio_discrete(
    edges: np.ndarray,
    log_ratio: np.ndarray,
    title: str = "Estimated Log Ratio"
) -> None:
    """
    Plot the log ratio of two discrete distributions on a grid.

    Parameters:
    - edges: array of grid points (length V)
    - log_ratio: array of log ratios. Shape (V,) for 1D, (V,V) for 2D.
    - title: plot title.
    """
    if log_ratio.ndim == 1:
        plt.figure()
        plt.plot(edges, log_ratio, marker='o')
        plt.xlabel('x')
        plt.ylabel('Log Ratio')
        plt.title(title)
        plt.grid(True)
        save_fig()
        plt.show()
    elif log_ratio.ndim == 2:
        fig, ax = plt.subplots()
        im = ax.imshow(log_ratio.T, origin="lower",
                       cmap="viridis", interpolation="nearest", vmin=-3, vmax=3)
        ax.set_xlabel("x₁ index")
        ax.set_ylabel("x₂ index")
        if title:
            ax.set_title(title)
            # lowest is -3 and highest is 3
        fig.colorbar(im, ax=ax, label="log(ratio)")
        save_fig()
        plt.show()
    else:
        raise ValueError("plot_log_ratio only supports 1D or 2D arrays.")



# =======================================================================
#  Function to Evaluate ratio estimator
# =======================================================================
def true_discrete_log_ratio_grid(
    src:   torch.Tensor,
    tgt:   torch.Tensor,
    edges: torch.Tensor,
    eps:   float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the log‑ratio log( q / p ) on a discrete (x₁,x₂) grid.

    Parameters
    ----------
    src   : (N,2) LongTensor – source samples as integer bin indices.
    tgt   : (M,2) LongTensor – target samples as integer bin indices.
    edges : 1‑D tensor/array  – bin edges (length  B+1  ⇒  B×B grid).
    eps   : float            – small constant for numerical stability.

    Returns
    -------
    p_map      : (B,B) NumPy array – empirical source probabilities  p(x).
    q_map      : (B,B) NumPy array – empirical target probabilities  q(x).
    log_ratio  : (B,B) NumPy array – log( q / (p+eps) ).
    """
    # ----------- set up ------------------------------------------------
    B        = len(edges) - 1                    # number of bins per axis
    p_counts = torch.zeros(B, B, dtype=torch.long)
    q_counts = torch.zeros_like(p_counts)

    # ----------- accumulate counts ------------------------------------
    def add_counts(samples, canvas):
        # keep only samples that fall inside the grid
        mask = (samples[:, 0] >= 0) & (samples[:, 0] < B) & \
               (samples[:, 1] >= 0) & (samples[:, 1] < B)
        valid = samples[mask].long()
        if valid.numel():                         # no‑op if all filtered out
            canvas.index_put_(
                (valid[:, 0], valid[:, 1]),
                torch.ones(valid.size(0), dtype=canvas.dtype),
                accumulate=True
            )

    add_counts(src, p_counts)
    add_counts(tgt, q_counts)

    # ----------- normalise to probabilities ---------------------------
    p_map = p_counts.double() / p_counts.sum().clamp(min=1)   # avoid /0 if empty
    q_map = q_counts.double() / q_counts.sum().clamp(min=1)

    p_np  = p_map.numpy()
    q_np  = q_map.numpy()

    # ----------- log‑ratio --------------------------------------------
    log_ratio = np.log( (q_np + eps) / (p_np + eps) )

    return p_np, q_np, log_ratio

def plot_log_ratio_map(source_map: np.ndarray,
                       target_map: np.ndarray,
                       eps: float = 1e-8,
                       title: str = "") -> None:
    """
    Plot log( target / (source + ε) ) on a heat‑map.

    Parameters
    ----------
    source_map, target_map : same shape, each summing to 1.
    eps                    : small constant to keep the denominator positive.
    title                  : optional figure title.
    """
    if source_map.shape != target_map.shape:
        raise ValueError("source_map and target_map must have identical shapes")

    ratio      = target_map / (source_map + eps)
    log_ratio  = np.log(ratio + eps)        # add eps so log(0) cannot occur

    fig, ax = plt.subplots()
    im = ax.imshow(log_ratio.T, origin="lower",
                   cmap="viridis", interpolation="nearest")
    ax.set_xlabel("x₁ index")
    ax.set_ylabel("x₂ index")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log(target / (source + ε))")
    save_fig()
    plt.show()

def plot_ratio_estimator(source_samples,target_samples,ratio_net, extras, config, extra_title=""):
    """
    Compute and visualize quality of ratio estimates for different data types.
    """
    ratio_net.eval()

    if config.data_type == "gaussian" or config.data_type == "point_forms":
        # True continuous ratio and log-ratio grid
        ratio_true, edges_ratio, ratio_grid_log = true_ratio_grid(
            mu_S=extras["mu_s"],
            mu_T=extras["mu_t"],
            grid_max=config.grid_size,
            grid_min= -config.grid_size,
            step=config.quantization_step,
            device=config.device,
            num_mixtures=config.num_mixtures,
        )

        edges = extras['edges']

        #ratio_true_discrete = true_discrete_log_ratio_grid(src=source_samples, tgt=target_samples, edges=edges)

        # Estimated ratio grid q/p on continuous grid
        if ratio_net.__class__.__name__ == "RatioNetAdaLNVector":
            idx_of_grids = [0, 1]
        else:
            idx_of_grids = [0]

        for i in idx_of_grids:
            est_grid = ratio_net_on_grid(ratio_net,pos=i, edges=edges_ratio, device=config.device)
            # Plot heatmaps side by side for true and estimated log-ratio
            if config.data_type == "gaussian":
                plot_ratio_heatmaps(
                ratio_grid_log, est_grid, edges_ratio,
                title_true="True log ratio p/q",
                title_est="Estimated log ratio p/q " + extra_title)
            plot_ratio_single_grid(est_grid, edges_ratio, title="Estimated log ratio p/q " + extra_title)

    elif config.data_type == "discrete":
        # True discrete ratio = target_probs / source_probs
        src_map = extras['src_probs']
        tgt_map = extras['tgt_probs']
        if "+ ratio net guided src and tgt" == extra_title:
            plot_log_ratio_map(src_map, tgt_map, eps=1e-8, title="Log‑ratio with ε‑stabilisation")

        # Estimated ratio grid (q/p) on discrete indices
        edges = extras['edges']
        est_ratio = ratio_net_on_grid( ratio_net, edges=edges, device=config.device ).numpy()
        est_ratio = est_ratio.T

        # Plot discrete log-ratio for true and estimated
        plot_log_ratio_discrete(edges, est_ratio, title="Estimated log ratio p/q " + extra_title)

    else:
        # do nothing
        print("")


def plot_prob_map(prob_map: np.ndarray, title: str = "") -> None:
    """
    Display a heat‑map of a discrete 2‑D probability array.

    Parameters
    ----------
    prob_map : 2‑D NumPy array whose entries sum to 1.
    title    : optional title for the figure.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(prob_map.T, origin="lower", interpolation="nearest")
    ax.set_xlabel("x₁ index")
    ax.set_ylabel("x₂ index")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Probability")
    save_fig()
    plt.show()

def plot_sample_counts(samples,
                       grid_shape: Union[tuple[int, int], None] = None,
                       title: Union[str,None] = None) -> None:
    """
    Heat‑map of raw counts from (x₁, x₂) samples.

    samples    – (N,2) integer coordinates (NumPy array or torch tensor)
    grid_shape – (nx,ny); if None, taken from max(sample)+1 in each axis
    """
    # accept either backend
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()

    samples = samples.astype(int)          # ensure integer indices

    if grid_shape is None:                 # deduce full canvas
        x_max = samples[:, 0].max()
        y_max = samples[:, 1].max()
        grid_shape = (x_max + 1, y_max + 1)

    # filter out the samples with Mask token
    num_samples = samples.shape[0]
    samples = samples[(samples[:, 0] < grid_shape[0]) & (samples[:, 1] < grid_shape[0])]
    num_samples_after = samples.shape[0]
    if num_samples_after < num_samples:
        print(f"Filtered out {num_samples - num_samples_after}, because they contain the Mask token")

    counts = np.zeros(grid_shape, dtype=int)
    np.add.at(counts, (samples[:, 0], samples[:, 1]), 1)

    fig, ax = plt.subplots()
    im = ax.imshow(counts.T, origin="lower", interpolation="nearest")
    ax.set_xlabel("x₁ index")
    ax.set_ylabel("x₂ index")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Count")
    save_fig()
    plt.show()



