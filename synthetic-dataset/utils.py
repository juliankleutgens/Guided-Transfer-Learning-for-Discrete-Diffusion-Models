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


def sample_markov_utils(T, P, start=None):
    K = P.shape[0]
    s  = torch.empty(T, dtype=torch.long)
    s[0] = torch.randint(0, K, (1,)) if start is None else start
    for t in range(1, T):
        s[t] = torch.multinomial(P[s[t-1]], 1)
    return s

def build_simple_data(n_src=20_000, n_tgt=2_000, L=20):
    P = torch.tensor([[0.7,0.1,0.1,0.1],
                      [0.1,0.7,0.1,0.1],
                      [0.1,0.1,0.7,0.1],
                      [0.1,0.1,0.1,0.7]])
    Q = torch.tensor([[0.4, 0.2, 0.2, 0.2],
                      [0.2, 0.4, 0.2, 0.2],
                      [0.2, 0.2, 0.4, 0.2],
                      [0.2, 0.2, 0.2, 0.4]])

    def sample_markov(T, P, start=None):
        K = P.shape[0]
        s = torch.empty(T, dtype=torch.long)
        s[0] = torch.randint(0, K, (1,)) if start is None else start
        for t in range(1, T):
            s[t] = torch.multinomial(P[s[t - 1]], 1)
        return s
    src = torch.stack([sample_markov(L, P) for _ in range(n_src)])
    tgt = torch.stack([sample_markov(L, Q) for _ in range(n_tgt)])
    return src, tgt, P, Q, L, P.shape[0]


def print_seq(seq):
    print(" ".join([str(i.item()) for i in seq]))


def true_ratio(seq, Q, P):
    r = 1.0
    for t in range(1, len(seq)):
        r *= (Q[seq[t-1], seq[t]] / P[seq[t-1], seq[t]]).item()
    return r


def example_seq_ratio(seq, domain: str):
    domain = "source" if domain == "src" else "target"
    print_seq(seq)
    print("Example sequence: ",domain , "True ratio:  ", true_ratio(seq))



def get_avaverage_true_ratio(seqs, Q, P, domain: str):
    r = 0
    domain = "source" if domain == "src" else "target"
    for seq in seqs:
        r += true_ratio(seq, Q, P)
    print("Average true ratio in ", domain, "domain:", r / len(seqs))
    return r / len(seqs)




# ==============================================================
# Discrete GMM — independent tokens per dimension
# ==============================================================
# seqs : [B, L]  long   (mini‑batch of token indices)
@torch.no_grad()
def hamming_matrix(seqs: torch.Tensor) -> torch.Tensor:
    """
    Return a [B,B] integer matrix D where D[i,j] = Hamming(seqs[i], seqs[j]).
    """
    # broadcast‑compare and count unequal positions
    return (seqs.unsqueeze(1) != seqs.unsqueeze(0)).sum(-1)



def _sample_categorical(logits_or_probs: torch.Tensor) -> torch.Tensor:
    """Draws one sample along last dim using Gumbel–max (no gradients)."""
    if logits_or_probs.dtype in (torch.float16, torch.bfloat16):
        logits_or_probs = logits_or_probs.float()
    if logits_or_probs.min() < 0:  # assume logits
        logits = logits_or_probs
    else:                          # assume probs
        logits = logits_or_probs.log()
    gumbel = -torch.empty_like(logits).exponential_().log()  # −log U
    return (logits + gumbel).argmax(dim=-1)



def augment_hamming1(
    seqs: torch.LongTensor,
    mask_idx: int,
    vocab_size: int,
    include_original: bool = False,
) -> torch.LongTensor:
    """
    Return every Hamming‑1 mutant obtained by changing ONE non‑mask token
    in each sequence to every other vocabulary symbol (except <mask>).

    • Sequences consisting only of <mask> tokens are left unchanged
      (no valid position to flip).
    • Output shape:
        [ Σ_i (V‑2)·1{seq_i has ≥1 non‑mask} ]      (+ B originals if requested)
    """
    device = seqs.device
    B, L   = seqs.shape

    mutants = []
    keep_orig = [] if include_original else None

    for b in range(B):
        x = seqs[b]

        # -----------------------------------------------
        # 1. find indices that are NOT mask tokens
        # -----------------------------------------------
        non_mask_pos = (x != mask_idx).nonzero(as_tuple=True)[0]

        if non_mask_pos.numel() == 0:               # all‑mask → no mutants
            if include_original:
                keep_orig.append(x.unsqueeze(0))
            continue

        # pick ONE random non‑mask position to flip
        pos = non_mask_pos[torch.randint(0, non_mask_pos.numel(), (1,))]

        orig_tok = x[pos].item()

        # -----------------------------------------------
        # 2. generate all replacements in that position
        # -----------------------------------------------
        for v in range(vocab_size):
            if v in (mask_idx, orig_tok):
                continue
            x_mut = x.clone()
            x_mut[pos] = v
            mutants.append(x_mut.unsqueeze(0))

        if include_original:
            keep_orig.append(x.unsqueeze(0))

    if not mutants and not keep_orig:
        return seqs.new_empty(0, L)                 # nothing to return

    out = torch.cat(mutants, dim=0) if mutants else seqs.new_empty(0, L)
    if include_original and keep_orig:
        out = torch.cat([torch.cat(keep_orig, 0), out], dim=0)
    return out


def transition_stats(P_pred: torch.Tensor,
                     P_true: torch.Tensor,
                     config
) -> Tuple[torch.Tensor, float, float, float, float]:
    """
    Element‑wise error + mean of diff and a neat side‑by‑side print‑out.

    Returns
    -------
    abs_diff : (V,V) tensor of absolute errors
    mean_diff : float
    """
    diff      = P_pred - P_true
    abs_diff  = diff.abs()
    mean_diff = torch.mean(abs_diff)



    # ── pretty print for small V ───────────────────────────────────────
    V = P_pred.size(0)
    if V <= 10:                                     # keep console readable
        print("\nRow‑stochastic transition matrices (pred | true):")
        for i in range(V):
            pred_row = " ".join(f"{p:7.4f}" for p in P_pred[i])
            true_row = " ".join(f"{p:7.4f}" for p in P_true[i])
            print(f"  {pred_row}   |   {true_row}")

    # mean of the diagonal values of the predicted matrix
    mean_diag = torch.mean(P_pred.diagonal())
    print_statement = f"P error max={abs_diff.max():.4f}  µ_diff={mean_diff:.4f} "

    mean_extra = 0.0
    if config.extra_target_tokens > 0:
        last_col = diff[:, -config.extra_target_tokens:]
        last_row = (diff[-config.extra_target_tokens:, :])[:, :-config.extra_target_tokens]
        mean_extra = torch.mean(torch.cat([last_col.flatten(), last_row.flatten()]).abs())
        print_statement += f"mean_extra={mean_extra:.4f} "

    if config.type_of_T_matrix == "diagonal":
        print_statement += f"mean_diag={mean_diag:.4f}"
    print(print_statement)


    # KL Divergence between predicted and true transition matrices
    P_true = P_true + 1e-10  # avoid log(0)
    P_pred = P_pred + 1e-10  # avoid log(0)
    kl_div = F.kl_div(P_pred.log(), P_true, reduction='batchmean')
    print(f"KL Divergence: {kl_div:.4f}")

    return abs_diff, mean_diff, mean_diag, mean_extra, kl_div



def print_trace(trace: torch.Tensor,
                sample_idx: int = 0,
                mask_idx: Union[int, None] = None,
                idx2sym: Union[Dict[int, str], None] = None,
                vocab_size: int = 26) -> None:
    """
    Nicely print the evolution of *one* sequence.

    Parameters
    ----------
    trace      : (T+1, B, L) tensor returned by `sample_trace`
    sample_idx : which of the B sequences to visualise
    mask_idx   : optional token id reserved for <mask>;
                 it will be shown as '□' for clarity
    idx2sym    : optional mapping int → str for custom symbols
    """
    if idx2sym is None:
        # make 0: A, 1: B, 2: C, 3: D 5: E 6: .....
        idx2sym = {i: chr(i + 65) for i in range(0, vocab_size)}
        idx2sym = {**idx2sym, mask_idx: "□"}

    traj = trace[:, sample_idx]          # (T+1, L)
    T, L = traj.size(0) - 1, traj.size(1)

    def tok2str(tok: int) -> str:
        if idx2sym is not None:
            return idx2sym[int(tok)]
        if mask_idx is not None and tok == mask_idx:
            return "□"                  # visual blank
        return str(int(tok))

    print(f"\nDenoising trajectory for sample {sample_idx} "
          f"(length {L}, {T} steps):\n" + "-"*40)
    for step, row in enumerate(traj):
        line = " ".join(tok2str(t) for t in row)
        print(f"{step:02d}: {line}")

# utils.py
import os, random, numpy as np, torch

def seed_everything(seed: int = 0, deterministic: bool = False) -> None:
    """
    Set all relevant RNG seeds so results are repeatable.

    Parameters
    ----------
    seed : int
        Seed value to use for Python, NumPy and PyTorch.
    deterministic : bool
        If True, turns off CUDNN benchmarking and forces deterministic
        convolution algorithms (slower but bit‑exact).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # For multi‑GPU / CUDA / MPS
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def adjust_transition_matrix_size(S, T):
    """Adjust the size of the transition matrix to include extra target tokens."""
    if S.shape[0] == T.shape[0]:
        return S, T
    if S.shape[0] < T.shape[0]:
        S_zeros = torch.zeros_like(T)
        S_zeros[:S.shape[0], :S.shape[0]] = S
        return S_zeros, T
    elif S.shape[0] > T.shape[0]:
        T_zeros = torch.zeros_like(S)
        T_zeros[:T.shape[0], :T.shape[0]] = T
        return S, T_zeros
    return S, T