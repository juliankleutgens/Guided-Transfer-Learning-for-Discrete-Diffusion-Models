import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, random
import torchsummary
from torchinfo import (
    summary)
from typing import Union, Optional, Dict, Tuple
import math

# ==================================================================================
# 1.  Synthetic Markov-shift data
# ==================================================================================
def sample_markov(T, P, start=None):
    K = P.shape[0]
    s  = torch.empty(T, dtype=torch.long)
    s[0] = torch.randint(0, K, (1,)) if start is None else start
    for t in range(1, T):
        s[t] = torch.multinomial(P[s[t-1]], 1)
    return s
def make_unigram_distribution(
    vocab_size: int,
    high_prob_tokens: torch.Tensor,
    high_prob: float,
    low_prob: float,
) -> torch.Tensor:
    """Return a length‑``vocab_size`` tensor of unigram probabilities.

    ``high_prob`` is assigned to each index in ``high_prob_tokens`` and
    ``low_prob`` to every other index; the vector is renormalised to sum to 1.
    """
    probs = torch.full((vocab_size,), low_prob, dtype=torch.float32)
    probs[high_prob_tokens] = high_prob
    probs /= probs.sum()  # exact normalisation
    return probs


def sample_unigram_sequences(
    num_sequences: int,
    seq_len: int,
    probs: torch.Tensor,
    *,
    device: str = "cpu",
) -> torch.Tensor:
    """IID sampling of categorical tokens.

    Returns a ``[num_sequences, seq_len]`` tensor of ``torch.long``.
    """
    categorical = torch.distributions.Categorical(probs=probs.to(device))
    return categorical.sample((num_sequences, seq_len))


def build_unigram_data(
    *,
    n_src: int = 20_000,
    n_tgt: int = 2_000,
    L: int = 64,
    vocab_size: int = 100,
    topic_split: int = 50,
    high: float = 0.016,
    low: float = 0.004,
    device: str = "cpu",
    include_oov: bool = False,
    oov_extra: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate source/target datasets for the bag‑of‑words topic‑shift case.

    Topic A tokens are ``0 … topic_split‑1``; Topic B are the remainder.  If
    ``include_oov`` is *True*, we append ``oov_extra`` new tokens that belong
    exclusively to Topic B, expanding the vocabulary.

    Returns
    -------
    src_seqs : ``[n_src, L]`` long tensor
    tgt_seqs : ``[n_tgt, L]`` long tensor
    p_probs  : ``[V]`` float tensor (source unigram)
    q_probs  : ``[V]`` float tensor (target unigram)
    """
    if  topic_split >= vocab_size:
        topic_split = int(vocab_size / 2)
    if include_oov:
        vocab_size += oov_extra
        topic_b_tokens = torch.arange(topic_split, vocab_size)
    else:
        topic_b_tokens = torch.arange(topic_split, vocab_size)

    topic_a_tokens = torch.arange(0, topic_split)

    p_probs = make_unigram_distribution(
        vocab_size,
        high_prob_tokens=topic_a_tokens,
        high_prob=high,
        low_prob=low,
    )

    q_probs = make_unigram_distribution(
        vocab_size,
        high_prob_tokens=topic_b_tokens,
        high_prob=high,
        low_prob=low,
    )

    src = sample_unigram_sequences(n_src, L, p_probs, device=device)
    tgt = sample_unigram_sequences(n_tgt, L, q_probs, device=device)

    return src, tgt, p_probs, q_probs

def make_transition_matrix(vocab_size: int,
                           diag_prob: float,
                           device: str = "cpu") -> torch.Tensor:
    """
    Construct a [V,V] transition matrix whose diagonal entries are
    `diag_prob` and every off-diagonal entry is equal so that
    each row sums to 1.
    """
    off_prob = (1.0 - diag_prob) / (vocab_size - 1)
    T = torch.full((vocab_size, vocab_size), off_prob, device=device)
    idx = torch.arange(vocab_size, device=device)
    T[idx, idx] = diag_prob
    return T

def random_transition_matrix(
        vocab_size: int,
        dirichlet_alpha: float = 1.0,
        device: str = "cpu") -> torch.Tensor:
    """
    Return a [V,V] matrix where each row is sampled from
    Dirichlet(alpha * 1_V).  alpha=1 ⇒ uniform over the simplex.
    """
    alpha_vec = torch.full((vocab_size,), dirichlet_alpha, device=device)
    d = torch.distributions.Dirichlet(alpha_vec)
    T = d.sample((vocab_size,))            # shape [V,V]
    return T


def build_markov_transfer(
    n_src: int                = 20_000,
    n_tgt: int                = 2_000,
    L: int                    = 128,
    vocab_size: int           = 20,
    type_of_T_matrix: str     = "diagonal", # "diagonal" or "random" or "anti-diagonal"
    diag_src: float           = 0.7,
    diag_tgt: float           = 0.4,
    dirichlet_alpha_src: float = 1.0,
    dirichlet_alpha_tgt: float = 1.0,
    extra_target_tokens: int  = 0,
    device: str               = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
            int]:
    """
    Generate source / target Markov datasets where the **target**
    domain may contain `extra_target_tokens` symbols that never
    appear in the source domain.

    Returns
    -------
    src_seq : [n_src, L] source sequences
    tgt_seq : [n_tgt, L] target sequences
    P_full  : [(V+K),(V+K)]   source transition matrix (zero rows/cols
               for the extra K tokens)
    Q_full  : [(V+K),(V+K)]   target transition matrix
    V       : int   original vocabulary size
    V_tgt   : int   target-domain vocabulary size (V + K)
    """
    # -------- core vocab matrices --------
    if type_of_T_matrix == "diagonal"  or type_of_T_matrix == "anti-diagonal":
        P_core = make_transition_matrix(vocab_size, diag_src, device)
        Q_core = make_transition_matrix(vocab_size + extra_target_tokens,diag_tgt, device)
        if type_of_T_matrix == "anti-diagonal":
           P_core = torch.flip(P_core, dims=[1])
    elif type_of_T_matrix == "random":
        P_core = random_transition_matrix(vocab_size,dirichlet_alpha=dirichlet_alpha_src, device=device)
        Q_core = random_transition_matrix(vocab_size + extra_target_tokens,dirichlet_alpha=dirichlet_alpha_tgt,
                                          device=device)
    else:
        raise ValueError("unknown type_of_T_matrix")

    # -------- sample datasets --------
    src_seq = torch.stack([sample_markov(L, P_core)
                           for _ in range(n_src)]).to(device)

    tgt_seq = torch.stack([sample_markov(L, Q_core)
                           for _ in range(n_tgt)]).to(device)
    if extra_target_tokens > 0:
        vocab_size += extra_target_tokens
    return src_seq, tgt_seq, P_core, Q_core, vocab_size


class SeqDS(TensorDataset):
    def __init__(self, seqs): self.seqs = seqs
    def __len__(self):  return self.seqs.size(0)
    def __getitem__(self, i): return self.seqs[i]



def orth_means_random(dimension: int,num_mixtures: int = 2, device: str = "cpu"):
    """
    Draws two random orthogonal unit vectors in ℝᵈ and rescales each
    to length 1/√2 so that their distance is exactly 1.
    d: dimension of the vectors
    """

    A = torch.randn(dimension, 2, device=device)   # random d×2 matrix
    Q, _ = torch.linalg.qr(A)              # Q has orthonormal columns
    r = 1 / math.sqrt(2)
    mu_s = r * Q[:, 0]                     # first orthonormal column
    mu_t = r * Q[:, 1]                     # second orthonormal column
    if dimension == 2:
        mu_s = torch.tensor([0.5, 0.5], device=device)
        mu_t = torch.tensor([0.5, -0.5], device=device)
    if dimension == 2 and num_mixtures == 1:
        mu_s = torch.tensor([0.5, 0.5])
        mu_t = torch.tensor([-0.5, -0.5])

    return mu_s, mu_t



def build_iid_gmm_data(
    n_src: int           = 20_000,
    n_tgt: int           = 2_000,
    mu_src: torch.Tensor = torch.tensor([0.5,  0.5]),
    mu_tgt: torch.Tensor = torch.tensor([0.5, -0.5]),
    sigma2: float        = 0.1,
    grid_min: float      = -1.5,
    grid_max: float      =  1.5,
    step: float          = 0.01,
    device: str          = "cpu",
    num_mixtures: int = 2,
):
    """
    Returns
    -------
    src_seqs : [n_src, d]  long   -- token ids for source mixture
    tgt_seqs : [n_tgt, d]  long   -- token ids for target mixture
    vocab_sz : int                -- same for every position
    edges    : 1-D numpy array    -- grid centres for plotting
    num_mixtures : int            -- number of mixtures, but can not be higher than 2
    """

    d = mu_src.numel()                          # sequence length
    assert mu_tgt.numel() == d, "mu vectors must match in length"
    if num_mixtures > d:
        num_mixtures = d

    edges  = np.arange(grid_min, grid_max + step * 0.5, step)
    vocab_sz = len(edges)                       # e.g. 301

    # --- helper: N samples from 0.5 N(μ,σ²I) + 0.5 N(−μ,σ²I)
    def _sample_two_mixtures(n, mu):
        z = torch.randn(n, d) * math.sqrt(sigma2)
        signs = torch.randint(0, 2, (n, 1)).float() * 2 - 1
        return signs * mu + z

    # -- . helper: N samples from 0.5 N([1 , 1] μ,σ²I) + 0.5 N([-1 , 1] μ,σ²I)
    def _sample_two_mixtures_2(n, mu):
        z = torch.randn(n, d) * math.sqrt(sigma2)
        signs = torch.randint(0, 2, (n, 1)).float() * 2 - 1
        dirs = torch.cat([signs, torch.ones_like(signs)], dim=1)
        return dirs * mu + z


    # --. helper one mixture 0.5 N(μ,σ²I)
    def _sample_one_mixture(n, mu):
        z = torch.randn(n, d) * math.sqrt(sigma2)
        return mu + z

    if num_mixtures == 1:
        src_cont = _sample_one_mixture(n_src, mu_src)
        tgt_cont = _sample_one_mixture(n_tgt, mu_tgt)
    elif num_mixtures == 2:
        src_cont = _sample_two_mixtures(n_src, mu_src)
        tgt_cont = _sample_two_mixtures(n_tgt, mu_tgt)

    # --- discretise each coordinate independently
    def _to_tokens(xy: torch.Tensor) -> torch.Tensor:
        idx = torch.round((xy - grid_min) / step).long()          # [N,d]
        idx.clamp_(0, vocab_sz - 1)
        return idx                                               # already long

    src_ids = _to_tokens(src_cont)
    tgt_ids = _to_tokens(tgt_cont)

    return src_ids, tgt_ids, vocab_sz, edges


def estimate_transition_matrix(
    seqs: torch.LongTensor,
    vocab_size: int,
    mask_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Empirical one‑step transition matrix P̂  where
    P̂[i, j] = Pr(token_{t+1}=j | token_t=i).

    Parameters
    ----------
    seqs       : (N, L) LongTensor of tokens in [0, vocab_size)
    vocab_size : number of distinct tokens V
    mask_idx   : id of a “mask” token to ignore (None → no masking)

    Returns
    -------
    P_hat      : (V, V) tensor with each row summing to 1
    """
    mask_idx = vocab_size if mask_idx is None else mask_idx
    # ── drop any sequence containing the mask token ─────────────────────
    M = seqs.size(0)                      # number of sequences
    if mask_idx is not None:
        seqs = seqs[~seqs.eq(mask_idx).any(dim=1)]
    if seqs.numel() == 0:
        print("No valid sequences after mask filtering.")
        return torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
    if seqs.ndim != 2:
        print("No valid sequences after mask filtering.")
        return torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
    N = seqs.size(0)                      # number of valid sequences
    print(f"Number of sequences dropped: {M - N} out of {M}. ({(M-N)/M*100:.2f}%)")


    # ── collect adjacent pairs (x_t, x_{t+1}) ───────────────────────────
    src = seqs[:, :-1].reshape(-1)              # token_t
    dst = seqs[:,  1:].reshape(-1)              # token_{t+1}
    flat_idx = src * vocab_size + dst           # linearised (i, j)

    counts = torch.bincount(flat_idx, minlength=vocab_size**2).float()
    counts = counts.view(vocab_size, vocab_size)

    row_sums = counts.sum(dim=1, keepdim=True).clamp_min(1e-8)
    P_hat = counts / row_sums
    return P_hat


def create_probability_maps(grid_max: int,
                            source_dist: str,
                            target_dist: str,
                            distance: int = 0) -> Tuple[np.ndarray, np.ndarray, int, int]:

    new_x_max = grid_max + max(distance, 0)          # union domain in x₁
    xs_big, ys_big = np.meshgrid(np.arange(new_x_max + 1),
                                 np.arange(new_x_max + 1),
                                 indexing="ij")

    def _pdf(kind: str, xs, ys):
        if kind == "uniform":
            pdf = np.ones_like(xs, dtype=float)
        elif kind == "gaussian":
            c = grid_max / 2
            s = max(grid_max, 1) / 4
            pdf = np.exp(-((xs - c)**2 + (ys - c)**2) / (2 * s**2))
        elif kind == "x2":
            pdf = xs.astype(float)**2
        else:
            raise ValueError("unknown distribution")
        return pdf / pdf.sum()

    # build small reference maps on 0 … grid_max
    xs_small, ys_small = np.meshgrid(np.arange(grid_max + 1),
                                     np.arange(grid_max + 1),
                                     indexing="ij")
    src_small   = _pdf(source_dist, xs_small, ys_small)
    tgt_small   = _pdf(target_dist, xs_small, ys_small)

    # embed them in the larger canvas
    source_map = np.zeros_like(xs_big, dtype=float)
    source_map[:grid_max + 1, :grid_max + 1] = src_small       # same placement

    target_map = np.zeros_like(xs_big, dtype=float)
    target_map[distance:grid_max + distance + 1, :grid_max + 1] = tgt_small

    vocab_size = grid_max + distance + 1
    sequence_length = 2
    return source_map, target_map, vocab_size, sequence_length



def sample_from_prob_map(prob_map: np.ndarray,
                         n_samples: int,
                         rng: Optional[np.random.Generator] = None,
                         ) -> np.ndarray:
    """
    Draw `n_samples` points (x₁, x₂) from a discrete probability map.

    Returns
    -------
    samples : (n_samples, 2) array of integer coordinates.
    """
    if rng is None:
        rng = np.random.default_rng()

    flat_p = prob_map.ravel()
    flat_p = flat_p / flat_p.sum()           # just in case

    idx = rng.choice(flat_p.size, size=n_samples, p=flat_p)
    x1, x2 = divmod(idx, prob_map.shape[1])  # row, column
    samples = np.column_stack((x1, x2))
    return torch.tensor(samples)




# =============================================================================
#  Discrete Gray-code toy shapes (2-spirals, 8-Gaussians, …)
# =============================================================================

def _discretise(points: torch.Tensor,
                grid_min: float,
                discrete_step: float) -> torch.LongTensor:
    """
    Convert continuous coords -> [0 … V-1] integer ids independently per axis.
    """
    idx = torch.round((points - grid_min) / discrete_step).long()
    return idx.clamp(min=0)

def _make_2spirals(n: int, noise: float = .2) -> torch.Tensor:
    theta = torch.sqrt(torch.rand(n)) * 4 * math.pi      # ~ [0, 4π]
    r     = 2.0 * theta
    x1    =  r * torch.cos(theta) + torch.randn(n) * noise
    y1    =  r * torch.sin(theta) + torch.randn(n) * noise
    x2, y2 = -x1, -y1                                     # second arm
    pts = torch.stack([torch.cat([x1, x2]), torch.cat([y1, y2])], dim=1)
    return pts

def _make_8gaussians(n: int, radius: float = 4.0, noise: float = .2) -> torch.Tensor:
    centers = torch.tensor([(1,0), (-1,0), (0,1), (0,-1),
                            (1/ math.sqrt(2),  1/ math.sqrt(2)),
                            (1/ math.sqrt(2), -1/ math.sqrt(2)),
                            (-1/ math.sqrt(2), 1/ math.sqrt(2)),
                            (-1/ math.sqrt(2),-1/ math.sqrt(2))]) * radius
    idx     = torch.randint(0, 8, (n,))
    pts     = centers[idx] + torch.randn(n, 2) * noise
    return pts

def _make_circles(n: int, r1=2.0, r2=5.0, noise=.2) -> torch.Tensor:
    n1 = n // 2
    n2 = n - n1
    th1 = torch.rand(n1) * 2 * math.pi
    th2 = torch.rand(n2) * 2 * math.pi
    inner = torch.stack([r1*torch.cos(th1), r1*torch.sin(th1)], dim=1)
    outer = torch.stack([r2*torch.cos(th2), r2*torch.sin(th2)], dim=1)
    pts   = torch.cat([inner, outer]) + torch.randn(n,2)*noise
    return pts

def _make_moons(n: int, noise=.2) -> torch.Tensor:
    n1 = n // 2
    n2 = n - n1
    # upper moon
    th1 = torch.rand(n1) * math.pi
    x1  = torch.stack([torch.cos(th1), torch.sin(th1)], dim=1)
    # lower moon (shifted)
    th2 = torch.rand(n2) * math.pi
    x2  = torch.stack([1 - torch.cos(th2),  -torch.sin(th2) - .5], dim=1)
    pts = torch.cat([x1, x2]) * 5.0 + torch.randn(n,2)*noise
    return pts

def _make_pinwheel(n: int, noise=.2, radial_std=.3,
                   tangential_std=.05, num_arms=5) -> torch.Tensor:
    r = torch.randn(n) * radial_std + 1.0
    theta = torch.randint(0, num_arms, (n,)).float() * 2*math.pi/num_arms \
            + r * 1.5
    pts = torch.stack([r*torch.cos(theta), r*torch.sin(theta)], dim=1)
    pts += torch.randn(n,2)*tangential_std
    return pts * 3.0 + torch.randn(n,2)*noise


def _make_swissroll(n: int, noise=0.6) -> torch.Tensor:
    t = 1.5 * math.pi * (1 + 2 * torch.rand(n))  # Spiral angle
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    pts = torch.stack([x, y], dim=1)
    pts += noise * torch.randn_like(pts)
    return pts

def _make_checkerboard(n: int) -> torch.Tensor:
    # 1) build list of all cell‐corner coordinates whose (i+j) is even
    cells = [(i, j)
             for i in range(-4, 4)
             for j in range(-4, 4)
             if (i + j) % 2 == 0]
    corners = torch.tensor(cells, dtype=torch.float)    # shape (32,2)

    # 2) pick n cells at random (with replacement)
    idx   = torch.randint(0, corners.size(0), (n,))     # shape (n,)
    base  = corners[idx]                                # shape (n,2)

    # 3) add a uniform offset in [0,1)×[0,1) to land inside each 1×1 square
    pts   = base + torch.rand(n, 2)                     # shape (n,2)

    return pts


# ---------------------------------------------------------------------

Array = Union[np.ndarray, torch.Tensor]

def rescale_to_box_torch(
    data: torch.Tensor,
    grid_min: float = -1.0,
    grid_max: float = 1.0,
    per_dimension: bool = True,
) -> torch.Tensor:
    """
    Affine-rescale a tensor so every entry lies in [grid_min, grid_max].

    Parameters
    ----------
    data           : (*, d) float tensor  (last dim = coordinates)
    grid_min/max   : target interval
    per_dimension  : True  → rescale each coordinate axis separately
                     False → use one global min/max (keeps aspect ratio)

    Returns
    -------
    scaled_data    : tensor with same shape & dtype as `data`
    """
    if not data.is_floating_point():
        data = data.float()

    # --- compute span --------------------------------------------------
    if per_dimension:
        lo = data.amin(dim=0, keepdim=True)
        hi = data.amax(dim=0, keepdim=True)
    else:
        lo = data.amin()
        hi = data.amax()

    span = torch.clamp(hi - lo, min=torch.finfo(data.dtype).eps)

    # --- map to [0,1] then to [grid_min, grid_max] ---------------------
    scaled = (data - lo) / span
    scaled = scaled * (grid_max - grid_min) + grid_min
    return scaled


def discrete_gray_codes_forms(
        form          : str   = "2spirals",
        grid_min      : float = -6.0,
        grid_max      : float =  6.0,
        discrete_step : float =  0.02,
        n_samples     : int   = 1000,
        device        : str   = "cpu",
) -> Tuple[torch.LongTensor, int, torch.LongTensor]:
    """
    Sample `n_samples` points from one of the toy 2-D shapes, then
    **quantise** each coordinate to a vocabulary index.

    Parameters
    ----------
    form          : one of {"2spirals","8gaussians","circles","moons",
                            "pinwheel","swissroll","checkerboard"}
    grid_min/max  : span of the square domain (same for x & y)
    discrete_step : width of a voxel → vocab size ≈ (grid_max-grid_min)/step
    n_samples     : total points to sample

    Returns
    -------
    tokens  : (n_samples,2) LongTensor of token IDs
    vocab   : int  (same for both positions)
    """
    # ---- draw continuous points centred at 0 --------------------------------
    if   form == "2spirals"   : pts = _make_2spirals  (n_samples)
    elif form == "8gaussians" : pts = _make_8gaussians(n_samples)
    elif form == "circles"    : pts = _make_circles   (n_samples)
    elif form == "moons"      : pts = _make_moons     (n_samples)
    elif form == "pinwheel"   : pts = _make_pinwheel  (n_samples)
    elif form == "swissroll"  : pts = _make_swissroll (n_samples)
    elif form == "checkerboard":pts = _make_checkerboard(n_samples)
    else:
        raise ValueError(f"Unknown form '{form}'.")

    # ---- shift so that the shape’s centre is midway between grid-min/max ----
    centre_shift = (grid_min + grid_max) / 2.0
    pts = pts + centre_shift
    # rescale to fit in the square [grid_min, grid_max]²
    pts = rescale_to_box_torch(pts, grid_min, grid_max, per_dimension=True)

    # ---- discretise ---------------------------------------------------------
    tokens = _discretise(pts, grid_min, discrete_step).to(device)

    # clamp to the square [grid_min,grid_max]¹ to avoid stray indices
    vmax = math.ceil((grid_max - grid_min) / discrete_step)
    tokens = tokens.clamp(0, vmax)

    edges = np.arange(grid_min, grid_max + discrete_step * 0.5, discrete_step)
    vocab_size = vmax + 1
    return tokens.long(), vocab_size, edges




