import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, random
import torchsummary
from torchinfo import (
    summary)
import torch.nn.functional as F
from typing import Union, Optional, Dict, Tuple
import math
import time
from tqdm.auto import tqdm
from models import ClassifierNet, RatioNetAdaLN

class LogLinearNoise(nn.Module):
    """σ(t) = exp(log σ_min + t (log σ_max − log σ_min)),   t∈[0,1]"""
    def __init__(self, sigma_min: float = 1e-4, sigma_max: float = 20.0):
        super().__init__()
        self.register_buffer("sigma_min", torch.tensor(sigma_min))
        self.register_buffer("sigma_max", torch.tensor(sigma_max))
    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_sigma = torch.log(self.sigma_min) + t * (torch.log(self.sigma_max) - torch.log(self.sigma_min))
        sigma = log_sigma.exp()
        dsigma = sigma * (torch.log(self.sigma_max) - torch.log(self.sigma_min))  # dσ/dt
        return sigma, dsigma


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


def joint_sample_two_positions(log1: torch.Tensor,
                               log2: torch.Tensor,
                               max_chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Jointly sample (x1,x2) for a batch using the log‑probs of
    position 0 and position 1.
    log1 , log2 : (B, V)      – logits or log‑probs for the two slots
    Returns
    -------
    x1, x2 : LongTensor (B,)  – drawn coordinate indices
    """
    B, V = log1.shape
    out1 = torch.empty(B,  dtype=torch.long, device=log1.device)
    out2 = torch.empty_like(out1)

    # process the batch in chunks to avoid allocating (B,V,V) at once
    for start in range(0, B, max_chunk):
        end   = min(start + max_chunk, B)

        # log P(i,j)  =  log p1(i) + log p2(j)
        joint_log = log1[start:end, :, None] + log2[start:end, None, :]  # (b, V, V)
        joint_log = joint_log.view(end-start, -1)                        # (b, V²)

        # convert to probabilities, draw one index, map back to (i,j)
        probs = joint_log.softmax(-1)
        idx   = torch.multinomial(probs, 1).squeeze(1)                  # (b,)

        out1[start:end] = idx // V
        out2[start:end] = idx %  V

    return out1, out2


class DiffusionConfig:
    batch_size: int = 512 * 4
    vocab_size: int
    seq_len: int
    mask_idx: Union[int, None] = None  # absorbing state index; if None use uniform diffusion
    T_sampling: int = 30  # # steps for deterministic sampler
    gamma: float = 2.0  # guidance strength (γ)
    use_approx: bool = True  # first‑order guidance if True
    end_time: float = 1e-5  # end time for sampling
    use_plg: bool = False  # use Posterior Logit Guidance (PLG) (positive logit guidance)
    eta: float = 1  # PLG strength (η)
    stochastic_sampling_jitter_mask: float = 0.0  # jitter mask probability (for absorbing state diffusion)
    stochastic_sampling_eps_noise: float = 0.0  # noise fraction (for uniform diffusion)
    k_best_sampling: int = 0  # k-best sampling (0 = no k-best sampling)
    top_n_ratio: int = -1  # top-n ratio evaluation (−1 = full eval)



class Diffusion(nn.Module):
    """Discrete diffusion with ratio‑based guidance.

    * `denoiser` maps (xt, sigma_t) ➜ (B,L,V) **logits**.
    * `ratio_model` is a `RatioGuidance` wrapper.
    """

    def __init__(self, denoiser: nn.Module, ratio_model: RatioNetAdaLN, cfg: DiffusionConfig, use_approx: bool = False):
        super().__init__()
        self.denoiser = denoiser
        self.ratio_model = ratio_model.eval()
        self.cfg = cfg
        self.noise = LogLinearNoise()  # could be swapped
        self.vocab_size = cfg.vocab_size
        self.mask_idx = cfg.mask_idx if cfg.mask_idx is not None else self.vocab_size  # sentinel
        self.batch_size = cfg.batch_size
        self.end_time = cfg.end_time
        self.use_plg = cfg.use_plg
        self.k_best_sampling = cfg.k_best_sampling if hasattr(cfg, "k_best_sampling") else 0
        self.jitter_mask = cfg.stochastic_sampling_jitter_mask if hasattr(cfg, "stochastic_sampling_jitter_mask") else 0.0
        self.eps_noise = cfg.stochastic_sampling_eps_noise if hasattr(cfg, "stochastic_sampling_eps_noise") else 0.0
        self.top_n_ratio = cfg.top_n_ratio if hasattr(cfg, "top_n_ratio") else -1
        self.use_approx = use_approx
        self.neg_infinity = -1e9
        if cfg.mask_idx is None:
            self.diffusion = "uniform"
        else:
            self.diffusion = "absorbing_state"

    # ---------------------------------------------------------------------
    # forward: one network pass ⇒ log‑probs over vocabulary
    # ---------------------------------------------------------------------
    def forward(self, x: torch.LongTensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            sigma = sigma[:, None]
        logits = self.denoiser(x, sigma)  # (B,L,V)
        return logits.log_softmax(dim=-1)  # log‑probs

    def model_prediction(self, xt: torch.LongTensor, sigma_t: torch.Tensor) -> torch.Tensor:
        """
        Perform chunked denoiser forward passes to avoid MPS out-of-memory errors.

        Args:
            xt (torch.LongTensor): current sequences tensor of shape (B, L).
            sigma_t (torch.Tensor): noise levels tensor of shape (B,).
            max_chunk (int): maximum number of sequences per chunk.

        Returns:
            torch.Tensor: log-probabilities tensor of shape (B, L, V).
        """
        outputs = []
        B = xt.size(0)
        # Process in chunks
        for start in range(0, B, self.batch_size):
            end = start + self.batch_size
            x_chunk = xt[start:end]
            sigma_chunk = sigma_t[start:end]
            # Use the existing forward method for denoiser + log_softmax
            logits_chunk = self.forward(x_chunk, sigma_chunk)
            outputs.append(logits_chunk)
        # Concatenate all chunks back into a full batch
        return torch.cat(outputs, dim=0)

    # ---------------------------------------------------------------------
    def _compute_posterior(self, log_x_theta: torch.Tensor, xt: torch.LongTensor,
                           alpha_s: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
        """Uniform diffusion posterior p(x_s | x_t, x_theta)."""
        # x_theta : probs . We get exp(...) outside.
        x_theta = log_x_theta.exp()
        xt_onehot = F.one_hot(xt, self.vocab_size)
        post = (
                (alpha_t * self.vocab_size * x_theta * xt_onehot +
                 (alpha_t / alpha_s - alpha_t) * xt_onehot +
                 (alpha_s - alpha_t) * x_theta +
                 (1 - alpha_t / alpha_s) * (1 - alpha_s)) /
                (alpha_t * self.vocab_size * torch.gather(x_theta, -1, xt[..., None]) + (1 - alpha_t))
        )
        return post

    def _preserve_mask_mass(
            self,
            logits: torch.Tensor,  # (B, L, V)  log‑probs, −∞ where tokens are pruned
            xt: torch.Tensor,  # (B, L)     current sequence (mask = self.mask_idx)
            mask_logits: torch.Tensor,  # (B, L)     log p_M (mask log‑prob)
    ) -> torch.Tensor:
        """
        Renormalise logits so that  q(z_s=m|z_t) == p_M  is kept intact.
        Works position‑wise and supports top‑k pruning (−∞ logits).
        """
        m = self.mask_idx
        mask_pos = (xt == m)  # positions we are sampling
        p_M = mask_logits.exp().clamp_(1e-12, 1. - 1e-6)

        # build a boolean mask for *finite* non‑mask logits
        nonmask_mask = torch.isfinite(logits)
        nonmask_mask[..., m] = False

        # log ∑_{v≠m} e^{ℓ_v}
        logsum_nonmask = torch.logsumexp(
            logits.masked_fill(~nonmask_mask, -float('inf')), dim=-1
        )

        shift = (torch.log1p(-p_M) - logsum_nonmask)  # (B,L)
        shift = shift.masked_fill(~mask_pos, 0.0)  # no shift where xt ≠ MASK

        # add the shift only to valid non‑mask logits
        logits = logits + shift.unsqueeze(-1) * nonmask_mask.float()
        logits[..., m] = mask_logits  # keep mask log‑prob

        return logits


    def _get_ratio(self, xt: torch.LongTensor, t: torch.Tensor) -> torch.Tensor:
        """
        Return log-ratio scores rψ(xₜ,ℓ,v,t)  ∈ ℝ^{B×L×V}.

        • If cfg.use_approx == True  ➜ first-order Taylor approximation
          (one network pass, one backward pass, O(B·L) memory).
        • Otherwise                ➜ exact but expensive enumeration.

        The approximation follows Eq. (8) of the paper:
            rψ(xₜ[ℓ←v]) ≈ rψ(xₜ) + ⟨∇_{xₜ} rψ(xₜ), e_{ℓ,v} − e_{ℓ,xₜ[ℓ]}⟩
        """
        # ------------------------------------------------------------------ #
        # 1. Fast first-order approximation (one fwd + bwd)                  #
        # ------------------------------------------------------------------ #
        if self.use_approx:
            with torch.enable_grad():  # ← turn grads back on
                xt_onehot = F.one_hot(xt, self.vocab_size).float()
                xt_onehot.requires_grad_(True)

                base = self.ratio_model(xt_onehot, t)  # forward pass
                grad, = torch.autograd.grad(base.sum(), xt_onehot, create_graph=False)
            # grad  ≜ ∂ rψ / ∂ xₜ           shape: (B,L,V)

            # baseline scores broadcast to token dimension
            base = base[:, None, None]  # (B,1,1)

            # For each position, substitute current token → build delta
            idx = xt.unsqueeze(-1)  # (B,L,1)
            scatter_grad = grad.gather(-1, idx)  # grad at current tokens (B,L,1)

            # rψ(xₜ[ℓ←v]) ≈ base + grad_{ℓ,v} − grad_{ℓ,xℓ}
            log_ratio = base + grad - scatter_grad  # (B,L,V)
            # grad - scatter_grad   ↔ their classifier_log_prob_ratio
            # base + grad - scatter_grad  ↔ their classifier_log_prob

            with torch.no_grad():
                log_ratio_exact = self._batched_ratio(xt, t)
            # print a set of diagnostics of how different the two methods are
            if not torch.allclose(log_ratio, log_ratio_exact, atol=1e-3):
                print(f"Warning: log_ratio and log_ratio_ differ significantly: "
                      f"{(log_ratio - log_ratio_exact).abs().mean().item():.3f} (mean abs diff)")

        # ------------------------------------------------------------------ #
        # 2. Exact enumeration (vector or scalar model)                      #
        # ------------------------------------------------------------------ #
        else:
            if self.ratio_model.__class__.__name__ == "RatioNetAdaLNVector":
                log_ratio = self._batched_ratio_vector(xt, t)
            else:
                log_ratio = self._batched_ratio(xt, t)
            log_ratio_exact = log_ratio

        # ------------------------------------------------------- #
        # 3. Diagnostics (unchanged)                              #
        # ------------------------------------------------------- #
        mask_mean = log_ratio[..., self.mask_idx:self.mask_idx + 1].mean(-1, keepdim=True)
        token_mean = log_ratio[..., :-1].mean(-1, keepdim=True)
        return log_ratio

    @torch.no_grad()
    def _batched_ratio(self, xt, t):
        """Return log‑ratio tensor[B,L,V] without overloading GPURAM."""
        B, L = xt.shape
        N = B * L * self.vocab_size  # total candidates
        out = torch.empty(N, device=xt.device)

        # build the index tensor only once
        jump_idx = torch.arange(N, device=xt.device)
        pos = jump_idx // self.vocab_size  # [N]
        tok = jump_idx % self.vocab_size  # [N]
        t_expand = t.repeat_interleave(L * self.vocab_size)

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)

            # slice indices for this mini‑batch
            b_slice = (pos[start:end] // L)  # which original seq
            p_slice = pos[start:end] % L  # which position in seq
            tok_slice = tok[start:end]

            # construct mutated sequences on the fly (no full xt_expand)
            seq_batch = xt[b_slice].clone()
            seq_batch[torch.arange(seq_batch.size(0), device=xt.device), p_slice] = tok_slice

            # forward pass
            out[start:end] = self.ratio_model(seq_batch,t_expand[start:end])

        return out.view(B, L, self.vocab_size)

    @torch.no_grad()
    def _batched_ratio_vector(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute rψ(xₜ, ℓ, t) for *every* position ℓ without blowing up GPU RAM.

        Args
        ----
        xt : LongTensor  – current noised sequences   (B, L)
        t  : Tensor      – times *per-sequence*       (B,)

        Returns
        -------
        log_ratio : Tensor  – log-ratio scores        (B, L, V)
                     for each sequence, each position, and every token.
        """
        B, L = xt.shape
        V    = self.vocab_size
        device = xt.device

        # where we'll write the per-position vectors
        log_ratio = torch.empty(B, L, V, device=device)

        # the ratio net is already placed on the correct device & in eval() by __init__
        # We stream the computation over positions and over batch chunks
        # to keep peak memory <= self.batch_size × V floats.
        for pos in range(L):
            # A position tensor (B,) filled with the current position index
            pos_vec_full = torch.full((B,), pos, dtype=torch.long, device=device)

            # Optional: split over the *sequence* dimension as well
            for start in range(0, B, self.batch_size):
                end   = min(start + self.batch_size, B)

                # Slice the current chunk
                x_chunk   = xt[start:end]                # (b′, L)
                pos_chunk = pos_vec_full[start:end]      # (b′,)
                t_chunk   = t[start:end]                 # (b′,)

                # Forward pass – returns (b′, V)
                vec = self.ratio_model(x_chunk, pos_chunk, t_chunk)

                # Write into the correct slot of the output tensor
                log_ratio[start:end, pos, :] = vec

        return log_ratio


    # ---------------------------------------------------------------------
    # denoising step with ratio guidance
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _one_timestep(
            self,
            xt: torch.LongTensor,
            t: torch.Tensor,  # shape (B,) or (B,1)
            dt: float,
            cache: Optional[dict],
            caching: bool = False
    ) -> Tuple[torch.LongTensor, torch.Tensor, Optional[dict]]:
        # ---- normalise time shape ----
        if t.ndim > 1:
            t = t.view(-1)

        reuse = (caching and cache is not None)

        if reuse:
            log_x_theta = cache["log_x_theta"]
            log_ratio = cache["log_ratio"]
        else:
            # ------- denoiser logits -------
            sigma_t, _ = self.noise(t)
            log_x_theta = self.model_prediction(xt, sigma_t)

            # ------- ratio logits (top-n, neutral outside the set) -------
            if self.cfg.gamma > 0 and self.cfg.top_n_ratio > 0:
                log_ratio = self.get_ratio_log_topk_stream(
                    xt=xt,
                    sigma=t,
                    log_x_theta=log_x_theta,
                    k=int(self.cfg.top_n_ratio),
                    ratio_bs=int(getattr(self.cfg, "batch_size_ratio", 1024)),
                    normalize=True
                )
            elif self.cfg.gamma > 0:
                log_ratio = self._get_ratio(xt, t)  # full-vocab path
            else:
                log_ratio = torch.zeros_like(log_x_theta)

        # ---- posterior (unguided) in log-space ----
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        move_t = (1 - torch.exp(-sigma_t)).clamp_(min=1e-8)
        move_s = (1 - torch.exp(-sigma_s)).clamp_(min=0.0, max=move_t.min().item())

        move_t = move_t[:, None, None]  # (B,1,1)
        move_s = move_s[:, None, None]

        if self.diffusion == "absorbing_state":
            ratio_ms_mt = (move_s / move_t).clamp_(0.0, 1.0 - 1e-6)
            q_log = log_x_theta + torch.log1p(-ratio_ms_mt)
            q_log[..., self.mask_idx] = torch.log(ratio_ms_mt)[:, :, 0]
        else:  # "uniform"
            # _compute_posterior expects probabilities
            q_log = self._compute_posterior(
                x=log_x_theta.exp(),
                xt=xt,
                alpha_s=1 - move_s,
                alpha_t=1 - move_t
            ).log()

        # ---- combine & (optionally) prune for sampling on masked only ----
        guided_log = q_log + float(self.cfg.gamma) * log_ratio

        if getattr(self, "k_best_sampling", 0) > 0:
            k = int(self.k_best_sampling)
            if k < self.vocab_size:
                mask = (xt == self.mask_idx)  # only masked rows
                if mask.any():
                    topk_vals, topk_ids = guided_log[mask].topk(k, dim=-1)
                    pruned = guided_log.new_full(guided_log[mask].shape, guided_log.dtype.new_tensor(-1e9))
                    pruned.scatter_(-1, topk_ids, topk_vals)
                    guided_log = guided_log.clone()
                    guided_log[mask] = pruned

        # ---- preserve mask mass (stability) ----
        guided_log = self._preserve_mask_mass(guided_log, xt, q_log[..., self.mask_idx])

        # ---- sample ----
        guided_probs = torch.softmax(guided_log.float(), dim=-1)
        xs = _sample_categorical(guided_probs)

        if self.diffusion == "absorbing_state":
            xs = torch.where(xt != self.mask_idx, xt, xs)

        # ---- cache policy (KEEP iff nothing changed AND caching enabled) ----
        changed = not torch.equal(xs, xt)
        if caching and not changed:
            cache = {"log_x_theta": log_x_theta, "log_ratio": log_ratio, "t": t.clone()}
        else:
            cache = None

        return xs, guided_probs, cache

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def sample_ancestral(self, num_of_samples: int,caching=False, device: Union[torch.device, str] = "cpu") -> torch.LongTensor:
        """Deterministic DDPM‑like sampler (fixed #steps) with RBG."""
        device = torch.device(device)
        xt = (torch.full((num_of_samples, self.cfg.seq_len), self.mask_idx, dtype=torch.long, device=device)
              if self.diffusion == "absorbing_state"
              else torch.randint(self.vocab_size, (num_of_samples, self.cfg.seq_len), device=device))

        timesteps = torch.linspace(1.0, self.cfg.end_time, self.cfg.T_sampling + 1, device=device)
        dt = (1.0 - self.cfg.end_time) / self.cfg.T_sampling

        cache = None

        for i in tqdm(range(self.cfg.T_sampling), desc="Sampling", unit="step", total=self.cfg.T_sampling):
            self.i = i
            t = timesteps[i].expand(num_of_samples)
            xt, _, cache = self._one_timestep(xt, t, dt, cache, caching=caching)

        return xt  # (B,L)

    def sample_trace(self,
                     num_samples: int,
                     device: Union[str, torch.device] = "cpu"
                     ) -> torch.LongTensor:
        """
        Return a full trajectory tensor with shape
            (T_sampling+1,num_samples,seq_len)

        traj[0] is the initial noise / mask state,
        traj[-1] is the final denoised sequence.
        """
        device = torch.device(device)
        T = self.cfg.T_sampling
        traj = torch.empty(T + 1, num_samples, self.cfg.seq_len,
                           dtype=torch.long, device=device)

        # --- initialise x_T -------------------------------------------------
        xt = (torch.full((num_samples, self.cfg.seq_len), self.mask_idx,
                         dtype=torch.long, device=device)
              if self.diffusion == "absorbing_state"
              else torch.randint(self.vocab_size,(num_samples, self.cfg.seq_len), device=device))
        traj[0] = xt

        # --- deterministic DDPM schedule -----------------------------------
        timesteps = torch.linspace(1.0, self.cfg.end_time, self.cfg.T_sampling, device=device)
        dt = (1.0 - self.cfg.end_time) / self.cfg.T_sampling
        cache = None

        for i in range(T):
            t = timesteps[i].expand(num_samples)
            xt, _, cache = self._one_timestep(xt, t, dt, cache)
            traj[i + 1] = xt  # save x_{t‑1}

        return traj.cpu()  # (T+1, B, L)

# ──────────────────────────────────────────────────────────────────────────────
# Add these methods & helpers inside the short Diffusion class
# ──────────────────────────────────────────────────────────────────────────────

    # ---------- small helper: keep only top-k logits (−∞ elsewhere) ----------
    def _keep_top_k_tokens(self, log_x_theta: torch.Tensor, xt: torch.Tensor,
                           k: int, *, always_keep: Optional[int] = None) -> torch.Tensor:
        if k <= 0:
            return log_x_theta
        mask = (xt == self.mask_idx)
        if not mask.any():
            return log_x_theta
        kept = torch.zeros_like(log_x_theta, dtype=torch.bool)
        topk_idx = torch.topk(log_x_theta[mask], k, dim=-1).indices
        kept_rows = kept[mask]
        kept_rows.scatter_(1, topk_idx, True)
        if always_keep is not None and always_keep < log_x_theta.size(-1):
            kept_rows[:, always_keep] = True
        kept[mask] = kept_rows
        neg_inf = torch.finfo(log_x_theta.dtype).min
        pruned = torch.full_like(log_x_theta, neg_inf)
        pruned[kept] = log_x_theta[kept]
        return pruned


    # ---------- posterior (q_log) builder reused below ------------------------
    def _posterior_log(self, log_x_theta: torch.Tensor, xt: torch.LongTensor,
                       move_t: torch.Tensor, move_s: torch.Tensor) -> torch.Tensor:
        """Return log q(x_s | x_t, x_theta) for both diffusion types."""
        if self.diffusion == "absorbing_state":
            # q_log[..., m] = log(move_s / move_t); others = log_x_theta + log(1 - move_s/move_t)
            q_log = log_x_theta + torch.log1p(-(move_s / move_t))
            q_log[..., self.mask_idx] = torch.log(move_s / move_t)[:, :, 0]
        else:
            q_log = self._compute_posterior(
                log_x_theta, xt, 1 - move_s, 1 - move_t
            ).log()
        return q_log


    # ======================================================================
    # 3) Planner sampling + Top-n ratio (position-wise revealing)
    # ======================================================================
    @torch.no_grad()
    def _planner_propose_positions(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Return per-row positions (B,) from self.planner(xt).
        Expected planner output: (B,L) logits. Falls back to random masked per row.
        """
        B, L = xt.shape
        device = xt.device

        if hasattr(self, "planner") and (self.planner is not None):
            pos_logits = self.planner(xt)  # ideally (B, L)
            if pos_logits.ndim == 2 and pos_logits.size(1) == L:
                return pos_logits.softmax(-1).argmax(-1)  # (B,)
            if pos_logits.ndim == 1:
                return pos_logits.view(-1)  # (B,)
            # Any other shape → fallback below

        # Fallback: random masked per row (or 0 if none masked)
        pos = torch.zeros(B, dtype=torch.long, device=device)
        mask_any = (xt == self.mask_idx)
        for b in range(B):
            cols = torch.nonzero(mask_any[b], as_tuple=False).flatten()
            pos[b] = cols[torch.randint(0, cols.numel(), (1,), device=device)] if cols.numel() > 0 else 0
        return pos

    def _ensure_masked_per_row(self, xt: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Ensure each pos[b] points to a masked column; if not, pick a random masked one.
        If a row has no masks left, keep pos[b] (it will be ignored downstream).
        """
        B, L = xt.shape
        device = xt.device
        fixed = pos.clone()
        mask_map = (xt == self.mask_idx)

        for b in range(B):
            j = int(fixed[b].item())
            if j < 0 or j >= L or not mask_map[b, j]:
                cols = torch.nonzero(mask_map[b], as_tuple=False).flatten()
                if cols.numel() > 0:
                    fixed[b] = cols[torch.randint(0, cols.numel(), (1,), device=device)]
                # else: keep as-is
        return fixed

    @torch.no_grad()
    def sample_planner_topn_ratio(self,
                                        num_samples: int,
                                        n_ratio: int = 1024,
                                        gamma: Optional[float] = None,
                                        ratio_bs: int = 8192,
                                        device: Union[str, torch.device] = "cpu"):
        """
        Planner sampling in exactly L steps:
          1) propose one position per row,
          2) force it to be masked (row-wise),
          3) reveal it with Top-n ratio guidance (streamed).
        Shows a step-based tqdm bar (L iterations).
        """
        device = torch.device(device)
        B, L, V = num_samples, self.cfg.seq_len, self.vocab_size

        # init x_T
        xt = (torch.full((B, L), self.mask_idx, dtype=torch.long, device=device)
              if self.diffusion == "absorbing_state"
              else torch.randint(V, (B, L), device=device))

        zeros = torch.zeros(B, device=device)  # sigma≈0 for planner mode
        gamma_eff = self.cfg.gamma if gamma is None else float(gamma)

        ratio_model = self.ratio_model.module if hasattr(self.ratio_model, "module") else self.ratio_model
        rdev = next(ratio_model.parameters()).device

        nfe_den, nfe_rat = 0, 0

        t0 = time.time()
        for step in tqdm(range(L), desc="Planner steps", unit="step"):
            # 1) propose positions and ensure they refer to masks
            pos = self._planner_propose_positions(xt)  # (B,)
            pos = self._ensure_masked_per_row(xt, pos)  # (B,)

            row_idx = torch.arange(B, device=device)

            # rows that are actually masked at pos
            row_mask = (xt[row_idx, pos] == self.mask_idx)
            if not row_mask.any():
                continue  # nothing to do this step

            # 2) base logits at those positions (sigma=0)
            log_x_theta = self.model_prediction(xt, zeros)  # (B,L,V) log-probs
            nfe_den += 1
            logits_pos = log_x_theta[row_idx, pos, :]  # (B,V)
            logits_pos[:, self.mask_idx] = -float("inf")  # never propose [MASK]

            # 3) top-n candidates per row
            K = int(max(1, min(n_ratio, V)))
            cand_idx = torch.topk(logits_pos, K, dim=-1).indices  # (B,K)

            # 4) streamed ratio over candidates: build (T=B*K) mutated seqs in chunks
            b_all = row_idx.repeat_interleave(K)  # (T,)
            pos_all = pos.repeat_interleave(K)  # (T,)
            tok_all = cand_idx.reshape(-1)  # (T,)
            Ttot = b_all.numel()

            ratio_log = torch.full_like(logits_pos, -float("inf"))
            start = 0
            while start < Ttot:
                end = min(start + ratio_bs, Ttot)
                tlen = end - start

                bs = b_all[start:end]
                ps = pos_all[start:end]
                ts = tok_all[start:end]

                seq_mb = xt[bs].clone().to(rdev, non_blocking=True)  # (tlen, L)
                seq_mb[torch.arange(tlen, device=rdev), ps.to(rdev)] = ts.to(rdev)
                sig_mb = zeros[bs].to(rdev, non_blocking=True)

                with torch.inference_mode():
                    out = ratio_model(seq_mb, sig_mb)  # shape flexible

                # reduce to 1-D scores
                if out.dim() == 0:
                    scores = out.view(1)
                elif out.dim() == 1:
                    scores = out
                elif out.size(-1) == 1:
                    scores = out.view(-1)
                else:
                    scores = out.gather(1, ts.to(out.device).view(-1, 1)).squeeze(1)

                ratio_log[bs, ts] = scores.to(ratio_log.device, dtype=ratio_log.dtype)
                nfe_rat += 1
                start = end

            ratio_log = torch.log_softmax(ratio_log, dim=-1)
            ratio_log[:, self.mask_idx] = 0.0

            # 5) combine + preserve mask mass at pos (use base mask logit at pos)
            guided_pos = logits_pos + gamma_eff * ratio_log  # (B,V)
            guided3d = guided_pos.unsqueeze(1)  # (B,1,V)
            xt_pos = xt[row_idx, pos].unsqueeze(1)  # (B,1)
            mask_logits = logits_pos[:, self.mask_idx].unsqueeze(1)
            guided3d = self._preserve_mask_mass(guided3d, xt_pos, mask_logits)
            guided_pos = guided3d.squeeze(1)

            # 6) sample only on rows where pos is masked
            probs = guided_pos[row_mask].softmax(dim=-1)
            new_tok = torch.multinomial(probs, 1).squeeze(1)
            xt[row_idx[row_mask], pos[row_mask]] = new_tok

        elapsed = time.time() - t0
        stats = {
            "wall_s": elapsed,
            "nfe_denoiser": nfe_den,
            "nfe_ratio": nfe_rat,
            "planner_steps": L,
        }
        return xt, stats

    def get_ratio_log_topk_stream_(
            self, xt, sigma, log_x_theta, k: int = 1024, ratio_bs: int = 8192,
            normalize: bool = True
    ) -> torch.Tensor:
        if k <= 0:
            k = min(int(self.vocab_size//2), 1024)
        B, L, V = log_x_theta.shape
        mask = (xt == self.mask_idx)
        rows = mask.nonzero(as_tuple=False)  # (Nmask,2)
        if rows.numel() == 0:
            return torch.zeros_like(log_x_theta)

        topk = torch.topk(log_x_theta[mask], k, dim=-1).indices  # (Nmask,k)


        # Build candidate triples (b,pos,tok), ensuring label is included
        cand_tuples = []
        for i, ((b, pos), toks) in enumerate(zip(rows.tolist(), topk.tolist())):
            for tok in toks:
                cand_tuples.append((int(b), int(pos), int(tok)))

        r_dev = next(self.ratio_model.parameters()).device
        r_dtype = next(self.ratio_model.parameters()).dtype
        ratio_log = torch.full((B, L, V), self.neg_infinity, device=r_dev, dtype=r_dtype)

        base_cpu = xt.to("cpu", non_blocking=True)
        sigma_cpu = sigma.to("cpu").flatten()

        loop =  range
        for i in loop(0, len(cand_tuples), ratio_bs):
            batch_slice = cand_tuples[i: i + ratio_bs]
            n = len(batch_slice)
            seq_batch = torch.empty((n, L), dtype=torch.long)
            for j, (b, pos, tok) in enumerate(batch_slice):
                seq = base_cpu[b].clone(); seq[pos] = tok; seq_batch[j] = seq

            idx = torch.tensor([b for (b, _, _) in batch_slice])
            sig_batch = sigma_cpu.index_select(0, idx).to(r_dev)

            seq_dev = seq_batch.to(r_dev, non_blocking=True)
            with torch.inference_mode():
                logits = self.ratio_model(seq_dev, sig_batch)  # (n,L,V) or (n,Vpos) or (n,1)

            for j, (b, pos, tok) in enumerate(batch_slice):
                if logits.dim() == 3:
                    score = logits[j, pos, tok]
                elif logits.dim() == 1:
                    score = logits[j]
                else:
                    score = logits[j, tok]
                ratio_log[b, pos, tok] = score

            #del seq_dev, logits

        if normalize:
            ratio_rows = ratio_log[mask]
            ratio_log[mask] = torch.log_softmax(ratio_rows, dim=-1)
        ratio_log[..., self.mask_idx] = 0.0
        return ratio_log.to(log_x_theta.device)

    @torch.inference_mode()
    def get_ratio_log_topk_stream(
            self, xt, sigma, log_x_theta, k: int = 1024, ratio_bs: int = 16384,
            normalize: bool = True
    ) -> torch.Tensor:
        if k <= 0:
            k = min(int(self.vocab_size // 2), 1024)

        dev = xt.device
        dtype = next(self.ratio_model.parameters()).dtype
        B, L, V = log_x_theta.shape

        mask = (xt == self.mask_idx)
        rows = mask.nonzero(as_tuple=False)  # (Nmask,2)
        if rows.numel() == 0:
            return torch.zeros_like(log_x_theta)

        # top-k per masked row (all on device)
        topk_idx = torch.topk(log_x_theta[mask], k, dim=-1).indices  # (Nmask,k)

        # Build candidate tensors (b,pos,tok) without Python loops
        b_rows = rows[:, 0]
        p_rows = rows[:, 1]
        Nmask = rows.size(0)

        b = b_rows.repeat_interleave(k)  # (Nmask*k,)
        p = p_rows.repeat_interleave(k)  # (Nmask*k,)
        tok = topk_idx.reshape(-1)  # (Nmask*k,)

        ratio_log = torch.full((B, L, V), self.neg_infinity, device=dev, dtype=dtype)

        # Batch over candidates
        for start in range(0, b.numel(), ratio_bs):
            end = min(start + ratio_bs, b.numel())

            b_slice = b[start:end]
            p_slice = p[start:end]
            tok_slice = tok[start:end]

            # Build mutated sequences on device
            seq_batch = xt[b_slice].clone()
            row_idx = torch.arange(seq_batch.size(0), device=dev)
            seq_batch[row_idx, p_slice] = tok_slice

            # Match sigma per example
            sig_batch = sigma[b_slice]

            # Forward once for the whole chunk
            logits = self.ratio_model(seq_batch, sig_batch)

            # Extract the score at the mutated position/token (vectorized)
            if logits.dim() == 3:  # (n,L,V)
                scores = logits[row_idx, p_slice, tok_slice]
            elif logits.dim() == 2:  # (n,Vpos) -> assume it's exactly for p_slice
                scores = logits[row_idx, tok_slice]
            else:  # (n,)
                scores = logits

            # Scatter back in one op
            ratio_log[b_slice, p_slice, tok_slice] = scores

        # Optional: normalize over top-k only
        if normalize:
            # Gather top-k scores and normalize along k
            gathered = ratio_log[mask].gather(1, topk_idx)  # (Nmask,k)
            normed = torch.log_softmax(gathered, dim=-1)  # (Nmask,k)
            # Write back the normalized top-k in one go
            ratio_log[mask] = torch.full((Nmask, V), self.neg_infinity, device=dev, dtype=dtype)
            ratio_log[mask].scatter_(1, topk_idx, normed)

        # Avoid forcing a dense sweep; set this only if truly required
        # ratio_log[..., self.mask_idx] = 0.0

        return ratio_log.to(log_x_theta.device)
