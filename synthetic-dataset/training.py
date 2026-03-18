from typing import Union, Optional, Dict, Tuple,  List
import math
import time
from tqdm.auto import tqdm
from itertools import cycle
from utils import hamming_matrix
from diffusion import LogLinearNoise
from inspect import signature
import torch.nn.functional as F
import torch, random
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Sampler



# -----------------------------------------------------------------------
# 0. Helper: CutMix, Laplacian penalty, and balanced sampler
# -----------------------------------------------------------------------
def _cutmix_seq(src, tgt, λ_min=0.3, λ_max=0.9):
    L = src.size(0)
    lam = random.uniform(λ_min, λ_max)           # λ∈[0.3,0.7]
    span = max(1, int(L * lam))
    i_src = random.randint(0, L - span)
    i_tgt = random.randint(0, L - span)
    mixed = src.clone()
    mixed[i_src:i_src + span] = tgt[i_tgt:i_tgt + span]
    mixed_label = 1.0 - lam                      # soft label
    return mixed, mixed_label

class _BalancedSampler(Sampler):
    """
    Always yields batches with equal numbers of source and target rows.
    Target indices are cycled so that every source row can be paired,
    even when N_tgt < N_src.
    """
    def __init__(self, n_src: int, n_tgt: int, batch_size: int):
        if batch_size % 2:
            raise ValueError("batch_size must be even.")
        self.n_src   = n_src
        self.n_tgt   = n_tgt
        self.batch_size = batch_size
        self.bs_half = batch_size // 2          # src rows per batch
        self.bs_split = min(self.bs_half, self.n_tgt)  # cap to N_tgt

    def __iter__(self):
        src_perm = torch.randperm(self.n_src)
        tgt_perm = torch.randperm(self.n_tgt) + self.n_src  # offset
        tgt_iter = cycle(tgt_perm.tolist())                 # endless stream

        ptr = 0
        while ptr < self.n_src:
            src_slice = src_perm[ptr:ptr + int(self.batch_size - self.bs_split)]
            ptr += len(src_slice)

            tgt_slice = [next(tgt_iter) for _ in range(int(self.batch_size - len(src_slice)))]
            yield torch.cat([src_slice, torch.tensor(tgt_slice)]).tolist()

    def __len__(self):
        return math.ceil(self.n_src / self.bs_half)


def sequence_augmentation(batch_inputs, # [B,L]  long
                         batch_labels,  # [B,]  float
                         cutmix_p,
                         device: str = "cpu"):
    src_idx = (batch_labels == 1).nonzero(as_tuple=True)[0]  # source rows
    tgt_idx = (batch_labels == 0).nonzero(as_tuple=True)[0]  # target rows

    # **skip when either domain is absent**
    if src_idx.numel() and tgt_idx.numel():
        k = int(cutmix_p * src_idx.numel())
        if k:  # nothing to mix → skip

            pick_src=src_idx[torch.randperm(src_idx.size(0), device=device)[:k]]
            pick_tgt = tgt_idx[torch.randint(0, tgt_idx.size(0), (k,), device=device)]


            L = batch_inputs.size(1)
            lam = torch.empty(k, device=device).uniform_(0.3, 0.7)
            span = torch.clamp((lam * L).long(), min=1)

            # span already k×1
            max_start = L - span  # k×1 tensor of per-row maxima
            start_src = (torch.rand(k, device=device) * (max_start + 1)).floor().long()
            start_tgt = (torch.rand(k, device=device) * (max_start + 1)).floor().long()

            pos = torch.arange(L, device=device).expand(k, L)
            mask = (pos >= start_src.unsqueeze(1)) & (pos < (start_src + span).unsqueeze(1))

            src_seqs = batch_inputs[pick_src].clone()
            tgt_seqs = batch_inputs[pick_tgt]
            src_seqs = torch.where(mask, tgt_seqs, src_seqs)

            batch_inputs[pick_src] = src_seqs
            batch_labels[pick_src] = 1.0 - lam
            return batch_inputs, batch_labels

# --- helper: single-span CutMix for sequences (base ← donor) ---
def cutmix_seq(base: torch.Tensor, donor: torch.Tensor, p: float, device):
    """
    base:  [B,L]  sequences to modify (e.g., source)
    donor: [B',L] sequences to paste from     (e.g., target)
    p:     per-row probability to CutMix
    returns x_mix [B,L]
    """
    B, L = base.size()
    x = base.clone()
    do = (torch.rand(B, device=device) < p)
    k = int(do.sum().item())
    if k == 0:
        return x  # no-op

    idx = do.nonzero(as_tuple=True)[0]
    # choose donors with replacement
    donors = donor[torch.randint(0, donor.size(0), (k,), device=device)]

    # span length ∈ [0.3, 0.7] of L
    lam   = torch.empty(k, device=device).uniform_(0.3, 0.7)
    span  = torch.clamp((lam * L).long(), min=1)
    max_s = L - span
    start = (torch.rand(k, device=device) * (max_s + 1)).floor().long()

    pos   = torch.arange(L, device=device).expand(k, L)
    mask  = (pos >= start.unsqueeze(1)) & (pos < (start + span).unsqueeze(1))

    x_sel = x[idx]                 # [k,L]
    d_sel = donors                 # [k,L]
    x_sel[mask] = d_sel[mask]      # paste at same columns
    x[idx] = x_sel
    return x


def laplacian_penalty(preds: torch.Tensor,
                      seqs: torch.Tensor,
                      radius: int = 1) -> torch.Tensor:
    """
    preds : [B]  real     - model outputs r_ϕ(x) (or log‑ratio)
    seqs  : [B,L] long    – same mini‑batch the preds came from
    radius: int           – connect edges with Hamming distance ≤ radius
    returns a scalar penalty  λ · Σ_(i~j) (pred_i − pred_j)² / |E|
    """
    # 1. build adjacency mask ─────────────────────────────────────────
    D   = hamming_matrix(seqs)                 # [B,B]  integer
    mask = (D > 0) & (D <= radius)             # exclude self‑edges

    # 2. compute squared diffs only for edges in the mask ────────────
    diff = preds.unsqueeze(1) - preds.unsqueeze(0)   # [B,B]
    if mask.any():
        penalty = (diff[mask] ** 2).mean()
    else:                                            # rare if batch very small
        penalty = torch.tensor(0., device=preds.device)

    return penalty

# =====================================================================
# 1. Forward diffusion for discrete sequences
# =====================================================================
def corrupt(
    x0: torch.LongTensor,
    sigma: torch.Tensor,
    *,
    diffusion: str = "absorbing_state",
    vocab_size: Optional[int] = None,
    mask_idx: Optional[int] = None,
):
    """Forward‑diffuse *discrete* sequences.

    Parameters
    ----------
    x0         : (B, L) long        – clean tokens
    sigma      : (B,)  float        – noise level produced by noise schedule
    diffusion  : "absorbing_state" | "uniform"
    vocab_size : required if `diffusion == 'uniform'`
    mask_idx   : required if `diffusion == 'absorbing_state'`

    Returns
    -------
    x_t        : (B, L) long        – corrupted sequence at time‑step *t*
    move_mask  : (B, L) bool        – True where corruption happened
    """
    move_prob = 1 - torch.exp(-sigma)              # (B,)
    B, L = x0.shape
    move_mask = (torch.rand_like(x0.float()) < move_prob[:, None])

    if diffusion == "absorbing_state":
        assert mask_idx is not None, "mask_idx must be set for absorbing‑state diffusion"
        x_t = x0.clone()
        x_t[move_mask] = mask_idx
    else:  # uniform diffusion
        assert vocab_size is not None, "vocab_size must be set for uniform diffusion"
        x_t = x0.clone()
        x_t[move_mask] = torch.randint(0, vocab_size, (move_mask.sum(),), device=x0.device)

    return x_t, move_mask


# ==================================================================================
# 2. Train time‑independent classifier  d_ω(x)
# ==================================================================================
def train_domain_classifier(
    model,
    source_data,            # LongTensor [N_src,L]
    target_data,            # LongTensor [N_tgt,L]
    epochs           = 10,
    batch_size       = 256,
    lr               = 1e-4,
    device           = "cpu",
    unbalance_data   = False,
    classifier_output_with_sigmoid = False,
    *,                            # new kwargs must be passed by name
    balanced_batches  = False,    # enable 50/50 sampler
    cutmix_p          = 0.0      # 10 % of source rows get CutMix (0 → off)
):

    N_src, N_tgt = source_data.size(0), target_data.size(0)

    # ----------------------------------------------------------------
    # dataset & loader
    # ----------------------------------------------------------------
    data   = torch.cat([source_data, target_data])
    labels = torch.cat([torch.ones(N_src), torch.zeros(N_tgt)])
    dataset = TensorDataset(data, labels)

    if balanced_batches:
        sampler = _BalancedSampler(N_src, N_tgt, batch_size)
        loader  = DataLoader(dataset, batch_sampler=sampler)
    else:
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, drop_last=True)

    # ----------------------------------------------------------------
    # loss
    # ----------------------------------------------------------------
    pos_weight = torch.tensor(1.0, device=device)
    if classifier_output_with_sigmoid:
        bce_loss = nn.BCELoss()
    else:
        if unbalance_data:
            pos_weight_ = min(N_tgt / N_src, 0.05)
            pos_weight = torch.tensor(pos_weight_, device=device)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        pos_weight = pos_weight.item() if isinstance(pos_weight, torch.Tensor) else pos_weight
    # ----------------------------------------------------------------
    # training loop
    # ----------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device).train()

    for epoch in range(epochs):
        running = 0.0
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            # ---------------------  CutMix on 10 % of source rows  ----------------
            if cutmix_p > 0:
                batch_inputs, batch_labels = sequence_augmentation(
                    batch_inputs=batch_inputs,
                    batch_labels=batch_labels,
                    cutmix_p=cutmix_p,
                    device=device
                )

            # ---------------------  optimisation step  ---------------------------
            optimizer.zero_grad()
            logits = model(batch_inputs)
            if classifier_output_with_sigmoid:
                loss = bce_loss(logits.squeeze(-1), batch_labels)
            else:
                bce = F.binary_cross_entropy_with_logits(logits.squeeze(-1), batch_labels, reduction="none")
                if unbalance_data:  # pos_weight = N_tgt / N_src
                    w = torch.where(batch_labels != 0, pos_weight, 1.0)
                    w_tgt = torch.where(batch_labels == 0, 2.0, 1.0)
                    w = w * w_tgt
                    bce = bce * w
                loss = bce.mean()
            loss.backward()
            optimizer.step()
            running += loss.item()

        print(f"[Classifier] epoch {epoch+1}/{epochs}, "
              f"loss={running/len(loader):.4f}")

    model.to("cpu")
    return model


# =====================================================================
# 3. Training the time-dependent domain classifier from the TLDM paper (Pseudo-Code 2 in the Appendix)
# =====================================================================
def train_time_dependent_classifier(
    model: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    noise_sched: nn.Module = LogLinearNoise(),
    *,
    diffusion: str = "absorbing_state",
    mask_idx: Optional[int] = None,
    vocab_size: Optional[int] = None,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-4,
    device: Union[str, torch.device] = "cpu",
    balanced_batches: bool = False,     # 50 / 50 sampler
    unbalance_data: bool = False,       # weighted BCE
    cutmix_p: float = 0.0,              # CutMix %
    classifier_output_with_sigmoid: bool = False,
):
    """Train d_ω(xₜ , t) with balanced batches, weighted loss, CutMix."""
    N_src, N_tgt = source_data.size(0), target_data.size(0)

    # ── dataset & loader ────────────────────────────────────────────
    data   = torch.cat([source_data, target_data])
    labels = torch.cat([torch.ones(N_src), torch.zeros(N_tgt)])
    ds = TensorDataset(data, labels)

    if balanced_batches:
        sampler = _BalancedSampler(N_src, N_tgt, batch_size)
        loader  = DataLoader(ds, batch_sampler=sampler)
    else:
        loader  = DataLoader(ds, batch_size=batch_size,
                             shuffle=True, drop_last=True)

    # ── loss -----------------------------------------------------------------
    if classifier_output_with_sigmoid:
        crit = nn.BCELoss()
        pos_weight = 1.0
    else:
        pos_weight = 1.0
        if unbalance_data:
            pos_weight = min(N_tgt / N_src, 0.05)      # cap, like before
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,
                                                            device=device))

    # ── optimiser ------------------------------------------------------------
    model.to(device).train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # ── training loop --------------------------------------------------------
    for ep in range(1, epochs + 1):
        total = 0.0
        for x0, y in loader:               # x₀ rows
            x0, y = x0.to(device), y.to(device)

            # data-aug (CutMix on source rows)
            if cutmix_p > 0.0:
                x0, y = sequence_augmentation(x0, y, cutmix_p, device=device)

            # sample timestep t and corrupt
            t  = torch.rand(x0.size(0), device=device)
            σt, _ = noise_sched(t)
            x_t, _ = corrupt(x0, σt, diffusion=diffusion,
                             mask_idx=mask_idx, vocab_size=vocab_size)

            # forward + loss
            out = model(x_t, t).squeeze(-1)
            if classifier_output_with_sigmoid:
                loss = crit(out, y)
            else:
                bce = F.binary_cross_entropy_with_logits(out, y, reduction="none")
                if unbalance_data:
                    w = torch.where(y != 0, pos_weight, 1.0)
                    w_tgt = torch.where(y == 0, 2.0, 1.0)  # double weight for target
                    w = w * w_tgt
                    bce = bce * w
                loss = bce.mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()

        print(f"[Time-Cond Clf] epoch {ep}/{epochs}, "
              f"loss={total / len(loader):.4f}")

    model.to("cpu")
    return model

# =====================================================================
# 4.  Validation for the domain-classifier dω
# =====================================================================
@torch.no_grad()
def validate_domain_classifier(
    classifier: nn.Module,
    src_val: torch.Tensor,
    tgt_val: torch.Tensor,
    batch_size: int  = 512,
    device: str      = "cpu",
    *,
    classifier_output_with_sigmoid: bool = False,  # <-- NEW
):
    """
    Draws new sequences from P (label=1) and Q (label=0) and evaluates
    the classifier. Uses BCEWithLogitsLoss if the classifier outputs logits,
    otherwise BCELoss if it outputs probabilities.
    """
    classifier.eval().to(device)

    # pick correct criterion & post-processing
    if classifier_output_with_sigmoid:
        crit = torch.nn.BCELoss(reduction="sum")
        apply_sigmoid_for_loss = False   # model already outputs probs
    else:
        crit = torch.nn.BCEWithLogitsLoss(reduction="sum")
        apply_sigmoid_for_loss = True    # ONLY for metrics

    # 1) fresh validation data
    seqs   = torch.cat([src_val, tgt_val])                           # [N,L]
    labels = torch.cat([torch.ones(src_val.shape[0]),
                        torch.zeros(tgt_val.shape[0])]).to(device)    # float
    labels = labels.to(dtype=torch.float32)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(seqs, labels),
        batch_size=batch_size, shuffle=False
    )

    total_loss = 0.0
    TP = FP = TN = FN = 0
    n_inputs = None  # set after first forward

    for batch_seqs, batch_labels in loader:
        batch_seqs  = batch_seqs.to(device)
        batch_labels = batch_labels.to(device)

        # handle time-dependent vs static classifier
        if n_inputs is None:
            from inspect import signature
            n_inputs = len(signature(classifier.forward).parameters)

        if n_inputs == 2:
            t = torch.zeros(batch_seqs.shape[0], device=device)
            out = classifier(batch_seqs, t).squeeze(-1)  # logits or probs
        else:
            out = classifier(batch_seqs).squeeze(-1)

        # loss
        loss_input = out
        if classifier_output_with_sigmoid:
            # out are probabilities already
            pass
        else:
            # out are logits (correct for BCEWithLogitsLoss)
            pass
        total_loss += crit(loss_input, batch_labels).item()

        # metrics (threshold on probabilities)
        probs = torch.sigmoid(out) if apply_sigmoid_for_loss else out
        preds = (probs >= 0.5).float()
        TP += ((preds == 1) & (batch_labels == 1)).sum().item()
        FP += ((preds == 1) & (batch_labels == 0)).sum().item()
        TN += ((preds == 0) & (batch_labels == 0)).sum().item()
        FN += ((preds == 0) & (batch_labels == 1)).sum().item()

    N = src_val.shape[0] + tgt_val.shape[0]
    acc  = (TP + TN) / N
    loss = total_loss / N

    which = "[Time-Dependent Classifier-Val]" if (n_inputs == 2) else "[Classifier-Val]"
    print(f"{which}  loss = {loss:.4f}  acc = {acc:.2%} TP={TP} FP={FP} TN={TN} FN={FN}")
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    print(f"{which}  Recall = {recall:.2%}  Precision = {precision:.2%}  F1 = {f1:.2%}")

    return {"bce": loss, "accuracy": acc, "TP": TP, "FP": FP, "TN": TN, "FN": FN}



# =====================================================================
# 5. Train the denoiser p(x₀ | x_t, σ_t)
# =====================================================================
def train_denoiser(
    denoiser: nn.Module,
    train_seqs: torch.Tensor,                          # (N, L) integer tokens
    val_seqs: Union[torch.Tensor, None] = None,
    *,
    diffusion: str = "absorbing_state",               # "absorbing_state" | "uniform"
    mask_idx: Union[int, None] = None,                # required for absorbing_state
    vocab_size: int,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    noise_sched = LogLinearNoise(),                   # can plug your own
    pad_val: Union[int, None] = None,                 # if you need to ignore PAD in CE
    device: Union[str, torch.device] = "cpu",
    print_every: int = 1,
    print_name: Union[str, None] = None,  # for logging purposes, e.g. "Denoiser"
):
    """
    Trains `denoiser` to predict p(x₀ | x_t, σ_t).

    Loss = cross‑entropy between model logits and the *clean* token x₀.
    """

    assert diffusion in {"absorbing_state", "uniform"}
    if diffusion == "absorbing_state":
        assert mask_idx is not None, "Need a dedicated <mask> token id"

    denoiser = denoiser.to(device)
    denoiser.train()
    opt = torch.optim.AdamW(denoiser.parameters(), lr=lr)

    ds     = TensorDataset(train_seqs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    if val_seqs is not None:
        val_loader = DataLoader(TensorDataset(val_seqs),
                                batch_size=batch_size)

    ce = nn.CrossEntropyLoss(
        ignore_index=pad_val if pad_val is not None else -100)

    print_label = f"[Denoiser]" if print_name is None else f"[Denoiser {print_name}]"
    for ep in range(1, epochs+1):
        running = 0.0
        for (x0,) in loader:
            x0 = x0.to(device)

            # 1) sample continuous timestep t ∼ U(0,1)
            t = torch.rand(x0.size(0), device=device)

            # 2) get sigma(t) for conditioning
            sigma_t, sigma_prime = noise_sched(t)

            # 3) corrupt x₀ → x_t
            x_t, move_mask_t = corrupt(x0=x0, sigma=sigma_t,
                                       diffusion=diffusion,
                                       vocab_size=vocab_size,
                                       mask_idx=mask_idx)

            # 4.1) Loss with weight and masking
            logits = denoiser(x_t, sigma_t)
            logits[..., mask_idx] = -1e9  # zero-masking prob
            nll = F.cross_entropy(logits.transpose(1, 2),x0,reduction='none' ) # keep every token
            nll = nll * move_mask_t         # (B,L)  # mask out non-moved tokens
            weight = sigma_prime.unsqueeze(1)  # σ′(t)  ==  α′/(1-α)
            loss1 = (nll * weight).sum() / move_mask_t.sum()

            # 4.2) forward + loss
            logits = denoiser(x_t, sigma_t)            # (B,L,V)
            loss2   = ce(logits.view(-1, vocab_size), x0.view(-1))

            opt.zero_grad()
            loss2.backward()
            opt.step()
            running += loss2.item() * x0.size(0)

        if ep % print_every == 0:
            print(f"{print_label} epoch {ep}/{epochs} "
                  f"trainCE = {running / len(ds):.4f}", end="")
            #wandb.log({"Denoiser/train_loss": running / len(ds), "epoch": ep})

            # quick val metrics if provided
            if val_seqs is not None:
                denoiser.eval()
                with torch.no_grad():
                    val_loss, correct_tok, correct_seq, total_tok, total_seq = 0, 0, 0, 0, 0
                    for (x0,) in val_loader:
                        x0 = x0.to(device)  # (B,L)
                        B, L = x0.shape
                        t = torch.rand(B, device=device)
                        sigma_t, _ = noise_sched(t)
                        x_t, move_mask_t = corrupt(x0=x0, sigma=sigma_t,
                                       diffusion=diffusion,
                                       vocab_size=vocab_size,mask_idx=mask_idx)  # noisy input

                        logits = denoiser(x_t, sigma_t)  # (B,L,V)
                        val_loss += ce(logits.view(-1, vocab_size),
                                       x0.view(-1)).item() * B * L

                        # ------------------------------
                        # accuracy metrics
                        # ------------------------------
                        pred = logits.argmax(dim=-1)  # (B,L)
                        match = pred.eq(x0)  # bool mask

                        correct_tok += match[move_mask_t].sum().item()
                        total_tok += move_mask_t.sum().item()

                        correct_seq += match.all(dim=1).sum().item()
                        total_seq += B

                print(f"    • val CE = {val_loss / total_tok:.4f}"
                      f"    • token acc = {100 * correct_tok / total_tok:.2f}%"
                      f"    • seq acc = {100 * correct_seq / total_seq:.2f}%")
            else:
                print()
    return denoiser



# =====================================================================
# 6. Train the ratio-estimator rφ without time-conditioning on clean data (just for experimentation)
# =====================================================================
def train_ratio_estimator_on_clean_data(
    model: nn.Module,
    domain_classifier: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    epochs: int      = 10,
    batch_size: int  = 256,
    lr: float        = 1e-4,
    classifier_output_with_sigmoid: bool  = True,          # train on log-ratio instead of ratio
    device: str      = "cpu",
    lambda_lap : float = 0,   # Laplacian penalty
):
    """
    Parameters
    ----------
    model              : RatioNet          (outputs rφ(x) or log rφ(x))
    domain_classifier  : already-trained dω(x) ∈ (0,1) (1 = source)
    source_data        : [N_src, L] long
    target_data        : [N_tgt, L] long
    """
    # freeze the classifier
    domain_classifier.eval().to(device)
    for p in domain_classifier.parameters():
        p.requires_grad_(False)

    # put the ratio network into training mode
    model.train().to(device)
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-4,lr=lr)
    mse_loss  = nn.MSELoss()

    # unlabeled mixture dataset
    seqs = torch.cat([source_data, target_data])      # [N_src+N_tgt, L]
    loader = DataLoader(seqs, batch_size=batch_size,
                        shuffle=True, drop_last=True)
    eps = 1e-8
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_seqs in loader:                     # batch_seqs: [B,L]
            batch_seqs = batch_seqs.to(device)

            # ----------------------------------------------------------
            # 1.  Compute pseudo-targets with the fixed classifier
            with torch.no_grad():
                c_out = domain_classifier(batch_seqs).squeeze(-1)  # [B]
                if classifier_output_with_sigmoid:
                    ratio_target = (1.0 - c_out) / (c_out + eps)  # avoid /0
                    ratio_target = torch.log(ratio_target + eps)  # [B]
                else:
                    ratio_target = -c_out                            # [B]

            # ----------------------------------------------------------
            # 2.  Forward pass through ratio network
            ratio_pred = model(batch_seqs)            # [B]
            loss = mse_loss(ratio_pred, ratio_target)

            # ----------------------------------------------------------
            # 3. Laplacian penalty
            lap = laplacian_penalty(ratio_pred, batch_seqs, radius=2)
            loss = loss + lambda_lap * lap

            # ----------------------------------------------------------
            # 4.  Optimise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_seqs.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        print(f"[Ratio] epoch {epoch:2d}/{epochs}, loss = {epoch_loss:.6f}")
        #wandb.log({"Ratio/train_loss": epoch_loss, "epoch": epoch})

    # move back to CPU for convenience
    model.to("cpu")
    return model


# ==================================================================================
# 7. Train time‑dependent ratio estimator  r_ψ(x_t , t)
# ==================================================================================
def train_ratio_estimator(
    model: nn.Module,
    domain_classifier: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    noise_sched: nn.Module,
    *,
    diffusion: str = "absorbing_state",
    mask_idx: Optional[int] = None,
    vocab_size: Optional[int] = None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    classifier_output_with_sigmoid: bool = False,
    device: Union[str, torch.device] = "cpu",
    lambda_lap: float = 0.0,
    only_source_domain: bool = False,                 #  training on source domain only - like in the TLDM paper
):
    """Train a *time‑conditioned* density‑ratio network on noisy sequences.

    The network learns  r_ψ(x_t , t) ≈ q(x_0) / p(x_0)  from pairs (x_t , t)
    produced by the discrete forward process.  Pseudo‑targets come from a
    fixed binary domain classifier d_ω(x_0).
    """

    # ───────────────────── 1.  prepare models ────────────────────────────    domain_classifier.eval().to(device)
    for p in domain_classifier.parameters():
        p.requires_grad_(False)

    model.train().to(device)
    domain_classifier.eval().to(device)  # freeze the classifier
    opt = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=lr)
    mse = nn.MSELoss()

    # Combined unlabeled pool (x_0 ,) – only x_0 used here.
    seqs = source_data if only_source_domain else torch.cat([source_data, target_data])         # (N, L)
    loader = DataLoader(seqs, batch_size=batch_size, shuffle=True, drop_last=True)

    for ep in range(1, epochs + 1):
        running = 0.0
        for x0 in loader:                                   # x0 : (B, L)
            x0 = x0.to(device)
            B = x0.size(0)

            # ---------------- 2. sample timestep & corrupt ----------------
            t = torch.rand(B, device=device)                # U(0,1)
            sigma_t, _ = noise_sched(t)                     # σ(t)
            x_t, _ = corrupt(
                x0,
                sigma_t,
                diffusion=diffusion,
                vocab_size=vocab_size,
                mask_idx=mask_idx,
            )

            # ---------------- 3. build pseudo targets --------------------
            with torch.no_grad():
                c_out = domain_classifier(x0).squeeze(-1)          # (B,)
                if classifier_output_with_sigmoid:
                    ratio_target = (1.0 - c_out) / (c_out + 1e-8)
                    ratio_target = torch.log(ratio_target + 1e-8)
                else:
                    ratio_target = -c_out                            # (B,)

            # ---------------- 4. forward + loss --------------------------
            ratio_pred = model(x_t, t)                              # (B,)
            loss = mse(ratio_pred, ratio_target)

            # optional Laplacian regulariser for smoothness
            if lambda_lap > 0:
                lap = laplacian_penalty(ratio_pred, x_t, radius=2)
                loss = loss + lambda_lap * lap

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item() * B

        print_statement = "[Ratio‑TD]" if not only_source_domain else "[Ratio‑TD-only-src]"
        print(f"{print_statement} epoch {ep}/{epochs}, loss = {running / len(loader.dataset):.6f}")

    model.to("cpu")
    return model




# =====================================================================
# 8. Train the ratio with the Regularization from the TLDM paper (Pseudo-Code 4 in the Appendix)
# =====================================================================
def mutate_sparse(
    xt: torch.Tensor,       # (B,L)   current noisy tokens
    mask: torch.Tensor,     # (B,L,V) boolean; True = need exact score
) -> Tuple[torch.Tensor,    # xt_mut  (B',L)  mutated copies
           List[Tuple[int,int,int]]]:
    """
    For every True entry (b,ℓ,v) in `mask`, clone xt[b] and replace
    position ℓ with vocabulary id v.  Return the stacked clones and
    the list of their original indices so we can scatter the results
    back later.
    """
    device = xt.device
    B, L, V = mask.shape
    where = mask.nonzero(as_tuple=False)              # (B',3)
    clones: List[torch.Tensor] = []
    idx_list: List[Tuple[int,int,int]] = []

    for (b, l, v) in where.tolist():
        x_clone = xt[b].clone()
        x_clone[l] = v
        clones.append(x_clone)
        idx_list.append((b, l, v))

    if len(clones) == 0:
        return torch.empty(0, L, dtype=xt.dtype, device=device), idx_list

    xt_mut = torch.stack(clones, dim=0).to(device)    # (B',L)
    return xt_mut, idx_list


@torch.no_grad()
def _batched_ratio(
    xt_mut: torch.Tensor,        # (B',L)
    t_mut: torch.Tensor,         # (B',)   same dtype/device as training t
    ratio_model: nn.Module,      # your *exact* ratio net
    vocab_size: int,
) -> torch.Tensor:              # (B',)
    """
    Forward pass of the exact ratio model on a batch of mutated
    sequences.  Converts ints→one-hot automatically.
    """
    if xt_mut.numel() == 0:          # early-out when nothing selected
        return torch.zeros(0, device=xt_mut.device)

    one_hot = F.one_hot(xt_mut, vocab_size).float()   # (B',L,V)
    return ratio_model(one_hot, t_mut)                # (B',)

def train_ratio_network_with_regularization(
    model: nn.Module,
    domain_classifier: nn.Module,
    domain_classifier_t: nn.Module,
    denoiser_model: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    noise_sched: nn.Module,
    *,
    diffusion: str = "absorbing_state",
    mask_idx: Optional[int] = None,
    vocab_size: Optional[int] = None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    classifier_output_with_sigmoid: bool = False,  # if True, use sigmoid output
    eta1and2: Tuple[float, float] = (0.1, 0.1),
    device: Union[str, torch.device] = "cpu",
):
    """
    Train a time‐conditioned ratio network r_ψ(x_t, t) with:
      L_ratio + η1 · L_cycle + η2 · L_consistency
    as in Appendix Pseudo-Code 4.
    """
    # 1) Prepare
    loss_consistency, loss_cycle, loss_ratio = 0.0, 0.0, 0.0
    device = torch.device(device)
    η1, η2 = eta1and2
    use_cycle = η1 > 0.0
    use_consistency = η2 > 0.0

    # Freeze pre-trained nets
    for net in (domain_classifier, domain_classifier_t, denoiser_model):
        net.eval().to(device)
    model.train().to(device)

    # 2) DataLoaders for source & target x₀
    src_loader = DataLoader(
        TensorDataset(source_data),
        batch_size=min(batch_size, len(source_data)),
        shuffle=True, drop_last=True
    )
    tgt_loader = DataLoader(
        TensorDataset(target_data),
        batch_size=min(batch_size, len(target_data)),
        shuffle=True, drop_last=True
    )
    # cycle target if needed
    if len(src_loader) > len(tgt_loader):
        from itertools import cycle
        tgt_loader = cycle(tgt_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # 3) Train
    for epoch in range(1, epochs + 1):
        for (x0_src,), (x0_tgt,) in zip(src_loader, tgt_loader):
            x0_src = x0_src.to(device)
            x0_tgt = x0_tgt.to(device)

            # —– sample times for src and tgt
            B_src = x0_src.size(0)
            t_src = torch.rand(B_src, device=device)  # U(0,1)
            sigma_src, _ = noise_sched(t_src)
            x_t_src, _ = corrupt(x0_src, sigma_src,
                                 diffusion=diffusion,
                                 mask_idx=mask_idx,
                                 vocab_size=vocab_size)

            # —– static‐classifier pseudo‐target
            with torch.no_grad():
                c_src = domain_classifier(x0_src).squeeze(-1)
                if classifier_output_with_sigmoid:
                    r_src = (1. - c_src) / (c_src + 1e-8)
                    r_src = torch.log(r_src + 1e-8)
                else:
                    r_src = -c_src

            # —– compute L_ratio
            r_pred_src = model(x_t_src, t_src)
            loss_ratio = mse(r_pred_src, r_src)

            # —– cycle loss on target
            B_tgt = x0_tgt.size(0)
            t_tgt = torch.rand(B_tgt, device=device)  # U(0,1)
            sigma_tgt, _ = noise_sched(t_tgt)
            x_t_tgt, _ = corrupt(x0_tgt, sigma_tgt,
                                 diffusion=diffusion,
                                 mask_idx=mask_idx,
                                 vocab_size=vocab_size)

            if use_cycle:
                with torch.no_grad():
                    c_tdep = domain_classifier_t(
                       x_t_tgt, t_tgt).squeeze(-1)
                    if classifier_output_with_sigmoid:
                        r_tdep = (1. - c_tdep) / (c_tdep + 1e-8)
                        r_tdep = torch.log(r_tdep + 1e-8)
                    else:
                        r_tdep = -c_tdep

                r_pred_tgt = model(x_t_tgt, t_tgt)
                loss_cycle = mse(r_pred_tgt, r_tdep)
            else:
                loss_cycle = torch.tensor(0., device=device)

            # —– consistency regulariser, no gradient is possible on discrete space
            if use_consistency and False:
                # 1) build CutMix-on-clean inputs (source as base, target as donor)
                cons_cutmix_p = 1.0  # tune: e.g., start 0.2 → 0.5 warm-up
                x0_cons = cutmix_seq(x0_src, x0_tgt, cons_cutmix_p, device)

                # 2) corrupt the mixed sequences
                t_cons = torch.rand(x0_cons.size(0), device=device)
                sigma_cons, _ = noise_sched(t_cons)
                x_t_cons, _ = corrupt(x0_cons, sigma_cons,
                                    diffusion=diffusion,
                                    mask_idx=mask_idx,
                                    vocab_size=vocab_size)

                # 3) "teacher" target from the CLEAN classifier (no grad)
                with torch.no_grad():
                    c_clean = domain_classifier(x0_cons).squeeze(-1)
                    if classifier_output_with_sigmoid:
                        r_cons_tgt = torch.log(((1. - c_clean) / (c_clean + 1e-8)) + 1e-8)
                    else:
                        r_cons_tgt = -c_clean

                # 4) student prediction on corrupted mixed inputs
                r_cons_pred = model(x_t_cons, t_cons)

                # 5) consistency loss
                loss_consistency = mse(r_cons_pred, r_cons_tgt)
            else:
                loss_consistency = torch.tensor(0., device=device)

            # —– total
            total = loss_ratio + η1 * loss_cycle + η2 * loss_consistency
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        print(f"[Ratio-Reg] epoch {epoch}/{epochs}  "
              f"L_ratio={loss_ratio.item():.4f}  "
              f"L_cycle={loss_cycle.item():.4f}  "
              f"L_consistency={loss_consistency.item():.4f}")

    model.to("cpu")
    return model


