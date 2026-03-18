import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
import typing

# -----------------------------------------------------------------------------
# 1.  Time‑step → vector embedding
# -----------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """Turn a 1‑D tensor of timesteps/noise levels into a feature vector."""

    def __init__(self, cond_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000):
        """Sinusoidal embedding (same formula as in diffusion/Transformers)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 != 0:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: [B]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)  # [B, cond_dim]


# -----------------------------------------------------------------------------
# 2.  AdaLN-Encoder layer (Zero‑init)
# -----------------------------------------------------------------------------
class AdaLNEncoderLayer(nn.Module):
    """Transformer encoder block with AdaLN‑Zero conditioning (shift, scale, gate)."""

    def __init__(self, d: int, heads: int, cond_dim: int, mlp_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        # Core sub‑layers
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

        # 6·d modulation parameters per block (shift, scale, gate) × 2
        self.mod = nn.Linear(cond_dim, 6 * d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x : [B, L, d]
        c : [B, cond_dim]
        """
        B, L, d = x.shape
        # Derive modulation parameters for this sample
        shift1, scale1, gate1, shift2, scale2, gate2 = self.mod(c).chunk(6, dim=-1)  # each [B, d]

        # ---- Self‑attention sub‑layer ----
        x_norm = self.norm1(x)
        x_mod = (1 + scale1[:, None, :]) * x_norm + shift1[:, None, :]
        attn_out, _ = self.self_attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate1[:, None, :] * self.dropout(attn_out)

        # ---- Feed‑forward / MLP sub‑layer ----
        y_norm = self.norm2(x)
        y_mod = (1 + scale2[:, None, :]) * y_norm + shift2[:, None, :]
        mlp_out = self.mlp(y_mod)
        x = x + gate2[:, None, :] * mlp_out

        return x


# -----------------------------------------------------------------------------
# 4.  RatioNet with per‑sample AdaLN conditioning
# -----------------------------------------------------------------------------
class RatioNetAdaLN(nn.Module):
    """Transformer encoder that predicts a scalar or classifier, conditioned on timestep."""

    def __init__(
        self,
        vocab_sz: int,
        seq_len: int,
        d: int = 256,
        heads: int = 4,
        layers: int = 2,
        output_with_sigmoid: bool = False,
        dropout: float = 0.1,
        out_dim: int = 1,
        time_conditioning: bool = False
    ):
        super().__init__()
        self.output_with_sigmoid = output_with_sigmoid

        # Token & positional embeddings
        self.emb = nn.Embedding(vocab_sz, d)
        self.pos = nn.Parameter(torch.randn(seq_len, d))

        # Time‑step embedder (same width as model)
        self.time_emb = TimestepEmbedder(cond_dim=d)

        # Stack of AdaLN encoder layers
        self.layers = nn.ModuleList([
            AdaLNEncoderLayer(d, heads, cond_dim=d, dropout=dropout) for _ in range(layers)
        ])

        # Output head
        self.fc = nn.Linear(d, out_dim)
        self.time_conditioning = time_conditioning

    # ---------------------------------------------------------------------
    def forward(self, seq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        seq : (B, L)  **or**  (B, L, V)
        """
        if seq.ndim == 3:
            x = nn.functional.linear(seq.float(), self.emb.weight.T) + self.pos
        else:
            x = self.emb(seq) + self.pos
        if not self.time_conditioning:
            t = torch.zeros_like(t)
        c = self.time_emb(t)
        for layer in self.layers:
            x = layer(x, c)

        g = x.mean(dim=1)
        out = self.fc(g)
        return torch.sigmoid(out).squeeze(-1) if self.output_with_sigmoid else out.squeeze(-1)

# ================================================================
# 2.  AdaLN Transformer backbone for the denoiser
# ================================================================
class AdaLNDenoiser(nn.Module):
    """
    Forward  =  (tokens x_t , sigma_t)  ➜  logits over vocab.
    Matches the signature expected inside `train_denoiser(...)`.
    """
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d: int = 256,
        heads: int = 4,
        layers: int = 4,
        dropout: float = 0.1,
        time_conditioning: bool = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb   = nn.Parameter(torch.randn(seq_len, d))
        self.time_emb  = TimestepEmbedder(cond_dim=d)

        self.blocks = nn.ModuleList([
            AdaLNEncoderLayer(d, heads, cond_dim=d, dropout=dropout)
            for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)
        self.time_conditioning = time_conditioning

    # -----------------------------------------------------------
    def forward(self, x_t: torch.LongTensor, sigma_t: torch.Tensor) -> torch.Tensor:
        """
        x_t     : (B, L)   integer indices
        sigma_t : (B,) or (B,1)   noise level – *not* log‑sigma
        returns : (B, L, vocab_size)   **logits** (no softmax)
        """
        if sigma_t.ndim > 1:                       # squeeze if (B,1)
            sigma_t = sigma_t.squeeze(-1)
        if not self.time_conditioning:
            sigma_t = torch.zeros_like(sigma_t)

        h = self.token_emb(x_t) + self.pos_emb     # (B,L,d)
        c = self.time_emb(sigma_t)                 # (B,d)

        for blk in self.blocks:
            h = blk(h, c)

        h = self.ln_f(h)
        return self.head(h)                        # raw scores



# =====================================================================
# 5.  Define time independent Network: - Sequence-level ratio estimator rφ : Σ^L → ℝ₊
#                     - Domain classifier  dφ : Σ^L → {0,1}
# =====================================================================
class ClassifierNet(nn.Module):
    def __init__(
        self,
        vocab_sz: int,
        seq_len: int,
        d: int = 256,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,          # NEW
        output_with_sigmoid: bool = False,
    ):
        super().__init__()
        self.emb  = nn.Embedding(vocab_sz, d)
        self.pos  = nn.Parameter(torch.randn(seq_len, d))      # learned positional enc.

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,nhead=heads,dim_feedforward=1024,batch_first=True,dropout=dropout,)
        self.enc   = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.drop  = nn.Dropout(dropout)                        # NEW
        self.fc    = nn.Linear(d, 1)
        self.output_with_sigmoid = output_with_sigmoid

    # ---------------------------------------------------------
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq : (B, L) token indices
        """
        x = self.emb(seq) + self.pos            # (B, L, d)
        x = self.drop(x)                        # NEW – token-level dropout

        h = self.enc(x)                         # (B, L, d)
        g = h.mean(dim=1)                       # global average pooling
        g = self.drop(g)                        # NEW – sequence-level dropout

        out = self.fc(g)                        # (B, 1)
        return torch.sigmoid(out) if self.output_with_sigmoid else out

    # Convenience wrapper (unchanged) -------------------------
    def get_log_ratio(self, seq: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(seq)


class QueryPosEmbed(nn.Module):
    """
    Embed a *single* target position ℓ into a cond-dim vector.
    """
    def __init__(self, max_len: int, cond_dim: int):
        super().__init__()
        self.table = nn.Embedding(max_len, cond_dim)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos : (B,)  int
        return self.table(pos)         # (B, cond_dim)


class RatioNetAdaLNVector(nn.Module):
    """
    r_ψ(x_t, ℓ, t)  →  (B, V)
    Conditioning vector  c = time_emb(t) + query_pos_emb(ℓ)
    Output is a full vocab-length log-ratio vector for each example.
    """

    def __init__(
        self,
        vocab_sz: int,
        seq_len: int,
        d: int = 256,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_sz, d)
        self.abs_pos   = nn.Parameter(torch.randn(seq_len, d))

        self.time_emb  = TimestepEmbedder(cond_dim=d)
        self.qpos_emb  = QueryPosEmbed(max_len=seq_len, cond_dim=d)

        self.blocks = nn.ModuleList(
            [AdaLNEncoderLayer(d, heads, cond_dim=d, dropout=dropout)
             for _ in range(layers)]
        )

        self.fc = nn.Linear(d, vocab_sz, bias=False)
        if tie_weights:
            self.fc.weight = self.token_emb.weight

    # --------------------------------------------------------------- #
    def forward(
        self,
        seq: torch.Tensor,       # (B, L)   noisy sequence  x_t
        pos: torch.Tensor,       # (B,)     target indices  ℓ
        t:   torch.Tensor,       # (B,)     timestep        t
    ) -> torch.Tensor:           # (B, V)   log-ratio vector
        B, L = seq.shape

        x = self.token_emb(seq) + self.abs_pos                  # (B, L, d)
        c = self.time_emb(t) + self.qpos_emb(pos)               # (B, d)

        for blk in self.blocks:
            x = blk(x, c)                                       # (B, L, d)

        g = x.mean(dim=1)                                       # (B, d)
        log_r = self.fc(g)                                      # (B, V)
        return log_r



class PlannerNet(nn.Module):
    """
    Per-position planner: (B, L) logits.
    Same encoder style as ClassifierNet, but with a *token* head instead of pooling.
    """
    def __init__(
        self,
        vocab_sz: int,
        seq_len: int,
        d: int = 256,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        time_conditioning: bool = False,   # optional
    ):
        super().__init__()
        self.time_conditioning = time_conditioning

        self.emb  = nn.Embedding(vocab_sz, d)
        self.pos  = nn.Parameter(torch.randn(seq_len, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=1024, batch_first=True, dropout=dropout
        )
        self.enc  = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.drop = nn.Dropout(dropout)
        self.token_head = nn.Linear(d, 1)        # <— per-token head
        # optional time embedding if you later want t-conditioning
        if self.time_conditioning:
            self.time_emb = TimestepEmbedder(cond_dim=d)

    def forward(self, seq: torch.Tensor, t: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        seq : (B, L) token ids
        t   : (B,) or None — ignored unless time_conditioning=True
        returns: (B, L) logits
        """
        x = self.emb(seq) + self.pos          # (B, L, d)
        if self.time_conditioning:
            assert t is not None, "Provide t when time_conditioning=True"
            c = self.time_emb(t)              # (B, d)
            x = x + c[:, None, :]             # simple additive conditioning

        x = self.drop(x)
        h = self.enc(x)                       # (B, L, d)
        logits = self.token_head(h).squeeze(-1)  # (B, L)
        return logits