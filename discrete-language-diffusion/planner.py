import itertools
import typing

import hydra.utils
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
import omegaconf

import models.dit
import noise_schedule
from base_dm_model import BaseDMModel


# ----------------------- metrics -----------------------

class MicroAveragingMetric(torchmetrics.Metric):
  def __init__(self, class_idx: typing.Optional[int] = 1, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.class_idx = torch.tensor(class_idx) if class_idx is not None else None
    self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

  def _update(self, numerator, denominator, preds, y) -> tuple:
    raise NotImplementedError

  def update(self, logits: torch.Tensor, y: torch.Tensor):
    if logits.size(-1) == 1:
      preds = (logits > 0).long().squeeze(-1)
    else:
      preds = torch.argmax(logits, dim=-1)
    y = y.view(-1)
    assert preds.shape == y.shape, f"preds shape {preds.shape} != y shape {y.shape}"
    self.numerator, self.denominator = self._update(self.numerator, self.denominator, preds, y)

  def compute(self):
    return self.numerator.float() / self.denominator if self.denominator.item() > 0. else torch.tensor(0.0)

  def reset(self):
    self.numerator = torch.tensor(0.0).to(self.device)
    self.denominator = torch.tensor(0.0).to(self.device)


class CrossEntropy(MicroAveragingMetric):
  def _update(self, numerator, denominator, logits, y) -> tuple:
    with torch.no_grad():
      if logits.size(-1) == 1:
        numerator += F.binary_cross_entropy_with_logits(logits.view(-1), y.float(), reduction='sum')
      else:
        numerator += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                     ignore_index=-100, reduction='sum')
      denominator += y.numel()
    return numerator, denominator

  def update(self, logits: torch.Tensor, y: torch.Tensor):
    y = y.view(-1)
    self.numerator, self.denominator = self._update(self.numerator, self.denominator, logits, y)


class Accuracy(MicroAveragingMetric):
  def _update(self, numerator, denominator, preds, y) -> tuple:
    if self.class_idx is None:
      numerator += (preds == y).sum()
      denominator += y.numel()
    else:
      class_idx = self.class_idx
      rel = (y == class_idx)
      numerator += (preds[rel] == class_idx).sum()
      denominator += rel.sum()
      rel = (y != class_idx)
      numerator += (preds[rel] != class_idx).sum()
      denominator += rel.sum()
    return numerator, denominator


class Precision(MicroAveragingMetric):
  def _update(self, numerator, denominator, preds, y) -> tuple:
    class_idx = self.class_idx
    rel = (preds == class_idx)
    numerator += (y[rel] == class_idx).sum()
    denominator += rel.sum()
    return numerator, denominator


class Recall(MicroAveragingMetric):
  def _update(self, numerator, denominator, preds, y) -> tuple:
    class_idx = self.class_idx
    rel = (y == class_idx)
    numerator += (preds[rel] == class_idx).sum()
    denominator += rel.sum()
    return numerator, denominator


# ----------------------- planner -----------------------

class Planner(BaseDMModel):
  def __init__(
      self,
      config,
      diffusion_model: BaseDMModel,                         
      tokenizer: transformers.PreTrainedTokenizer,
      pretrained_backbone: typing.Optional[torch.nn.Module] = None,
  ):
    # 0) Base
    self.config = config
    super().__init__()   # gives self.dtype, device helpers

    # 1) Flags
    self.train_time_independent = True
    self.is_eval_planner = getattr(config, "is_eval_planner", False)

    # 2) Tokenizer / vocab
    self.tokenizer = tokenizer
    self.vocab_size = tokenizer.vocab_size
    if not getattr(tokenizer, "mask_token", None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id

    # 3) Training schedule flags
    tr_cfg = config.training_planner
    self.antithetic_sampling = tr_cfg.antithetic_sampling
    self.importance_sampling = tr_cfg.importance_sampling
    self.change_of_variables = tr_cfg.change_of_variables
    self.sampling_eps = tr_cfg.sampling_eps

    self.T = config.T
    self.lr = config.optim.lr
    self.time_conditioning = config.time_conditioning

    # 4) Noise schedule
    self.noise = noise_schedule.get_noise(config, dtype=self.dtype)

    # 5) Backbone: per-position logit (B,L) or (B,L,1)
    if config.planner_backbone == "dit":
      self.planner_model = models.dit.DITClassifier(
        config,
        vocab_size=self.vocab_size,
        time_conditioning=not self.train_time_independent,
        output_dim=config.model.length,    # one logit per position
      )
    else:
      raise NotImplementedError(f"planner backbone {config.planner_backbone} not implemented.")
    if pretrained_backbone is not None:
      self.planner_model.load_pretrained_encoder(pretrained_backbone)

    # 6) Metrics
    metrics = torchmetrics.MetricCollection({
      "cross_entropy": CrossEntropy(),
      "accuracy": Accuracy(class_idx=None),
      "precision": Precision(class_idx=1),
      "recall": Recall(class_idx=1),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix="train/")
    self.valid_metrics = metrics.clone(prefix="val/")
    self.test_metrics = metrics.clone(prefix="test/")

    # 7) Misc
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

    # 8) Frozen denoiser for labels
    if self.config.mode == "train_planner":
      self.diffusion_model = diffusion_model
      self.diffusion_model.eval()  # ensure denoiser is frozen during planner training
      for p in self.diffusion_model.parameters():
        p.requires_grad_(False)
    else:
      self.diffusion_model = None

  # ---- forward ----
  def forward(self, x, sigma=None, x_emb=None, attention_mask=None):
    """Return per-position logits. If time-independent, sigma can be None."""
    if self.is_eval_planner:
      logits = self.planner_model(x)
      if hasattr(logits, 'logits'):
        logits = logits.logits
      return logits
    sigma = self._process_sigma(sigma) if (sigma is not None and not self.train_time_independent) else sigma
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.planner_model(x, sigma, x_emb=x_emb, attention_mask=attention_mask)
    return logits

  # ---- train/val/test steps ----
  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log('trainer/loss', loss.item(), on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
    self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'],
             on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=False)
    return loss

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  # ---- optim + scheduler ----
  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      itertools.chain(self.planner_model.parameters(), self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1, self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay,
    )
    total_steps = self.config.trainer_planner.max_steps
    warmup_steps = int(total_steps * 0.1)
    sched_cfg = omegaconf.OmegaConf.merge(
      self.config.lr_scheduler,
      {"warmup_t": warmup_steps, "t_initial": total_steps - warmup_steps},
    )
    scheduler = hydra.utils.instantiate(sched_cfg, optimizer=optimizer)
    return [optimizer], [{
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }]

  # ---- core loss ----
  def _compute_loss(self, batch, prefix):
    device = self.device

    # A) data
    x0 = batch['input_ids'].to(device)
    attention_mask = batch.get('attention_mask', torch.ones_like(x0, device=device))

    # Always create a corrupted input x_t (labels depend on it).
    B = x0.size(0)
    t = self._sample_t(B)                                 # (B,)
    if self.T > 0:                                        # map to {1/T, ..., 1}
      t = (t * self.T).to(torch.int) / self.T + (1.0 / self.T)

    time_cond, move_chance, dsigma = self._get_time_conditioning_and_move_chance(t)
    x_t = self._q_xt(x0, move_chance)                     # (B,L)

    # B) labels (binary) on masked positions using frozen denoiser
    with torch.no_grad():
      log_x = self.diffusion_model.forward(x_t, time_cond)   # (B,L,V)
      z = log_x.argmax(dim=-1)                               # (B,L)
    mask_M = (x_t == self.mask_index) & attention_mask.bool()
    y = ((z == x0) & mask_M).float()                         # (B,L), 1 only where masked & correct-now

    # C) planner logits (per position)
    logits = self.forward(x_t, None if self.train_time_independent else time_cond,
                          attention_mask=attention_mask)     # (B,L) or (B,L,1)
    if logits.dim() == 3 and logits.size(-1) == 1:
      logits_pos = logits.squeeze(-1)                        # (B,L)
    else:
      logits_pos = logits

    # D) BCE over masked positions only
    if mask_M.any():
      loss_core = F.binary_cross_entropy_with_logits(logits_pos[mask_M], y[mask_M], reduction='mean')
    else:
      loss_core = logits_pos.new_tensor(0.0)

    # E) weight by σ′(t)
    if self.change_of_variables:
      weight = 1.0  # plug the correct weight for your CoV parameterization if needed
    else:
      # dsigma is dσ/dt from the noise schedule (shape (B,))
      weight = dsigma.mean().clamp_min(1e-8)
    loss = loss_core * weight

    # F) metrics (masked positions only, treat as single-logit binary)
    logits_m = logits_pos[mask_M].unsqueeze(-1) if mask_M.any() else logits_pos.new_zeros((0, 1))
    y_m = y[mask_M].long() if mask_M.any() else y.new_zeros((0,), dtype=torch.long)
    if prefix == 'train':
      self.train_metrics.update(logits_m, y_m)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(logits_m, y_m)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(logits_m, y_m)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')
    self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

    return loss


  @staticmethod
  def detokenize_batch(x0: torch.Tensor, tokenizer, *, skip_special_tokens: bool = True):
    input_ids = x0.detach().cpu().tolist()
    return tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)

  @torch.no_grad()
  def get_next_step(
    self,
    xt: torch.Tensor,                                # (B, L)
    attention_mask: typing.Optional[torch.Tensor] = None,
    sigma: typing.Optional[torch.Tensor] = None,
    *,
    strategy: str = "sample",                        # "argmax" | "sample"
    temperature: typing.Optional[float] = 0.5,      # only for "sample"
    return_scores: bool = False,
      ) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:
    B, L = xt.shape

    # 1) per-position logits -> (B, L) (squeeze if model returns (B, L, 1))
    attention_mask = attention_mask if attention_mask is not None else xt.new_ones(xt.shape, dtype=torch.bool)
    logits = self.forward(
        xt,
        None if self.train_time_independent else sigma,
        attention_mask=attention_mask,
    )
    if logits.dim() == 3 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)  # (B, L)
    assert logits.shape == (B, L)

    # 2) eligibility: masked tokens & inside attention
    eligible = (xt == self.mask_index)
    if attention_mask is not None:
        eligible = eligible & attention_mask.bool()

    if not eligible.any():
        return (None, logits.new_full((B, L), float("-inf"))) if return_scores else None

    # 3) scores: set ineligible to −inf
    scores = logits.masked_fill(~eligible, float("-inf"))  # (B, L)

    # 4) pick one column per eligible row
    temp = float(temperature) if (strategy == "sample" and temperature is not None and temperature > 0) else 1.0
    idx_list = []

    for b in range(B):
        row = scores[b]  # (L,)
        valid_idx = torch.nonzero(torch.isfinite(row), as_tuple=True)[0]
        if valid_idx.numel() == 0:
            # no eligible position in this row -> skip
            continue

        row_logits = row[valid_idx]  # logits over valid positions only

        if strategy == "argmax":
            j_local = int(row_logits.argmax().item())
        else:  # "sample"
            # handle near-uniform / degenerate safely
            if (row_logits.max() - row_logits.min()) < 1e-6:
                j_local = int(torch.randint(0, valid_idx.numel(), (1,), device=xt.device).item())
            else:
                probs = torch.softmax(row_logits / temp, dim=0)
                if torch.isnan(probs).any() or float(probs.sum().item()) <= 0:
                    j_local = int(torch.randint(0, valid_idx.numel(), (1,), device=xt.device).item())
                else:
                    j_local = int(torch.multinomial(probs, 1).item())

        j = int(valid_idx[j_local].item())
        idx_list.append((b, j))

    if len(idx_list) == 0:
        return (None, scores) if return_scores else None

    # (M, 2) LongTensor of (b, j) pairs, one per eligible row
    idx = torch.tensor(idx_list, device=xt.device, dtype=torch.long)
    return (idx, scores) if return_scores else idx[:,1].unsqueeze(-1)