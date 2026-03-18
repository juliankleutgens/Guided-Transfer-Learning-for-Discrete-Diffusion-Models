import itertools
import math
import os
from tqdm import tqdm, trange
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import models
import ratio
import planner
import noise_schedule
import utils
import time
import random
from base_dm_model import BaseDMModel

LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
  gumbel_norm = (1e-10- (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


# =========================================================
# Table of Contents
# =========================================================
# 0. Initialization
# 1. Configuration validation and checkpointing
# 2. Optim/EMA
# 3. Forward & parameterizations
#     3.1 SUBS parameterization
#     3.2 SEDD parameterization
# 4. Loss computation
#     4.1 Compute the loss on the input batch
#     4.2 Compute the loss depending on the parameterization: D3PM, SEDD, or SUBS
#     4.3 Guided Loss computation on for validation
# 5. Train/Val/Test hooks
# 6. Normal Sampling (core)
#     6.1 Sampling without guidance
#     6.2 Sampling with guidance
# 7. Sampling with Planner
#     7.1 Denoiser step at a single position
#     7.2 Guided sampling at a single position
# 8. Helper functions
#     8.1 Sampling helper
#     8.2 Ratio model helper functions
#     8.3 Top-k helper function
#     8.4 Preserve mask mass helper function
#     8.5 Planner helper: selection of next position by confidence
# 9. PPL / Evaluation
# 10. Utilities for logging


class Diffusion(BaseDMModel):
  # =========================================================
  # 0. Initialization
  # =========================================================
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    self.config = config
    super().__init__()
    self.save_hyperparameters()

    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    if (not hasattr(self.tokenizer, 'mask_token') or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token =\
          self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id =\
          self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.inference_mode = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()
    self.diffusion = config.diffusion

    # sample congiguration
    self.validation_mode = config.mode == "ppl_eval"
    self.guided_sampling = self.config.sampling.guided_sampling
    self.guided_validation = self.config.eval.guided_validation
    self.guidance_scale = self.config.gamma.get('guidance_scale', 1.0)
    if self.guided_sampling or (self.guided_validation and self.validation_mode):
      self.ratio_model, ratio_loaded = utils.build_or_load(
        model_=ratio.RatioEstimator,
        init_kwargs=dict(
            config=config,
            tokenizer=tokenizer,
            domain_classifier=None,
            domain_classifier_time_dependent=None,
            inference_mode=True
        ),
        ckpt_path=config.ratio_model.ckpt_path,
        freeze=True,)
      self.ratio_model_loaded = ratio_loaded
      if not ratio_loaded:
         print("WARNING: Ratio model not loaded, using a Random Weights Ratio instead!")
      self.ratio_model.eval()           # inference only
    self._ratio_flat_num = 0  # int
    self._ratio_flat_den = 0  # int
    TOPK_LIST = (4,8,25,50, 100,250, 500, 1000, 2000, 5000, 10000, 15000, 20000)
    self.TOPK_LIST = [k for k in TOPK_LIST if k <= self.vocab_size and k <= self.config.sampling.compute_top_k_ratio]

    self._topk_ratio_vs_source    = {k: [] for k in self.TOPK_LIST}  # diffusion vs ratio
    self._topk_diff_vs_guidance = {k: [] for k in self.TOPK_LIST}  # diffusion vs guided

    self._t_records = []  # float t per (informative) batch
    self.gamma_sched = utils.get_gamma_scheduler(config)
    self.show_loading = False
    self.planner = None
    if self.config.sampling.plan_random == False and self.planner == None and self.config.mode == "sample_planner_eval":
      self.planner, planner_loaded = utils.build_or_load(
          model_=planner.Planner,  
          init_kwargs=dict(
              config=config,
              diffusion_model=self,         # the denoiser is used only for labels at train time; ok for eval too
              tokenizer=tokenizer,
              pretrained_backbone=None,
          ),
          ckpt_path=getattr(config, "planner", {}).get("ckpt_path", None),
          freeze=True,
      )
      if planner_loaded:
        self.planner.eval()
        print("Planner loaded and frozen for sampling.")
    self.hf_cache_root = self.config.system.hf_cache_root
    utils.set_hf_cache_envs(self.hf_cache_root)

  # =========================================================
  # 1. Configuration validation and checkpointing
  # =========================================================
  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  # =========================================================
  # 2. Optim/EMA
  # =========================================================
  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  def optimizer_step(self, *args, **kwargs):
    """The optimizer step for the Hydra."""
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
  # =========================================================
  # 3. Forward & parameterizations
  # =========================================================
  def forward(self, x, sigma):
    """Returns log score."""
    sigma = self._process_sigma(sigma) # if time_conditioning==False, then sigma=zeros 
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    return logits

  def _subs_parameterization(self, logits, xt):
    logits[..., self.mask_index] += self.neg_infinity

    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0

    return logits.log_softmax(dim=-1)

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(sigma < 0.5,torch.expm1(sigma),sigma.exp() - 1).log().to(logits.dtype) # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(logits.shape[-1] - 1)

    # The below scatter operation sets the log score for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],torch.zeros_like(logits[..., :1]))
    return logits

  # =========================================================
  # 4.  Loss computation
  # =========================================================
  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['input_ids'], attention_mask)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss
  
  # ------------------------------------------------
  # 4.1 Compute the loss on the input batch
  def _loss(self, x0, attention_mask):
    (input_tokens, output_tokens,attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    loss = self._forward_pass_diffusion(input_tokens)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)
  
  def _forward_pass_diffusion(self, x0):
    t = self._sample_t(x0.shape[0])
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self._q_xt(x0, move_chance)
    model_output = self.forward(xt, unet_conditioning)
    topk = getattr(self.config.eval, "topk_remass", 64)
    if topk > 0 and self.validation_mode:
      model_output = self._truncate_and_renorm_logprobs(
          model_output, xt, x0, k=topk, only_masked=True, include_mask=True
      )
    # mask = (xt == self.mask_index); pos_idx_ = mask.nonzero(); i = 1
    # model_output[pos_idx_[i,0],pos_idx_[i,1], x0[pos_idx_[i,0],pos_idx_[i,1]]]


    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(model_output=model_output, xt=xt, x0=x0, t=t)
      return diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(- torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]
  
  # ------------------------------------------------
  # 4.2 Compute the loss depending on the parameterization: D3PM, SEDD, or SUBS
  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy
  
  # ------------------------------------------------
  # 4.3 Guided Loss computation for validation
  def _compute_guided_loss(self, batch: typing.Dict[str, torch.Tensor]) -> Loss:
    assert self.guided_validation
    device = self.device

    x0 = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask", torch.ones_like(x0, device=device))
    input_tokens, _, attention_mask = self._maybe_sub_sample(x0, attention_mask)
    gamma = float(self.guidance_scale)
    B = input_tokens.size(0)
    t = self._sample_t(B)
    if self.T > 0:
        t = (t * self.T).to(torch.int) / self.T + (1.0 / self.T)

    if self.change_of_variables:
        time_conditioning = t[:, None]
        f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
        f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
        move_chance = torch.exp(f_0 + t * (f_T - f_0))[:, None]
        sigma, dsigma = None, None
    else:
        sigma, dsigma = self.noise(t)
        time_conditioning = sigma[:, None]
        move_chance = (1.0 - torch.exp(-sigma))[:, None]

    # forward noising
    xt = self._q_xt(input_tokens, move_chance)

    # base logits/log-scores
    log_x_theta = self.forward(xt, time_conditioning)  # (B,L,V)

    # --- top-k remass & ratio on the same candidate set ---
    topk = int(getattr(self.config.eval, "topk_remass", 0) or 0)
    if topk > 0:
        # Build keep-set from base logits; include label; for SUBS loss do NOT include [MASK]
        include_mask = (self.diffusion == "absorbing_state")
        keep_mask = self._build_keep_mask_from_base(
            base_logits=log_x_theta, xt=xt, x0=input_tokens,
            k=topk, include_mask=include_mask
        )  # (B,L,V) bool

        # Ratio scores only for those candidates (plus label if needed)
        k_ratio = min(topk, self.vocab_size)
        ratio_log = self.get_ratio_log_topk_stream(
            xt=xt, sigma=time_conditioning, log_x_theta=log_x_theta,
            k=k_ratio, ratio_bs=getattr(self.config.sampling, "batch_size_ratio", 1024),
            x0=input_tokens, normalize=True,
        )
        # mask = (xt == self.mask_index); pos_idx_ = mask.nonzero(); i = 1
        # ratio_log[pos_idx_[i,0],pos_idx_[i,1], x0[pos_idx_[i,0],pos_idx_[i,1]]]
        """
        mask = (xt == self.mask_index); pos_idx_ = mask.nonzero(); i = -20
        v = ratio_log[pos_idx_[i, 0], pos_idx_[i, 1], :]
        mask = v > -100          # False only for -inf (also False for NaN)
        idx  = mask.nonzero(as_tuple=True)[0]
        vals = v[idx]
        vals
        """
        ratio_log = ratio_log.to(log_x_theta.device)
        # neutral outside candidate set (will be pruned anyway)
        ratio_log = ratio_log.masked_fill(~keep_mask, 0.0)

        guided_raw = log_x_theta + gamma * ratio_log
        # Renormalize ONLY over kept tokens

        guided_logprob = self._renorm_with_keep_mask(guided_raw, keep_mask)
        # For absorbing-state, if you want to preserve mask prob *during loss*,
        # you can overwrite the mask column post-hoc; for SUBS it’s not needed.
        if self.diffusion == "absorbing_state":
            guided_logprob = self._preserve_mask_mass(
                guided_logprob, xt,
                mask_logits=(log_x_theta + 0.0)[..., self.mask_index]  # base mask logits
            )
    else:
        # Full-vocab path (no pruning): compute ratio for all (or stream) and softmax globally
        ratio_log = self.get_ratio_log_stream(
            xt, time_conditioning.squeeze(-1),
            chunk_v=getattr(self.config.sampling, "batch_size_ratio", 1024)
        )
        ratio_log = torch.where(torch.isfinite(ratio_log), ratio_log, torch.zeros_like(ratio_log))
        ratio_log = ratio_log.to(log_x_theta.device)

        if self.diffusion == "absorbing_state":
            ratio_log[..., self.mask_index] = 0.0
        guided_logprob = torch.log_softmax(log_x_theta + gamma * ratio_log, dim=-1)
        if self.diffusion == "absorbing_state":
            guided_logprob = self._preserve_mask_mass(
                guided_logprob, xt, mask_logits=log_x_theta[..., self.mask_index]
            )

    # ---- same objective as in _forward_pass_diffusion ----
    if self.parameterization == 'sedd':
        assert dsigma is not None
        per_token = dsigma[:, None] * self._score_entropy(
            guided_logprob, sigma[:, None], xt, input_tokens)
    elif self.T > 0:
        per_token = self._d3pm_loss(model_output=guided_logprob, xt=xt, x0=input_tokens, t=t)
    else:
        log_p_theta = torch.gather(guided_logprob, -1, input_tokens.unsqueeze(-1)).squeeze(-1)
        if self.change_of_variables or self.importance_sampling:
            per_token = log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))
        else:
            per_token = -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

    nlls = per_token * attention_mask
    token_nll = nlls.sum() / attention_mask.sum().clamp_min(1)
    self.valid_metrics.update(nlls, attention_mask)
    self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, sync_dist=True)
    return Loss(loss=token_nll, nlls=nlls, token_mask=attention_mask)
  # =========================================================
  # 5. Train/Val/Test hooks
  # =========================================================
  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    #x_clean_text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
    #perplexity_of_clean_text = self.get_generative_perplexity(x_clean_text, retokenize=True,max_length=self.config.model.length)
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    self._ratio_flat_num = 0
    self._ratio_flat_den = 0
    self._topk_ratio_vs_source    = {k: [] for k in self.TOPK_LIST}  # diffusion vs ratio
    self._topk_diff_vs_guidance = {k: [] for k in self.TOPK_LIST}  # diffusion vs guided
    self._t_records = []
    if self.ema:
      self.ema.store(itertools.chain(self.backbone.parameters(),self.noise.parameters()))
      self.ema.copy_to(itertools.chain(self.backbone.parameters(),self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0
    if self.guided_validation and self.validation_mode and self.trainer.is_global_zero:
        print("[val] Guided validation active → using _compute_guided_loss")

  def validation_step(self, batch, batch_idx):
    self.show_loading = batch_idx == 0
    if self.guided_validation and self.validation_mode:
      torch.cuda.reset_peak_memory_stats(self.device)
      return self._compute_guided_loss(batch).loss
    else:
      return self._compute_loss(batch, prefix='val')

  def on_validation_epoch_end(self):
    if self.guided_validation and self.validation_mode:
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(),
                                self.noise.parameters()))
        return
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples and not self.parameterization == 'ar'):
      samples, text_samples = None, None
      torch.cuda.empty_cache()          
      torch.cuda.reset_peak_memory_stats()
      for _ in range(self.config.sampling.num_sample_batches):
        samples = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[: self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))
  # =========================================================
  # 6. Normal Sampling (core) 
  # =========================================================
  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    xt = self._sample_prior(batch_size_per_gpu,self.config.model.length).to(self.device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    pbar = tqdm(range(self.config.sampling.steps),desc='Sampling',leave=False)
    NFEs = 0
    cache = None
    q_xs = xt.new_zeros(1)

    for i in pbar:
      t, sigma_t, sigma_s, move_chance_t, move_chance_s = \
          self._compute_move_chances(timesteps[i], dt, xt.size(0))
      NFEs += 1 if cache is None else 0
      if self.sampler == 'ddpm_cache' and not self.config.sampling.guided_sampling:
        xs, q_xs, cache = self._mdlm_denoise(
          xt=xt,
          time_conditioning=sigma_t,
          move_chance_t=move_chance_t,
          move_chance_s=move_chance_s,
          cache=cache)
      elif self.sampler == 'ddpm_cache' and self.config.sampling.guided_sampling:
        xs, q_xs, cache = self._ratio_guidance_denoise(
            xt=xt,
            timestep=timesteps[i],
            time_conditioning=sigma_t,
            move_chance_t=move_chance_t,
            move_chance_s=move_chance_s,
            cache=cache)      
      else:
        xs = self._analytic_update(xt, t, dt)
      
      pbar.set_postfix(NFEs=NFEs,prob_check=(q_xs.sum() / xt.numel()).item(),nan_check=bool(q_xs.isnan().sum() > 0))
      if (not torch.allclose(xs, xt) or self.time_conditioning):
        cache = None
      xt = xs
      if self.config.sampling.print_all_noise_sequences:
        decoded_xt = self.tokenizer.batch_decode(xt, skip_special_tokens=True)
        print(f"Step {i+1}/{num_steps} - Sampled tokens: {decoded_xt}")
          # just one last denoising step to get rid of remaining MASK tokens
    if self.config.sampling.noise_removal:
      # print how many mask tokens are left
      print(f"Number of mask tokens left: {xt.eq(self.mask_index).sum().item()}")
      t = timesteps[-1] * torch.ones(xt.shape[0], 1,device=self.device)
      if self.sampler == 'analytic':
        xs = self._denoiser_update(xt, t)
      else:
        unet_conditioning = self.noise(t)[0]
        xs = self.forward(xt, unet_conditioning).argmax(dim=-1)
      xt = xs
    return xt
  
  # ------------------------------------------------
  # 6.1. Sampling without guidance 
  def _mdlm_denoise(
    self,
    xt: torch.tensor,
    time_conditioning: torch.tensor,
    move_chance_t: torch.tensor,
    move_chance_s: torch.tensor,
    cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
     ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:

    # Compute x_theta
    if cache is not None:
      log_x_theta = cache['log_x_theta']
      x_theta = log_x_theta.exp()
    else:
      with torch.cuda.amp.autocast(): 
        log_x_theta = self.forward(xt, time_conditioning)
        if self.config.sampling.top_k_sampling > 0:
          log_x_theta = self.keep_top_k_tokens_rest_minus_infinity(log_x_theta=log_x_theta, xt=xt,k=self.config.sampling.top_k_sampling)
        x_theta = log_x_theta.exp()

    # Compute posterior
    if self.diffusion == 'absorbing_state':
      q_xs = x_theta * (move_chance_t - move_chance_s)
      q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
      q_xs /= move_chance_t
    elif self.diffusion == 'uniform':
      q_xs = self._compute_posterior(
        x=x_theta,
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t)
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented.")
    # Sample from posterior
    xs = _sample_categorical(q_xs)
    if self.diffusion == 'absorbing_state':
      copy_flag = (xt != self.mask_index).to(torch.bool)
      q_xs[copy_flag] = 0.0
      q_xs[copy_flag, xt[copy_flag]] = 1.0
      xs = torch.where(copy_flag, xt, xs)

    return xs, q_xs, {'log_x_theta': log_x_theta}
  
  # ------------------------------------------------
  # 6.2. Sampling with guidance 
  def _ratio_guidance_denoise(
      self,
      xt: torch.tensor,
      time_conditioning: torch.tensor,
      timestep: torch.tensor,
      move_chance_t: torch.tensor,
      move_chance_s: torch.tensor,
      cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
    ) -> typing.Tuple[torch.tensor, torch.tensor, typing.Dict[str, torch.tensor]]:
    seq_len = xt.shape[1]
    gamma = self.guidance_scale
    if cache is not None:
      log_x_theta = cache['log_x_theta']
      ratio_log = cache['ratio_log']
    else:
      # Diffusion model
      log_x_theta = self.forward(xt, time_conditioning)

      # Ratio model
      number_sequences_in_a_chunk = self.config.sampling.batch_size_ratio
      if self.config.sampling.compute_top_k_ratio > 0:
        # Use top-k sampling to reduce the number of sequences in a chunk
        ratio_log = self.get_ratio_log_topk_stream(xt=xt, 
              sigma=time_conditioning, 
              log_x_theta=log_x_theta,  
              k=self.config.sampling.compute_top_k_ratio,
              ratio_bs=number_sequences_in_a_chunk)
        log_x_theta = self.keep_top_k_tokens_rest_minus_infinity(
              log_x_theta=log_x_theta, xt=xt,k=self.config.sampling.compute_top_k_ratio)
      else:
        #xt_jumps = self._expand_with_single_token_replacements(xt=xt) # (B, L) -> (B · L · V, L) tensor.
        #batch_size = self.config.sampling.batch_size_ratio
        #ratio_log = self.ratio_model.get_log_probs(xt_jumps, batch_size ,time_conditioning.repeat(seq_len * self.vocab_size)) 
        ratio_log = self.get_ratio_log_stream(xt, time_conditioning, number_sequences_in_a_chunk)

    # Compute unguided posterior
    if self.diffusion == 'absorbing_state':
      diffusion_log_probs = log_x_theta + torch.log(1. - (move_chance_s / move_chance_t))
      diffusion_log_probs[..., self.mask_index] = torch.log(move_chance_s / move_chance_t)[:, :, 0]
      diffusion_log_probs.detach()
    elif self.diffusion == 'uniform':
      diffusion_log_probs = self._compute_posterior(x=log_x_theta.exp(),xt=xt,alpha_s=1 - move_chance_s,alpha_t=1 - move_chance_t).log()
    else:
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented.")

    if not self.config.gamma.type == 'constant':
       gamma = self.gamma_sched(1-timestep)     
    # Apply guidance
    with torch.no_grad():
      ratio_log[..., self.mask_index] = 0.0   
      if self.diffusion == 'absorbing_state':
        guided_log_probs = (gamma * ratio_log) + diffusion_log_probs
        copy_flag = (xt != self.mask_index)
        guided_log_probs[copy_flag] = self.neg_infinity
        guided_log_probs[copy_flag, xt[copy_flag]] = 0.0
      elif self.diffusion == 'uniform':
        guided_log_probs = (gamma * ratio_log) + diffusion_log_probs
      else:
        raise NotImplementedError(
          f"Diffusion type {self.diffusion} not implemented.")
    
    K = self.config.sampling.top_k_sampling

    # this is for logging purposes
    if self.diffusion == 'absorbing_state':
      guided_log_probs_buffer_for_table = self._preserve_mask_mass(guided_log_probs, xt, mask_logits=diffusion_log_probs[..., self.mask_index])
      self._log_table_wandb_overlap(log_x_theta, ratio_log,guided_log_probs_buffer_for_table ,xt, timestep)

    if K > 0 and K < self.vocab_size:
        guided_log_probs = self.keep_top_k_tokens_rest_minus_infinity(log_x_theta=guided_log_probs,xt=xt,k=K,always_keep=self.mask_index)    # <- mask survives
    
    # --- keep mask probability fixed ------------------------------------------
    guided_log_probs = self._preserve_mask_mass(guided_log_probs, xt, mask_logits=diffusion_log_probs[..., self.mask_index])
    guided_probs = guided_log_probs.softmax(dim=-1)

    # Sample from guided posterior
    xs = _sample_categorical(guided_probs)
    if self.diffusion == 'absorbing_state':
      xs = torch.where(copy_flag.to(bool), xt, xs)
    return xs, guided_probs, {'log_x_theta': log_x_theta,
                              'ratio_log': ratio_log}

  # ==========================================================
  # 7. Sampling with Planner
  # ==========================================================
  def _sample_with_planner(self):
      """Sample planner for the diffusion model."""
      num_steps = self.config.model.length
      pbar = tqdm(range(num_steps),desc='Sampling with planner',leave=False)
      batch_size_per_gpu = self.config.loader.eval_batch_size 
      xt = self._sample_prior(batch_size_per_gpu,self.config.model.length).to(self.device)#
      overall_time = 0 
      for _ in pbar:
        # Sample next position from the planner
        if self.planner is not None and not self.config.sampling.plan_random:
            next_pos_pair = self.planner.get_next_step(xt)
            if next_pos_pair is None:
                # nothing masked left
                break
            # Your step functions expect a single index or [b, j]; both are handled
            next_pos = next_pos_pair # [b , 1]
        else:  # random planning
            #next_pos2 = self.select_next_pos_by_confidence(xt)
            all_mask = (xt == self.mask_index).nonzero(as_tuple=False)  # (M, 2) => (b, j) pairs
            if all_mask.numel() == 0:
                break
            idx = torch.randint(0, all_mask.size(0), (1,), device=xt.device).item()
            next_pos = all_mask[idx]  # tensor([b, j])
            #next_pos = next_pos2[-1]
        if self.config.sampling.guided_sampling:
          xs, timer = self.ratio_guided_step_at_position(xt, next_pos)
        else:
          xs = self.denoiser_step_at_position(xt, next_pos)
        xt = xs
        #overall_time += timer
      average_time_on_ratio = overall_time / num_steps 
      #print(f"Overall time taken: {average_time_on_ratio:.4f}s")
      #print(f"Even when the batch ({self.config.loader.eval_batch_size}) * top_k_ratio ({self.config.sampling.compute_top_k_ratio}) < batch_size_ratio ({self.config.sampling.batch_size_ratio}), the average time is {average_time_on_ratio:.4f}s")
      return xt

  # ------------------------------------------------
  # 7.1 Denoiser step at a single position
  @torch.no_grad()
  def denoiser_step_at_position(
      self,
      xt: torch.Tensor,
      next_pos: typing.Union[int, torch.Tensor, typing.Sequence[int]],
  ) -> torch.Tensor:
    """
    Reveal positions using the denoiser only.

    Accepts:
      - int or 0-D tensor: column j (update all rows where xt[:, j] is masked)
      - (M, 1) tensor: per-row positions for rows b = 0..M-1
      - (M, 2) tensor: explicit (b, j) pairs
      - (2,) tensor/seq: single (b, j) pair
    """
    device = xt.device
    B, L = xt.shape
    V = self.vocab_size
    np = next_pos.to(device)
    M = np.size(0)
    rows = torch.arange(M, device=device, dtype=torch.long)
    cols = np.view(-1).long()

    # Bounds check and drop invalid entries
    valid = (rows >= 0) & (rows < B) & (cols >= 0) & (cols < L)
    if not valid.any():
        return xt
    rows = rows[valid]
    cols = cols[valid]

    # Only update where masked
    masked = (xt[rows, cols] == self.mask_index)
    if not masked.any():
        return xt
    rows = rows[masked]
    cols = cols[masked]
    M = rows.numel()

    # ---- model forward once ----
    zeros = torch.zeros(B, device=device, dtype=self.dtype)
    log_x_theta = self.forward(xt, zeros)  # (B, L, V)

    # ---- gather logits at requested positions ----
    vecs = log_x_theta[rows, cols, :].clone()  # (M, V)
    vecs[:, self.mask_index] = -float("inf")

    # Optional local top-k
    K = int(getattr(self.config.sampling, "top_k_sampling", 0) or 0)
    if 0 < K < V:
        topk_vals, topk_idx = torch.topk(vecs, K, dim=-1)
        keep = torch.zeros_like(vecs, dtype=torch.bool)
        keep.scatter_(1, topk_idx, True)
        vecs = torch.where(keep, vecs, vecs.new_full(vecs.shape, -float("inf")))

    # Sample one token per row
    probs = torch.softmax(vecs, dim=-1).to(torch.float32)  # stable weights
    toks = torch.multinomial(probs, num_samples=1).squeeze(1)  # (M,)

    # ---- write back ----
    xt_new = xt.clone()
    xt_new[rows, cols] = toks.to(xt_new.dtype)
    return xt_new
  
  # ------------------------------------------------
  # 7.2 Guided sampling at a single position
  @torch.no_grad()
  def ratio_guided_step_at_position(self, xt, next_pos, sigma=None):
    """
    xt:        (B, L) token ids
    next_pos:  int | Tensor[1] | Tensor[B] | Tensor[B,1]   # per-row target positions
    sigma:     None | Tensor[B] | Tensor[B,1]              # planner noise (for ratio_model)
    """
    device = xt.device
    B, L = xt.shape
    V = self.vocab_size

    # --- normalize next_pos -> (B,) long ---
    if isinstance(next_pos, int):
        pos_vec = torch.full((B,), next_pos, device=device, dtype=torch.long)
    else:
        pos_vec = next_pos.to(device).long().view(-1)
        if pos_vec.numel() == 1:
            pos_vec = pos_vec.expand(B)
        elif pos_vec.numel() != B:
            raise ValueError("next_pos must be scalar or have shape [B] / [B,1].")

    # --- normalize sigma -> (B,) ---
    if sigma is None:
        sigma_vec = torch.zeros(B, device=device, dtype=self.dtype)
    else:
        sigma_vec = sigma.to(device).view(B, -1).squeeze(-1).to(self.dtype)

    # Base logits from the model (keep your original behavior: sigma=0 here)
    zeros = torch.zeros(B, device=device, dtype=self.dtype)
    log_x_theta = self.forward(xt, zeros)  # (B, L, V)

    # Planner ratio config
    k_ratio  = int(getattr(self.config.sampling, "compute_top_k_ratio", 0) or 0)
    ratio_bs = int(getattr(self.config.sampling, "batch_size_ratio", 1024) or 1024)
    gamma    = float(self.guidance_scale)

    # Ratio scores for each row at its own position (single batched call)
    ratio_log, time_ = self.get_ratio_log_topk_stream_at_pos(
        xt=xt, sigma=sigma_vec, log_x_theta=log_x_theta,
        pos=pos_vec, k=min(k_ratio, V), ratio_bs=ratio_bs
    )  # (B, V)
    ratio_log[:, self.mask_index] = 0.0

    # Combine logits with guidance, all rows at once
    row_idx = torch.arange(B, device=device)
    v = log_x_theta[row_idx, pos_vec, :].to(torch.float32)  # (B, V)
    r = ratio_log.to(torch.float32)                          # (B, V)
    guided = v + gamma * r                                   # (B, V)

    # Optional top-k on combined logits (per row)
    K = int(getattr(self.config.sampling, "top_k_sampling", 0) or 0)
    if 0 < K < V:
        g3d = guided.unsqueeze(1)  # (B,1,V)
        xt_seg = xt[row_idx, pos_vec].unsqueeze(1)  # (B,1)
        g3d = self.keep_top_k_tokens_rest_minus_infinity(
            log_x_theta=g3d, xt=xt_seg, k=K, always_keep=self.mask_index
        )
        guided = g3d.squeeze(1)

    # Preserve mask probability (per row)
    mask_logits = v[:, self.mask_index].unsqueeze(1)         # (B,1)
    g3d = guided.unsqueeze(1)                                 # (B,1,V)
    xt_seg = xt[row_idx, pos_vec].unsqueeze(1)                # (B,1)
    g3d = self._preserve_mask_mass(g3d, xt_seg, mask_logits=mask_logits)
    guided = g3d.squeeze(1)                                   # (B,V)

    # Sample only rows whose target position is currently masked
    row_mask = (xt[row_idx, pos_vec] == self.mask_index)      # (B,)
    if not row_mask.any():
        return xt

    probs = guided[row_mask].softmax(dim=-1).to(torch.float32)        # (Bm,V)
    toks = torch.multinomial(probs, num_samples=1).squeeze(1)         # (Bm,)

    xt_new = xt.clone()
    rows = torch.nonzero(row_mask, as_tuple=False).flatten()          # (Bm,)
    xt_new[rows, pos_vec[rows]] = toks
    return xt_new, time_
    
  # ========================================================================
  # 8. Helper function: 
  # ========================================================================
  # -------------------------------------------------
  # 8.1 Sampling Helper function
  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _compute_move_chances(
    self,
    t_scalar: torch.Tensor,   # the current scalar timestep (shape: ())
    dt:       float,
    batch_sz: int, ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """
    Return the tensors needed for a single sampling step.
    """
    # Map to training grid when using discrete-time diffusion.
    if self.T > 0:
        t_scalar = ((t_scalar * self.T).to(torch.int) / self.T) + 1 / self.T

    t = t_scalar.expand(batch_sz, 1).to(self.device)

    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    sigma_t = sigma_t.squeeze(-1)
    sigma_s = sigma_s.squeeze(-1)

    move_t = (1 - torch.exp(-sigma_t)).unsqueeze(-1).unsqueeze(-1)
    move_s = (1 - torch.exp(-sigma_s)).unsqueeze(-1).unsqueeze(-1)

    return t, sigma_t, sigma_s, move_t, move_s

  # -------------------------------------------------
  # 8.2 Get Ratio model helper function: 
  def get_ratio_log_stream(self, xt, sigma, chunk_v: int = 1024):
    """
    Compute log-prob correction terms from the ratio model, but *only* for
    sequence positions that are currently MASK tokens in at least one item
    of the batch. Unmasked positions are left at 0.0 log-ratio (i.e., neutral).
    """
    B, L = xt.shape
    V = self.vocab_size

    # Handle DataParallel-wrapped ratio_model
    ratio_model = self.ratio_model.module if isinstance(self.ratio_model, torch.nn.DataParallel) else self.ratio_model
    device = next(ratio_model.parameters()).device
    param_dtype = next(ratio_model.parameters()).dtype

    # Allocate result (neutral log-factor = 0)
    ratio_log = torch.full((B, L, V), self.neg_infinity, device=device, dtype=param_dtype)  # (B, L, V)

    # Move inputs to CPU for expansion
    base_cpu  = xt.to("cpu", non_blocking=True)
    sigma_cpu = sigma.to("cpu", non_blocking=True)

    # Identify which columns contain at least one MASK
    mask_cols = (base_cpu == self.mask_index).any(dim=0).nonzero(as_tuple=False).flatten().tolist()
    if len(mask_cols) == 0:
        # Nothing to guide; return neutral ratios.
        return torch.zeros_like(ratio_log, dtype=param_dtype, device=device)

    # Pre-build a [0..n-1] index once per chunk_v loop (built in loop below)
    for pos in tqdm(mask_cols, desc="ratio-mask-cols"):
        for v0 in range(0, V, chunk_v):
            v1 = min(v0 + chunk_v, V)
            n  = v1 - v0

            # (B, n, L) -> (B*n, L)
            cand = torch.arange(v0, v1, dtype=base_cpu.dtype, device=base_cpu.device)
            tmp = base_cpu.unsqueeze(1).expand(B, n, L).clone().reshape(-1, L)
            tmp[:, pos] = cand.repeat(B)

            tmp_dev = tmp.to(device, non_blocking=True)
            sig_dev = sigma_cpu.repeat_interleave(n).to(device, dtype=param_dtype, non_blocking=True)

            with  torch.inference_mode():
                logits = ratio_model(tmp_dev,sig_dev)

            # Expect (B*n, L, V) or (B*n, V_pos) fallback
            logits_pos = logits[:, pos, :] if logits.dim() == 3 else logits
            #if pos == mask_cols[0] and v0 == 0:
            #  print("DEBUG ratio_model out:", logits.shape)
            #  print("DEBUG logits_pos:", logits_pos.shape)
            cand_scores = logits_pos.reshape(B, n)   # works if logits_pos numel == B*n
            ratio_log[:, pos, v0:v1] = cand_scores
            del tmp_dev, sig_dev, logits_pos, logits, cand_scores

        # Normalize *this* column over vocab.
        ratio_log[:, pos, :] = torch.log_softmax(ratio_log[:, pos, :], dim=-1)
        torch.cuda.empty_cache()
    return ratio_log
  

  def get_ratio_log_topk_stream(
    self, xt, sigma, log_x_theta, k: int = 1024, ratio_bs: int = 8192, x0: torch.Tensor = None, normalize: bool = True
    ) -> torch.Tensor:
    B, L, V = log_x_theta.shape
    mask = (xt == self.mask_index)
    rows = mask.nonzero(as_tuple=False)  # (Nmask,2)
    if rows.numel() == 0:
        return torch.zeros_like(log_x_theta)

    topk = torch.topk(log_x_theta[mask], k, dim=-1).indices  # (Nmask,k)
    labels = x0[mask] if x0 is not None else None

    # Build candidate triples (b,pos,tok), ensuring label is included
    cand_tuples = []
    for i, ((b, pos), toks) in enumerate(zip(rows.tolist(), topk.tolist())):
        if labels is not None:
            lbl = int(labels[i])
            if lbl not in toks:
                toks.append(lbl)
        for tok in toks:
            cand_tuples.append((int(b), int(pos), int(tok)))

    r_dev  = next(self.ratio_model.parameters()).device
    r_dtype = next(self.ratio_model.parameters()).dtype
    ratio_log = torch.full((B, L, V), self.neg_infinity, device=r_dev, dtype=r_dtype)

    base_cpu  = xt.to("cpu", non_blocking=True)
    sigma_cpu = sigma.to("cpu").flatten()

    loop = trange if self.show_loading else range
    for i in loop(0, len(cand_tuples), ratio_bs):
        batch_slice = cand_tuples[i : i + ratio_bs]
        n = len(batch_slice)
        seq_batch = torch.empty((n, L), dtype=base_cpu.dtype)
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
            elif logits.size(1) == 1:
                score = logits[j, 0]
            else:
                score = logits[j, tok]
            ratio_log[b, pos, tok] = score

        del seq_dev, logits

    if normalize:
        ratio_rows = ratio_log[mask]
        ratio_log[mask] = torch.log_softmax(ratio_rows, dim=-1)
    return ratio_log.to(log_x_theta.device)

  @torch.no_grad()
  def get_ratio_log_topk_stream_at_pos(
      self,
      xt: torch.Tensor,                 # (B, L)
      sigma: torch.Tensor,              # (B,) or (B,1)
      log_x_theta: torch.Tensor,        # (B, L, V)
      pos,                               # int | Tensor[B] | Tensor[B,1]
      k: int = 1024,
      ratio_bs: int = 8192,
  ) -> torch.Tensor:
      """
      Returns (B, V) of log-softmaxed planner scores, computed simultaneously for
      each row at its own target position `pos[b]`. Unmasked rows receive zeros.
      """
      device = log_x_theta.device
      B, L = xt.shape
      V = log_x_theta.size(-1)

      # Normalize pos -> (B,) long on CPU/GPU device
      if isinstance(pos, int):
          pos_vec = torch.full((B,), pos, device=device, dtype=torch.long)
      else:
          pos_vec = pos.to(device).long().view(-1)
          if pos_vec.numel() == 1:
              pos_vec = pos_vec.expand(B)
          elif pos_vec.numel() != B:
              raise ValueError("pos must be scalar or have shape [B] / [B,1].")

      # Guard for k<=0
      if int(k) <= 0:
          k = V 

      row_idx = torch.arange(B, device=device)
      at_pos_tokens = xt[row_idx, pos_vec]                    # (B,)
      mask_rows = (at_pos_tokens == self.mask_index)          # (B,)
      if not mask_rows.any():
          return torch.zeros(B, V, device=device, dtype=log_x_theta.dtype)

      K = min(int(k), V)

      # Get top-K candidate tokens per row at its own position
      logits_at_pos = log_x_theta[row_idx, pos_vec, :]        # (B, V)
      topk_idx_all = torch.topk(logits_at_pos, K, dim=-1).indices  # (B, K)

      # Slice only masked rows
      rows_sel = torch.nonzero(mask_rows, as_tuple=False).flatten()      # (Bm,)
      Bm = rows_sel.numel()

      r_dev  = next(self.ratio_model.parameters()).device
      r_dtype = next(self.ratio_model.parameters()).dtype

      base = xt[rows_sel].to(r_dev, non_blocking=True)                    # (Bm, L)
      pos_m = pos_vec[rows_sel].to(r_dev, non_blocking=True)              # (Bm,)
      topk_m = topk_idx_all[rows_sel].to(r_dev, non_blocking=True).long() # (Bm, K)

      # Build candidate sequences: repeat each masked row K times and set its pos
      seq_all = base.unsqueeze(1).expand(Bm, K, L).clone().reshape(Bm*K, L)       # (T,L)
      tok_all = topk_m.reshape(Bm*K)                                              # (T,)
      pos_rep = pos_m.repeat_interleave(K)                                        # (T,)

      seq_all[torch.arange(Bm*K, device=r_dev), pos_rep] = tok_all

      # Map back to original batch rows and prepare sigma
      b_all = rows_sel.to(r_dev).repeat_interleave(K)                              # (T,)
      sig_rows = sigma.view(B, -1).squeeze(-1).to(r_dev, non_blocking=True)       # (B,)
      sig_all = sig_rows[b_all]                                                    # (T,)

      # Score candidates in chunks
      T = seq_all.size(0)
      scores_all = torch.empty(T, device=r_dev, dtype=r_dtype)
      step = int(max(1, ratio_bs))
      for start in range(0, T, step):
          end = min(start + step, T)
          with torch.inference_mode():
              with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = self.ratio_model(seq_all[start:end], sig_all[start:end])    # (N,L,V)|(N,1)|(N,V)

          if logits.dim() == 3:   # (N, L, V) -> take per-sample position, then chosen token
              sel = logits[torch.arange(end-start, device=r_dev), pos_rep[start:end], :]   # (N,V)
              scores = sel.gather(1, tok_all[start:end].view(-1,1)).squeeze(1)
          elif logits.size(1) == 1:  # (N,1)
              scores = logits.view(-1)
          else:                      # (N,V)
              scores = logits.gather(1, tok_all[start:end].view(-1,1)).squeeze(1)

          scores_all[start:end] = scores.to(r_dtype)
      # Fill ratio_log with -inf except the scored candidates, then log-softmax per masked row
      ratio_log = torch.full((B, V), self.neg_infinity, device=r_dev, dtype=log_x_theta.dtype)
      ratio_log[b_all, tok_all] = scores_all.to(ratio_log.dtype)

      ratio_log = ratio_log.to(device)
      ratio_log[mask_rows] = torch.log_softmax(ratio_log[mask_rows].float(), dim=-1).to(ratio_log.dtype)
      ratio_log[~mask_rows] = 0
      return ratio_log, 0


  @torch.no_grad()
  def get_ratio_log_at_true_token(
      self,
      xt: torch.Tensor,               # (B, L)
      x0: torch.Tensor,               # (B, L)
      sigma: torch.Tensor,            # (B,) or (B,1)
      ratio_bs: int = 8192,
  ) -> torch.Tensor:
      """
      Return a (B, L, V) tensor filled with zeros except at masked positions (xt == mask_index),
      where we place the ratio logit at the ground-truth token index.
      For each masked (b,l): build seq = xt[b] with seq[l]=x0[b,l], run ratio model, take
      logits at position l and index x0[b,l], scatter into output.
      """
      base_device = xt.device
      out_dtype = getattr(self, "dtype", torch.float32)
      B, L = xt.shape

      # Infer vocab size V
      V = getattr(self, "vocab_size", None)
      if V is None:
          if hasattr(self.ratio_model, "vocab_embed") and hasattr(self.ratio_model.vocab_embed, "num_embeddings"):
              V = int(self.ratio_model.vocab_embed.num_embeddings)
          elif hasattr(self.ratio_model, "vocab_size"):
              V = int(self.ratio_model.vocab_size)
          else:
              raise ValueError("Cannot infer vocab size. Set self.vocab_size or expose ratio_model.vocab_embed.num_embeddings.")

      # Masked positions
      mask = (xt == self.mask_index)  # (B, L)
      if not mask.any():
          return torch.zeros((B, L, V), device=base_device, dtype=out_dtype)

      b_idx, l_idx = mask.nonzero(as_tuple=True)   # (Nmask,), (Nmask,)
      Nmask = b_idx.numel()

      # Output tensor: zeros everywhere by default
      ratio_full = torch.zeros((B, L, V), device=base_device, dtype=out_dtype)

      # Flatten sigma to (B,)
      sigma_flat = sigma.view(-1)

      # Ratio model device
      ratio_dev = next(self.ratio_model.parameters()).device

      start = 0
      while start < Nmask:
          end = min(start + ratio_bs, Nmask)
          n = end - start

          bs = b_idx[start:end]                                   # (n,)
          ls = l_idx[start:end]                                   # (n,)
          row_local = torch.arange(n, device=xt.device)           # (n,)

          # Build sequences on xt's device, inject true token at the masked position
          seqs = xt[bs].clone()                                   # (n, L)
          true_tok = x0[bs, ls]                                   # (n,)
          seqs[row_local, ls] = true_tok

          # Move to ratio model device and set dtypes
          seqs = seqs.to(ratio_dev)
          if seqs.dtype != torch.long:
              seqs = seqs.long()
          sigma_chunk = sigma_flat[bs].to(ratio_dev).float()
          attention_mask = torch.ones_like(seqs, dtype=torch.long, device=ratio_dev)

          # Forward: expect (n, 1) logits
          scores = self.ratio_model(seqs, sigma_chunk, attention_mask=attention_mask)  # (n, 1)

          # Scatter into (B, L, V) output at (b, l, true_tok)
          ratio_full[bs, ls, true_tok.to(base_device)] = scores.squeeze(-1).to(base_device, dtype=out_dtype)

          start = end

      return ratio_full
  
  # ------------------------------------------------
  # 8.3 Top k helper function
  def keep_top_k_tokens_rest_minus_infinity(self, log_x_theta, xt, k: int, *, always_keep: int = None):
    if k <= 0:
        return log_x_theta

    mask = (xt == self.mask_index)
    if not mask.any():
        return log_x_theta

    keep = torch.zeros_like(log_x_theta, dtype=torch.bool)
    topk_idx = torch.topk(log_x_theta[mask], k, dim=-1).indices
    keep_rows = keep[mask]
    keep_rows.scatter_(1, topk_idx, True)
    if always_keep is not None:
        keep_rows[:, always_keep] = True          # guarantee mask token
    keep[mask] = keep_rows

    minus_inf = self.neg_infinity
    return torch.where(keep, log_x_theta, minus_inf)

  # ------------------------------------------------
  # 8.4 Preserve mask mass helper function
  def _preserve_mask_mass(
    self,
    logits: torch.Tensor,     # (B, L, V)  log‑probs, −∞ where tokens are pruned
    xt: torch.Tensor,         # (B, L)     current sequence (mask = self.mask_index)
    mask_logits: torch.Tensor,  # (B, L)     log p_M (mask log‑prob)
    ) -> torch.Tensor:
    """
    Renormalise logits so that  q(z_s=m|z_t) == p_M  is kept intact.
    Works position‑wise and supports top‑k pruning (−∞ logits).
    """
    m = self.mask_index
    mask_pos = (xt == m)                          # positions we are sampling
    p_M = mask_logits.exp().clamp_(1e-12, 1. - 1e-6)

    # build a boolean mask for *finite* non‑mask logits
    nonmask_mask = torch.isfinite(logits)
    nonmask_mask[..., m] = False

    # log ∑_{v≠m} e^{ℓ_v}
    logsum_nonmask = torch.logsumexp(
        logits.masked_fill(~nonmask_mask, -float('inf')), dim=-1
    )

    shift = (torch.log1p(-p_M) - logsum_nonmask)      # (B, L)
    shift = shift.masked_fill(~mask_pos, 0.0)         # no shift where xt ≠ MASK

    # add the shift only to valid non‑mask logits
    logits = logits + shift.unsqueeze(-1) * nonmask_mask.float()
    logits[..., m] = mask_logits                      # keep mask log‑prob

    return logits

  # ------------------------------------------------
  # 8.5: Planner helper: Selection of next position by confidence 
  @torch.no_grad()
  def select_next_pos_by_confidence(
    self,
    xt: torch.Tensor,
    *,
    guided: bool = False,
    metric: str = "maxprob",) -> torch.Tensor:
    """
    Returns a single (b, j) index of the masked position with highest confidence.
    Confidence is computed over the vocab at each masked position.
    """
    B, L = xt.shape
    if not (xt == self.mask_index).any():
        return None  # nothing left to reveal

    # Use sigma≈0 for "reveal now" decisions.
    zeros = torch.zeros(B, device=xt.device, dtype=self.dtype)

    # Base logits from the denoiser.
    log_x_theta = self.forward(xt, zeros)  # (B, L, V)

    # Optional ratio guidance.
    if guided:
        k_ratio = int(getattr(self.config.sampling, "compute_top_k_ratio", 0) or 0)
        ratio_bs = int(getattr(self.config.sampling, "batch_size_ratio", 1024) or 1024)
        if k_ratio > 0:
            ratio_log = self.get_ratio_log_topk_stream(
                xt=xt, sigma=zeros, log_x_theta=log_x_theta,
                k=min(k_ratio, self.vocab_size), ratio_bs=ratio_bs)
        else:
            ratio_log = self.get_ratio_log_stream(xt=xt, sigma=zeros, chunk_v=ratio_bs)
        logits = log_x_theta + float(self.guidance_scale) * ratio_log
    else:
        logits = log_x_theta

    # Never consider the MASK token for confidence.
    logits = logits.clone()
    logits[..., self.mask_index] = -float("inf")

    # Confidence per (b, j).
    if metric == "margin":
        top2 = torch.topk(logits, 2, dim=-1).values              # (B, L, 2)
        conf = top2[..., 0] - top2[..., 1]                       # (B, L)
    elif metric == "maxprob":
        conf = torch.log_softmax(logits, dim=-1).max(dim=-1).values
    elif metric == "neg_entropy":
        p = torch.softmax(logits, dim=-1)
        conf = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1)     # higher = more confident
    else:
        raise ValueError("metric must be one of {'margin','maxprob','neg_entropy'}")

    # Only allow masked positions to compete.
    conf = conf.masked_fill(xt != self.mask_index, -float("inf"))
    # if conf is all the same value, return random masked position
    if torch.all(conf == conf.view(-1)[0]):
        mask_cols = (xt == self.mask_index).nonzero(as_tuple=False)
        rand_idx = torch.randint(0, len(mask_cols), (1,))
        return mask_cols[rand_idx].squeeze(0).to(device=xt.device)
    # Global best over batch × length.
    flat = conf.view(-1).argmax()
    b = int(flat // L); j = int(flat % L)
    return torch.tensor([b, j], device=xt.device)
         



  # =========================================================
  # 9. PPL / evaluation
  # =========================================================
  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,}
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(self, text_samples, retokenize: bool = True, max_length=None) -> None:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    model_id = self.gen_ppl_eval_model_name_or_path
    cache_dir = self.hf_cache_root  # from config

    # Tokenizer first (same local_files_only logic)
    self.eval_model_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=False,
        use_fast=True,
     )
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token = self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id = self.eval_model_tokenizer.eos_token_id
      self.eval_model_tokenizer.padding_side = "left"  # better for decoder-only LMs
    try: 
      eval_model = transformers.AutoModelForCausalLM.from_pretrained(
          model_id,
          cache_dir=cache_dir,
          use_safetensors=True,
          local_files_only=False,
      ).eval()
    except:
      try: 
        eval_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=False,
        ).eval()
      except Exception as e:
             self.logger.error(f"Failed to load evaluation model: {e}")
             raise

    if eval_model is None:
        raise RuntimeError("Evaluation model could not be loaded.")

    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(self.config.eval.perplexity_batch_size,samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      # Splite the generated sample size into smaller sub batches  
      _samples = torch.split(samples[i * batch_size: (i + 1) * batch_size],eval_context_size,dim=-1)
      _attn_mask = torch.split(attn_mask[i * batch_size: (i + 1) * batch_size],eval_context_size,dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(_samples, _attn_mask):
        logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],sample_chunk[..., 1:],reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer.eos_token_id).cumsum(-1) == 1
        token_mask = (sample_chunk!= self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def detokenize_batch(x0: torch.Tensor, tokenizer, *, skip_special_tokens: bool = True):
      """
      How to use:
      texts = self.detokenize_batch(x0, self.tokenizer)

      for i, txt in enumerate(texts[:3]):   # show first few examples
        print(f"[sample {i}] {txt}")
      """

      input_ids = x0.detach().cpu().tolist()
      return tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)

  # ===============================================================================
  # 10. Utilities function for logging
  # ===============================================================================
  def _expand_with_single_token_replacements(
    self, xt: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each row is `xt` with one token replaced
    by every vocabulary symbol, **but keep it on CPU**.
    """
    bsz, seq_len = xt.shape
    V = self.vocab_size

    # 1) build on CPU to avoid a huge GPU allocation
    xt_cpu = xt.to("cpu")                                     # (B, L)
    xt_expand = (xt_cpu.unsqueeze(1).repeat(1, seq_len * V, 1).view(-1, seq_len)                              # (B·L·V, L)
                 )

    # 2) overwrite the chosen positions
    jump_idx   = torch.arange(seq_len * V).repeat(bsz, 1).flatten()
    jump_dims  = jump_idx // V
    jump_state = jump_idx %  V
    xt_expand[torch.arange(jump_idx.size(0)), jump_dims] = jump_state

    return xt_expand   

  def _topk_overlap_pct_batch(self, s1, s2, k_list, eps=1e-8):
    out = {}
    V = s1.size(-1)
    # drop constant rows in s2
    keep = (s2.max(-1).values - s2.min(-1).values) > eps
    if not keep.any():
        for K in k_list:
            out[K] = float('nan')
        return out
    s1 = s1[keep]; s2 = s2[keep]
    for K in k_list:
        k = min(K, V)
        idx1 = torch.topk(s1, k, dim=-1).indices
        idx2 = torch.topk(s2, k, dim=-1).indices
        isin = torch.isin(idx1, idx2) if hasattr(torch, "isin") else (idx1.unsqueeze(-1) == idx2.unsqueeze(-2)).any(-1)
        pct = (isin.sum(-1).float() / k * 100.0).mean().item()
        out[K] = pct
    return out
        
  
  def _log_topk_table_wandb(self):
    """Push top-k overlap vs t tables to W&B (if available & rank0)."""
    if not getattr(self.trainer, "is_global_zero", False):
        return
    exp = getattr(self.logger, "experiment", None)
    if exp is None:
        return
    try:
        import wandb
    except ImportError:
        return
    if not self._t_records:
        return  # nothing to plot

    def _mk_rows(buf_dict):
        rows = []
        for i, t_val in enumerate(self._t_records):
            row = [t_val]
            for k in self.TOPK_LIST:
                lst = buf_dict[k]
                row.append(lst[i] if i < len(lst) else float("nan"))
            rows.append(row)
        return rows

    # diffusion vs ratio
    cols_dr = ["t"] + [f"topk@{k}" for k in self.TOPK_LIST]
    tbl_dr = wandb.Table(columns=cols_dr, data=_mk_rows(self._topk_ratio_vs_source))
    exp.log({"val/topk_vs_t": tbl_dr})

    # diffusion vs guided
    cols_dg = ["t"] + [f"topk_dvsg@{k}" for k in self.TOPK_LIST]
    tbl_dg = wandb.Table(columns=cols_dg, data=_mk_rows(self._topk_diff_vs_guidance))
    exp.log({"val/topk_dvsg_vs_t": tbl_dg})

    # optional line plots
    for k in self.TOPK_LIST:
        exp.log({
            f"val/topk_vs_t_{k}": wandb.plot.line(tbl_dr, "t", f"topk@{k}",
                                                  title=f"Top-{k} overlap (diffusion vs ratio)"),
            f"val/topk_dvsg_vs_t_{k}": wandb.plot.line(tbl_dg, "t", f"topk_dvsg@{k}",
                                                       title=f"Top-{k} overlap (diffusion vs guided)"),})


  def _log_table_wandb_overlap(self, log_x_theta, ratio_log,guided_log_probs ,xt, t):
        mask_positions = (xt == self.mask_index)
        s_diff  = log_x_theta[mask_positions]         # diffusion rows
        s_ratio = ratio_log[mask_positions]           # ratio rows
        s_guided = guided_log_probs[mask_positions]   # guided rows
        overlap_dr = self._topk_overlap_pct_batch(s_diff, s_ratio, self.TOPK_LIST, eps=1e-6)
        overlap_dg = self._topk_overlap_pct_batch(s_diff, s_guided, self.TOPK_LIST, eps=1e-6)

        # record t
        t_val = float(t.mean().item())
        self._t_records.append(t_val)

        # buffer + log (per-step)
        for K, pct in overlap_dr.items():
            self._topk_ratio_vs_source[K].append(pct)
            self.log(f"val/topk_overlap_ratio_source@{K}", pct,
                    on_step=False, on_epoch=True, sync_dist=True)

        for K, pct in overlap_dg.items():
            self._topk_diff_vs_guidance[K].append(pct)
            self.log(f"val/topk_overlap_dvsg@{K}", pct,
                    on_step=False, on_epoch=True, sync_dist=True)
            

  @torch.no_grad()
  def compute_topk_overlap_means(self) -> None:
    """
    Print & (optionally) log the mean Top‑K overlaps collected
    during validation.  Simpler than building big tables.
    """
    # nothing collected?
    if not self._t_records:
        print("[compute_topk_overlap_means]  no overlap data")
        return

    # try to get a live wandb run (works whether using Lightning's WandbLogger
    # or calling wandb.init() yourself)
    try:
        import wandb
        exp = wandb.run if wandb.run and wandb.run.is_running() else None
    except ImportError:
        exp = None

    # iterate over each K in your configured list
    for k in self.TOPK_LIST:
        vals_dr = np.array(self._topk_ratio_vs_source[k], dtype=float)
        vals_dg = np.array(self._topk_diff_vs_guidance[k], dtype=float)

        mean_dr = np.nanmean(vals_dr) if vals_dr.size else float("nan")
        mean_dg = np.nanmean(vals_dg) if vals_dg.size else float("nan")

        # nicely formatted console output
        print(
            f"[Top {k:5d}]  diff vs ratio: {mean_dr:6.2f}%   "
            f"diff vs guided: {mean_dg:6.2f}%   "
            f"(N = {len(vals_dr)})")

        # optional W&B scalars (so you can plot them later)
        if exp is not None:
            exp.log({
                f"val/topk_mean/ratio_source@{k}": mean_dr,
                f"val/topk_mean/dvsg@{k}":        mean_dg,})
            

  def _truncate_and_renorm_logprobs(
    self,
    logits: torch.Tensor,     # (B,L,V) raw model scores
    xt: torch.Tensor,         # (B,L)   noised tokens
    x0: torch.Tensor,         # (B,L)   ground-truth tokens
    k: int = 64,
    only_masked: bool = True, # apply truncation only where xt==[MASK]
    include_mask: bool = True # force-include [MASK]
) -> torch.Tensor:
    """
    Returns (B,L,V) log-probs where truncated rows are renormalized within
    the candidate set (top-K ∪ {x0} ∪ {[MASK]}). Untruncated rows are the
    original log-softmax logits.
    """
    B, L, V = logits.shape
    # base log-probs (for rows we won't truncate)
    logp_full = torch.log_softmax(logits, dim=-1)

    if k <= 0:
        return logp_full

    apply_rows = (xt == self.mask_index) if only_masked else torch.ones_like(xt, dtype=torch.bool)
    if not apply_rows.any():
        return logp_full

    # Work only on affected rows to save memory
    idx = apply_rows.view(-1)
    logits_flat = logits.view(B * L, V)
    x0_flat     = x0.view(B * L)
    rows_logits = logits_flat[idx]          # (Nm, V)
    rows_x0     = x0_flat[idx]              # (Nm,)

    Nm = rows_logits.size(0)
    K  = min(k, V)

    # Top-K indices by logits
    topk_idx = torch.topk(rows_logits, K, dim=-1).indices  # (Nm, K)

    # Build a boolean keep mask for these rows
    keep_rows = torch.zeros_like(rows_logits, dtype=torch.bool)  # (Nm, V)
    keep_rows.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_idx, dtype=torch.bool))

    # Force-include the true label
    keep_rows.scatter_(dim=-1, index=rows_x0[:, None], src=torch.ones((Nm, 1), dtype=torch.bool, device=rows_logits.device))

    # Force-include [MASK] if requested
    if include_mask:
        keep_rows[:, self.mask_index] = True

    # Build pruned logits: -inf outside kept set, original logits inside
    neg_inf = torch.finfo(rows_logits.dtype).min
    pruned_rows = torch.full_like(rows_logits, neg_inf)
    pruned_rows[keep_rows] = rows_logits[keep_rows]

    # Renormalize within the kept set → log-probs
    pruned_logp = pruned_rows - torch.logsumexp(pruned_rows, dim=-1, keepdim=True)

    # Stitch back into full tensor
    out = logp_full.view(B * L, V).clone()
    out[idx] = pruned_logp
    return out.view(B, L, V)



  def _build_keep_mask_from_base(self, base_logits, xt, x0, k=64, include_mask=False):
      B,L,V = base_logits.shape
      apply_rows = (xt == self.mask_index)
      if not apply_rows.any():
          return torch.zeros_like(base_logits, dtype=torch.bool)

      flat = apply_rows.view(-1)
      rows_logits = base_logits.view(B*L, V)[flat]
      rows_x0     = x0.view(B*L)[flat]
      K = min(k, V)

      topk_idx = torch.topk(rows_logits, K, dim=-1).indices
      keep_rows = torch.zeros_like(rows_logits, dtype=torch.bool)
      keep_rows.scatter_(1, topk_idx, True)
      keep_rows.scatter_(1, rows_x0[:,None], True)  # force-in label
      if include_mask:
          keep_rows[:, self.mask_index] = True

      keep = torch.zeros_like(base_logits, dtype=torch.bool).view(B*L, V)
      keep[flat] = keep_rows
      return keep.view(B,L,V)

  def _renorm_with_keep_mask(self, logits: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
      # logits: (B,L,V), keep_mask: (B,L,V) bool
      neg_inf = -float("inf")

      any_keep = keep_mask.any(dim=-1)        # (B,L)
      out = torch.full_like(logits, neg_inf)  # default

      # Rows that have at least one kept token
      if any_keep.any():
          pruned = logits.masked_fill(~keep_mask, neg_inf)
          logZ = torch.logsumexp(pruned, dim=-1, keepdim=True)      # (B,L,1)

          # Where logZ is finite: normalize; else keep −inf (not NaN)
          normed = pruned - logZ
          normed = torch.where(torch.isfinite(logZ), normed, torch.full_like(pruned, neg_inf))

          out[any_keep] = normed[any_keep]

      # Rows with no kept tokens → fall back to full softmax
      if (~any_keep).any():
          out[~any_keep] = torch.log_softmax(logits[~any_keep], dim=-1)

      return out








# ==========================================================
# Never used functions
# ==========================================================
  def _ddpm_update(self, x, t, dt):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t- move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x
    

  def restore_model_and_sample(self, num_steps=512,use_planner_sampling=False, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(self.backbone.parameters(),self.noise.parameters()))
      self.ema.copy_to(itertools.chain(self.backbone.parameters(),self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    if use_planner_sampling:
      samples = self._sample_with_planner()
    else:
      samples = self._sample(num_steps=num_steps, eps=eps)

    if self.ema:
      self.ema.restore(itertools.chain(self.backbone.parameters(),self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':     
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(model_output)
      unmasked_score = torch.scatter(unmasked_score,-1,x[..., None],torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
      model_output = (masked_score * masked_indices + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask


  @torch.no_grad
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype,device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self._sample_prior(n_samples,self.config.model.length).to(self.device)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ddpm_caching_update(x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x) or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((np.concatenate(intermediate_tokens, axis=1)[:, 1:] == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append( self.tokenizer.batch_decode(np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples, sequence_lengths)

  @torch.no_grad()
  def update_gen_ppl_metric_from_text(
    self,
    text_samples: typing.List[str],
    *,
    max_length: typing.Union[int, None] = None,
      ) -> float:
    """
    Compute NLLs + mask for a batch of text, update `self.gen_ppl_metric`,
    and return the corresponding perplexity.

    This is suitable for calling inside **training** or **validation** loops.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if max_length is None:
        max_length = self.config.model.length

    # 1. Load the (frozen) evaluator language model.
    scorer_name = self.gen_ppl_eval_model_name_or_path
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(scorer_name).eval()
    if "llama2" not in scorer_name:
        eval_model = eval_model.to(self.device)

    # 2. Re-tokenise with the scorer’s tokenizer.
    ids, attn_mask, ctx = self.eval_retokenize(text_samples, max_length=max_length)

    # 3. Walk through the sequence in context-length chunks.
    eos_id = self.eval_model_tokenizer.eos_token_id
    bsz = min(self.config.eval.perplexity_batch_size, ids.size(0))

    total_nll, total_tokens = 0.0, 0.0
    for i in range(0, ids.size(0), bsz):
        chunk_ids = ids[i : i + bsz]
        chunk_msk = attn_mask[i : i + bsz]

        for ids_split, msk_split in zip(
            torch.split(chunk_ids, ctx, dim=-1),
            torch.split(chunk_msk, ctx, dim=-1),
        ):
            logits = eval_model(
                ids_split.to(eval_model.device),
                attention_mask=msk_split.to(eval_model.device),
            ).logits.transpose(1, 2)  # (B, V, L)

            nlls = F.cross_entropy(
                logits[..., :-1], ids_split[..., 1:], reduction="none"
            )

            first_eos = (ids_split == eos_id).cumsum(-1) == 1
            token_mask = (ids_split != eos_id)
            mask = (first_eos | token_mask)[..., 1:]  # align with nlls

            # 4. Metric update expected by Lightning.
            self.gen_ppl_metric.update(nlls, mask)

            total_nll += (nlls * mask).sum().item()
            total_tokens += mask.sum().item()

    ppl = math.exp(total_nll / max(total_tokens, 1e-8))
    return ppl
  
  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),self.noise.parameters()))
      self.ema.copy_to(itertools.chain( self.backbone.parameters(), self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    (sampling_steps, samples, sequence_lengths) = self.sample_subs_guidance(
          n_samples=self.config.loader.eval_batch_size, stride_length=stride_length, num_strides=num_strides, dt=dt)
    if self.ema:
      self.ema.restore(itertools.chain(self.backbone.parameters(),self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths
  

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      p_x0 = self.forward(x, sigma_t).exp()
    
    assert move_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x