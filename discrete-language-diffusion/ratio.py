import itertools, typing, torch, torch.nn as nn, torchmetrics
import lightning as L
import hydra.utils
import dataloader, noise_schedule, utils
import models
from base_dm_model import BaseDMModel
import omegaconf

# -------------------------------------------------------------------

class RatioEstimator(BaseDMModel):
    """
    Time-conditioned ratio network r_ψ(x_t, t) with cycle- and
    consistency-regularisation (see TLDM Appendix, pseudo-code 4).
    """
    def __init__(
        self,
        config,
        tokenizer,
        domain_classifier: typing.Union[nn.Module, None],
        domain_classifier_time_dependent: typing.Union[nn.Module, None],
        pretrained_backbone: typing.Optional[nn.Module] = None,
        inference_mode = False,
    ):
        self.config = config
        super().__init__()

        # 1 ▸ High-level flags & helpers
        self.tokenizer = tokenizer
        self.train_time_independent = False

        # 2 ▸ Vocabulary & special tokens
        self.vocab_size = tokenizer.vocab_size
        if getattr(tokenizer, "mask_token", None) is None:
            self.mask_index = self.vocab_size
            self.vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id

        # 3 ▸ Training-ratio hyper-parameters
        tr_cfg = config.training_ratio
        self.antithetic_sampling = tr_cfg.antithetic_sampling
        self.importance_sampling = tr_cfg.importance_sampling
        self.change_of_variables = tr_cfg.change_of_variables
        self.eta1, self.eta2 = tr_cfg.eta1, tr_cfg.eta2
        self.sampling_eps       = tr_cfg.sampling_eps
        self.consistency        = bool(self.eta2 and self.eta2 > 0)
        self.cycle              = bool(self.eta1 and self.eta1 > 0)

        self.diffusion = config.diffusion
        self.time_conditioning = config.time_conditioning
        self.T         = config.T

        # ### NEW: flag to enable mixed domain batches
        self.mixed_domain_batches = bool(getattr(tr_cfg, "mixed_domain_batches", False))

        # 4 ▸ Frozen auxiliary classifiers
        if not inference_mode:
            self.domain_classifier = domain_classifier.eval().requires_grad_(False)
            self.domain_classifier_t = domain_classifier_time_dependent.eval().requires_grad_(False)

        # 5 ▸ Ratio-network backbone
        if config.ratio_backbone == "dit":
            self.ratio_model = models.dit.DITRatio(
                config, vocab_size=self.vocab_size, time_conditioning=True
            )
        else:
            raise NotImplementedError(
                f"Ratio backbone '{config.ratio_backbone}' not implemented.")
        if pretrained_backbone is not None:
            self.ratio_model.load_pretrained_encoder(pretrained_backbone)

        # 6 ▸ Noise schedule
        self.noise = noise_schedule.get_noise(config, dtype=self.dtype)

        # 7 ▸ Metrics & loss
        self.mse = nn.MSELoss()
        base_metrics = torchmetrics.MetricCollection({"loss": torchmetrics.MeanMetric()})
        self.train_metrics = base_metrics.clone(prefix="train/")
        self.valid_metrics = base_metrics.clone(prefix="val/")

        # 8 ▸ Fast-forward placeholders
        self.fast_forward_epochs  = None
        self.fast_forward_batches = None

    # ================================================================
    # ▸ forward interface
    def forward(self, x_t, sigma, attention_mask=None, x_emb=None):
        sigma = self._process_sigma(sigma) if sigma is not None else sigma
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.ratio_model(x_t, sigma, x_emb=x_emb, attention_mask=attention_mask)
        return logits

    # ================================================================
    # ▸ util classifiers forward (unchanged)
    def _frozen_clf_forward(self, model, *tensors, **kwargs):
        tensors_cpu = [t.cpu() for t in tensors]
        kwargs_cpu  = {k: v.cpu() for k, v in kwargs.items()}
        with torch.no_grad():
            out = model(*tensors_cpu, **kwargs_cpu)
        return out.to(self.device)

    def _cpu_forward(self, model: nn.Module, *args, **kwargs):
        def to_cpu(x):
            return x.cpu() if torch.is_tensor(x) else x
        args_cpu  = [to_cpu(a) for a in args]
        kwargs_cpu = {k: to_cpu(v) for k, v in kwargs.items()}
        prev_device = getattr(model, "device", None)
        if prev_device is not None:
            model.device = torch.device("cpu")
            print(f"Moved model {model.__class__.__name__} to CPU for inference.")
        with torch.no_grad():
            out = model(*args_cpu, **kwargs_cpu)
        return out.to(self.device)

    # ================================================================
    # ▸ helpers for mixing + masked MSE
    def _build_mixed_groups(self, x0_src, attn_src, x0_tgt, attn_tgt):
        """Return two groups (for ratio- and cycle-loss) each mixed from src+tgt."""
        B_src, B_tgt = x0_src.size(0), x0_tgt.size(0)
        x0_all  = torch.cat([x0_src, x0_tgt], dim=0)
        att_all = torch.cat([attn_src, attn_tgt], dim=0)
        # 0 = src, 1 = tgt
        dom_all = torch.cat([
            torch.zeros(B_src, dtype=torch.long, device=x0_all.device),
            torch.ones(B_tgt,  dtype=torch.long, device=x0_all.device)
        ], dim=0)
        perm = torch.randperm(B_src + B_tgt, device=x0_all.device)
        idx_ratio = perm[:B_src]            # same size as original src branch
        idx_cycle = perm[B_src:]            # same size as original tgt branch
        grp_ratio = (x0_all[idx_ratio], att_all[idx_ratio], dom_all[idx_ratio])
        grp_cycle = (x0_all[idx_cycle], att_all[idx_cycle], dom_all[idx_cycle])
        return grp_ratio, grp_cycle

    def _per_sample_se(self, pred, target):
        """Per-sample squared error reduced across non-batch dims to shape (B,)."""
        se = (pred - target) ** 2
        while se.dim() > 1:
            se = se.mean(dim=-1)
        return se  # (B,)

    def _masked_mean(self, values, mask):
        denom = mask.float().sum().clamp_min(1.0)
        return (values[mask].sum() / denom)

    # ================================================================
    # ▸ core loss computation
    def _compute_losses(self, batch):
        batch_src = batch["src"]
        batch_tgt = batch["tgt"]

        x0_src, x0_tgt = batch_src["input_ids"], batch_tgt["input_ids"]
        attention_mask_src = batch_src["attention_mask"]
        attention_mask_tgt = batch_tgt["attention_mask"]

        # -------- decide groups (possibly mixed) --------
        if self.mixed_domain_batches:
            # two mixed groups: one for L_ratio, one for L_cycle
            (x0_r, att_r, dom_r), (x0_c, att_c, dom_c) = self._build_mixed_groups(
                x0_src, attention_mask_src, x0_tgt, attention_mask_tgt
            )
        else:
            # original behavior: ratio on source, cycle on target
            x0_r, att_r, dom_r = x0_src, attention_mask_src, torch.zeros(
                x0_src.size(0), dtype=torch.long, device=x0_src.device
            )
            x0_c, att_c, dom_c = x0_tgt, attention_mask_tgt, torch.ones(
                x0_tgt.size(0), dtype=torch.long, device=x0_tgt.device
            )

        # ---------- ratio (time-independent target) ----------
        B_r = x0_r.size(0)
        t_r = self._sample_t(B_r)
        sigma_r, _ = self.noise(t_r)
        move_r = 1 - torch.exp(-sigma_r)
        x_t_r = self._q_xt(x0_r, move_r[:, None])

        with torch.no_grad():
            c_r = self.domain_classifier(x0_r, attention_mask=att_r).squeeze(-1)
            r_r = (-c_r if not self.config.training_ratio.classifier_output_with_sigmoid
                   else torch.log((1 - c_r) / (c_r + 1e-8) + 1e-8))

        r_pred_r = self.forward(x_t_r, sigma=sigma_r, attention_mask=att_r)
        loss_ratio = self.mse(r_pred_r, r_r)

        # per-domain diagnostics
        se_r = self._per_sample_se(r_pred_r, r_r)
        mask_r_src = dom_r == 0
        mask_r_tgt = dom_r == 1
        l_ratio_src = self._masked_mean(se_r, mask_r_src)
        l_ratio_tgt = self._masked_mean(se_r, mask_r_tgt)

        # ---------- cycle (time-dependent target) ----------
        if self.cycle:
            B_c = x0_c.size(0)
            t_c = self._sample_t(B_c)
            sigma_c, _ = self.noise(t_c)
            move_c = 1 - torch.exp(-sigma_c)
            x_t_c = self._q_xt(x0_c, move_c[:, None])

            with torch.no_grad():
                c_tdep = self.domain_classifier_t(x_t_c, sigma_c, attention_mask=att_c).squeeze(-1)
                r_c = (-c_tdep if not self.config.training_ratio.classifier_output_with_sigmoid
                       else torch.log((1 - c_tdep) / (c_tdep + 1e-8) + 1e-8))

            r_pred_c = self(x_t_c, sigma=sigma_c, attention_mask=att_c)
            loss_cycle = self.mse(r_pred_c, r_c)

            # per-domain diagnostics
            se_c = self._per_sample_se(r_pred_c, r_c)
            mask_c_src = dom_c == 0
            mask_c_tgt = dom_c == 1
            l_cycle_src = self._masked_mean(se_c, mask_c_src)
            l_cycle_tgt = self._masked_mean(se_c, mask_c_tgt)
        else:
            loss_cycle = torch.zeros((), device=self.device)
            l_cycle_src = torch.zeros((), device=self.device)
            l_cycle_tgt = torch.zeros((), device=self.device)

        total = loss_ratio + self.eta1 * loss_cycle

        # ### NEW: return extra per-domain metrics for logging
        extras = {
            "L_ratio_src": l_ratio_src,
            "L_ratio_tgt": l_ratio_tgt,
            "L_cycle_src": l_cycle_src,
            "L_cycle_tgt": l_cycle_tgt,
        }
        return total, loss_ratio, loss_cycle, extras

    # ================================================================
    # ▸ Lightning overrides
    def training_step(self, batch, _):
        total, l_r, l_cy, extras = self._compute_losses(batch)
        self.train_metrics.update(torch.tensor([total.detach()]))

        log_dict = {
            "train/total": total,
            "train/L_ratio": l_r,
            "train/L_cycle": l_cy,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        # ### NEW: also log per-domain when available (esp. useful with mixing)
        log_dict.update({
            "train/L_ratio_src": extras["L_ratio_src"],
            "train/L_ratio_tgt": extras["L_ratio_tgt"],
            "train/L_cycle_src": extras["L_cycle_src"],
            "train/L_cycle_tgt": extras["L_cycle_tgt"],
        })

        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        return total

    def validation_step(self, batch, _):
        total, l_r, l_cy, extras = self._compute_losses(batch)
        self.valid_metrics.update(torch.tensor([total.detach()]))

        # ### NEW: per-domain validation metrics to inspect source vs target
        self.log_dict({
            "val/total": total,
            "val/L_ratio": l_r,
            "val/L_cycle": l_cy,
            "val/L_ratio_src": extras["L_ratio_src"],
            "val/L_ratio_tgt": extras["L_ratio_tgt"],
            "val/L_cycle_src": extras["L_cycle_src"],
            "val/L_cycle_tgt": extras["L_cycle_tgt"],
        }, prog_bar=True, sync_dist=True)

    # ----------------- optimiser & sched -----------------------------
    def configure_optimizers(self):
        optim_args = {
            'lr': self.config.optim.lr,
            'betas': (self.config.optim.beta1, self.config.optim.beta2),
            'eps': self.config.optim.eps,
            'weight_decay': self.config.optim.weight_decay,
        }
        optimizer = torch.optim.AdamW(
            itertools.chain(self.ratio_model.parameters(), self.noise.parameters()),
            **optim_args
        )
        total_steps = self.config.trainer_ratio.max_steps
        warmup_steps = int(total_steps * 0.1)
        sched_cfg = omegaconf.OmegaConf.merge(
            self.config.lr_scheduler,
            {"warmup_t": warmup_steps, "t_initial": total_steps - warmup_steps},
        )
        scheduler = hydra.utils.instantiate(sched_cfg, optimizer=optimizer)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }
        return [optimizer], [scheduler_dict]

    def get_log_probs(self, x_t: torch.Tensor, batch_size: int, sigma: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        chunks = []
        for start in range(0, x_t.size(0), batch_size):
            end = start + batch_size
            ids   = x_t[start:end].to(device, non_blocking=True)
            sig   = sigma[start:end].to(device, non_blocking=True)
            with torch.no_grad():
                logits = self.forward(ids, sig)
            chunks.append(torch.log_softmax(logits, dim=-1).cpu())
        return torch.cat(chunks, dim=0)