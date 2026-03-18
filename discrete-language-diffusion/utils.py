"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import logging
import math
import os
import json
import numpy as np
from datasets import concatenate_datasets
from torch.utils.data import DataLoader, RandomSampler
import fsspec
import lightning
import torch
from numpy.random import Generator, PCG64
from timm.scheduler import CosineLRScheduler
import copy
import typing
from pathlib import Path
import yaml, hydra, os
import lightning.pytorch as pl
from typing import List, Optional, Tuple, Dict, Any
import classifier
import re
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from transformers import AutoTokenizer, AutoModel
import tokenizers, safetensors, transformers
from typing import Iterable, Dict, List, Union, Optional
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
import numpy as np
from collections import Counter
import numpy as np
import torch

def filter_arxiv_dataset_by_domain(dataset, domain_keys):
    """
    Filter an arXiv metadata Dataset to only include records whose
    *primary* category prefix matches domain_key.
    domain_keys   : str | Iterable[str] | None
          e.g. "cs" or ["cs", "stat", "math"]
    Returns: filtered Dataset
    -------
    """
    if not domain_keys:  # None, "", [], …
        return dataset

    # normalise to a set of prefixes
    if isinstance(domain_keys, str):
        domain_keys = {domain_keys}
    else:
        domain_keys = set(domain_keys)

    def _matches_primary(example):
        cats = example.get("categories")
        tags = (cats.split() if isinstance(cats, str)
                else cats if isinstance(cats, list) else [])
        if not tags:
            return False
        primary_prefix = tags[0].split('.', 1)[0]
        return primary_prefix in domain_keys

    return dataset.filter(_matches_primary)


def fsspec_exists(filename):
  """Check if a file exists using fsspec."""
  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def fsspec_listdir(dirname):
  """Listdir in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
  """Mkdirs in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
  if torch.isnan(tensor).any():
    print(name, tensor)


class CosineDecayWarmupLRScheduler(
  CosineLRScheduler,
  torch.optim.lr_scheduler._LRScheduler):
  """Wrap timm.scheduler.CosineLRScheduler
  Enables calling scheduler.step() without passing in epoch.
  Supports resuming as well.
  Adapted from:
    https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_epoch = -1
    self.step(epoch=0)

  def step(self, epoch=None):
    if epoch is None:
      self._last_epoch += 1
    else:
      self._last_epoch = epoch
    # We call either step or step_update, depending on
    # whether we're using the scheduler every epoch or every
    # step.
    # Otherwise, lightning will always call step (i.e.,
    # meant for each epoch), and if we set scheduler
    # interval to "step", then the learning rate update will
    # be wrong.
    if self.t_in_epochs:
      super().step(epoch=self._last_epoch)
    else:
      super().step_update(num_updates=self._last_epoch)


class LoggingContext:
  """Context manager for selective logging."""
  def __init__(self, logger, level=None, handler=None, close=True):
    self.logger = logger
    self.level = level
    self.handler = handler
    self.close = close

  def __enter__(self):
    if self.level is not None:
      self.old_level = self.logger.level
      self.logger.setLevel(self.level)
    if self.handler:
      self.logger.addHandler(self.handler)

  def __exit__(self, et, ev, tb):
    if self.level is not None:
      self.logger.setLevel(self.old_level)
    if self.handler:
      self.logger.removeHandler(self.handler)
    if self.handler and self.close:
      self.handler.close()

# ── helpers ─────────────────────────────────────────────────────────────
def _unwrap_dataset(ds):
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds

def _infer_wrap_from_filename(loader):
    """
    Return True / False if the cached filename tells us,
    otherwise return None (meaning “unknown”).
    """
    ds = _unwrap_dataset(loader.dataset)
    if getattr(ds, "cache_files", None):
        fname = ds.cache_files[0]["filename"]
        if "_wrapped.dat"   in fname: return True
        if "_unwrapped.dat" in fname: return False
    return None   # no hint found
# ── main check ──────────────────────────────────────────────────────────
def make_checks_if_config_and_loader_is_synchronized(config, *dataloaders):
    loaders = [dl for dl in dataloaders if dl is not None]
    if not loaders:
        raise ValueError("No dataloaders provided.")

    # 1 ▸ tokenizer / model_max_length consistency
    ref_tok = loaders[0].tokenizer
    for i, dl in enumerate(loaders[1:], 1):
        tok = dl.tokenizer
        if tok.name_or_path   != ref_tok.name_or_path \
        or tok.vocab_size     != ref_tok.vocab_size \
        or tok.model_max_length != ref_tok.model_max_length:
            raise ValueError(f"Tokenizer mismatch between loader[0] and loader[{i}].")

    # 2 ▸ wrap check (filename-only, no data access)
    inferred = _infer_wrap_from_filename(loaders[0])
    if inferred is not None and inferred != config.data.wrap:
        raise ValueError(f"Config says wrap={config.data.wrap} but cache looks "
                         f"{'wrapped' if inferred else 'unwrapped'}.")

    # 3 ▸ simple batch-size sanity
    if getattr(config.loader, "batch_size", 1) <= 0:
        raise ValueError("Batch size must be positive.")

    return {
        "wrap"          : inferred if inferred is not None else "unknown",
        "tokenizer"     : ref_tok.name_or_path,
        "model_max_len" : ref_tok.model_max_length,
    }


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
  """Initializes multi-GPU-friendly python logger."""

  logger = logging.getLogger(name)
  logger.setLevel(level)

  # this ensures all logging levels get marked with the rank zero decorator
  # otherwise logs would get multiplied for each GPU process in multi-GPU setup
  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(
              getattr(logger, level)))

  return logger


class Sampler:
  def __init__(self, shape):
    self.shape = shape

  def _sampling_noise(self):
    pass
  
  def _hard_sample(self, logits):
    pass

  def _soft_sample(self, logits):
    return 0

  def sample(self, logits):
    noise = self._sampling_noise()
    noise = noise[: logits.shape[0], :]
    logits = logits + noise.to(
      dtype=logits.dtype, device=logits.device)
    hard_sample = self._hard_sample(logits)
    soft_sample = self._soft_sample(logits)
    return soft_sample + (hard_sample - soft_sample).detach()


class TopKSampler(Sampler):
  def __init__(self, k, shape, gamma_tau=1.0):
    super().__init__(shape)
    self.k = k
    self.gamma_tau = gamma_tau
    self.num_betas = 10
    self.sampler = torch.distributions.gamma.Gamma(
      1 / k * torch.ones(self.num_betas, * self.shape), 1.0)

  def _sampling_noise(self):
    noise = self.sampler.sample()
    beta = self.k / torch.arange(1, self.num_betas + 1, 1,
                                 dtype=torch.float32)
    beta = beta[:, None, None]
    assert beta.ndim == noise.ndim
    s = noise / beta
    s = torch.sum(s, axis=0)
    s = s - math.log(10.0)
    s = self.gamma_tau * (s / self.k)
    return s

  def _hard_sample(self, logits):
    assert logits.ndim == 2
    thresholds, _ = torch.sort(logits, dim=-1)
    thresholds = thresholds[:, - self.k][:, None]
    return (logits >= thresholds).type(logits.dtype)

  def _soft_sample(self, logits):
    soft_top_k = logits - torch.mean(logits, dim=-1,
                                     keepdim=True)
    return soft_top_k / torch.norm(soft_top_k, dim=-1,
                                   keepdim=True)


class DeterministicTopK(TopKSampler):
  def __init__(self, k):
    super().__init__(k, shape=(1, 1))

  def _sampling_noise(self):
    return 0

  def discreize(self, x):
    hard_sample = self._hard_sample(x)
    soft_sample = self._soft_sample(x)
    return soft_sample + (hard_sample - soft_sample).detach()

class GumbelSampler(Sampler):

  def __init__(self, shape, temperature=1.0):
    super().__init__(shape)
    self.temperature = temperature

  def _sampling_noise(self):
    return - (1e-10 - (
      torch.rand(* self.shape) + 1e-10).log()).log()

  def _hard_sample(self, logits):
    assert logits.ndim == 2
    indices = torch.argmax(logits, dim=-1)
    zeros = logits * 0
    ones = torch.ones_like(logits[:, :, :1])
    return torch.scatter(zeros, -1, indices[:, :, None],
                         ones)

  def _soft_sample(self, logits):
    return torch.nn.functional.softmax(
      logits / self.temperature, dim=-1)


class BinarySampler(GumbelSampler):

  def sample(self, probs):
    # TODO(subhamsahoo): use the temperature parameter.
    pos_noise = self._sampling_noise().to(
      dtype=probs.dtype, device=probs.device)
    neg_noise = self._sampling_noise().to(
      dtype=probs.dtype, device=probs.device)
    del_noise_exp = (neg_noise - pos_noise).exp()
    hard_sample = (probs * (1 + del_noise_exp)
                   > 1).to(probs.dtype)
    soft_sample = probs / (probs + (1 - probs) * del_noise_exp)
    return soft_sample + (hard_sample - soft_sample).detach()


class GaussianSampler:
  def __init__(self):
    self.softplus = torch.nn.Softplus()

  def sample(self, x):
    assert x.ndim == 2
    n = x.shape[-1] // 2
    mu = x[:, :n]
    sigma = self.softplus(x[:, n:]).sqrt()
    return mu + sigma * torch.randn_like(mu)



@lightning.pytorch.utilities.rank_zero_only
def print_num_parameters(model: torch.nn.Module, verbose: bool = True, print_prefix="") -> None:
    """
    Prints the total and trainable number of parameters in a model.

    Args:
        model: the torch.nn.Module whose parameters to count.
        verbose: if True, prints the counts; otherwise returns nothing.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"{print_prefix}Total parameters: {total_params:,}")
        print(f"{print_prefix}Trainable parameters: {trainable_params:,}")



def build_or_load(
    model_, 
    init_kwargs: Dict[str, Any], 
    ckpt_path: Optional[str] = None, 
    freeze: bool = False
    ) -> Tuple[torch.nn.Module, bool]:
    """
    Load `model_` from `ckpt_path` if the file exists; otherwise create a
    fresh instance.

    * Extra tensors present in the checkpoint but absent in the current
      model definition are silently discarded.
    * The original attribute-consistency check remains unchanged.
    """
    try: 
      if ckpt_path and os.path.exists(ckpt_path):
        model = model_.load_from_checkpoint(
            ckpt_path,
            strict=False,          # ignore extra / missing keys
            **init_kwargs          # passed to __init__
        )
        loaded = True
      else:                          # 2. fresh instance
        model  = model_(**init_kwargs)
        loaded = False

      # 3. optionally freeze
      if freeze and loaded:
          for p in model.parameters():
              p.requires_grad = False
      if loaded:
          print(f"Loaded model from {ckpt_path}.")
      else:
          print("Created a fresh model instance without loading weights!")
      return model, loaded
    except Exception as e:             # any load problem → fall back
      # ---------------------------------------------------- 1) create model
      model = model_(**init_kwargs)
      loaded = False

      # ---------------------------------------------------- 2) restore weights (if any)
      if ckpt_path and os.path.exists(ckpt_path):
          checkpoint = torch.load(ckpt_path, map_location="cpu")
          sd_ckpt: Dict[str, torch.Tensor] = checkpoint["state_dict"]

          # keep only parameters that exist in the current model
          current_keys = set(model.state_dict().keys())
          sd_filtered = {k: v for k, v in sd_ckpt.items() if k in current_keys}

          model.load_state_dict(sd_filtered, strict=False)
          loaded = True

      # ---------------------------------------------------- 3) sanity-check unchanged
      if loaded:
          model_scratch = model_(**init_kwargs)
          keys_to_check = ["time_conditioning", "T", "change_of_variables"]
          for key in keys_to_check:
              if hasattr(model_scratch, key) and hasattr(model, key):
                  before = getattr(model_scratch, key)
                  after  = getattr(model, key)
                  if before != after:
                      raise ValueError(
                          f"Checkpoint at '{ckpt_path}' has a different value for "
                          f"'{key}': checkpoint={after} vs fresh={before}"
                      )
          print(f"Loaded model from {ckpt_path}.")

      # ---------------------------------------------------- 4) optionally freeze
      if freeze and loaded:
          for p in model.parameters():
              p.requires_grad = False

      return model, loaded



def _callbacks_for(subdir: str, checkpoint_save_dir: str, callbacks, monitor_metric: str = 'gen_ppl'):
    #Return a deep‑copied callbacks list whose ModelCheckpoint
    #saves to .../<subdir>/checkpoints/.
    #config.checkpointing.save_dir
    
    cbs = []
    for cb in callbacks:
        cb_new = copy.deepcopy(cb)
        if isinstance(cb_new, lightning.pytorch.callbacks.ModelCheckpoint):
            cb_new.dirpath = os.path.join(
                checkpoint_save_dir, subdir, "checkpoints"
            )
            # Special‑case: for the ratio model we want to track a different metric
            # ToDo: !!this is a hack, should be fixed in the config make a new config callbacks files for ratio model!!
            if getattr(cb_new, 'monitor', None) is not None:
                cb_new.monitor = f'val/{monitor_metric}'
        cbs.append(cb_new)
    return cbs


def build_subset_key(
    config: typing.Any,                     # your Hydra/argparse config object
    domain: typing.Union[str, None]         # "src", "tgt", or None
    ) -> typing.Optional[str]:
    """
    Derive a deterministic `subset_key` for dataset filtering.

    Rules
    -----
    1. If `domain` is None                   → return None.
    2. Base key:  config.data["<domain>_domain"]
       (ignored if missing, empty, or the string "none").
    3. Optional fraction:
       * If config.data["domain_fraction"] exists and is in (0, 1]:
         - For "src": use `fraction`
         - For "tgt": use `1 - fraction`
         - Append "<domain>_<percent>" (e.g. "src_70") to the base key.
    """
    if domain is None:
        return None

    data_cfg = getattr(config, "data", {})          # safe even if .data absent

    # --- base key -----------------------------------------------------------
    raw_val = str(data_cfg.get(f"{domain}_domain", "")).strip()
    base_key: Optional[str] = None
    if raw_val and raw_val.lower() != "none":
        base_key = raw_val

    # --- optional fraction --------------------------------------------------
    frac = data_cfg.get("domain_fraction")          # may be None
    if frac is None:
        return base_key

    if not 0 < frac <= 1:
        raise ValueError("domain_fraction must be in (0, 1].")

    frac = 1 - frac if domain == "tgt" else frac
    suffix = f"{domain}_{int(frac * 100)}"          # e.g. "tgt_30"

    return f"{base_key}_{suffix}" if base_key else suffix


class PiecewiseGamma:
    def __init__(self, g0=1.0, g1=2.0, s1=0.33, s2=0.66):
        self.g0, self.g1 = float(g0), float(g1)
        self.s1, self.s2 = float(s1), float(s2)

    def __call__(self, t):
        # float input
        if not isinstance(t, torch.Tensor):
            s = float(t)
            if s <= self.s1:   return self.g0
            if s >= self.s2:   return self.g1
            w = (s - self.s1) / (self.s2 - self.s1)
            return (1.0 - w) * self.g0 + w * self.g1

        # tensor input (broadcast-friendly)
        s = t.clamp(0.0, 1.0)
        w = ((s - self.s1) / (self.s2 - self.s1)).clamp(0.0, 1.0)
        return (1.0 - w) * self.g0 + w * self.g1


def get_gamma_scheduler(config):
    """
    Get the gamma scheduler based on the configuration.
    """
    if config.gamma.type == "piecewise_linear":
        return PiecewiseGamma(
            g0=config.gamma.g0,
            g1=config.gamma.g1,
            s1=config.gamma.s1,
            s2=config.gamma.s2)
    elif config.gamma.type == "constant":
        return lambda _: config.gamma.guidance_scale
    else:
        raise ValueError(f"Unknown gamma scheduler: {config.sampling.gamma_scheduler}")
    

def set_hf_cache_envs(cache_root: str):
    cache_root = str(Path(cache_root).expanduser())
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_root, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_root, "datasets"))

def model_is_cached(model_id_or_path: str, cache_root: str) -> bool:
    """True if the model is already present locally under cache_root (or a local path given)."""
    p = Path(model_id_or_path).expanduser()
    if p.is_dir():  # user passed a local dir
        return True
    try:
        # Will return immediately if already cached; raises if not found locally.
        snapshot_download(
            repo_id=model_id_or_path,
            cache_dir=os.path.join(cache_root, "hub"),
            local_files_only=True,
            allow_patterns="*",
        )
        return True
    except LocalEntryNotFoundError:
        return False
    


def _clone_loader_like(dl: DataLoader, new_dataset) -> DataLoader:
    from torch.utils.data import DataLoader
    kwargs = dict(
        dataset=new_dataset,
        batch_size=dl.batch_size,
        sampler=RandomSampler(new_dataset),
        drop_last=dl.drop_last,
        num_workers=dl.num_workers,
        pin_memory=dl.pin_memory,
        persistent_workers=getattr(dl, "persistent_workers", False),
        timeout=getattr(dl, "timeout", 0),
        generator=getattr(dl, "generator", None),
        collate_fn=getattr(dl, "collate_fn", None),
        worker_init_fn=getattr(dl, "worker_init_fn", None),
    )
    pf = getattr(dl, "prefetch_factor", None)
    if dl.num_workers and pf is not None:
        kwargs["prefetch_factor"] = pf
    pmd = getattr(dl, "pin_memory_device", None)
    if pmd:
        kwargs["pin_memory_device"] = pmd
    return DataLoader(**kwargs)


def _split_cache_path(
    tgt_ds,
    pct: float,
    seed: int,
    cache_dir: str,
    split_id: Optional[str] = None,
    ) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    fp = getattr(tgt_ds, "_fingerprint", f"rows{len(tgt_ds)}")
    tag = split_id or fp
    fname = f"tgt_split_{tag}_pct{int(round(pct))}_seed{seed}.json"
    return os.path.join(cache_dir, fname)
    
def apply_tgt_split_indices(
    train_ds_src: DataLoader,
    train_ds_tgt: DataLoader,
    idx_to_src: List[int],
    ) -> Tuple[DataLoader, DataLoader]:
    """Apply a precomputed split to rebuild loaders."""
    tgt_ds = train_ds_tgt.dataset
    n = len(tgt_ds)
    move = set(idx_to_src)
    if any(i < 0 or i >= n for i in move):
        raise ValueError("Some saved indices are out of range for the current target dataset.")

    idx_remain = [i for i in range(n) if i not in move]

    if len(move) == 0:
        return train_ds_src, train_ds_tgt
    if len(move) == n:
        new_src_dataset = concatenate_datasets([train_ds_src.dataset, tgt_ds])
        empty_tgt = tgt_ds.select([])
        return _clone_loader_like(train_ds_src, new_src_dataset), _clone_loader_like(train_ds_tgt, empty_tgt)

    tgt_chunk_for_src = tgt_ds.select(sorted(move))
    tgt_chunk_remain  = tgt_ds.select(idx_remain)
    new_src_dataset = concatenate_datasets([train_ds_src.dataset, tgt_chunk_for_src])

    new_train_src = _clone_loader_like(train_ds_src, new_src_dataset)
    new_train_tgt = _clone_loader_like(train_ds_tgt, tgt_chunk_remain)
    return new_train_src, new_train_tgt


def compute_tgt_split_indices_rng(tgt_ds, pct: float, seed: int = 42, sort_indices: bool = True):
    #  deterministic across devices on the same NumPy version. Usually enough.
    n = len(tgt_ds)
    n_to_src = int(round(n * (pct / 100.0)))
    rng = Generator(PCG64(seed))           # explicit, cross-platform deterministic
    perm = rng.permutation(n)
    idx = perm[:n_to_src]
    if sort_indices:
        idx = np.sort(idx)
    return idx.tolist()

def drop_target_samples(
    train_ds_tgt: DataLoader,
    idx_keep: List[int]
    ) -> DataLoader:
    """
    Drop `pct_to_drop` percent from the target DataLoader and return a new target loader.
    Deterministic with `seed`. If `persist=True`, the kept indices are saved and reused.
    """
    tgt_ds = train_ds_tgt.dataset
    subset = tgt_ds.select(idx_keep)
    return _clone_loader_like(train_ds_tgt, subset)


def print_i_beginning_and_ending_samples(dataloader_, i, tokenizer):
    train_dataset_tgt = dataloader_.dataset
    for i in range(i):
       beginning_sample = train_dataset_tgt[i]['input_ids'][:30]
       beginning_sample_detokenized = tokenizer.decode(beginning_sample, skip_special_tokens=True)
       last_sample = train_dataset_tgt[-(i+1)]['input_ids'][:30]
       last_sample_detokenized = tokenizer.decode(last_sample, skip_special_tokens=True)

       print(f"Beginning sample {i}: {beginning_sample_detokenized}")
       print(f"Last sample {i}: {last_sample_detokenized}")