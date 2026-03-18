"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import logging
from logging import config
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
import mauve
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
import utils
import dataloader


class RewardModelScorer:
    def __init__(
        self,
        model_id: str = "openbmb/Eurus-RM-7b",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True,use_fast=False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"  # Eurus-RM trained with left padding.
        # Eurus-RM ships custom code; trust_remote_code is required.
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=self.dtype
        )
        # Push to device (or let accelerate handle it if you prefer device_map="auto")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def score(self, texts: List[str], batch_size: int = 8, max_length: int = 4096) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,  # Eurus config allows very long ctx; cap for speed.
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            # EurusRewardModel returns rewards shaped [batch, 1] or [batch]
            r = self.model(**enc).squeeze(-1).detach().cpu().tolist()
            scores.extend(r)
        return scores
    


def _subset_samples(generated, idxs: typing.Sequence[int]):
    """Pick a subset of generated samples by integer indices, preserving type."""
    if not idxs:
        return None
    if isinstance(generated, torch.Tensor):
        return generated[idxs]
    # list[str] or list[list[int]] or list[Tensor]
    return [generated[i] for i in idxs]

def _choose_labels_for_bucketing(
    n: int,
    cls_probs: Optional[torch.Tensor],
    dom: Optional[Dict[str, Any]],
    threshold: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Prefer classifier: src if prob>=threshold else tgt.
    Fallback to embeddings by argmax (no 'ambiguous' labels in the output).
    """
    if isinstance(cls_probs, torch.Tensor) and cls_probs.numel():
        p = cls_probs.squeeze(-1).detach().to(dtype=torch.float32, device="cpu").numpy()
        return np.where(p >= threshold, "src", "tgt")
    if dom and ("cos_src" in dom) and ("cos_tgt" in dom):
        # resolve any 'ambiguous' by argmax
        cos_src = np.asarray(dom["cos_src"])
        cos_tgt = np.asarray(dom["cos_tgt"])
        return np.where(cos_src >= cos_tgt, "src", "tgt")
    return None


def eval_sample_gen_ppl(
    samples: Iterable[str],
    models: Union[str, Iterable[str]] = ("gpt2", "facebook/opt-2.7b"),
    *,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    use_safetensors: bool = True,
    allow_bin: bool = False,
    stride: Optional[int] = None,  # None => use model context length
    continue_on_error: bool = True,  # NEW: keep going on errors
) -> Dict[str, Dict]:
    """
    Compute perplexity for each sample under each model.

    Returns:
      {
        <model_id>: {
          "avg_ppl": float,
          "details": [{"ppl": Optional[float], "n_tokens": int}],  # ppl may be None on per-sample failure
          "n_total_tokens": int,  # only counts successfully-scored tokens
        },
        ...
      }
    """
    if isinstance(models, str):
        model_ids = [models]
    else:
        model_ids = list(models)

    base_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    results: Dict[str, Dict] = {}

    for model_id in model_ids:
        # --- Load tokenizer ---
        try:
            tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
        except Exception as e:
            print(f"[WARN] Tokenizer load failed for {model_id}: {e}")
            results[model_id] = {"avg_ppl": float("nan"), "details": [], "n_total_tokens": 0}
            if not continue_on_error:
                raise
            continue

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # --- Load model (prefer safetensors) ---
        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=False,
                use_safetensors=use_safetensors,
                torch_dtype="auto",
            )
        except ValueError as e:
            msg = str(e)
            if "upgrade torch to at least v2.6" in msg and not allow_bin:
                print(f"[WARN] {model_id} only provides .bin; set allow_bin=True to allow.")
                results[model_id] = {"avg_ppl": float("nan"), "details": [], "n_total_tokens": 0}
                if not continue_on_error:
                    raise
                continue
            # Fallback to .bin if allowed
            try:
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    local_files_only=False,
                    use_safetensors=False,
                    torch_dtype="auto",
                )
            except Exception as e2:
                print(f"[WARN] Model load failed for {model_id}: {e2}")
                results[model_id] = {"avg_ppl": float("nan"), "details": [], "n_total_tokens": 0}
                if not continue_on_error:
                    raise
                continue
        except Exception as e:
            print(f"[WARN] Model load failed for {model_id}: {e}")
            results[model_id] = {"avg_ppl": float("nan"), "details": [], "n_total_tokens": 0}
            if not continue_on_error:
                raise
            continue

        # --- Device placement with CPU fallback instead of skipping ---
        dev = base_dev
        try:
            mdl.to(dev).eval()
        except Exception as e:
            print(f"[WARN] Could not move {model_id} to {dev} ({e}); retrying on CPU.")
            dev = "cpu"
            try:
                mdl.to(dev).eval()
            except Exception as e2:
                print(f"[WARN] CPU placement also failed for {model_id}: {e2}")
                results[model_id] = {"avg_ppl": float("nan"), "details": [], "n_total_tokens": 0}
                if not continue_on_error:
                    raise
                del mdl
                continue

        # --- Context/stride ---
        ctx = getattr(mdl.config, "max_position_embeddings", None)
        if ctx is None or ctx > 10_000_000_000_000_000:
            ctx = 2048 if "opt" in model_id.lower() else 1024
        win = stride or ctx

        per_sample: List[Dict[str, Optional[float]]] = []
        n_total = 0
        sum_neg_log_lik = 0.0

        # --- Per-sample scoring with robust error handling ---
        for text in samples:
            try:
                enc = tok(text, return_tensors="pt")
                input_ids = enc["input_ids"].to(dev)
                attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(dev)
                L = input_ids.size(1)

                if L <= ctx:
                    labels = input_ids.clone()
                    with torch.no_grad():
                        loss = mdl(input_ids=input_ids, attention_mask=attn, labels=labels).loss
                    n_tokens = int(attn.sum().item())
                    ppl = math.exp(float(loss))
                    per_sample.append({"ppl": ppl, "n_tokens": n_tokens})
                    n_total += n_tokens
                    sum_neg_log_lik += float(loss) * n_tokens
                else:
                    nll = 0.0
                    seen = 0
                    i = 0
                    with torch.no_grad():
                        while i < L:
                            begin = max(i + ctx - win, 0)
                            end = min(i + ctx, L)
                            trg_len = end - (0 if i == 0 else begin)
                            chunk_ids = input_ids[:, begin:end]
                            chunk_attn = attn[:, begin:end]

                            labels = chunk_ids.clone()
                            if trg_len < labels.size(1):
                                labels[:, :-trg_len] = -100  # score only new tokens

                            out = mdl(input_ids=chunk_ids, attention_mask=chunk_attn, labels=labels)
                            nll += float(out.loss) * trg_len
                            seen += trg_len
                            i += win

                    n_tokens = seen
                    ppl = math.exp(nll / max(1, n_tokens))
                    per_sample.append({"ppl": ppl, "n_tokens": n_tokens})
                    n_total += n_tokens
                    sum_neg_log_lik += nll

            except RuntimeError as e:
                # Handle CUDA OOM gracefully
                if "out of memory" in str(e).lower() and dev.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(f"[WARN] Sample scoring failed on {model_id}: {e}")
                per_sample.append({"ppl": None, "n_tokens": 0})
                if not continue_on_error:
                    # Clean up before re-raising
                    del mdl
                    if dev.startswith("cuda"):
                        torch.cuda.empty_cache()
                    raise
                continue
            except Exception as e:
                print(f"[WARN] Sample scoring failed on {model_id}: {e}")
                per_sample.append({"ppl": None, "n_tokens": 0})
                if not continue_on_error:
                    del mdl
                    if dev.startswith("cuda"):
                        torch.cuda.empty_cache()
                    raise
                continue

        avg_ppl = math.exp(sum_neg_log_lik / max(1, n_total)) if n_total > 0 else float("nan")
        results[model_id] = {
            "avg_ppl": avg_ppl,
            "details": per_sample,
            "n_total_tokens": n_total,
        }

        # Free VRAM between models
        del mdl
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    return results


def _to_texts(
    generated: Union[List[str], torch.Tensor, Iterable[Iterable[int]]],
    tokenizer,
    skip_special_tokens: bool = True,
) -> List[str]:
    if isinstance(generated, list) and (len(generated) == 0 or isinstance(generated[0], str)):
        return list(generated)
    if isinstance(generated, torch.Tensor):
        return tokenizer.batch_decode(generated, skip_special_tokens=skip_special_tokens)
    # list of id sequences
    return [tokenizer.decode(g, skip_special_tokens=skip_special_tokens) for g in generated]

def _collect_validation_texts(valid_loader, tokenizer, skip_special_tokens=True) -> List[str]:
    """Deterministic given a fixed loader order (typical for val/test)."""
    texts = []
    for batch in valid_loader:
        ids = batch["input_ids"]
        if isinstance(ids, torch.Tensor):
            texts.extend(tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens))
        else:
            # fallback for lists/np arrays
            texts.extend(tokenizer.batch_decode(torch.as_tensor(ids), skip_special_tokens=skip_special_tokens))
    return texts

def _token_lengths(tokenizer, texts: List[str]) -> np.ndarray:
    # Fast and consistent length proxy; includes BOS/EOS if tokenizer adds them.
    return np.array([len(tokenizer.encode(t)) for t in texts], dtype=np.int32)


def mauve_balanced_bootstrap(
    valid_loader,
    tokenizer,
    generated_samples: Union[List[str], torch.Tensor, Iterable[Iterable[int]]],
    num_runs: int = 5,
    base_seed: int = 1234,
    n_bins: int = 8,
    device_id: int = None,
    max_text_length: int = 1024,
    skip_special_tokens: bool = True,
) -> Dict[str, Any]:
    """
    Build length-matched reference subsets from the validation split and compute MAUVE.
    Lazily decodes only sampled references (with cross-run decode cache).
    Deterministic across runs for a fixed (base_seed, num_runs).
    """
    # --- Q: generated texts & lengths (decode Q once; it's small) -------------
    # Reuse your helper to get texts; for lengths prefer IDs if available.
    if isinstance(generated_samples, torch.Tensor):
        q_ids = generated_samples.detach().cpu()
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            q_len = (q_ids != pad_id).sum(dim=1).numpy()
        else:
            q_len = np.full((q_ids.size(0),), q_ids.size(1))
        q_texts = tokenizer.batch_decode(q_ids, skip_special_tokens=skip_special_tokens)
    elif isinstance(generated_samples, (list, tuple)) and generated_samples and not isinstance(generated_samples[0], str):
        # list of id sequences
        q_ids = [torch.as_tensor(x) for x in generated_samples]
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            q_len = np.array([int((t != pad_id).sum().item()) for t in q_ids], dtype=np.int32)
        else:
            q_len = np.array([int(t.numel()) for t in q_ids], dtype=np.int32)
        q_texts = tokenizer.batch_decode([t.tolist() for t in q_ids], skip_special_tokens=skip_special_tokens)
    else:
        # strings
        q_texts = _to_texts(generated_samples, tokenizer, skip_special_tokens=skip_special_tokens)
        q_len = np.array([len(tokenizer.encode(t)) for t in q_texts], dtype=np.int32)

    N = len(q_texts)
    if N == 0:
        return {"mean": float("nan"), "std": float("nan"), "scores": [], "seeds": [], "n_gen": 0, "n_ref_total": 0}

    # --- P pool: reference IDs + lengths (no decoding) ------------------------
    pad_id = tokenizer.pad_token_id
    ref_ids: List[List[int]] = []
    r_len_list: List[int] = []
    for batch in valid_loader:
        ids = batch["input_ids"]
        am = batch.get("attention_mask", None) if isinstance(batch, dict) else None
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu()
            if am is not None and isinstance(am, torch.Tensor):
                am = am.detach().cpu()
            for i in range(ids.size(0)):
                seq = ids[i]
                ref_ids.append(seq.tolist())
                if am is not None:
                    r_len_list.append(int(am[i].sum().item()))
                elif pad_id is not None:
                    r_len_list.append(int((seq != pad_id).sum().item()))
                else:
                    r_len_list.append(int(seq.numel()))
        else:
            # fallback for list/np arrays
            ids = torch.as_tensor(ids)
            for i in range(ids.size(0)):
                seq = ids[i]
                ref_ids.append(seq.tolist())
                if pad_id is not None:
                    r_len_list.append(int((seq != pad_id).sum().item()))
                else:
                    r_len_list.append(int(seq.numel()))

    if len(ref_ids) == 0:
        return {"mean": float("nan"), "std": float("nan"), "scores": [], "seeds": [], "n_gen": N, "n_ref_total": 0}

    r_len = np.asarray(r_len_list, dtype=np.int32)

    # --- Length bins from Q (enforce strict monotonic edges) ------------------
    n_bins = max(1, int(n_bins))
    edges = np.quantile(q_len, np.linspace(0, 1, n_bins + 1)).astype(float)
    # ensure strictly increasing to satisfy np.histogram
    eps = 1e-6
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps

    q_hist, _ = np.histogram(q_len, bins=edges)

    # pre-index refs per bin
    ref_bins: List[np.ndarray] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (r_len.astype(float) >= lo) & (r_len.astype(float) <= hi)
        else:
            mask = (r_len.astype(float) >= lo) & (r_len.astype(float) <  hi)
        ref_bins.append(np.nonzero(mask)[0])

    # --- Device default -------------------------------------------------------
    if device_id is None:
        device_id = 0 if torch.cuda.is_available() else -1

    seeds = [base_seed + i for i in range(num_runs)]
    scores: List[float] = []

    # decode cache to avoid re-decoding the same ref multiple times
    decoded_cache: Dict[int, str] = {}

    # --- Run K times with deterministic seeds --------------------------------
    for s in seeds:
        rs = np.random.RandomState(s)
        chosen: List[int] = []

        for i in range(n_bins):
            need = int(q_hist[i])
            if need == 0:
                continue
            pool = ref_bins[i]
            if pool.size == 0:
                # backfill uniformly from full pool
                chosen.extend(rs.randint(0, len(ref_ids), size=need).tolist())
            elif pool.size >= need:
                chosen.extend(rs.choice(pool, size=need, replace=False).tolist())
            else:
                # with replacement to hit the count
                chosen.extend(rs.choice(pool, size=need, replace=True).tolist())

        # exact N
        if len(chosen) < N:
            chosen.extend(rs.randint(0, len(ref_ids), size=(N - len(chosen))).tolist())
        elif len(chosen) > N:
            chosen = chosen[:N]

        # lazy decode only what's new, then assemble p_text
        needed = [idx for idx in set(chosen) if idx not in decoded_cache]
        if needed:
            dec_texts = tokenizer.batch_decode([ref_ids[idx] for idx in needed],
                                               skip_special_tokens=skip_special_tokens)
            decoded_cache.update({i: t for i, t in zip(needed, dec_texts)})

        p_text = [decoded_cache[i] for i in chosen]

        # compute MAUVE (safe)
        try:
            res = mauve.compute_mauve(
                p_text=p_text,
                q_text=q_texts,
                device_id=device_id,
                max_text_length=max_text_length,
                verbose=False,
            )
            scores.append(float(res.mauve))
        except Exception as e:
            print(f"[WARN] MAUVE run failed (seed={s}): {e.__class__.__name__}: {e}")
            scores.append(float("nan"))

    return {
        "mean": float(np.nanmean(scores)) if scores else float("nan"),
        "std": float(np.nanstd(scores)) if scores else float("nan"),
        "scores": scores,
        "seeds": seeds,
        "n_gen": N,
        "n_ref_total": len(ref_ids),
    }



def format_float(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "N/A"

def print_ppl_report(results: dict, model_ids):
    """Pretty, aligned console report for perplexity outputs."""
    print("\nPerplexity — Averages")
    width = max(len(m) for m in model_ids) + 2
    for m in model_ids:
        avg = results.get(m, {}).get("avg_ppl")
        print(f"{m:<{width}}: {format_float(avg)}")

    print("\nPerplexity — Full Stats")
    for m in model_ids:
        stats = results.get(m, {})
        print(f"{m}: {json.dumps(stats, sort_keys=True)}")


def _fmt(x, nd=3):
        return "N/A" if (x is None or not math.isfinite(x)) else f"{x:.{nd}f}"
    
def print_ppl_table(out, out_ref, model_ids):
        rows = []
        for m in model_ids:
            my = float(out.get(m, {}).get("avg_ppl", float("nan")))
            ref = float(out_ref.get(m, {}).get("avg_ppl", float("nan")))
            rel = (my / ref) if (math.isfinite(my) and math.isfinite(ref) and ref > 0) else float("nan")
            rows.append((m, my, ref, rel))

        name_w = max(5, max(len(m) for m, *_ in rows))
        header = f"{'Model':<{name_w}}  {'PPL(yours)':>12}  {'PPL(real)':>12}  {'RelPPL':>9}"
        line   = "-" * len(header)
        print("\nPerplexity — Averages")
        print(header)
        print(line)
        for m, my, ref, rel in rows:
            print(f"{m:<{name_w}}  {_fmt(my):>12}  {_fmt(ref):>12}  {_fmt(rel):>9}")



def assign_domain_by_embeddings(
    src_val_texts: List[str],
    tgt_val_texts: List[str],
    sample_texts: List[str],
    embedder_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    margin: float = 0.02,
    device: str = None,
) -> Dict[str, np.ndarray]:
    """
    Classify each sample as 'src', 'tgt', or 'ambiguous' by cosine similarity to
    domain centroids built from the provided validation texts.

    Returns a dict with:
      - 'labels': np.ndarray[str] of length len(sample_texts)
      - 'cos_src': np.ndarray[float] cosine to source centroid
      - 'cos_tgt': np.ndarray[float] cosine to target centroid
      - 'src_proto': np.ndarray[float] source centroid embedding (L2-normalized)
      - 'tgt_proto': np.ndarray[float] target centroid embedding (L2-normalized)
    """
    if not src_val_texts or not tgt_val_texts:
        raise ValueError("Both src_val_texts and tgt_val_texts must be non-empty.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _l2n(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

    # Build an encoder with a clean .encode API and mean-pool fallback.
    def _get_encoder():
        try:
            from sentence_transformers import SentenceTransformer
            enc = SentenceTransformer(embedder_model_id, device=device)
            def encode(texts: List[str]) -> np.ndarray:
                embs = enc.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                return embs
            return encode
        except Exception:
            from transformers import AutoTokenizer, AutoModel
            tok = AutoTokenizer.from_pretrained(embedder_model_id, use_fast=True)
            mdl = AutoModel.from_pretrained(embedder_model_id).to(device).eval()
            @torch.no_grad()
            def encode(texts: List[str]) -> np.ndarray:
                out = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    inputs = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                    reps = mdl(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    reps = (reps * mask).sum(1) / mask.sum(1).clamp_min(1e-9)  # mean pool
                    out.append(reps.cpu().numpy())
                return np.concatenate(out, axis=0)
            return encode

    encode = _get_encoder()

    # Embed and build L2-normalized centroids.
    src_embs = _l2n(encode(src_val_texts))
    tgt_embs = _l2n(encode(tgt_val_texts))
    src_proto = _l2n(src_embs.mean(axis=0, keepdims=True))[0]
    tgt_proto = _l2n(tgt_embs.mean(axis=0, keepdims=True))[0]

    # Classify samples by cosine similarity to both prototypes.
    samp_embs = _l2n(encode(sample_texts))
    cos_src = samp_embs @ src_proto
    cos_tgt = samp_embs @ tgt_proto
    diff = cos_src - cos_tgt

    labels = np.where(diff >  margin, "src",
              np.where(diff < -margin, "tgt", "ambiguous"))

    return {
        "labels": labels,
        "cos_src": cos_src,
        "cos_tgt": cos_tgt,
        "src_proto": src_proto,
        "tgt_proto": tgt_proto,
    }


def print_domain_results(results, sample_texts, truncate=120):
    labels  = results["labels"]
    cos_src = results["cos_src"]
    cos_tgt = results["cos_tgt"]
    diff    = cos_src - cos_tgt  # positive => closer to src

    # Summary
    counts = Counter(labels)
    total = len(labels)
    print(f"Counts: {dict(counts)}  (total={total})")
    print(f"Means:  cos_src={cos_src.mean():.3f}  cos_tgt={cos_tgt.mean():.3f}  Δ={diff.mean():.3f}\n")

    # Table (strongest decisions first)
    header = f"{'idx':>4}  {'label':>10}  {'cos_src':>7}  {'cos_tgt':>7}  {'Δ':>6}  text"
    print(header)
    print("-" * len(header))
    for i in range(len(labels)):
        text = sample_texts[i].replace("\n", " ")
        if len(text) > truncate:
            text = text[:truncate-1] + "…"
        print(f"{i:4d}  {labels[i]:>10}  {cos_src[i]:7.3f}  {cos_tgt[i]:7.3f}  {diff[i]:6.3f}  {text}")


# --- Whole-sample diversity & memorization metrics ---------------------------
def compute_self_bertscore_f1(texts: List[str],
                              model_type: str = "roberta-large",
                              batch_size: int = 16,
                              sample_hyps: int = 200,
                              sample_refs: int = 200,
                              seed: int = 0) -> float:
    """
    Self-BERTScore-F1 (↓ = more diverse): for each sample, score against the rest as references, then average.
    Uses bert-score if available; otherwise returns NaN.
    """
    try:
        from bert_score import score as bertscore
    except Exception as e:
        print(f"[WARN] Self-BERTScore unavailable (bert-score not installed?): {e}")
        return float("nan")

    import random, numpy as np
    n = len(texts)
    if n < 3:
        return float("nan")
    rng = random.Random(seed)
    idx_hyps = rng.sample(range(n), k=min(sample_hyps, n))
    f1_all = []
    for i in idx_hyps:
        pool = [j for j in range(n) if j != i]
        idx_refs = rng.sample(pool, k=min(sample_refs, len(pool)))
        # Repeat candidate once per reference; mean F1 across refs
        cands = [texts[i]] * len(idx_refs)
        refs  = [texts[j] for j in idx_refs]
        try:
            _, _, F1 = bertscore(cands, refs, lang="en", rescale_with_baseline=False,
                                 model_type=model_type, batch_size=batch_size, verbose=False)
            f1_all.append(float(F1.mean().item()))
        except Exception as e:
            print(f"[WARN] BERTScore failed for one hypothesis: {e}")
    return float(np.mean(f1_all)) if f1_all else float("nan")

def _ks(real_n): 
        base = (1,5,10)
        return tuple(max(1, min(k, max(1, real_n-1))) for k in base)

def compute_miss_from_embeddings(emb: np.ndarray,
                                 sample_pairs: int = 20000,
                                 seed: int = 0) -> Dict[str, float]:
    """
    MISS (↓ = more diverse): mean pairwise cosine similarity and 95th percentile.
    `emb` must be L2-normalized sentence embeddings (use embed_texts provided above).
    """
    import numpy as np, random
    m = emb.shape[0]
    if m < 2:
        return {"mean": float("nan"), "p95": float("nan")}
    rng = random.Random(seed)
    pairs = set()
    cap = min(sample_pairs, m * (m - 1) // 2)
    while len(pairs) < cap:
        i = rng.randrange(m); j = rng.randrange(m)
        if i == j: 
            continue
        if i > j: 
            i, j = j, i
        pairs.add((i, j))
    sims = np.fromiter((float(emb[i] @ emb[j]) for (i, j) in pairs), dtype=np.float32)
    return {"mean": float(np.mean(sims)), "p95": float(np.percentile(sims, 95))}


def compute_topic_entropy(emb: np.ndarray, K: Optional[int] = None, seed: int = 0) -> float:
    """
    Topic/Cluster Entropy (↑ = broader coverage):
      - KMeans on embeddings into K topics (default K = 20 or sqrt(N), capped to N).
      - Normalized entropy H / log K.
    """
    import numpy as np, math
    if emb.shape[0] == 0:
        return float("nan")
    if K is None:
        K = 20 if emb.shape[0] >= 20 else max(2, int(round(math.sqrt(emb.shape[0]))))
        K = min(K, emb.shape[0])
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        print(f"[WARN] sklearn not available for KMeans: {e}")
        return float("nan")
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels = km.fit_predict(emb)
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    p = counts / max(1.0, counts.sum())
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    return H / float(np.log(K)) if K > 1 else float("nan")


def _words(text: str) -> List[str]:
    import re
    return re.findall(r"\w+", text.lower())


def compute_mattr(texts: List[str], window: int = 50) -> float:
    """
    MATTR (↑ = more diverse vocabulary). Word-level moving-average TTR with window size.
    Average across samples.
    """
    import numpy as np
    vals = []
    for t in texts:
        toks = _words(t)
        L = len(toks)
        if L == 0:
            continue
        if L <= window:
            vals.append(len(set(toks)) / L)
            continue
        scores = []
        for i in range(0, L - window + 1):
            w = toks[i:i+window]
            scores.append(len(set(w)) / float(window))
        if scores:
            vals.append(float(np.mean(scores)))
    return float(np.mean(vals)) if vals else float("nan")


def compute_repetition_stats(idxs_samples: Iterable[Iterable[int]], n: int = 8, pad_id: Optional[int] = None
                            ) -> Dict[str, Any]:
    """
    Repetition-8 (↓): % tokens covered by any repeated n-gram within a sample; also max contiguous repeated span.
    Returns mean and per-sample list.
    """
    import numpy as np
    rates, spans = [], []
    for ids_ in idxs_samples:
        ids = list(ids_.tolist() if hasattr(ids_, "tolist") else ids_)
        if pad_id is not None:
            ids = [x for x in ids if x != pad_id]
        L = len(ids)
        if L < n:
            rates.append(0.0)
            spans.append(0)
            continue
        starts = {}
        for i in range(L - n + 1):
            ng = tuple(ids[i:i+n])
            starts.setdefault(ng, []).append(i)
        mask = [False] * L
        for pos in starts.values():
            if len(pos) >= 2:
                for s in pos:
                    for j in range(s, s + n):
                        mask[j] = True
        # longest contiguous True-span
        best = cur = 0
        for m in mask:
            if m:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        rate = sum(mask) / float(L) if L else 0.0
        rates.append(rate)
        spans.append(best)
    return {
        "mean_rate": float(np.mean(rates)) if rates else float("nan"),
        "p95_rate": float(np.percentile(rates, 95)) if rates else float("nan"),
        "max_repeat_span": int(max(spans) if spans else 0),
        "per_sample": rates,
    }


def _iter_token_ids_from_any(idxs_samples, pad_id: Optional[int]) -> List[List[int]]:
    """Normalize idxs_samples into a list of token-id lists with pads removed."""
    out = []
    if isinstance(idxs_samples, torch.Tensor):
        for row in idxs_samples.detach().cpu():
            ids = row.tolist()
            if pad_id is not None:
                ids = [x for x in ids if x != pad_id]
            out.append(ids)
    elif isinstance(idxs_samples, (list, tuple)) and idxs_samples and not isinstance(idxs_samples[0], str):
        for seq in idxs_samples:
            ids = list(seq.tolist() if hasattr(seq, "tolist") else seq)
            if pad_id is not None:
                ids = [x for x in ids if x != pad_id]
            out.append(ids)
    else:
        out = []
    return out


def build_training_ngram_index(train_ids: Iterable[Iterable[int]], L: int, pad_id: Optional[int]) -> set:
    """Hash set of all length-L token n-grams in the training set (for copy/memorization checks)."""
    idx = set()
    for seq in train_ids:
        ids = list(seq.tolist() if hasattr(seq, "tolist") else seq)
        if pad_id is not None:
            ids = [x for x in ids if x != pad_id]
        for i in range(0, max(0, len(ids) - L + 1)):
            idx.add(tuple(ids[i:i+L]))
    return idx


def compute_memorization_rate(idxs_samples, train_ids, L: int = 50, pad_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Memorization (↑ = worse): % of generated samples that contain ANY exact length-L substring present in training.
    Also returns per-sample boolean flags.
    """
    import numpy as np
    idx = build_training_ngram_index(train_ids, L=L, pad_id=pad_id)
    flags = []
    for seq in _iter_token_ids_from_any(idxs_samples, pad_id):
        found = any(tuple(seq[i:i+L]) in idx for i in range(0, max(0, len(seq) - L + 1)))
        flags.append(bool(found))
    rate = float(np.mean(flags)) if flags else float("nan")
    return {"rate": rate, "flags": flags, "L": L}


def print_overall_summary_in_table(
    out: Dict[str, Dict],
    cls_probs: Optional[torch.Tensor],
    cls_logits: Optional[torch.Tensor],
    mauve_weighted: Optional[float],
    model_ids: Optional[Iterable[str]] = None,
    *,
    prob_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Prints a single-row summary:
      Weighted MAUVE | Target % (<thr) | Avg logit | Avg prob | PPL[model]... (one column per model)

    Returns:
      {
        "weighted_mauve": float,
        "pct_target": float,  # 0-100
        "avg_classifier_logit": float,
        "avg_classifier_prob": float,
        "ppl_by_model": {model_id: float},
      }
    """
    def _fmt(x, nd=3):
        try:
            xf = float(x)
            return "N/A" if not math.isfinite(xf) else f"{xf:.{nd}f}"
        except Exception:
            return "N/A"

    def _mean_tensor(x) -> float:
        if x is None:
            return float("nan")
        if isinstance(x, torch.Tensor):
            return float(x.detach().to(dtype=torch.float32, device="cpu").mean().item()) if x.numel() else float("nan")
        try:
            arr = np.asarray(x, dtype=np.float32)
            return float(np.nanmean(arr)) if arr.size else float("nan")
        except Exception:
            return float("nan")

    def _sigmoid(t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(t) if isinstance(t, torch.Tensor) else None

    # --- Per-model avg PPLs (already aggregated in `out`) ---
    mids = list(model_ids) if model_ids is not None else list(out.keys())
    ppl_by_model: Dict[str, float] = {}
    for m in mids:
        v = (out.get(m) or {}).get("avg_ppl")
        try:
            v = float(v)
            ppl_by_model[m] = v if math.isfinite(v) else float("nan")
        except Exception:
            ppl_by_model[m] = float("nan")

    # --- Classifier stats: avg logit, avg prob, and % target (< threshold) ---
    if cls_probs is not None and isinstance(cls_probs, torch.Tensor) and cls_probs.numel():
        probs = cls_probs.detach().to(dtype=torch.float32, device="cpu").squeeze(-1)
    elif cls_logits is not None and isinstance(cls_logits, torch.Tensor) and cls_logits.numel():
        probs = _sigmoid(cls_logits.detach().to(dtype=torch.float32, device="cpu").squeeze(-1))
    else:
        probs = None

    avg_logit = _mean_tensor(cls_logits)
    avg_prob  = float(probs.mean().item()) if isinstance(probs, torch.Tensor) and probs.numel() else float("nan")
    if isinstance(probs, torch.Tensor) and probs.numel():
        pct_target = float((probs < prob_threshold).float().mean().item() * 100.0)
    else:
        pct_target = float("nan")

    # --- Weighted MAUVE (already computed upstream) ---
    w_mauve = float(mauve_weighted) if mauve_weighted is not None else float("nan")

    # --- Build and print header dynamically ---
    model_cols = [f"PPL[{m}]" for m in mids]
    header = (
        f"{'Weighted MAUVE':>15}  {'Target %':>9}  {'Avg logit':>10}  {'Avg prob':>10}  "
        + "  ".join(f"{c:>20}" for c in model_cols)
    )
    line = "-" * len(header)
    print("\nOverall summary")
    print(header)
    print(line)
    row = (
        f"{_fmt(w_mauve):>15}  {_fmt(pct_target):>9}  {_fmt(avg_logit):>10}  {_fmt(avg_prob):>10}  "
        + "  ".join(f"{_fmt(ppl_by_model[m]):>20}" for m in mids)
    )
    print(row)

    return {
        "weighted_mauve": w_mauve,
        "pct_target": pct_target,
        "avg_classifier_logit": avg_logit,
        "avg_classifier_prob": avg_prob,
        "ppl_by_model": ppl_by_model,
    }

# --- Diversity metrics --------------------------------------------------------
def distinct_n_from_ids(idxs_samples, n=2, pad_id=None):
    """Return (micro, macro) distinct-n over token ids."""
    def ngrams(seq, n):
        return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]
    per_frac, all_ngrams, total_all = [], [], 0
    for t in idxs_samples:
        ids = t.tolist() if hasattr(t, "tolist") else list(t)
        if pad_id is not None:
            ids = [x for x in ids if x != pad_id]
        ng = ngrams(ids, n)
        total = len(ng)
        if total == 0:
            per_frac.append(0.0)
            continue
        uniq = len(set(ng))
        per_frac.append(uniq / total)
        all_ngrams.extend(ng)
        total_all += total
    micro = (len(set(all_ngrams)) / total_all) if total_all else 0.0
    macro = float(sum(per_frac) / max(1, len(per_frac)))
    return micro, macro

def _l2_normalize(a):
    import numpy as np
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / n

def embed_texts(texts, model_id="sentence-transformers/all-MiniLM-L6-v2",
                batch_size=64, device=None):
    """Sentence embeddings with ST; fallback to mean-pooled HF encoder."""
    import torch, numpy as np
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer(model_id, device=device)
        embs = enc.encode(texts, batch_size=batch_size, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=False)
        return _l2_normalize(embs)
    except Exception:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        mdl = AutoModel.from_pretrained(model_id).to(device).eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inp = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                h = mdl(**inp).last_hidden_state
                m = inp["attention_mask"].unsqueeze(-1)
                reps = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)  # mean pool
                outs.append(reps.cpu().numpy())
        return _l2_normalize(np.concatenate(outs, 0))
    
def nearest_neighbor_similarities(emb: np.ndarray) -> np.ndarray:
    """For each sample, max cosine sim to any other sample (↑ = more redundancy)."""
    import numpy as np
    if emb.shape[0] < 2:
        return np.full((emb.shape[0],), np.nan, dtype=np.float32)
    # emb is L2-normalized -> cosine = dot
    S = emb @ emb.T
    np.fill_diagonal(S, -1.0)  # exclude self
    return S.max(axis=1).astype(np.float32)

def redundancy_at_thresholds(nns: np.ndarray, thresholds=(0.85, 0.90, 0.95)) -> dict:
    import numpy as np
    out = {"mean": float(np.nanmean(nns)), "p95": float(np.nanpercentile(nns, 95))}
    for t in thresholds:
        out[f"rate@{t:.2f}"] = float(np.nanmean(nns >= t))
    return out

def knn_coverage_precision(real_emb, gen_emb, k=5, metric="cosine"):
    """
    Coverage (recall): % real points with a generated neighbor inside a generated k-NN radius.
    Precision: % generated points that fall inside a real k-NN radius.
    """
    import numpy as np
    try:
        from sklearn.neighbors import NearestNeighbors
        # radii around REAL points (k-th neighbor among REAL)
        k_r = min(k+1, len(real_emb))
        nn_r = NearestNeighbors(n_neighbors=k_r, metric=metric).fit(real_emb)
        dist_rr, _ = nn_r.kneighbors(real_emb)                 # [N_r, k_r]
        r_radii = dist_rr[:, -1] if dist_rr.shape[1] > 1 else dist_rr[:, 0]

        # precision: gen -> nearest real; within that real's radius?
        dist_gr, idx_gr = nn_r.kneighbors(gen_emb, n_neighbors=1, return_distance=True)
        precision = float(np.mean(dist_gr[:, 0] <= r_radii[idx_gr[:, 0]])) if len(gen_emb) else float("nan")

        # radii around GENERATED points (k-th neighbor among GEN)
        if len(gen_emb) >= 2:
            k_g = max(1, min(k, len(gen_emb)))
            nn_g = NearestNeighbors(n_neighbors=k_g, metric=metric).fit(gen_emb)
            dist_gg, _ = nn_g.kneighbors(gen_emb, n_neighbors=k_g)
            g_radii = dist_gg[:, -1]
            # coverage: real -> nearest gen; within that gen's radius?
            dist_rg, idx_rg = nn_g.kneighbors(real_emb, n_neighbors=1, return_distance=True)
            coverage = float(np.mean(dist_rg[:, 0] <= g_radii[idx_rg[:, 0]]))
        else:
            coverage = float("nan")
        return {"precision": precision, "coverage": coverage}
    except Exception:
        # Fallback (O(N^2)) cosine distance
        def cdist_cos(A, B):
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
            return 1.0 - A @ B.T
        D_rr = cdist_cos(real_emb, real_emb)
        np.fill_diagonal(D_rr, np.inf)
        r_radii = np.partition(D_rr, kth=min(k, D_rr.shape[1]-1), axis=1)[:, min(k, D_rr.shape[1]-1)]
        D_gr = cdist_cos(gen_emb, real_emb)
        precision = float((D_gr.min(axis=1) <= r_radii[D_gr.argmin(axis=1)]).mean()) if len(gen_emb) else float("nan")
        D_gg = cdist_cos(gen_emb, gen_emb)
        np.fill_diagonal(D_gg, np.inf)
        kth = min(k-1, max(0, D_gg.shape[1]-1))
        g_radii = np.partition(D_gg, kth=kth, axis=1)[:, kth] if len(gen_emb) >= 2 else None
        if g_radii is None:
            coverage = float("nan")
        else:
            D_rg = cdist_cos(real_emb, gen_emb)
            coverage = float((D_rg.min(axis=1) <= g_radii[D_rg.argmin(axis=1)]).mean())
        return {"precision": precision, "coverage": coverage}

def cluster_concentration_stats(emb: np.ndarray, K: Optional[int] = None, seed: int = 0) -> Dict[str, float]:
    """
    Cluster embeddings then measure concentration:
      - norm_entropy (↑ = broader spread), HHI (↑ = more concentrated), effective_K = 1/HHI.
    """
    import numpy as np, math
    if emb.shape[0] < 2:
        return {"K": float("nan"), "norm_entropy": float("nan"), "hhi": float("nan"),
                "effective_K": float("nan"), "used_K": 0}
    if K is None:
        K = 20 if emb.shape[0] >= 20 else max(2, int(round(math.sqrt(emb.shape[0]))))
        K = min(K, emb.shape[0])

    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        print(f"[WARN] sklearn not available for KMeans (cluster concentration skipped): {e}")
        return {"K": float("nan"), "norm_entropy": float("nan"), "hhi": float("nan"),
                "effective_K": float("nan"), "used_K": 0}

    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels = km.fit_predict(emb)
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    p = counts / max(1.0, counts.sum())
    used = int((p > 0).sum())

    # Entropy and concentration
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    norm_entropy = H / float(np.log(K)) if K > 1 else float("nan")
    hhi = float((p * p).sum())                                # Herfindahl–Hirschman Index
    effective_K = float(1.0 / hhi) if hhi > 0 else float("nan")
    return {"K": float(K), "norm_entropy": norm_entropy, "hhi": hhi, "effective_K": effective_K, "used_K": used}


def _print_diversity_summary(distinct, prd):
    d1, d2, d3 = distinct["d1"], distinct["d2"], distinct["d3"]
    print("\nDiversity — distinct-n (micro, macro)")
    print(f"distinct-1: {d1['micro']:.3f}, {d1['macro']:.3f}")
    print(f"distinct-2: {d2['micro']:.3f}, {d2['macro']:.3f}  <-- watch for mode collapse")
    print(f"distinct-3: {d3['micro']:.3f}, {d3['macro']:.3f}")
    print(f"Coverage (recall): {prd['coverage']:.3f}   Precision: {prd['precision']:.3f}")

# ---------- Self-BLEU (sentence-level, smoothed) ----------
def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _sent_bleu_smooth(hyp, refs, max_n=4):
    import math, collections
    # Modified n-gram precision with smoothing (+1)
    log_p = 0.0
    for n in range(1, max_n+1):
        hyp_ngr = _ngrams(hyp, n)
        if len(hyp_ngr) == 0:
            return 0.0
        hyp_counts = collections.Counter(hyp_ngr)
        # reference max counts
        max_ref = collections.Counter()
        for r in refs:
            rc = collections.Counter(_ngrams(r, n))
            for k, v in rc.items():
                if v > max_ref[k]:
                    max_ref[k] = v
        clipped = sum(min(c, max_ref[k]) for k, c in hyp_counts.items())
        total = len(hyp_ngr)
        p = (clipped + 1.0) / (total + 1.0)  # smoothing
        log_p += (1.0/max_n) * math.log(p)

    # Brevity penalty
    hyp_len = len(hyp)
    ref_lens = [len(r) for r in refs]
    closest = min(ref_lens, key=lambda L: (abs(L - hyp_len), L))
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > closest else math.exp(1.0 - closest / max(hyp_len, 1))
    return bp * math.exp(log_p)

def _decode_first(dataset, tokenizer, cap=-1, skip_special_tokens=True):
    """Decode up to `cap` items from a HF dataset's 'input_ids' column; cap<0 -> all."""
    if dataset is None:
        return []
    if cap is None or cap < 0:
        ids = dataset["input_ids"]
    else:
        ids = dataset[:cap]["input_ids"]
    return tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)


def self_bleu_from_ids(idxs_samples, *, tokenizer, max_n=4, sample_hyps=200, sample_refs=200, seed=0):
    import numpy as np, random
    pad_id = tokenizer.pad_token_id
    seqs = []
    for t in idxs_samples:
        ids = t.tolist() if hasattr(t, "tolist") else list(t)
        if pad_id is not None:
            ids = [x for x in ids if x != pad_id]
        seqs.append(ids)
    n = len(seqs)
    if n < 3:
        return float("nan")
    rng = random.Random(seed)
    hyps = rng.sample(range(n), k=min(sample_hyps, n))
    scores = []
    for i in hyps:
        pool = [j for j in range(n) if j != i]
        refs_idx = rng.sample(pool, k=min(sample_refs, len(pool)))
        score = _sent_bleu_smooth(seqs[i], [seqs[j] for j in refs_idx], max_n=max_n)
        scores.append(score)
    return float(np.mean(scores))

# ---------- Pairwise self-similarity on embeddings ----------
def pairwise_self_similarity_stats(emb, num_pairs=10000, seed=0, dup_thresh=0.95):
    import numpy as np, random
    m = emb.shape[0]
    if m < 2:
        return {"mean": float("nan"), "p90": float("nan"), "max": float("nan"), "dup_rate": float("nan")}
    rng = random.Random(seed)
    pairs = set()
    for _ in range(min(num_pairs, m*(m-1)//2)):
        i = rng.randrange(m); j = rng.randrange(m)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))
        if len(pairs) >= num_pairs:
            break
    sims = [float(emb[i] @ emb[j]) for (i, j) in pairs]  # cosine since emb are L2-normalized
    sims = np.asarray(sims, dtype=np.float32)
    return {
        "mean": float(np.mean(sims)),
        "p90": float(np.percentile(sims, 90)),
        "max": float(np.max(sims)),
        "dup_rate": float(np.mean(sims >= dup_thresh)),
    }

# ---------- Multi-k PR (precision/coverage) ----------
def multi_k_pr(real_emb, gen_emb, ks=(1,3,5,10,20)):
    out = []
    for k in ks:
        pr = knn_coverage_precision(real_emb, gen_emb, k=k, metric="cosine")
        out.append({"k": k, "precision": pr["precision"], "coverage": pr["coverage"]})
    return out

def compute_mauve_valid_vs_samples(
    valid_loader,
    tokenizer,
    generated,                      # list[str] OR Tensor[N, T] OR list[Tensor[T]]
    *,
    device_id: int = 0,             # set to 0,1,... for GPU; use -1 for CPU
    max_text_length: int = 1024,
    skip_special_tokens: bool = True,
):
    """
    Build human references from `valid_loader` and compare to `generated` with MAUVE.

    Returns:
        mauve_score (float), mauve_result (the full object returned by mauve.compute_mauve)
    """
    # 1) Normalize generated samples -> List[str]
    if isinstance(generated, torch.Tensor):
        if generated.ndim != 2:
            raise ValueError("`generated` Tensor must have shape [N, T].")
        gen_texts = tokenizer.batch_decode(
            generated, skip_special_tokens=skip_special_tokens
        )
    elif isinstance(generated, (list, tuple)):
        if len(generated) == 0:
            return float("nan"), None
        if isinstance(generated[0], str):
            gen_texts = list(generated)
        else:
            # list of token-id sequences (Tensor or list[int])
            gen_texts = [
                tokenizer.decode(g, skip_special_tokens=skip_special_tokens)
                for g in generated
            ]
    else:
        raise TypeError("`generated` must be list[str], Tensor[N,T], or list[Tensor[T]].")

    n_needed = len(gen_texts)

    # 2) Collect reference texts from the validation loader
    human_refs = []
    valid_iter = iter(valid_loader)
    while len(human_refs) < n_needed:
        try:
            batch = next(valid_iter)
        except StopIteration:
            # loop back through the loader if we ran out
            valid_iter = iter(valid_loader)
            batch = next(valid_iter)

        input_ids = batch["input_ids"]
        # Decode this batch and append only what's needed
        decoded = tokenizer.batch_decode(
            input_ids, skip_special_tokens=skip_special_tokens
        )
        remaining = n_needed - len(human_refs)
        human_refs.extend(decoded[:remaining])

    # 3) Compute MAUVE

    # If you want an automatic CPU fallback:
    # if (device_id is None) and (not torch.cuda.is_available()): device_id = -1
    result = mauve.compute_mauve(
        p_text=human_refs,
        q_text=gen_texts,
        device_id=device_id,
        max_text_length=max_text_length,
        verbose=False,
    )
    return result.mauve, result

# --- NEW: small utilities -----------------------------------------------------
def _safe(name: str, fn, *, fallback=None):
    """Run fn() and catch exceptions; print a compact warning and return fallback."""
    try:
        return fn()
    except Exception as e:
        print(f"[WARN] {name} failed: {e.__class__.__name__}: {e}")
        return fallback

def _fmtf(x, nd=3):
    if x is None:
        return "N/A"
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "N/A"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _truncate(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else (s[: n - 1] + "…")

# --- UPDATED: classifier helper (returns tensors, prints summary nicely) ------
def run_classifier_on_samples(config, tokenizer, idxs_samples):
    """
    Returns:
      probs:  Tensor[N, 1] on CUDA (or None on failure)
      logits: Tensor[N, 1] on CUDA (or None on failure)
    """
    def _do():
        ti_cls, loaded = utils.build_or_load(
            classifier.Classifier,
            dict(config=config, tokenizer=tokenizer, train_time_independent=True),
            ckpt_path=config.classifier_ti.ckpt_path,
            freeze=True,
        )
        if not loaded:
            print(
                "Time-independent classifier checkpoint not found. "
                f"Check path: {config.classifier_ti.ckpt_path}"
            )
        ti_cls.eval().to("cuda")

        probs = []
        logits = []
        with torch.no_grad():
            for idx_sample in idxs_samples:
                logit = ti_cls(idx_sample.unsqueeze(0))
                prob = torch.sigmoid(logit)
                logits.append(logit)
                probs.append(prob)
        probs = torch.cat(probs, dim=0) if probs else torch.empty(0, 1, device="cuda")
        logits = torch.cat(logits, dim=0) if logits else torch.empty(0, 1, device="cuda")

        # Compact summary
        print("Classifier: tgt=0, src=1")
        if probs.numel():
            print(
                f"avg prob(src)= {_fmtf(probs.mean().item(), 4)}   "
                f"avg logit= {_fmtf(logits.mean().item(), 4)}   "
                f"src≥0.5: {int((probs>=0.5).sum().item())}/{probs.shape[0]}"
            )
        else:
            print("No classifier outputs to summarize.")
        return probs, logits

    return _safe("Classifier inference", _do, fallback=(None, None))
def naeem_coverage(real_emb, gen_emb, k=5, metric="cosine"):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    if len(real_emb) == 0 or len(gen_emb) == 0:
        return float("nan")
    k_r = min(k+1, len(real_emb))  # +1 to include self in sklearn
    nn_r = NearestNeighbors(n_neighbors=k_r, metric=metric).fit(real_emb)
    dist_rr, _ = nn_r.kneighbors(real_emb)                # [N_real, k_r]
    r_radii = dist_rr[:, -1] if dist_rr.shape[1] > 1 else dist_rr[:, 0]
    nn_g = NearestNeighbors(n_neighbors=1, metric=metric).fit(gen_emb)
    dist_rg, _ = nn_g.kneighbors(real_emb, n_neighbors=1, return_distance=True)
    return float((dist_rg[:, 0] <= r_radii).mean())

def nearest_dist_to_set(A, B, metric="cosine"):
    # For each point in A, distance to the nearest point in B
    from sklearn.neighbors import NearestNeighbors
    if len(A) == 0 or len(B) == 0:
        return None
    nn = NearestNeighbors(n_neighbors=1, metric=metric).fit(B)
    dists, _ = nn.kneighbors(A, n_neighbors=1, return_distance=True)
    return dists[:, 0]
# --- NEW: unified per-sample table printer -----------------------------------
def print_per_sample_table(
    *,
    texts: List[str],
    labels: Optional[typing.Iterable[str]],
    cos_src: Optional[typing.Iterable[float]],
    cos_tgt: Optional[typing.Iterable[float]],
    cls_probs: Optional[torch.Tensor],
    cls_logits: Optional[torch.Tensor],
    per_model_ppl: Dict[str, List[Optional[float]]],
    text_width: int = 120,
):
    """
    Print a single table with metrics per sample.
    Columns:
      idx | label | cls_p(src) | cls_logit | PPL[model...] | cos_src | cos_tgt | Δ | text
    """
    n = len(texts)
    labels = list(labels) if labels is not None else ["?"] * n
    cos_src = list(cos_src) if cos_src is not None else [None] * n
    cos_tgt = list(cos_tgt) if cos_tgt is not None else [None] * n

    # Normalize classifier tensors to CPU lists (or Nones)
    def _to_list(x):
        if x is None:
            return [None] * n
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return [None] * n
            return x.squeeze(-1).detach().cpu().tolist()
        return list(x)

    cls_p = _to_list(cls_probs)
    cls_l = _to_list(cls_logits)

    # Ensure per-model PPL arrays of length n
    model_ids = list(per_model_ppl.keys())
    for m in model_ids:
        arr = per_model_ppl[m]
        if arr is None:
            per_model_ppl[m] = [None] * n
        elif len(arr) != n:
            # pad or trim to length n
            per_model_ppl[m] = (arr + [None] * n)[:n]

    # Build header
    model_cols = [f"PPL[{m}]" for m in model_ids]
    header_parts = ["idx", "label", "cls_p(src)", "cls_logit"] + model_cols + ["cos_src", "cos_tgt", "Δ", "text"]
    # Column widths
    w = {
        "idx": 4,
        "label": max(5, max(len(str(x)) for x in labels)),
        "cls_p(src)": 10,
        "cls_logit": 10,
        "cos_src": 7,
        "cos_tgt": 7,
        "Δ": 6,
    }
    model_w = 12
    text_w = text_width

    # Print header
    header = (
        f"{'idx':>{w['idx']}}  "
        f"{'label':>{w['label']}}  "
        f"{'cls_p(src)':>{w['cls_p(src)']}}  "
        f"{'cls_logit':>{w['cls_logit']}}  "
        + "  ".join(f"{c:>{model_w}}" for c in model_cols)
        + f"  {'cos_src':>{w['cos_src']}}  {'cos_tgt':>{w['cos_tgt']}}  {'Δ':>{w['Δ']}}  text"
    )
    print("\nPer-sample metrics")
    print(header)
    print("-" * len(header))

    for i in range(n):
        delta = None
        if cos_src[i] is not None and cos_tgt[i] is not None:
            delta = float(cos_src[i]) - float(cos_tgt[i])
        row = (
            f"{i:>{w['idx']}}  "
            f"{labels[i]:>{w['label']}}  "
            f"{_fmtf(cls_p[i]):>{w['cls_p(src)']}}  "
            f"{_fmtf(cls_l[i]):>{w['cls_logit']}}  "
            + "  ".join(f"{_fmtf(per_model_ppl[m][i]):>{model_w}}" for m in model_ids)
            + f"  {_fmtf(cos_src[i]):>{w['cos_src']}}  {_fmtf(cos_tgt[i]):>{w['cos_tgt']}}  {_fmtf(delta):>{w['Δ']}}  "
            f"{_truncate(texts[i], text_w)}"
        )
        print(row)

# --- UPDATED: main evaluation function ---------------------------------------
def evaluate_the_generated_samples(
    idxs_samples,  # Tensor[N, T] or list[str]/list[list[int]]
    config,
    tokenizer: AutoTokenizer,
    *,
    model_ids: Tuple[str, ...] = ("gpt2", "facebook/opt-2.7b", "facebook/opt-6.7b"),
    cache_dir: Optional[str] = "/n/netscratch/dominici_lab/Lab/juliank/hf_cache/huggingface",
    device_id: int = 0,
    ref_cap: int = 256,
    text_width: int = 120,
) -> Dict[str, Any]:
    """
    Evaluate generated samples with:
      1) PPL vs baseline LMs (robust to failures)
      2) Domain assignment by embeddings
      3) Optional classifier summary and per-sample scores
      4) MAUVE vs src validation set
    Then print a single per-sample table aggregating the metrics.
    """
    # Skip for semi-AR
    if getattr(config.sampling, "semi_ar", False):
        print("Semi-AR sampling enabled; skipping evaluation.")
        return {}

    # --- Load validation data (safe) ---
    loaders = _safe(
        "Load dataloaders",
        lambda: dataloader.get_dataloaders(config, tokenizer, domain="src"),
        fallback=(None, None),
    )
    _, valid_src = loaders if isinstance(loaders, tuple) else (None, None)

    loaders_tgt = _safe(
        "Load target dataloaders",
        lambda: dataloader.get_dataloaders(config, tokenizer, domain="tgt"),
        fallback=(None, None),
    )
    _, valid_tgt = loaders_tgt if isinstance(loaders_tgt, tuple) else (None, None)
    train_ds_src, valid_ds_src = dataloader.get_dataloaders(config, tokenizer, domain="src")
    # --- Decode samples (safe) ---
    gen_texts = _safe(
        "Decode generated samples",
        lambda: tokenizer.batch_decode(idxs_samples, skip_special_tokens=True),
        fallback=[],
    )
    n = len(gen_texts)

    # --- Build small target reference for PPL baselines ---
    tgt_ref_texts = _safe(
        "Prepare target reference texts",
        lambda: tokenizer.batch_decode(
            valid_tgt.dataset[:ref_cap]["input_ids"], skip_special_tokens=True
        ),
        fallback=[],
    )
    loaders_tgt = _safe(
        "Load target dataloaders",
        lambda: dataloader.get_dataloaders(config, tokenizer, domain="tgt"),
        fallback=(None, None),
    )
    train_tgt, valid_tgt = loaders_tgt if isinstance(loaders_tgt, tuple) else (None, None)
    train_texts = _safe( "Decode target TRAIN texts",lambda: _decode_first(train_tgt.dataset, tokenizer, cap=ref_cap), fallback=[] )

    # --- PPL (safe) ---
    out = _safe(
        "PPL on generated samples",
        lambda: eval_sample_gen_ppl(
            gen_texts, models=model_ids, cache_dir=cache_dir, use_safetensors=True, allow_bin=False
        ),
        fallback={},
    )
    out_ref = _safe(
        "PPL on reference texts",
        lambda: eval_sample_gen_ppl(
            tgt_ref_texts, models=model_ids, cache_dir=cache_dir, use_safetensors=True, allow_bin=False
        ),
        fallback={},
    )
    val_texts = tgt_ref_texts
    src_train_texts = _safe(
        "Decode SRC TRAIN texts",
        lambda: _decode_first(valid_ds_src.dataset, tokenizer, cap=ref_cap),
        fallback=[],
    )
    src_val_texts = _safe(
        "Decode SRC VAL texts",
        lambda: tokenizer.batch_decode(
            valid_src.dataset[:ref_cap]["input_ids"],  # reuse the valid_src you already loaded above
            skip_special_tokens=True,
        ),
        fallback=[],
    )
    # Average PPL tables if available
    if out and out_ref:
        print_ppl_table(out, out_ref, model_ids)
        print_ppl_report(out, model_ids)

    # --- Classifier (optional; safe) ---
    eval_cfg = getattr(config, "eval", None)
    if isinstance(eval_cfg, dict):
        classify_flag = eval_cfg.get("classify_samples", False)
    else:
        classify_flag = bool(getattr(eval_cfg, "classify_samples", False))

    cls_probs, cls_logits = (None, None)
    if classify_flag:
        cls_probs, cls_logits = run_classifier_on_samples(config, tokenizer, idxs_samples)

    # --- Domain assignment (safe) ---
    def _prep_domain_texts():
        number_of_examples = 16 if bool(getattr(config, "debug", False)) else -1
        src_val_ids = valid_src.dataset["input_ids"][:number_of_examples]
        tgt_val_ids = valid_tgt.dataset["input_ids"][:number_of_examples]
        src_val_texts = tokenizer.batch_decode(src_val_ids, skip_special_tokens=True)
        tgt_val_texts = tokenizer.batch_decode(tgt_val_ids, skip_special_tokens=True)
        return src_val_texts, tgt_val_texts

    dom = None
    src_texts, tgt_texts = _safe("Prepare domain texts", _prep_domain_texts, fallback=(None, None))
    if src_texts and tgt_texts and n:
        dom = _safe(
            "Domain assignment",
            lambda: assign_domain_by_embeddings(src_texts, tgt_texts, gen_texts),
            fallback=None,
        )
        # --- MAUVE (classifier-first, bucketed) -----------------------------------
    # Decide labels using classifier first; fallback to embeddings.
    labels = _choose_labels_for_bucketing(n, cls_probs, dom, threshold=0.5)

    # Build indices per bucket
    if labels is not None:
        idx_src = [i for i, l in enumerate(labels) if l == "src"]
        idx_tgt = [i for i, l in enumerate(labels) if l == "tgt"]
    else:
        idx_src, idx_tgt = [], []

    # Subset generated samples by bucket
    gen_src = _subset_samples(idxs_samples, idx_src) if idx_src else None
    gen_tgt = _subset_samples(idxs_samples, idx_tgt) if idx_tgt else None

    base_seed = int(getattr(config, "seed", 2025))
    num_runs = 5 if config.debug == False else 2
    # Compute MAUVE for each bucket against its own validation set
    if gen_src is not None and valid_src is not None:
        m_src = _safe(
            "Balanced MAUVE (src bucket vs src valid)",
            lambda: mauve_balanced_bootstrap(
                valid_src, tokenizer, gen_src, num_runs=num_runs, base_seed=base_seed
            ),
            fallback={"mean": float("nan"), "std": float("nan"), "scores": [], "seeds": [], 
                      "n_gen": len(idx_src), "n_ref_total": 0},
        )
    else:
        m_src = {"mean": float("nan"), "std": float("nan"), "scores": [], "seeds": [], 
                 "n_gen": 0, "n_ref_total": 0}

    if gen_tgt is not None and valid_tgt is not None:
        m_tgt = _safe(
            "Balanced MAUVE (tgt bucket vs tgt valid)",
            lambda: mauve_balanced_bootstrap(
                valid_tgt, tokenizer, gen_tgt, num_runs=num_runs, base_seed=base_seed
            ),
            fallback={"mean": float("nan"), "std": float("nan"), "scores": [], "seeds": [], 
                      "n_gen": len(idx_tgt), "n_ref_total": 0},
        )
    else:
        m_tgt = {"mean": float("nan"), "std": float("nan"), "scores": [], "seeds": [], 
                 "n_gen": 0, "n_ref_total": 0}

    # Combine with observed mix
    n_src = len(idx_src)
    n_tgt = len(idx_tgt)
    n_tot = n_src + n_tgt
    p_src = (n_src / n_tot) if n_tot else float("nan")

    if n_src == 0 and n_tgt == 0:
        mauve_weighted = float("nan")
    elif n_src == 0:
        mauve_weighted = m_tgt["mean"]
    elif n_tgt == 0:
        mauve_weighted = m_src["mean"]
    else:
        mauve_weighted = p_src * m_src["mean"] + (1.0 - p_src) * m_tgt["mean"]


    # --- MAUVE (safe) ---
    base_seed = int(getattr(config, "seed", 2025))
    m_src_whole = _safe(
        "Balanced MAUVE src",
        lambda: (mauve_balanced_bootstrap(valid_src, tokenizer, idxs_samples, num_runs=num_runs, base_seed=base_seed)
                 if (valid_src is not None and n) else
                 {"mean": float("nan"), "std": float("nan"), "scores": [], "seeds": [], "n_gen": n, "n_ref_total": 0}),
        fallback={"mean": float("nan"), "std": float("nan"), "scores": [], "seeds": [], "n_gen": n, "n_ref_total": 0},
    )
    m_tgt_whole = _safe(
        "Balanced MAUVE tgt",
        lambda: (mauve_balanced_bootstrap(valid_tgt, tokenizer, idxs_samples, num_runs=num_runs, base_seed=base_seed)
                 if (valid_tgt is not None and n) else
                 {"mean": float("nan"), "std": float("nan"), "scores": [], "seeds": [], "n_gen": n, "n_ref_total": 0}),
        fallback={"mean": float("nan"), "std": float("nan"), "scores": [], "seeds": [], "n_gen": n, "n_ref_total": 0},
    )
    print(f"Buckets: src={n_src}, tgt={n_tgt}, p_src={_fmtf(p_src, 3)}")
    print(f"MAUVE_src  mean±std: {_fmtf(m_src['mean'])} ± {_fmtf(m_src['std'])}  (N_gen={n_src})")
    print(f"MAUVE_tgt  mean±std: {_fmtf(m_tgt['mean'])} ± {_fmtf(m_tgt['std'])}  (N_gen={n_tgt})")
    print(f"MAUVE_weighted:      {_fmtf(mauve_weighted)}")
    print("MAUVE on full")
    print("MAUVE on src  mean±std:", m_src_whole["mean"], m_src_whole["std"])
    print("MAUVE on tgt  mean±std:", m_tgt_whole["mean"], m_tgt_whole["std"])

    # --- Build per-sample table aggregating metrics ---------------------------
    # Gather per-model PPL details aligned to samples
    per_model_ppl = {}
    for m in model_ids:
        details = (out.get(m, {}) or {}).get("details", []) if out else []
        per_model_ppl[m] = [ (d.get("ppl") if isinstance(d, dict) else None) for d in details ]

    labels = dom["labels"] if (dom and "labels" in dom) else None
    cos_src = dom["cos_src"] if (dom and "cos_src" in dom) else None
    cos_tgt = dom["cos_tgt"] if (dom and "cos_tgt" in dom) else None
    _ = print_overall_summary_in_table(out, cls_probs, cls_logits, mauve_weighted, model_ids)
    if n:
        print_per_sample_table(
            texts=gen_texts,
            labels=labels,
            cos_src=cos_src,
            cos_tgt=cos_tgt,
            cls_probs=cls_probs,
            cls_logits=cls_logits,
            per_model_ppl=per_model_ppl,
            text_width=text_width,
        )
    # summary on run
    # --- NEW: Whole-sample diversity metrics ---------------------------------
    # Reuse embeddings if available; otherwise compute once.
    gen_emb  = embed_texts(gen_texts,   model_id="sentence-transformers/all-MiniLM-L6-v2")
    val_emb   = embed_texts(val_texts,   model_id="sentence-transformers/all-MiniLM-L6-v2")
    train_emb = embed_texts(train_texts, model_id="sentence-transformers/all-MiniLM-L6-v2")

    src_train_emb = embed_texts(src_train_texts, model_id="sentence-transformers/all-MiniLM-L6-v2")
    src_val_emb   = embed_texts(src_val_texts,   model_id="sentence-transformers/all-MiniLM-L6-v2")

    print("\n[SRC] Coverage (Naeem et al. 2020)")
    for k in (1, 5, 10):
        cov_tr     = naeem_coverage(src_train_emb, gen_emb, k=k, metric="cosine")
        cov_val    = naeem_coverage(src_val_emb,   gen_emb, k=k, metric="cosine")
        cov_union  = naeem_coverage(np.vstack([src_train_emb, src_val_emb]), gen_emb, k=k, metric="cosine")
        print(f"[SRC] Coverage@{k}: train={cov_tr:.3f}  val={cov_val:.3f}  union={cov_union:.3f}  gap(val-train)={(cov_val-cov_tr):+.3f}")

    # Novel-val coverage for SRC: mask src-val points far from src-train
    src_val2train = nearest_dist_to_set(src_val_emb, src_train_emb, metric="cosine")
    if src_val2train is not None and len(src_val2train):
        tau_s = np.percentile(src_val2train, 70)  # top 30% most novel relative to SRC-train
        mask_s = src_val2train >= tau_s
        if mask_s.any():
            for k in (1, 5, 10):
                cov_src_val_novel = naeem_coverage(src_val_emb[mask_s], gen_emb, k=k, metric="cosine")
                print(f"[SRC] Coverage@{k} on novel-val (≥{tau_s:.3f} from train): {cov_src_val_novel:.3f}")
    # 1) Self-BERTScore-F1 (semantic self-similarity; ↓ better diversity)
    self_bert_f1 = 0

    # 2) MISS (mean intra-set similarity) using cosine on L2-normalized embeddings
    miss = compute_miss_from_embeddings(gen_emb, sample_pairs=20000, seed=base_seed)

    # 3) Topic/Cluster Entropy (normalized)
    topic_ent = compute_topic_entropy(gen_emb, K=None, seed=base_seed)
    cluster = cluster_concentration_stats(gen_emb, K=None, seed=base_seed)

    # 4) MATTR (word-level, length-robust lexical diversity; ↑ is better)
    mattr = compute_mattr(gen_texts, window=50)

    # 5) Repetition-8 (degeneration; ↓ is better)
    rep_stats = compute_repetition_stats(
        _iter_token_ids_from_any(idxs_samples, tokenizer.pad_token_id),
        n=8, pad_id=tokenizer.pad_token_id
    )

    # --- NEW: Optional memorization vs target-domain TRAIN set ----------------
    mem = {"rate": float("nan"), "L": 50, "flags": []}
    try:
        if train_tgt is not None:
            train_ids = train_tgt.dataset["input_ids"]
            mem = compute_memorization_rate(
                idxs_samples, train_ids, L=50, pad_id=tokenizer.pad_token_id
            )
    except Exception as e:
        print(f"[WARN] Memorization metric skipped: {e}")

    # --- Display compact row (requested columns) ------------------------------
    # Quality: MAUVE (weighted) ↑, Perplexity ↓  |  Diversity: Self-BERTScore-F1 ↓, MISS ↓, Topic Entropy ↑, MATTR ↑, Repetition-8 ↓
        # Choose an aggregate PPL (mean over available models)
    try:
        ppl_vals = [float((out.get(m) or {}).get("avg_ppl")) for m in model_ids if m in out]
        ppl_mean = float(np.nanmean(ppl_vals)) if ppl_vals else float("nan")
    except Exception:
        ppl_mean = float("nan")

    # --- Diversity: distinct-n and embedding PR metrics ----------------------
    distinct = {}
    for n_ in (1, 2, 3):
        micro, macro = distinct_n_from_ids(idxs_samples, n=n_, pad_id=tokenizer.pad_token_id)
        distinct[f"d{n_}"] = {"micro": micro, "macro": macro}

    # Use target validation texts as the "real" distribution
    real_emb = embed_texts(tgt_ref_texts, model_id="sentence-transformers/all-MiniLM-L6-v2")
    gen_emb  = embed_texts(gen_texts,     model_id="sentence-transformers/all-MiniLM-L6-v2")
    prd = knn_coverage_precision(real_emb, gen_emb, k=5, metric="cosine")
        # Self-BLEU-4
    self_bleu4 = self_bleu_from_ids(idxs_samples, tokenizer=tokenizer, max_n=4, sample_hyps=200, sample_refs=200)
    print(f"Self-BLEU-4 (↓ better diversity): {self_bleu4:.3f}")

    # Embeddings once
    real_emb = embed_texts(tgt_ref_texts, model_id="sentence-transformers/all-MiniLM-L6-v2")
    gen_emb  = embed_texts(gen_texts,     model_id="sentence-transformers/all-MiniLM-L6-v2")

    # Pairwise self-similarity
    try:
        selfsim = pairwise_self_similarity_stats(gen_emb, num_pairs=10000, dup_thresh=0.95)
        print(f"SelfSim mean={selfsim['mean']:.3f}  p90={selfsim['p90']:.3f}  max={selfsim['max']:.3f}  dup_rate@0.95={selfsim['dup_rate']:.3f}")
    except Exception as e:
        print(f"[WARN] Self-similarity computation failed: {e}")
        selfsim = {}
    train_emb = embed_texts(train_texts)    # or from your dataloader
    val_emb   = embed_texts(val_texts)
    gen_emb   = embed_texts(gen_texts)

    for k in (1, 5, 10):
        cov_tr   = naeem_coverage(train_emb, gen_emb, k=k, metric="cosine")
        cov_val  = naeem_coverage(val_emb,   gen_emb, k=k, metric="cosine")
        cov_union = naeem_coverage(
            np.vstack([train_emb, val_emb]), gen_emb, k=k, metric="cosine"
        )
        print(f"Coverage@{k}: train={cov_tr:.3f}  val={cov_val:.3f}  union={cov_union:.3f}  gap(val-train)={(cov_val-cov_tr):.3f}")

    # Novel-val coverage (mask out val points near train)
    val2train = nearest_dist_to_set(val_emb, train_emb, metric="cosine")
    if val2train is not None:
        tau = np.percentile(val2train, 70)  # top 30% most novel
        mask = val2train >= tau
        for k in (1, 5, 10):
            cov_val_novel = naeem_coverage(val_emb[mask], gen_emb, k=k, metric="cosine")
            print(f"Coverage@{k} on novel-val (≥{tau:.3f} from train): {cov_val_novel:.3f}")
    # Multi-k PR curve
    pr_curve = multi_k_pr(real_emb, gen_emb, ks=(1,3,5,10,20))
    print("\nQuality & Context Diversity")
    hdr = (
        f"{'MAUVE↑':>8}  {'PPL↓ (avg)':>12}  "
        f"{'Self-BERT-F1↓':>13}  {'MISS↓':>8}  {'p95↓':>8}  "
        f"{'TopicEnt↑':>10}  {'Eff-K↑':>8}  {'HHI↓':>8}  {'Dup@0.95↓':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    print(
        f"{_fmtf(mauve_weighted):>8}  {_fmtf(ppl_mean):>12}  "
        f"{_fmtf(self_bert_f1):>13}  {_fmtf(miss['mean']):>8}  {_fmtf(miss['p95']):>8}  "
        f"{_fmtf(topic_ent):>10}  {_fmtf(cluster.get('effective_K')):>8}  {_fmtf(cluster.get('hhi')):>8}  "
        #f"{_fmtf(selfsim.get('dup_rate')):>10}"
    )
    print(
        f"{_fmtf(mauve_weighted):>8}  {_fmtf(ppl_mean):>24}  "
        f"{_fmtf(self_bert_f1):>13}  {_fmtf(miss['mean']):>8}  {_fmtf(miss['p95']):>11}  "
        f"{_fmtf(topic_ent):>10}  {_fmtf(mattr):>8}  {_fmtf(rep_stats['mean_rate']):>8}"
    )
    if mem and math.isfinite(mem.get('rate', float('nan'))):
        print(f"Memorization (n-gram≥{mem['L']}) rate: {_fmtf(mem['rate']*100, 2)}%  (higher = more memorization)")

    nns = nearest_neighbor_similarities(gen_emb)
    redund = redundancy_at_thresholds(nns, thresholds=(0.80, 0.85, 0.90, 0.95))
    print("Nearest-neighbor redundancy:",
        f"mean={redund['mean']:.3f} p95={redund['p95']:.3f} "
        f"r@0.80={redund['rate@0.80']:.3f} r@0.85={redund['rate@0.85']:.3f} "
        f"r@0.90={redund['rate@0.90']:.3f} r@0.95={redund['rate@0.95']:.3f}")
    # res_small['diversity']['nns'] and res_full['diversity']['nns'] should be the arrays you stored
    print("PR(k): " + "  ".join([f"k={d['k']}: P={d['precision']:.3f}, C={d['coverage']:.3f}" for d in pr_curve]))

    _print_diversity_summary(distinct, prd)

    for k in _ks(len(train_emb)):
        cov_tr = naeem_coverage(train_emb, gen_emb, k=k, metric="cosine")
        print(f"Coverage@{k} (train): {cov_tr:.3f}")

    for k in _ks(len(val_emb)):
        cov_val = naeem_coverage(val_emb, gen_emb, k=k, metric="cosine")
        print(f"Coverage@{k} (val):   {cov_val:.3f}")

    # Union coverage (train ∪ val)
    if len(train_emb) and len(val_emb):
        union_emb = np.vstack([train_emb, val_emb])
        for k in _ks(len(union_emb)):
            cov_union = naeem_coverage(union_emb, gen_emb, k=k, metric="cosine")
            print(f"Coverage@{k} (union): {cov_union:.3f}")

    # Novel-val coverage: mask val points far from train
    val2train = nearest_dist_to_set(val_emb, train_emb, metric="cosine")
    if val2train is not None and len(val2train):
        tau = np.percentile(val2train, 70)  # top 30% most novel
        mask = val2train >= tau
        if mask.any():
            for k in _ks(mask.sum()):
                cov_val_novel = naeem_coverage(val_emb[mask], gen_emb, k=k, metric="cosine")
                print(f"Coverage@{k} on novel-val (≥{tau:.3f} from train): {cov_val_novel:.3f}")
    _ = print_overall_summary_in_table(out, cls_probs, cls_logits, mauve_weighted, model_ids)
    # --- Return all raw objects ----------------------------------------------
    return {
        "ppl": out,
        "ppl_ref": out_ref,
        "classifier": {
            "probs": cls_probs,
            "logits": cls_logits,
        },
        "domain": dom,
        "mauve_src": m_src["mean"],
        "mauve_tgt": m_tgt["mean"],
        "mauve": {"src": m_src, "tgt": m_tgt}
    }