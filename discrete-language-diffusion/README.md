# Guided Transfer Learning for Discrete Diffusion Models

> **GTL** adapts a pretrained discrete diffusion model to a target domain — without updating the denoiser — by learning a lightweight ratio network that reweights the source reverse transitions at sampling time.

📄 [Paper](https://arxiv.org/abs/XXXX.XXXXX) | 🤗 [arXiv Abstracts Dataset](https://huggingface.co/datasets/arxiv_org_submitters)

---

## Overview

GTL freezes a pretrained source denoiser and trains a small ratio estimator on mixed source/target data. At sampling time, the ratio network guides the reverse transitions toward the target distribution — no fine-tuning of the denoiser required.

**Key results:** GTL outperforms both vanilla and fine-tuned discrete diffusion across data-scarce regimes, while training only ~7% as many parameters.

---

## Code Architecture

All models are built on a shared PyTorch Lightning base class.

### `base_dm_model.py` — Shared Parent Class
`BaseDMModel` extends `lightning.LightningModule` and centralises all logic shared across models:
- **Corruption** (`_q_xt`): absorbing-state (masking) or uniform noise forward process
- **Time sampling** (`_sample_t`): antithetic and importance sampling support
- **Checkpoint fast-forwarding**: fault-tolerant resumption via `on_load_checkpoint` / `on_save_checkpoint`
- **Dataloader re-wiring**: swaps in `FaultTolerantDistributedSampler` at training start for reproducible, resumable data loading

Every model below inherits from `BaseDMModel`.

---

### Model Files

| File | Class | Role |
|---|---|---|
| `diffusion.py` | `Diffusion` | Source denoiser $p_\theta$. Trained on source data with the masked diffusion objective. Frozen during GTL. |
| `classifier.py` | `Classifier` | Domain classifier $d_\omega$. Trained with binary labels (source=1, target=0). Provides pseudo-ratio targets for the ratio network. Supports both time-independent and time-dependent variants. |
| `ratio.py` | `RatioEstimator` | Ratio network $r_\phi(x_t, t)$. Trained with a guidance loss (against the frozen classifier) and a cycle-consistency loss on target data. Core component of GTL at sampling time. |
| `planner.py` | `Planner` | Planner network $\rho_\vartheta$. Predicts which masked position to unmask next. Reduces per-step ratio evaluations from $\mathcal{O}(L \|\mathcal{V}\|)$ to $\mathcal{O}(n_\text{ratio})$. |
| `noise_schedule.py` | `LogLinearNoise`, etc. | Noise schedules (log-linear, cosine, geometric, linear). Shared across all models. |
| `base_dm_model.py` | `BaseDMModel` | Abstract parent — corruption helpers, time sampling, checkpointing, dataloader hooks. |
| `dataloader.py` | — | Fault-tolerant samplers and dataset utilities. |

### Training Order
```
1. Train Classifier     (classifier.py)   — time-independent + time-dependent
2. Train Source Denoiser (diffusion.py)   — freeze after training
3. Train Ratio Estimator (ratio.py)       — uses frozen classifier + denoiser
4. Train Planner        (planner.py)      — uses frozen denoiser for labels
5. Sample               (diffusion.py)    — guided by ratio + planner
```

---

## Environment Setup

**Step 1 — Create the environment**
```bash
mamba create -p /your/path/gtl_env python=3.9 pip
eval "$(conda shell.bash hook)"
conda activate /your/path/gtl_env
```

**Step 2 — Install PyTorch and core scientific packages**
```bash
mamba install -y -c conda-forge -c nvidia -c pytorch \
      pytorch=2.2.2 torchvision=0.17.2 pytorch-cuda=12.1 \
      jupyterlab git-lfs pandas=2.2 seaborn=0.13 scikit-learn=1.4

mamba install "numpy<2.0"
```

**Step 3 — Install remaining dependencies**
```bash
pip install datasets==2.18.0 einops==0.7.0 fsspec==2024.2.0 \
            h5py==3.10.0 hydra-core==1.3.2 ipdb==0.13.13 \
            lightning==2.2.1 nvitop==1.3.2 omegaconf==2.3.0 \
            packaging==23.2 rich==13.7.1 timm==0.9.16 \
            transformers==4.38.2 wandb==0.13.5

pip install -U sentencepiece tokenizers accelerate safetensors
```

**Activating an existing environment**
```bash
set +u
eval "$(conda shell.bash hook)"
mamba activate /your/path/gtl_env
set -u
```

---

## Dataset

We use the [arXiv abstracts dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv), tokenized with `bert-base-uncased` (vocab size $N = 30{,}522$) into segments of 512 tokens.
Then link the path "pathto/arxiv_abstracts/arxiv-metadata-oai-snapshot.json" in 

| Domain | Samples |
|---|---|
| Computer Science | 285,946 |
| Mathematics | 176,831 |
| Physics (target) | 79,631 |

The source domain is CS ∪ Math (+ an optional fraction $r$ of Physics). The target domain is the remaining Physics abstracts.

---

## Citation
```bibtex
@article{kleutgens2025gtl,
  title     = {Guided Transfer Learning for Discrete Diffusion Models},
  author    = {Kleutgens, Julian and Battiloro, Claudio and Kong, Lingkai and Grewe, Benjamin and Dominici, Francesca and Tec, Mauricio},
  year      = {2025},
}
```
