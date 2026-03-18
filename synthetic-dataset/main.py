#!/usr/bin/env python3
"""
Main entry point: data → training → sampling.
Keeps every original helper function name from the notebook/codebase.
"""

import time
import torch, random, numpy as np
from tqdm.auto import tqdm

import time, types
import torch, random, numpy as np
from tqdm.auto import tqdm

# ── editable configuration object ──────────────────────────────────────
config = types.SimpleNamespace(
    # experiment switches
    data_type="markov",                      # "markov" | "gaussian" | "discrete"
    unguided_sampling=False,
    skip_ratio_estimator_on_clean_data=True,
    test_run=True,                  # if True, use smaller data and fewer epochs

    # reproducibility
    seed=100,

    # data settings
    len_seqs=30,
    vocab_size=5,
    number_of_source_samples=10_000,
    number_of_target_samples=1_000,

    # data_type="gaussian",                         ´
    dimension_gaussian=2,
    num_mixtures=2,
    grid_size=2,
    quantization_step=0.01,

    # -- data_type="discrete",
    source_dist="uniform", # "uniform" | "gaussian" | "x2"
    target_dist="gaussian", # "uniform" | "gaussian" | "x2"
    distance=0, # distance between source and target distributions

    # -- data_type="markov",
    extra_target_tokens=0,  # how many extra tokens to add to the target sequences
    type_of_T_matrix="diagonal", # "diagonal" | "random" | "anti-diagonal"
    dirichlet_alpha=4.0,

    # training
    batch_size=256,
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    train_classifier_epochs=30,
    train_ratio_estimator_epochs=30,
    train_denoiser_epochs=20,

    # denoiser model dims
    denoiser_dim=256,
    denoiser_heads=4,
    denoiser_layers=4,

    # sampling
    sample_batch_size=512*2,
    sample_gammas=[0,1,2], # [0.0, .1,.2,.3,.4,.5,0.6,.7,.8,.9, 1.0, 2.0],
    sample_size=2048,
    denoising_steps=20,
    use_plg=False
)

# ── local modules (unchanged function names) ────────────────────────────
from data import (
    build_markov_transfer,      # Markov‑shift
    orth_means_random,          # Gaussian mixture helpers
    build_iid_gmm_data,
    estimate_transition_matrix,
    create_probability_maps,
    sample_from_prob_map,
)
from utils import transition_stats, seed_everything, print_trace
from plot_function import (
    plot_iid_gmm_points,
    plot_ratio_estimator,
    plot_prob_map,
    plot_sample_counts,
)
from models import *
from diffusion import LogLinearNoise, DiffusionConfig, Diffusion
from training import (
    train_domain_classifier,
    validate_domain_classifier,
    train_ratio_estimator_on_clean_data,
    train_ratio_estimator,
    train_ratio_network_with_regularization,
    train_denoiser,
)

# =======================================================================
#  Function to load data
# =======================================================================
def load_data():
    """Return (src, tgt, extras) depending on ``config.data_type``."""

    # ----------------------- markov -----------------------
    if config.data_type == "markov":
        src, tgt, P, Q, vc_sz = build_markov_transfer(
            L=config.len_seqs,
            vocab_size=config.vocab_size,
            n_src=config.number_of_source_samples,
            n_tgt=config.number_of_target_samples,
            extra_target_tokens=config.extra_target_tokens,
        )

        config.vocab_size = vc_sz
        return src, tgt, {"S": P, "T": Q}

    # ----------------------- gaussian -----------------------
    if config.data_type == "gaussian":
        mu_s, mu_t = orth_means_random(
            dimension=config.dimension_gaussian,)
        if config.num_mixtures == 1 and config.dimension_gaussian == 2:
            mu_s = torch.tensor([config.grid_size-1, config.grid_size-1])
            mu_t = torch.tensor([-(config.grid_size-1), -(config.grid_size-1)])

        src, tgt, vocab_sz, edges = build_iid_gmm_data(
            mu_src=mu_s,
            mu_tgt=mu_t,
            device=config.device,
            n_src=config.number_of_source_samples, n_tgt=config.number_of_target_samples,
            grid_min = -config.grid_size, grid_max=config.grid_size,
            num_mixtures = config.num_mixtures,)
        config.len_seqs = config.dimension_gaussian
        config.vocab_size = vocab_sz
        return src, tgt, {"edges": edges, "mu_s": mu_s, "mu_t": mu_t}

    # ----------------------- discrete -----------------------
    if config.data_type == "discrete":
        src_probs, tgt_probs, vocab_size, seq_len = create_probability_maps(grid_max=4,
                                           source_dist=config.source_dist, target_dist=config.target_dist,distance=config.distance,)
        src = sample_from_prob_map(src_probs, config.number_of_source_samples)
        tgt = sample_from_prob_map(tgt_probs, config.number_of_target_samples)
        config.len_seqs = seq_len
        config.vocab_size = vocab_size
        return src, tgt, {"grid_shape": src_probs.shape, "src_probs": src_probs, "tgt_probs": tgt_probs,
                          "edges": torch.arange(0, vocab_size, 1)}
    raise ValueError(f"Unknown data_type: {config.data_type}")


# =======================================================================
#  Function to evaluate raw and sampled data
# =======================================================================
def evaluate_raw_data(src, tgt, extras):
    """Quick sanity checks / visualisations before training."""
    if config.data_type == "markov":
        S_est = estimate_transition_matrix(src, config.vocab_size)
        abs_diff, norm, diag_mean,_,_= transition_stats(S_est, extras["S"], config)
        print(
            f"[Data] P_est error: max {abs_diff.max():.4f}  ‖·‖₁={norm:.4f} mean_diag={diag_mean:.4f}"
        )
        T_est = estimate_transition_matrix(tgt, config.vocab_size)
        abs_diff, norm, diag_mean,_,_ = transition_stats(T_est, extras["T"], config)
        print(
            f"[Data] Q_est error: max {abs_diff.max():.4f}  ‖·‖₁={norm:.4f} mean_diag={diag_mean:.4f}"
        )
    elif config.data_type == "gaussian":
        edges = extras["edges"]
        plot_iid_gmm_points(tgt, edges, title="Target samples")
        plot_iid_gmm_points(src, edges, title="Source samples")

    elif config.data_type == "discrete":
        plot_prob_map(extras["src_probs"], title="Source distribution")
        plot_prob_map(extras["tgt_probs"], title="Target distribution (shifted)")
        #plot_sample_counts(src,grid_shape=extras["grid_shape"],title="Observed sample counts")
        #plot_sample_counts(tgt,grid_shape=extras["grid_shape"], title="Observed sample counts")


def evaluate_sampled_data(samples, gamma, type, dt, extras):

    # ── evaluation / visualisation ──
    if config.data_type == "markov":
        print(f"Sampled sequences (γ={gamma}), type={type}")
        P_est = estimate_transition_matrix(samples.cpu(), config.vocab_size)
        P_true = extras["T"] if not type == "source" else extras["S"]
        abs_diff, norm, mean_diag, mean_extra,_ = transition_stats(P_est, P_true, config)
        print(f"P error max={abs_diff.max():.4f}  ‖·‖₁={norm:.4f} mean_diag={mean_diag:.4f} (t={dt:.1f}s)")
        if config.extra_target_tokens > 0:
            print(f"Average mean distance in extra token {mean_extra}")
    elif config.data_type == "gaussian":
        plot_iid_gmm_points(samples.cpu(), extras["edges"], title=f"Generated γ={gamma}, type={type}")
    elif config.data_type == "discrete":
        plot_sample_counts(samples.cpu(),
                           grid_shape=extras["grid_shape"],
                           title=f"Generated samples γ={gamma}, type={type}",
                           )

# =======================================================================
# Main function
# =======================================================================
def main() -> None:
    seed_everything(config.seed)

    # --------------------------------------------------------------------------
    # 1 ▸ Data -----------------------------------------------------------------
    src, tgt, extras = load_data()
    evaluate_raw_data(src, tgt, extras)
    if config.test_run:
        config.train_classifier_epochs = 3
        config.train_ratio_estimator_epochs = 3
        config.train_denoiser_epochs = 3
    if config.unguided_sampling:
        config.sample_gammas = [0.0]

    # --------------------------------------------------------------------------
    # 2 ▸ Domain classifier ----------------------------------------------------
    if not config.skip_ratio_estimator_on_clean_data or not config.unguided_sampling:
        classifier = train_domain_classifier(
            model=ClassifierNet(
                vocab_sz=config.vocab_size, seq_len=config.len_seqs, output_with_sigmoid=False
            ),
            source_data=src,
            target_data=tgt,
            epochs=config.train_classifier_epochs,
            batch_size=config.batch_size,
            device=config.device,
        )
        validate_domain_classifier(
            classifier,
            src,
            tgt,
            batch_size=config.batch_size,
            device=config.device,
        )


    # --------------------------------------------------------------------------
    # 4 ▸ Ratio estimator (noisy) ---------------------------------------------
    ratio_net_guided = RatioNetAdaLN(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)
    if not config.unguided_sampling:
        ratio_net_guided = train_ratio_estimator(
            model=ratio_net_guided,
            domain_classifier=classifier,
            source_data=src,
            target_data=tgt,
            noise_sched=LogLinearNoise(),
            diffusion="absorbing_state",
            mask_idx=config.vocab_size,
            vocab_size=config.vocab_size + 1,
            epochs=config.train_ratio_estimator_epochs,
            batch_size=config.batch_size,
            device=config.device,
        )
        plot_ratio_estimator(source_samples=src, target_samples=tgt,
            ratio_net=ratio_net_guided,extras=extras,config=config,)


    # --------------------------------------------------------------------------
    # 5 ▸ Denoiser -------------------------------------------------------------
    denoiser = AdaLNDenoiser(
        vocab_size=config.vocab_size + 1,
        seq_len=config.len_seqs,
        d=config.denoiser_dim,
        heads=config.denoiser_heads,
        layers=config.denoiser_layers,
    )
    denoiser_source = train_denoiser(
        denoiser=denoiser,
        train_seqs=src,
        diffusion="absorbing_state",
        mask_idx=config.vocab_size,
        vocab_size=config.vocab_size + 1,
        epochs=config.train_denoiser_epochs,
        batch_size=config.batch_size,
        device=config.device,
    )

    # --------------------------------------------------------------------------
    # 6 ▸ Sampling -------------------------------------------------------------
    cfg = DiffusionConfig()
    cfg.vocab_size = config.vocab_size + 1
    cfg.seq_len = config.len_seqs
    cfg.mask_idx = config.vocab_size
    cfg.batch_size = config.sample_batch_size
    cfg.T_sampling = config.denoising_steps
    for gamma in config.sample_gammas:
        cfg.gamma = gamma
        print(f"\n[Sampling] gamma={gamma}")
        sampler = Diffusion(denoiser_source, ratio_net_guided, cfg).to(config.device)
        with torch.inference_mode():
                sample_trace = sampler.sample_trace(num_samples=8,device=config.device,)
        print_trace(sample_trace,sample_idx=0, vocab_size=config.vocab_size+1, mask_idx=config.vocab_size)
        tic = time.time()
        samples = sampler.sample_ancestral(num_of_samples=config.sample_size, device=config.device)
        dt = time.time() - tic

        evaluate_sampled_data(
                samples=samples,
                gamma=gamma,
                type=type,
                dt=dt,
                extras=extras,)



if __name__ == "__main__":
    main()
