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
import copy
from tqdm.auto import tqdm
from torchinfo import summary

# ── editable configuration object ──────────────────────────────────────
config = types.SimpleNamespace(
    # experiment switches
    data_type="markov",                      # "markov" | "gaussian" | "discrete"
    unguided_sampling=False,
    skip_ratio_estimator_on_clean_data=True,
    test_run=True,                  # if True, use smaller data and fewer epochs

    # -- data_type="markov",
    extra_target_tokens=0,  # how many extra tokens to add to the target sequences
    type_of_T_matrix="anti-diagonal", # "diagonal" | "random" | "anti-diagonal"
    dirichlet_alpha=4.0,

    # reproducibility
    seed=100,

    # data settings
    len_seqs=20,
    vocab_size=5,
    number_of_source_samples=10_000,
    number_of_target_samples=5_000,

    # data_type="gaussian",                         ´
    dimension_gaussian=2,
    num_mixtures=2,
    grid_size=2,
    quantization_step=0.01,

    # -- data_type="discrete",
    source_dist="uniform", # "uniform" | "gaussian" | "x2"
    target_dist="gaussian", # "uniform" | "gaussian" | "x2"
    distance=0, # distance between source and target distributions


    # training
    batch_size=256,
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    train_classifier_epochs=20,
    train_ratio_estimator_epochs=30,
    train_denoiser_epochs=30,

    # training the classifier
    unbalance_data=True,
    balanced_batches=True,
    cutmix_p=0.0,
    use_time_dependent_classifier=True,


    # denoiser model dims
    denoiser_dim=256,
    denoiser_heads=4,
    denoiser_layers=4,

    # sampling
    k_best_sampling=-1,
    sample_batch_size=512*2,
    sample_gammas=[0,1], # [0.0, .1,.2,.3,.4,.5,0.6,.7,.8,.9, 1.0, 2.0],
    sample_size=2048*2,
    denoising_steps=20,
    use_plg=False,
    use_approx=True,
    output_with_sigmoid=False,  # whether to apply sigmoid to the output of the classifier
)

# ── local modules (unchanged function names) ────────────────────────────
from data import *
from utils import *
from plot_function import *
from models import *
from diffusion import LogLinearNoise, DiffusionConfig, Diffusion
from training import *

# =======================================================================
#  Function to load data
# =======================================================================
def load_data():
    """Return (src, tgt, extras) depending on ``config.data_type``."""

    # ----------------------- markov -----------------------
    if config.data_type == "markov":
        src, tgt, S, T, vc_sz = build_markov_transfer(
            L=config.len_seqs,
            type_of_T_matrix=config.type_of_T_matrix,
            dirichlet_alpha_src=config.dirichlet_alpha,
            dirichlet_alpha_tgt=config.dirichlet_alpha,
            diag_src=0.6,
            diag_tgt=0.6,
            vocab_size=config.vocab_size,
            n_src=config.number_of_source_samples,
            n_tgt=config.number_of_target_samples,
            extra_target_tokens=config.extra_target_tokens,
        )
        src_val = torch.stack([sample_markov(config.len_seqs, S)for _ in range(5000)]).to("cpu")
        tgt_val = torch.stack([sample_markov(config.len_seqs, T)for _ in range(5000)]).to("cpu")

        config.vocab_size = vc_sz
        S, T = adjust_transition_matrix_size(S, T)
        return src, tgt, {"S": S, "T": T, "src_val": src_val, "tgt_val": tgt_val}

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
    src_val, tgt_val = extras["src_val"], extras["tgt_val"]
    ratio_in_data_size = len(src) / len(tgt)
    unbalance_data = True if ratio_in_data_size**-1 < 0.1 else False
    evaluate_raw_data(src, tgt, extras)
    if config.test_run:
        config.train_classifier_epochs = 1
        config.train_ratio_estimator_epochs = 1
        config.train_denoiser_epochs = 1
        config.denoising_steps = 1
    if config.unguided_sampling:
        config.sample_gammas = [0.0]

    # --------------------------------------------------------------------------
    # 2 ▸ Domain classifier ----------------------------------------------------
    classifier = train_domain_classifier(
            model=ClassifierNet(
                vocab_sz=config.vocab_size, seq_len=config.len_seqs, output_with_sigmoid=config.output_with_sigmoid
            ),
            source_data=src,
            target_data=tgt,
            epochs=config.train_classifier_epochs,
            batch_size=config.batch_size,
            device=config.device,
            unbalance_data=config.unbalance_data,
            balanced_batches=config.balanced_batches,
            cutmix_p=config.cutmix_p,
            classifier_output_with_sigmoid=config.output_with_sigmoid,
        )
    validate_domain_classifier(
            classifier,
            src_val,
            tgt_val,
            batch_size=config.batch_size,
            device=config.device,
        )

    classifier_t = RatioNetAdaLN(
        vocab_sz=config.vocab_size, seq_len=config.len_seqs, output_with_sigmoid=config.output_with_sigmoid)
    classifier_t = train_time_dependent_classifier(
        model=classifier_t,
        noise_sched=LogLinearNoise(),
        source_data=src,
        target_data=tgt,
        mask_idx=config.vocab_size,
        vocab_size=config.vocab_size + 1,
        epochs=config.train_classifier_epochs,
        batch_size=config.batch_size,
        device=config.device,
        unbalance_data=config.unbalance_data,
        balanced_batches=config.balanced_batches,
        cutmix_p=config.cutmix_p,
        classifier_output_with_sigmoid=config.output_with_sigmoid,
    )
    validate_domain_classifier(
        classifier_t,
        src_val,
        tgt_val,
        batch_size=config.batch_size,
        device=config.device, )

    # --------------------------------------------------------------------------
    # 3 ▸ Denoiser - source -------------------------------------------------------------
    denoiser_source = AdaLNDenoiser(
        vocab_size=config.vocab_size + 1,
        seq_len=config.len_seqs,
        d=config.denoiser_dim,
        heads=config.denoiser_heads,
        layers=config.denoiser_layers,)

    # manual totals
    total_params = sum(p.numel() for p in denoiser_source.parameters())
    trainable_params = sum(p.numel() for p in denoiser_source.parameters() if p.requires_grad)
    print(f"Total params of source Denoiser: {total_params:,}")
    print(f"Trainable params of source Denoiser: {trainable_params:,}")
    denoiser_source = train_denoiser(
        denoiser=denoiser_source,
        train_seqs=src,
        diffusion="absorbing_state",
        mask_idx=config.vocab_size,
        vocab_size=config.vocab_size + 1,
        epochs=config.train_denoiser_epochs,
        batch_size=config.batch_size,
        device=config.device,
        print_name="Source",)

    # --------------------------------------------------------------------------
    # 3 ▸ Denoiser - target -------------------------------------------------------------
    denoiser_target = AdaLNDenoiser(
        vocab_size=config.vocab_size + 1,
        seq_len=config.len_seqs,
        d=config.denoiser_dim,
        heads=config.denoiser_heads,
        layers=config.denoiser_layers,)
    denoiser_target = train_denoiser(
        denoiser=denoiser_target,
        train_seqs=tgt,
        diffusion="absorbing_state",
        mask_idx=config.vocab_size,
        vocab_size=config.vocab_size + 1,
        epochs=int(config.train_denoiser_epochs * min(max(ratio_in_data_size, 1), 2)),
        batch_size=config.batch_size,
        device=config.device,
        print_name="Target",)

    ratio_net_guided = RatioNetAdaLN(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)

    # --------------------------------------------------------------------------
    # 4 ▸ Train finetuned denoiser ---------------------------------------------
    denoiser_finetuned = copy.deepcopy(denoiser_source)
    denoiser_finetuned = train_denoiser(
        denoiser=denoiser_finetuned,
        train_seqs=tgt,
        diffusion="absorbing_state",
        mask_idx=config.vocab_size,
        vocab_size=config.vocab_size + 1,
        epochs=int(config.train_denoiser_epochs * min(max(ratio_in_data_size, 1), 2)),
        batch_size=config.batch_size,
        device=config.device,
        print_name="Finetune",
    )


    # --------------------------------------------------------------------------
    # 5 ▸ Ratio estimator (noisy) ---------------------------------------------
    # --- ratio estimator with regularization like TLDM paper ---
    ratio_net_guided = RatioNetAdaLN(vocab_sz=config.vocab_size + 1, seq_len=config.len_seqs)
    ratio_net_guided = train_ratio_network_with_regularization(
        model=ratio_net_guided,
        domain_classifier=classifier,
        domain_classifier_t=classifier_t,
        denoiser_model=denoiser_source,
        eta1and2=(0.1, 0.0),
        source_data=src,
        target_data=tgt,
        noise_sched=LogLinearNoise(),
        diffusion="absorbing_state",
        mask_idx=config.vocab_size,
        vocab_size=config.vocab_size + 1,
        epochs=config.train_ratio_estimator_epochs,
        batch_size=config.batch_size,
        device=config.device,
        classifier_output_with_sigmoid=config.output_with_sigmoid,
    )
    plot_ratio_estimator(source_samples=src, target_samples=tgt,
            ratio_net=ratio_net_guided,extras=extras,config=config,)

    # --------------------------------------------------------------------------
    # 7 ▸ Sampling -------------------------------------------------------------
    cfg = DiffusionConfig()
    cfg.vocab_size = config.vocab_size + 1
    cfg.seq_len = config.len_seqs
    cfg.mask_idx = config.vocab_size
    cfg.batch_size = config.sample_batch_size
    cfg.T_sampling = config.denoising_steps

    cfg.gamma = 0.0
    sample_dict = {
        "source": Diffusion(denoiser_source, ratio_net_guided, cfg).to(config.device),
        "target": Diffusion(denoiser_target, ratio_net_guided, cfg).to(config.device),
        "finetuned": Diffusion(denoiser_finetuned, ratio_net_guided, cfg).to(config.device),
        #"guided_approx": Diffusion(denoiser_source, ratio_net_guided, cfg, use_approx=True).to(config.device),
        "guided": Diffusion(denoiser_source, ratio_net_guided, cfg).to(config.device),
    }
    for type, sampler in sample_dict.items():
        if "guided" in type:
            cfg.gamma = 2.0
        print("")
        print("-" * 50)
        print(f"Gamma is set to {sampler.cfg.gamma}, type={type}")
        t1 = time.time()
        samples = sampler.sample_ancestral(
                num_of_samples=config.sample_size,
                device=config.device,
            )
        dt = time.time() - t1
        evaluate_sampled_data(samples, sampler.cfg.gamma, type, dt, extras)



if __name__ == "__main__":
    main()
