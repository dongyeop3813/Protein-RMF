from functools import partial
from torch._tensor import Tensor
from typing import Callable, Literal
import torch
from torch import Tensor


def uniform_sample(batch_size: int, device: torch.device) -> Tensor:
    return torch.rand(batch_size, device=device)


def logit_norm_sample(
    batch_size: int,
    device: torch.device,
    mu: float = -0.4,
    sigma: float = 1.0,
) -> Tensor:
    normal_samples = mu + sigma * torch.randn(batch_size, device=device)
    return torch.sigmoid(normal_samples)


def mixture_sample(
    batch_size: int,
    device: torch.device,
    alpha: float = 1.9,
    beta: float = 1.0,
    eps_prob: float = 0.02,
):
    uniform_t = torch.rand(batch_size, device=device)
    beta_dist = torch.distributions.beta.Beta(alpha, beta)
    beta_t = beta_dist.sample((batch_size,)).squeeze().to(device)

    # 2% probability of uniform_t, 98% probability of beta_t
    flip_coin = torch.rand(batch_size, device=device) < eps_prob
    t = torch.where(flip_coin, uniform_t, beta_t)
    return t


def mixture_with_lognorm_sample(
    batch_size: int,
    device: torch.device,
    mu: float = -0.8,
    sigma: float = 1.0,
    alpha: float = 1.9,
    beta: float = 1.0,
    p_lognorm: float = 0.10,
) -> Tensor:
    """
    Mixture of Logit-Normal and Beta distributions.
    - With probability p_lognorm, sample from logit-normal (sigmoid of Normal(mu, sigma)).
    - With probability (1 - p_lognorm), sample from Beta(alpha, beta).
    """
    lognorm_t = logit_norm_sample(batch_size, device, mu=mu, sigma=sigma)
    beta_dist = torch.distributions.beta.Beta(alpha, beta)
    beta_t = beta_dist.sample((batch_size,)).squeeze().to(device)

    flip_coin = torch.rand(batch_size, device=device) < p_lognorm
    t = torch.where(flip_coin, lognorm_t, beta_t)
    return t


def rescale(t: Tensor, min_t: float = 0.0, max_t: float = 1.0) -> Tensor:
    return t * (max_t - min_t) + min_t


def sample_boundary_mask(
    batch_size: int, device: torch.device, boundary_ratio: float = 0.75
) -> Tensor:
    is_boundary = torch.rand(batch_size, device=device) < boundary_ratio
    return is_boundary


def interval_sample(
    batch_size: int,
    device: torch.device,
    single_t_sampler: Callable[[int, torch.device], Tensor],
    boundary_ratio: float = 0.75,
    order: Literal["ordered", "unordered"] = "ordered",
) -> tuple[Tensor, Tensor]:

    t_raw = single_t_sampler(batch_size, device)
    r_raw = single_t_sampler(batch_size, device)
    is_boundary = sample_boundary_mask(batch_size, device, boundary_ratio)

    assert order in ["ordered", "unordered"]
    if order == "ordered":
        t = torch.minimum(t_raw, r_raw)
        r = torch.maximum(t_raw, r_raw)
        t = torch.where(is_boundary, t_raw, t)
        r = torch.where(is_boundary, t_raw, r)
    elif order == "unordered":
        t = t_raw
        r = r_raw
        r = torch.where(is_boundary, t, r)

    return t, r


def beta_length_sampler(
    batch_size: int,
    device: torch.device,
    alpha: float = 0.5,
    beta: float = 0.5,
    min_length: float = 0.01,
    max_length: float = 1.0,
) -> Tensor:
    alpha = torch.tensor([alpha], device=device)
    beta = torch.tensor([beta], device=device)
    length = torch.distributions.beta.Beta(alpha, beta).sample((batch_size,)).squeeze()
    return rescale(length, min_length, max_length)


def length_first_interval_sample(
    batch_size: int,
    device: torch.device,
    length_sampler: Callable[[int, torch.device], Tensor],
    bias_with_mixture: bool = False,
) -> tuple[Tensor, Tensor]:
    length = length_sampler(batch_size, device)

    if bias_with_mixture:
        t = mixture_sample(batch_size, device) * (1 - length)
        r = t + length
    else:
        t = torch.rand(batch_size, device=device) * (1 - length)
        r = t + length

    return t, r


def three_point_sample(
    batch_size: int,
    device: torch.device,
    single_t_sampler: Callable[[int, torch.device], Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Sample three time points t, s, r such that t <= s <= r.
    The time points are sampled from the single time point sampler.
    """
    t_raw = single_t_sampler(batch_size, device)
    r_raw = single_t_sampler(batch_size, device)

    t = torch.minimum(t_raw, r_raw)
    r = torch.maximum(t_raw, r_raw)

    w = torch.rand(batch_size, device=device)
    s = w * r + (1 - w) * t

    return t, s, r


def length_first_three_point_sample(
    batch_size: int,
    device: torch.device,
    length_sampler: Callable[[int, torch.device], Tensor],
    is_mid_point: bool = False,
    bias_with_mixture: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:

    t, r = length_first_interval_sample(
        batch_size, device, length_sampler, bias_with_mixture
    )

    if is_mid_point:
        s = 0.5 * r + 0.5 * t
    else:
        w = torch.rand(batch_size, device=device) * 0.98 + 0.01  # [0.01, 0.99]
        s = w * r + (1 - w) * t

    return t, s, r


def create_single_t_sampler(cfg) -> Callable[[int, torch.device], Tensor]:
    type = cfg.get("type", None)
    if type == "uniform":
        fn = uniform_sample
    elif type == "lognorm":
        fn = partial(logit_norm_sample, mu=cfg.mu, sigma=cfg.sigma)
    elif type == "mixture":
        fn = mixture_sample
    elif type == "mixture_with_lognorm":
        fn = partial(
            mixture_with_lognorm_sample,
            mu=cfg.get("mu", -0.8),
            sigma=cfg.get("sigma", 1.0),
            alpha=cfg.get("alpha", 1.9),
            beta=cfg.get("beta", 1.0),
            p_lognorm=cfg.get("p_lognorm", 0.10),
        )
    else:
        raise ValueError(f"Invalid single time point sampler type: {type}")

    min_t = cfg.get("min_t", 0.0)
    max_t = cfg.get("max_t", 1.0)

    def sample_fn(batch_size: int, device: torch.device) -> Tensor:
        return rescale(fn(batch_size, device), min_t=min_t, max_t=max_t)

    return sample_fn


def create_interval_sampler(cfg) -> Callable[[int, torch.device], Tensor]:

    if cfg.interval_sample in ["ordered", "unordered"]:
        single_t_sampler = create_single_t_sampler(cfg)
        interval_sampler = partial(
            interval_sample,
            single_t_sampler=single_t_sampler,
            boundary_ratio=cfg.boundary_ratio,
            order=cfg.interval_sample,
        )
        return interval_sampler
    elif cfg.interval_sample == "interval_length_first":
        length_sampler = partial(beta_length_sampler, alpha=cfg.alpha, beta=cfg.beta)
        interval_sampler = partial(
            length_first_interval_sample,
            length_sampler=length_sampler,
        )
        return interval_sampler
    else:
        raise ValueError(f"Invalid interval sampler type: {cfg.interval_sample}")


def create_three_point_sampler(cfg) -> Callable[[int, torch.device], Tensor]:
    if cfg.three_sample in ["interval_length_first", "interval_length_first_midpoint"]:
        length_sampler = partial(
            beta_length_sampler,
            alpha=cfg.alpha,
            beta=cfg.beta,
            min_length=cfg.get("min_length", 0.01),
        )
        is_mid_point = cfg.three_sample == "interval_length_first_midpoint"
        return partial(
            length_first_three_point_sample,
            length_sampler=length_sampler,
            is_mid_point=is_mid_point,
            bias_with_mixture=cfg.get("bias_with_mixture", False),
        )
    elif cfg.three_sample == "ordered":
        single_t_sampler = create_single_t_sampler(cfg)
        return partial(three_point_sample, single_t_sampler=single_t_sampler)
    else:
        raise ValueError(f"Invalid three point sampler type: {cfg.three_sample}")
