import torch
from data import scheduler


class TransInterpolant:
    def __init__(self, cfg):
        self.cfg = cfg
        self.schedule = scheduler.make_schedule(cfg)

    def sample_xt(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        g_t = self.schedule.g(t)[..., None]
        x_t = x_1 * g_t + x_0 * (1 - g_t)
        return x_t

    def cond_vf(
        self, t: torch.Tensor, x_t: torch.Tensor, x_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Conditional vector field of the transition.
        v(t) = x_1 h(t) - x_0 h(t)
        x_0 = (x_t - x_1 * g_t) / (1 - g_t)
        """
        g_t = self.schedule.g(t)[..., None]
        h_t = self.schedule.h(t)[..., None]
        x_0 = (x_t - x_1 * g_t) / torch.clamp(1 - g_t, min=1e-6)
        v = x_1 * h_t - x_0 * h_t
        return v

    def euler_step_with_x1_prediction(
        self,
        d_t: torch.Tensor,
        t: torch.Tensor,
        x_1_pred: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        v = self.cond_vf(t, x_t, x_1_pred)
        return self.euler_step(d_t, x_t, v)

    def euler_step(
        self,
        d_t: torch.Tensor,
        x_t: torch.Tensor,
        vf: torch.Tensor,
    ) -> torch.Tensor:
        assert d_t > 0
        return x_t + vf * d_t
