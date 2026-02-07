import torch
from dataclasses import dataclass, field
from data import so3_utils


class ExpSchedule:
    def __init__(self, exp_rate):
        self.exp_rate = exp_rate

    def __call__(self, t):
        return 1 - torch.exp(-t * self.exp_rate)


class LinearSchedule:
    def __call__(self, t):
        return t


class QuadraticSchedule:
    def __call__(self, t):
        return t**2


class SO3Interpolant:
    def __init__(self, cfg):
        self.cfg = cfg

        if self.cfg.sample_schedule == "exp":
            self.kappa = ExpSchedule(self.cfg.exp_rate)
        elif self.cfg.sample_schedule == "linear":
            self.kappa = LinearSchedule()
        elif self.cfg.sample_schedule == "quadratic":
            self.kappa = QuadraticSchedule()
        else:
            raise ValueError(f"Invalid sample schedule: {self.cfg.sample_schedule}")

    def sample_xt(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        kappa_t = self.kappa(t)
        x_t = so3_utils.geodesic_t(kappa_t[..., None], x_1, x_0)
        return x_t

    def cond_vf(
        self, t: torch.Tensor, x_t: torch.Tensor, x_1: torch.Tensor
    ) -> torch.Tensor:
        if self.cfg.sample_schedule == "exp":
            return self.cond_exp_vf(t, x_t, x_1)
        elif self.cfg.sample_schedule == "linear":
            return self.cond_linear_vf(t, x_t, x_1)
        elif self.cfg.sample_schedule == "quadratic":
            return self.cond_quadratic_vf(t, x_t, x_1)
        else:
            raise ValueError(f"Invalid schedule: {self.cfg.sample_schedule}")

    def cond_linear_vf(
        self, t: torch.Tensor, x_t: torch.Tensor, x_1: torch.Tensor
    ) -> torch.Tensor:
        v = so3_utils.calc_rot_vf(x_t, x_1)
        return v / torch.clamp(1 - t, min=1e-6)[..., None]

    def cond_exp_vf(
        self, t: torch.Tensor, x_t: torch.Tensor, x_1: torch.Tensor
    ) -> torch.Tensor:
        v = so3_utils.calc_rot_vf(x_t, x_1)
        return self.cfg.exp_rate * v

    def cond_quadratic_vf(
        self, t: torch.Tensor, x_t: torch.Tensor, x_1: torch.Tensor
    ) -> torch.Tensor:
        v = so3_utils.calc_rot_vf(x_t, x_1)
        scale = 2 * t / torch.clamp(1 - t**2, min=1e-6)[..., None]
        return scale * v

    def euler_step_with_x1_prediction(
        self,
        d_t: torch.Tensor,
        t: torch.Tensor,
        x_1_pred: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.sample_schedule == "linear":
            scaling = 1 / (1 - t)
        elif self.cfg.sample_schedule == "exp":
            scaling = self.cfg.exp_rate
        elif self.cfg.sample_schedule == "quadratic":
            scaling = 2 * t / torch.clamp(1 - t**2, min=1e-6)[..., None]
        else:
            raise ValueError(f"Unknown sample schedule {self.cfg.sample_schedule}")
        return so3_utils.geodesic_t(scaling * d_t, x_1_pred, x_t)

    def euler_step(
        self,
        d_t: torch.Tensor,
        x_t: torch.Tensor,
        vf: torch.Tensor,
    ) -> torch.Tensor:
        return so3_utils.geodesic_t(d_t, None, x_t, vf)
