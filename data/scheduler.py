import torch
from dataclasses import dataclass, field
from data import so3_utils

import abc


class Schedule(abc.ABC):
    @abc.abstractmethod
    def g(self, t):
        """
        Noise schedule function g(t)
        """
        pass

    @abc.abstractmethod
    def h(self, t):
        """
        Derivative of the noise schedule function h(t)
        """
        pass


class ExpSchedule(Schedule):
    def __init__(self, exp_rate):
        self.exp_rate = exp_rate

    def g(self, t):
        """
        Noise schedule function g(t)
        """
        return 1 - torch.exp(-t * self.exp_rate)

    def h(self, t):
        """
        Derivative of the noise schedule function h(t)
        """
        return torch.exp(-t * self.exp_rate)


class LinearSchedule(Schedule):
    def g(self, t):
        """
        Noise schedule function g(t)
        """
        return t

    def h(self, t):
        """
        Derivative of the noise schedule function h(t)
        """
        return torch.ones_like(t)


class QuadraticSchedule(Schedule):
    def g(self, t):
        """
        Noise schedule function g(t)
        """
        return t**2

    def h(self, t):
        """
        Derivative of the noise schedule function h(t)
        """
        return 2 * t


def make_schedule(cfg) -> Schedule:
    if cfg.sample_schedule == "exp":
        return ExpSchedule(cfg.exp_rate)
    elif cfg.sample_schedule == "linear":
        return LinearSchedule()
    elif cfg.sample_schedule == "quadratic":
        return QuadraticSchedule()
    else:
        raise ValueError(f"Invalid sample schedule: {cfg.sample_schedule}")
