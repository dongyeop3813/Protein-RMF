import torch
from torch.optim import Optimizer


class CautiousAdamW(Optimizer):
    def __init__(
        self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                # state init
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)

                st["step"] += 1
                t = st["step"]
                m = st["exp_avg"]
                v = st["exp_avg_sq"]

                # Adam moments update (표준)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # bias correction
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                # AdamW의 "update 방향" u (weight decay 제외하면 보통 이 부분)
                u = m_hat / (v_hat.sqrt().add_(eps))

                # Cautious mask: u * g > 0 인 좌표만
                mask = (u * g > 0).to(u.dtype)
                scale = mask.mean().clamp_min(1e-8)

                # weight decay는 보통 안전하게 그대로 적용(혹은 u에 섞지 말고 별도 적용)
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)

                # cautious update
                p.add_(u * mask / scale, alpha=-lr)

        return loss
