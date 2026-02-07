import torch
from data import all_atom
from data.interpolant import make_feats, IGSO3Prior
from data import utils as du


class DDIMInference:
    def __init__(self, cfg, model, interpolant):
        self.cfg = cfg
        self.model = model
        self.interpolant = interpolant
        self.rot_prior = IGSO3Prior(cfg)

    def corrupt_trans(self, t, trans_1):
        num_batch, num_res = trans_1.shape[:2]
        trans_0 = self.interpolant.trans_noise(num_batch, num_res)
        if self.interpolant._trans_cfg.get("align", False):
            # Kabsh alignment for noise and target
            trans_0, _, _ = du.batch_align_structures(trans_0, trans_1)

        noise_scale = self.cfg.trans_gamma * (1 - t[..., None])
        trans_t = noise_scale * trans_0 + t[..., None] * trans_1
        return trans_t

    def corrupt_rotmats(self, t, rotmats_1):
        noise_scale = self.cfg.rot_gamma * 1.5
        rotmats_0 = self.rot_prior.sample_noise_like(rotmats_1, sigma=noise_scale)
        rotmats_t = self.interpolant.so3_interpolant.sample_xt(rotmats_0, rotmats_1, t)

        return rotmats_t

    @torch.no_grad()
    def n_step_sample(
        self,
        num_batch,
        num_res,
        n_steps,
        last_step=True,
        return_trans_rot=False,
    ):
        device = self.interpolant._device
        # Set-up initial prior samples
        trans_t, rotmats_t = self.interpolant.init_prior(num_batch, num_res)

        feats = make_feats(num_batch, num_res, device)

        if self.cfg.time_steps == "uniform":
            ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        elif self.cfg.time_steps == "exponential":
            ts = [0.0] + [1 - (0.5 ** (i + 1)) for i in range(n_steps - 1)] + [1.0]
            ts = torch.tensor(ts, device=device)
        else:
            raise ValueError(
                f"Invalid time step sampling scheme: {self.cfg.time_steps}"
            )

        for t in ts:
            t1 = torch.ones((num_batch, 1), device=device) * t
            t2 = torch.ones_like(t1)

            trans_1, rotmats_1 = self.model.forward_flow(
                trans_t, rotmats_t, t1, t2, feats
            )

            trans_t = self.corrupt_trans(t, trans_1)
            rotmats_t = self.corrupt_rotmats(t, rotmats_1)

        if last_step:
            trans_t, rotmats_t = self.interpolant.last_step_refinement(
                self.model, trans_t, rotmats_t, feats
            )

        if return_trans_rot:
            return trans_t, rotmats_t, feats["res_mask"]

        atom37 = all_atom.transrot_to_atom37([(trans_t, rotmats_t)], feats["res_mask"])
        samples = atom37[-1].numpy()
        return samples
