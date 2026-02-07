from collections import defaultdict
import os
import torch
from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from data import all_atom
from data.time_sampler import (
    create_single_t_sampler,
    create_interval_sampler,
    create_three_point_sampler,
)
import copy
from torch import autograd
from motif_scaffolding import twisting
from data.so3_interpolant import SO3Interpolant
from data.trans_interpolant import TransInterpolant
from data.so3_utils import calc_rot_vf, exp


DEBUG = False
LAST_STEP = True


def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)


def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch * num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return rotmats_t * diffuse_mask[..., None, None] + rotmats_1 * (
        1 - diffuse_mask[..., None, None]
    )


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE

        if self._trans_cfg.get("align", False):
            trans_0, _, _ = du.batch_align_structures(trans_0, trans_1)

        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]

    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_batch * num_res).to(
            self._device
        )
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = rotmats_t * res_mask[..., None, None] + identity[None, None] * (
            1 - res_mask[..., None, None]
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch["trans_1"]  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch["rotmats_1"]

        # [B, N]
        res_mask = batch["res_mask"]
        diffuse_mask = batch["diffuse_mask"]
        num_batch, _ = diffuse_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        so3_t = t
        r3_t = t
        noisy_batch["so3_t"] = so3_t
        noisy_batch["r3_t"] = r3_t

        # Apply corruptions
        if self._trans_cfg.corrupt:
            trans_t = self._corrupt_trans(trans_1, r3_t, res_mask, diffuse_mask)
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError("NaN in trans_t during corruption")
        noisy_batch["trans_t"] = trans_t

        if self._rots_cfg.corrupt:
            rotmats_t = self._corrupt_rotmats(rotmats_1, so3_t, res_mask, diffuse_mask)
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError("NaN in rotmats_t during corruption")
        noisy_batch["rotmats_t"] = rotmats_t
        return noisy_batch

    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == "exp":
            return 1 - torch.exp(-t * self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == "linear":
            return t
        else:
            raise ValueError(f"Invalid schedule: {self._rots_cfg.sample_schedule}")

    def _trans_vector_field(self, t, trans_1, trans_t):
        return (trans_1 - trans_t) / (1 - t)

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t > 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == "linear":
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == "exp":
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f"Unknown sample schedule {self._rots_cfg.sample_schedule}"
            )
        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)

    def sample(
        self,
        num_batch,
        num_res,
        model,
        num_timesteps=None,
        trans_potential=None,
        trans_0=None,
        rotmats_0=None,
        trans_1=None,
        rotmats_1=None,
        diffuse_mask=None,
        chain_idx=None,
        res_idx=None,
        verbose=False,
    ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        if trans_0 is None:
            trans_0 = (
                _centered_gaussian(num_batch, num_res, self._device)
                * du.NM_TO_ANG_SCALE
            )
        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        if res_idx is None:
            res_idx = torch.arange(num_res, device=self._device, dtype=torch.float32)[
                None
            ].repeat(num_batch, 1)
        batch = {"res_mask": res_mask, "diffuse_mask": res_mask, "res_idx": res_idx}

        motif_scaffolding = False
        if diffuse_mask is not None and trans_1 is not None and rotmats_1 is not None:
            motif_scaffolding = True
            motif_mask = ~diffuse_mask.bool().squeeze(0)
        else:
            motif_mask = None
        if motif_scaffolding and not self._cfg.twisting.use:  # amortisation
            diffuse_mask = diffuse_mask.expand(
                num_batch, -1
            )  # shape = (B, num_residue)
            batch["diffuse_mask"] = diffuse_mask
            rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)
            trans_0 = _trans_diffuse_mask(trans_0, trans_1, diffuse_mask)
            if torch.isnan(trans_0).any():
                raise ValueError("NaN detected in trans_0")

        logs_traj = defaultdict(list)
        if motif_scaffolding and self._cfg.twisting.use:  # sampling / guidance
            assert trans_1.shape[0] == 1  # assume only one motif
            motif_locations = torch.nonzero(motif_mask).squeeze().tolist()
            true_motif_locations, motif_segments_length = (
                twisting.find_ranges_and_lengths(motif_locations)
            )

            # Marginalise both rotation and motif location
            assert len(motif_mask.shape) == 1
            trans_motif = trans_1[:, motif_mask]  # [1, motif_res, 3]
            R_motif = rotmats_1[:, motif_mask]  # [1, motif_res, 3, 3]
            num_res = trans_1.shape[-2]
            with torch.inference_mode(False):
                motif_locations = (
                    true_motif_locations if self._cfg.twisting.motif_loc else None
                )
                F, motif_locations = twisting.motif_offsets_and_rots_vec_F(
                    num_res,
                    motif_segments_length,
                    motif_locations=motif_locations,
                    num_rots=self._cfg.twisting.num_rots,
                    align=self._cfg.twisting.align,
                    scale=self._cfg.twisting.scale_rots,
                    trans_motif=trans_motif,
                    R_motif=R_motif,
                    max_offsets=self._cfg.twisting.max_offsets,
                    device=self._device,
                    dtype=torch.float64,
                    return_rots=False,
                )

        if motif_mask is not None and len(motif_mask.shape) == 1:
            motif_mask = motif_mask[None].expand((num_batch, -1))

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for i, t_2 in enumerate(ts[1:]):
            if verbose:  # and i % 1 == 0:
                print(f"{i=}, t={t_1.item():.2f}")
                print(
                    torch.cuda.mem_get_info(trans_0.device),
                    torch.cuda.memory_allocated(trans_0.device),
                )
            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            if self._trans_cfg.corrupt:
                batch["trans_t"] = trans_t_1
            else:
                if trans_1 is None:
                    raise ValueError("Must provide trans_1 if not corrupting.")
                batch["trans_t"] = trans_1
            if self._rots_cfg.corrupt:
                batch["rotmats_t"] = rotmats_t_1
            else:
                if rotmats_1 is None:
                    raise ValueError("Must provide rotmats_1 if not corrupting.")
                batch["rotmats_t"] = rotmats_1
            batch["t"] = torch.ones((num_batch, 1), device=self._device) * t_1
            batch["so3_t"] = batch["t"]
            batch["r3_t"] = batch["t"]
            d_t = t_2 - t_1

            use_twisting = (
                motif_scaffolding
                and self._cfg.twisting.use
                and t_1 >= self._cfg.twisting.t_min
            )

            if use_twisting:  # Reconstruction guidance
                with torch.inference_mode(False):
                    batch, Log_delta_R, delta_x = twisting.perturbations_for_grad(batch)
                    model_out = model(batch)
                    t = batch["r3_t"]  # TODO: different time for SO3?
                    trans_t_1, rotmats_t_1, logs_traj = self.guidance(
                        trans_t_1,
                        rotmats_t_1,
                        model_out,
                        motif_mask,
                        R_motif,
                        trans_motif,
                        Log_delta_R,
                        delta_x,
                        t,
                        d_t,
                        logs_traj,
                    )

            else:
                with torch.no_grad():
                    model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out["pred_trans"]
            pred_rotmats_1 = model_out["pred_rotmats"]
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                if motif_scaffolding:
                    batch["trans_sc"] = pred_trans_1 * diffuse_mask[
                        ..., None
                    ] + trans_1 * (1 - diffuse_mask[..., None])
                else:
                    batch["trans_sc"] = pred_trans_1

            # Take reverse step

            trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            if trans_potential is not None:
                with torch.inference_mode(False):
                    grad_pred_trans_1 = (
                        pred_trans_1.clone().detach().requires_grad_(True)
                    )
                    pred_trans_potential = autograd.grad(
                        outputs=trans_potential(grad_pred_trans_1),
                        inputs=grad_pred_trans_1,
                    )[0]
                if self._trans_cfg.potential_t_scaling:
                    trans_t_2 -= t_1 / (1 - t_1) * pred_trans_potential * d_t
                else:
                    trans_t_2 -= pred_trans_potential * d_t
            rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            if motif_scaffolding and not self._cfg.twisting.use:
                trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
                rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        if LAST_STEP:
            # We only integrated to min_t, so need to make a final step
            t_1 = ts[-1]
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            if self._trans_cfg.corrupt:
                batch["trans_t"] = trans_t_1
            else:
                if trans_1 is None:
                    raise ValueError("Must provide trans_1 if not corrupting.")
                batch["trans_t"] = trans_1
            if self._rots_cfg.corrupt:
                batch["rotmats_t"] = rotmats_t_1
            else:
                if rotmats_1 is None:
                    raise ValueError("Must provide rotmats_1 if not corrupting.")
                batch["rotmats_t"] = rotmats_1
            batch["t"] = torch.ones((num_batch, 1), device=self._device) * t_1
            with torch.no_grad():
                model_out = model(batch)
            pred_trans_1 = model_out["pred_trans"]
            pred_rotmats_1 = model_out["pred_rotmats"]
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            prot_traj.append((pred_trans_1, pred_rotmats_1))

        if DEBUG:
            return prot_traj, res_mask

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return atom37_traj, clean_atom37_traj, clean_traj

    def guidance(
        self,
        trans_t,
        rotmats_t,
        model_out,
        motif_mask,
        R_motif,
        trans_motif,
        Log_delta_R,
        delta_x,
        t,
        d_t,
        logs_traj,
    ):
        # Select motif
        motif_mask = motif_mask.clone()
        trans_pred = model_out["pred_trans"][:, motif_mask]  # [B, motif_res, 3]
        R_pred = model_out["pred_rotmats"][:, motif_mask]  # [B, motif_res, 3, 3]

        # Proposal for marginalising motif rotation
        F = twisting.motif_rots_vec_F(
            trans_motif,
            R_motif,
            self._cfg.twisting.num_rots,
            align=self._cfg.twisting.align,
            scale=self._cfg.twisting.scale_rots,
            device=self._device,
            dtype=torch.float32,
        )

        # Estimate p(motif|predicted_motif)
        grad_Log_delta_R, grad_x_log_p_motif, logs = twisting.grad_log_lik_approx(
            R_pred,
            trans_pred,
            R_motif,
            trans_motif,
            Log_delta_R,
            delta_x,
            None,
            None,
            None,
            F,
            twist_potential_rot=self._cfg.twisting.potential_rot,
            twist_potential_trans=self._cfg.twisting.potential_trans,
        )

        with torch.no_grad():
            # Choose scaling
            t_trans = t
            t_so3 = t
            if self._cfg.twisting.scale_w_t == "ot":
                var_trans = ((1 - t_trans) / t_trans)[:, None]
                var_rot = ((1 - t_so3) / t_so3)[:, None, None]
            elif self._cfg.twisting.scale_w_t == "linear":
                var_trans = (1 - t)[:, None]
                var_rot = (1 - t_so3)[:, None, None]
            elif self._cfg.twisting.scale_w_t == "constant":
                num_batch = trans_pred.shape[0]
                var_trans = torch.ones((num_batch, 1, 1)).to(R_pred.device)
                var_rot = torch.ones((num_batch, 1, 1, 1)).to(R_pred.device)
            var_trans = var_trans + self._cfg.twisting.obs_noise**2
            var_rot = var_rot + self._cfg.twisting.obs_noise**2

            trans_scale_t = self._cfg.twisting.scale / var_trans
            rot_scale_t = self._cfg.twisting.scale / var_rot

            # Compute update
            trans_t, rotmats_t = twisting.step(
                trans_t,
                rotmats_t,
                grad_x_log_p_motif,
                grad_Log_delta_R,
                d_t,
                trans_scale_t,
                rot_scale_t,
                self._cfg.twisting.update_trans,
                self._cfg.twisting.update_rot,
            )

        # delete unsused arrays to prevent from any memory leak
        del grad_Log_delta_R
        del grad_x_log_p_motif
        del Log_delta_R
        del delta_x
        for key, value in model_out.items():
            model_out[key] = value.detach().requires_grad_(False)

        return trans_t, rotmats_t, logs_traj


class IGSO3Prior:
    def __init__(self, cfg):
        self._cfg = cfg
        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return self._igso3

    def sample_noise_like(self, rotmats_1, sigma=1.5):
        num_batch, num_res = rotmats_1.shape[:2]
        noise = self.igso3.sample(torch.tensor([sigma]), num_batch * num_res).to(
            rotmats_1.device
        )
        noise = noise.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noise)
        return rotmats_0


class UniformSO3Prior:
    def __init__(self, cfg):
        self._cfg = cfg
        self._uniform_so3 = None

    def sample_noise_like(self, rotmats_1):
        num_batch, num_res = rotmats_1.shape[:2]
        rotmats_0 = _uniform_so3(num_batch, num_res, rotmats_1.device)
        return rotmats_0


import abc


class FlowMapInterpolant(abc.ABC):
    def __init__(self, cfg):
        if cfg.prior_type == "igso3":
            self.prior = IGSO3Prior(cfg)
        elif cfg.prior_type == "uniform":
            self.prior = UniformSO3Prior(cfg)
        else:
            raise ValueError(f"Invalid prior type: {cfg.prior_type}")

        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        # self.trans_interpolant = TransInterpolant(self._trans_cfg)
        self.so3_interpolant = SO3Interpolant(self._rots_cfg)

    def set_device(self, device):
        self._device = device

    def trans_noise(self, num_batch, num_res):
        prior_type = self._trans_cfg.get("prior_type", "gaussian")
        if prior_type == "gaussian":
            trans_nm_0 = _centered_gaussian(num_batch, num_res, self._device)
            return (
                trans_nm_0
                * du.NM_TO_ANG_SCALE
                * self._cfg.trans.get("noise_scale", 1.0)
            )
        else:
            raise ValueError(f"Invalid prior type: {prior_type}")

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        trans_0 = self.trans_noise(*res_mask.shape)
        if self._trans_cfg.get("align", False):
            # Kabsh alignment for noise and target
            trans_0, _, _ = du.batch_align_structures(trans_0, trans_1)

        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]

    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        rotmats_0 = self.prior.sample_noise_like(rotmats_1)
        rotmats_t = self.so3_interpolant.sample_xt(rotmats_0, rotmats_1, t)

        # Apply masking if task is motif scaffolding or some residue is missing.
        identity = torch.eye(3, device=self._device)
        rotmats_t = rotmats_t * res_mask[..., None, None] + identity[None, None] * (
            1 - res_mask[..., None, None]
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def random_rotate(self, trans_1, rotmats_1):
        B = trans_1.shape[0]

        # Rotate the structure by a random rotation.
        rot = _uniform_so3(B, 1, rotmats_1.device)
        trans_1 = torch.einsum("...ij,...j->...i", rot, trans_1)
        rotmats_1 = so3_utils.rot_mult(rot, rotmats_1)

        # Zero CoM by default.
        trans_1 = trans_1 - trans_1.mean(dim=-2, keepdims=True)

        # Translate the structure by a random translation.
        random_com = (
            torch.randn(B, 1, 3, device=rotmats_1.device) * self._cfg.random_com_scale
        )
        trans_1 += random_com

        return trans_1, rotmats_1

    def augment_batch_if_needed(self, batch):
        if self._cfg.get("rotation_augmentation", False):
            trans_1, rotmats_1 = self.random_rotate(
                batch["trans_1"], batch["rotmats_1"]
            )
            batch["trans_1"] = trans_1
            batch["rotmats_1"] = rotmats_1
        else:
            # No rotation augmentation, then alignment must be disabled.
            assert (
                not self._trans_cfg.align
            ), "Alignment must be disabled if no rotation augmentation"

    def trans_cond_vf(self, t, trans_t, trans_1):
        return (trans_1 - trans_t) / (1 - t)
        # return self.trans_interpolant.cond_vf(t, trans_t, trans_1)

    def rots_cond_vf(self, t, rotmats_t, rotmats_1):
        return self.so3_interpolant.cond_vf(t, rotmats_t, rotmats_1)

    def rots_cond_exp_vf(self, t, rotmats_t, rotmats_1):
        return self.so3_interpolant.cond_exp_vf(t, rotmats_t, rotmats_1)

    def _trans_euler_step(self, d_t, trans_t, v_t):
        # assert d_t > 0
        return trans_t + v_t * d_t
        # return self.trans_interpolant.euler_step(d_t, trans_t, v_t)

    def _rots_euler_step(self, d_t, rotmats_t, v_t):
        return self.so3_interpolant.euler_step(d_t, rotmats_t, v_t)

    def last_step_refinement(self, model, trans_t, rotmats_t, batch):
        num_batch = trans_t.shape[0]
        r = t = torch.ones((num_batch, 1), device=self._device)
        pred_trans_1, pred_rotmats_1 = model(trans_t, rotmats_t, t, r, batch)
        return pred_trans_1, pred_rotmats_1

    def get_sigma(self, t):
        t = t[0].item()
        t = 1 - t

        beta_t = t
        alpha_t = 1 - t
        dalpha_t = -1
        dbeta_t = 1

        sigma_2 = 2 * ((dbeta_t / beta_t) * alpha_t**2 - dalpha_t * alpha_t)
        return sigma_2

    @torch.no_grad()
    def sample(
        self,
        num_batch,
        num_res,
        model,
        num_timesteps=None,
        trans_0=None,
        rotmats_0=None,
        res_idx=None,
        verbose=False,
        end_time=1.0,
        last_step=True,
        return_trans_rot=False,
    ):
        # Sampling by integrating the meanflow model
        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        if trans_0 is None:
            trans_0 = self.trans_noise(num_batch, num_res)
        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        if res_idx is None:
            res_idx = torch.arange(num_res, device=self._device, dtype=torch.float32)[
                None
            ].repeat(num_batch, 1)
        batch = {"res_mask": res_mask, "diffuse_mask": res_mask, "res_idx": res_idx}

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, end_time, num_timesteps + 1)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        print("helloooooo")
        for i, t_2 in enumerate(ts[1:]):
            if verbose:  # and i % 1 == 0:
                print(f"{i=}, t={t_1.item():.2f}")
                print(
                    torch.cuda.mem_get_info(trans_0.device),
                    torch.cuda.memory_allocated(trans_0.device),
                )

            # Run model.
            trans_t, rotmats_t = prot_traj[-1]
            r = t = torch.ones((num_batch, 1), device=self._device) * t_1
            d_t = t_2 - t_1

            v_trans, v_rot = model.avg_vel(trans_t, rotmats_t, t, r, batch)

            trans_t_2 = self._trans_euler_step(d_t, trans_t, v_trans)
            rotmats_t_2 = self._rots_euler_step(d_t, rotmats_t, v_rot)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        if self._sample_cfg.get("last_step", True) and last_step:
            trans_t, rotmats_t = prot_traj[-1]
            pred_trans_1, pred_rotmats_1 = self.last_step_refinement(
                model, trans_t, rotmats_t, batch
            )
            prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        if DEBUG or return_trans_rot:
            return prot_traj, res_mask
        return atom37_traj, None, None

    def init_prior(self, num_batch, num_res):
        # Set-up initial prior samples
        trans_0 = self.trans_noise(num_batch, num_res)
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        return trans_0, rotmats_0

    @torch.no_grad()
    def n_step_sample(
        self,
        num_batch,
        num_res,
        model,
        n_steps,
        trans_0=None,
        rotmats_0=None,
        ts=None,
        last_step=True,
        self_condition=False,
        rot_schedule="constant",  # "constant", "linear", "exp"
        rot_schedule_rate=10.0,
        return_trans_rot=False,
        use_inference_vf=False,
    ):
        # Set-up initial prior samples
        if trans_0 is None or rotmats_0 is None:
            trans_t, rotmats_t = self.init_prior(num_batch, num_res)
        else:
            trans_t = trans_0
            rotmats_t = rotmats_0

        # Set-up time
        if ts is None:
            ts = torch.linspace(0.0, 1.0, n_steps + 1)

        feats = make_feats(num_batch, num_res, self._device)

        trans_sc = None

        if rot_schedule == "linear":
            t1s = ts[:-1]
            dt = ts[1:] - t1s
            t2s = t1s + dt * rot_schedule_rate * (1 - t1s)
        elif rot_schedule == "exp":
            t1s = ts[:-1]
            dt = torch.exp(-rot_schedule_rate * t1s)
            dt /= dt.sum(dim=-1, keepdims=True)  # Normalize to sum to 1
            t2s = t1s + dt
        else:
            t1s = ts[:-1]
            t2s = ts[1:]

        for t1, t2 in zip(t1s, t2s):
            t = torch.ones((num_batch, 1), device=self._device) * t1
            r = torch.ones((num_batch, 1), device=self._device) * t2

            if self_condition:
                trans_sc, _ = model(trans_t, rotmats_t, t, r, feats)

            if use_inference_vf:
                trans_t, rotmats_t = model.inference_forward_flow(
                    trans_t, rotmats_t, t, r, feats
                )
            else:
                trans_t, rotmats_t = model.forward_flow(
                    trans_t, rotmats_t, t, r, feats, trans_sc=trans_sc
                )

        if last_step and self._sample_cfg.get("last_step", True):
            trans_t, rotmats_t = self.last_step_refinement(
                model, trans_t, rotmats_t, feats
            )

        if DEBUG or return_trans_rot:
            return trans_t, rotmats_t, feats["res_mask"]

        atom37 = all_atom.transrot_to_atom37([(trans_t, rotmats_t)], feats["res_mask"])
        samples = atom37[-1].numpy()
        return samples

    def n_step_sample_with_reward_guidance(
        self,
        num_batch,
        num_res,
        model,
        n_steps,
        reward_fn,
        guidance_scale=1.0,
        guidance_scale_trans=None,  # If None, uses guidance_scale
        guidance_scale_rot=None,  # If None, uses guidance_scale
        trans_0=None,
        rotmats_0=None,
        ts=None,
        last_step=True,
        self_condition=False,
        rot_schedule="constant",
        rot_schedule_rate=10.0,
        return_trans_rot=False,
        output_dir=None,
        setting="rX1",  # "rX1" uses trans_r1/rotmats_r1, "rXt" uses trans_t/rotmats_t for reward computation
        scale_grad_by_t=False,  # If True, scale the projected reward gradients by t
    ):
        """
        Sample with reward guidance applied at each step.

        Args:
            num_batch: Number of samples to generate
            num_res: Number of residues
            model: The flow model
            n_steps: Number of integration steps
            reward_fn: Function that takes (trans, rotmats) and returns per-sample
                rewards of shape (batch_size,). Should be differentiable.
                You can leave this as a placeholder and fill in later.
            guidance_scale: Scale factor for the reward gradient guidance
            trans_0: Optional initial translations
            rotmats_0: Optional initial rotation matrices
            ts: Optional time schedule
            last_step: Whether to do a final refinement step
            self_condition: Whether to use self-conditioning
            rot_schedule: Rotation schedule type ("constant", "linear", "exp")
            rot_schedule_rate: Rate for non-constant schedules
            return_trans_rot: Whether to return trans/rot instead of atom37

        Returns:
            Guided samples at t=1
        """
        # Set-up initial prior samples
        if trans_0 is None or rotmats_0 is None:
            trans_t, rotmats_t = self.init_prior(num_batch, num_res)
        else:
            trans_t = trans_0
            rotmats_t = rotmats_0

        # Set-up time
        if ts is None:
            ts = torch.linspace(0.0, 1.0, n_steps + 1)

        feats = make_feats(num_batch, num_res, self._device)

        trans_sc = None

        if rot_schedule == "linear":
            t1s = ts[:-1]
            dt = ts[1:] - t1s
            t2s = t1s + dt * rot_schedule_rate * (1 - t1s)
        elif rot_schedule == "exp":
            t1s = ts[:-1]
            dt = torch.exp(-rot_schedule_rate * t1s)
            dt /= dt.sum(dim=-1, keepdims=True)
            t2s = t1s + dt
        else:
            t1s = ts[:-1]
            t2s = ts[1:]

        prev_reward = None
        reward_history = []

        # Set up separate guidance scales (fall back to unified guidance_scale if not specified)
        scale_trans = (
            guidance_scale_trans if guidance_scale_trans is not None else guidance_scale
        )
        scale_rot = (
            guidance_scale_rot if guidance_scale_rot is not None else guidance_scale
        )

        print(f"\n{'='*60}")
        print(f"Starting reward-guided sampling with {len(t1s)} steps")
        print(f"Guidance scale (trans): {scale_trans}")
        print(f"Guidance scale (rot):   {scale_rot}")
        print(f"Scale grad by t:        {scale_grad_by_t}")
        print(f"{'='*60}")

        for step_idx, (t1, t2) in enumerate(zip(t1s, t2s)):
            t = torch.ones((num_batch, 1), device=self._device) * t1
            r = torch.ones((num_batch, 1), device=self._device) * t2
            # if t[0].item() >= 0.6:
            #    scale_trans = 0
            #    scale_rot = 0
            # else:
            #    print("starting guidance")
            #    scale_trans = guidance_scale_trans if guidance_scale_trans is not None else guidance_scale
            #    scale_rot = guidance_scale_rot if guidance_scale_rot is not None else guidance_scale

            if self_condition:
                with torch.no_grad():
                    trans_sc, _ = model(trans_t, rotmats_t, t, r, feats)

            trans_t = trans_t.detach().requires_grad_(True)
            rotmats_t = rotmats_t.detach().requires_grad_(True)

            # Compute reward and its gradient
            os.makedirs(f"{output_dir}/r1_preds", exist_ok=True)

            if setting == "rX1":
                trans_r1, rotmats_r1 = model.forward_flow(
                    trans_t, rotmats_t, t, 1, feats, trans_sc=trans_sc
                )
                atom37 = all_atom.transrot_to_atom37_grad(
                    [(trans_r1, rotmats_r1)], feats["res_mask"]
                )
            elif setting == "rXt":
                atom37 = all_atom.transrot_to_atom37_grad(
                    [(trans_t, rotmats_t)], feats["res_mask"]
                )
            else:
                raise ValueError(f"Unknown setting: {setting}. Must be 'rX1' or 'rXt'.")
            reward = reward_fn(
                atom37,
                feats["res_mask"],
                save_path=f"{output_dir}/r1_preds/sample_<IDX>_step_{step_idx}.pdb",
            )

            # reward_sum = reward.mean()
            # reward_sum = reward.sum()

            current_reward = reward.clone().detach().item()
            reward_history.append(current_reward)

            # Determine if reward increased
            if prev_reward is not None:
                delta = current_reward - prev_reward
                direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                print(
                    f"Step {step_idx:3d}/{len(t1s)} | t={t1.item():.4f}→{t2.item():.4f} | reward={current_reward:+.4f} | Δ={delta:+.4f} {direction}"
                )
            else:
                print(
                    f"Step {step_idx:3d}/{len(t1s)} | t={t1.item():.4f}→{t2.item():.4f} | reward={current_reward:+.4f} | (initial)"
                )

            prev_reward = current_reward

            # Compute gradients w.r.t. current state
            # Enable anomaly detection to find source of NaNs (disable for production)
            grad_trans, grad_rotmats = torch.autograd.grad(
                reward,
                [trans_t, rotmats_t],
                create_graph=False,
                retain_graph=False,
            )
            # Handle NaN values from autograd (can occur due to numerical instability)
            # print(f"grad_trans norm: {grad_trans.norm().item():.4f}")
            # print(f"grad_rotmats norm: {grad_rotmats.norm().item():.4f}")

            # Detach gradients
            grad_trans = grad_trans.detach()
            grad_rotmats = grad_rotmats.detach()

            # Check for NaN in gradients and zero them out if found
            # if torch.isnan(grad_trans).any():
            #    #print(f"Warning: NaN in grad_trans at step, zeroing out")
            #    grad_trans = torch.zeros_like(grad_trans)
            # if torch.isnan(grad_rotmats).any():
            #    #print(f"Warning: NaN in grad_rotmats at step, zeroing out")
            #    grad_rotmats = torch.zeros_like(grad_rotmats)

            # print(f"grad_rotmats norm: {grad_rotmats.norm().item():.4f}")
            grad_rotmats = so3_utils.project_to_so3(rotmats_t.detach(), grad_rotmats)
            # Handle NaN values that may arise from projection
            # print(f"grad_trans norm: {grad_trans.norm().item():.4f}")
            # print(f"grad_rotmats norm: {grad_rotmats.norm().item():.4f}")

            # Optionally scale gradients by t (current time step)
            if scale_grad_by_t:
                t_scale = t1.item()
                print("t_scale", t_scale)
                grad_trans = grad_trans * t_scale
                grad_rotmats = grad_rotmats * t_scale

            grad_trans = grad_trans * scale_trans
            grad_rotmats = grad_rotmats * scale_rot

            if scale_trans == 0.0 and scale_rot == 0.0:
                assert (grad_trans == 0).all() and (grad_rotmats == 0).all()
            # else:
            #    assert not ((grad_trans == 0).all() and (grad_rotmats == 0).all())

            # Forward flow step (with gradients)
            with torch.no_grad():
                trans_t, rotmats_t = model.forward_flow(
                    trans_t,
                    rotmats_t,
                    t,
                    r,
                    feats,
                    trans_sc=trans_sc,
                    trans_vf_add=grad_trans,
                    rot_vf_add=grad_rotmats,
                )

        # Print reward summary
        print(f"\n{'='*60}")
        print("Reward Guidance Summary:")
        print(f"  Initial reward: {reward_history[0]:+.4f}")
        print(f"  Final reward:   {reward_history[-1]:+.4f}")
        # print(f"  Total change:   {reward_history[-1] - reward_history[0]:+.4f}")
        num_increases = sum(
            1
            for i in range(1, len(reward_history))
            if reward_history[i] > reward_history[i - 1]
        )
        num_decreases = sum(
            1
            for i in range(1, len(reward_history))
            if reward_history[i] < reward_history[i - 1]
        )
        print(f"  Steps with ↑:   {num_increases}/{len(reward_history)-1}")
        print(f"  Steps with ↓:   {num_decreases}/{len(reward_history)-1}")
        print(
            f"  Max reward:     {max(reward_history):+.4f} (step {reward_history.index(max(reward_history))})"
        )
        print(
            f"  Min reward:     {min(reward_history):+.4f} (step {reward_history.index(min(reward_history))})"
        )
        print(f"{'='*60}\n")

        if last_step and self._sample_cfg.get("last_step", True):
            with torch.no_grad():
                trans_t, rotmats_t = self.last_step_refinement(
                    model, trans_t, rotmats_t, feats
                )

        # Build reward info dict
        reward_info = {
            "reward_history": reward_history,
            "initial_reward": reward_history[0],
            "final_reward": reward_history[-1],
            "total_change": reward_history[-1] - reward_history[0],
            "num_increases": num_increases,
            "num_decreases": num_decreases,
            "max_reward": max(reward_history),
            "max_reward_step": reward_history.index(max(reward_history)),
            "min_reward": min(reward_history),
            "min_reward_step": reward_history.index(min(reward_history)),
            "guidance_scale": guidance_scale,
            "guidance_scale_trans": scale_trans,
            "guidance_scale_rot": scale_rot,
            "scale_grad_by_t": scale_grad_by_t,
            "n_steps": n_steps,
        }

        if DEBUG or return_trans_rot:
            return trans_t, rotmats_t, feats["res_mask"], reward_info

        atom37 = all_atom.transrot_to_atom37([(trans_t, rotmats_t)], feats["res_mask"])
        samples = atom37[-1].numpy()
        return samples, reward_info


class MeanFlowInterpolant(FlowMapInterpolant):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.interval_sampler = create_interval_sampler(cfg.time_sampler)

    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)
        self.augment_batch_if_needed(batch)

        # [B, N, 3]
        trans_1 = batch["trans_1"]  # Angstrom
        # [B, N, 3, 3]
        rotmats_1 = batch["rotmats_1"]

        # [B, N]
        res_mask = batch["res_mask"]
        diffuse_mask = batch["diffuse_mask"]
        num_batch, _ = diffuse_mask.shape

        # [B, 1]
        t, r = self.interval_sampler(num_batch, self._device)
        t, r = t[:, None], r[:, None]
        so3_t = t
        r3_t = t
        noisy_batch["so3_t"] = so3_t
        noisy_batch["r3_t"] = r3_t
        noisy_batch["r"] = r

        # Apply corruptions
        trans_t = self._corrupt_trans(trans_1, t, res_mask, diffuse_mask)
        if torch.any(torch.isnan(trans_t)):
            raise ValueError("NaN in trans_t during corruption")
        noisy_batch["trans_t"] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask, diffuse_mask)
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError("NaN in rotmats_t during corruption")
        noisy_batch["rotmats_t"] = rotmats_t

        return noisy_batch


class SplitMeanFlowInterpolant(FlowMapInterpolant):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fm_time_sampler = create_single_t_sampler(cfg.time_sampler)
        self.sg_time_sampler = create_three_point_sampler(cfg.time_sampler)

    def sample_t_s_r(self, num_batch):
        t, s, r = self.sg_time_sampler(num_batch, self._device)
        t *= 1 - self._cfg.min_t
        s *= 1 - self._cfg.min_t
        return t, s, r

    def corrupt_batch(self, batch):
        self.set_device(batch["res_mask"].device)
        batch = copy.deepcopy(batch)
        self.augment_batch_if_needed(batch)

        # trans_1: [B, N, 3]
        # rotmats_1: [B, N, 3, 3]
        trans_1 = batch["trans_1"]  # Angstrom
        rotmats_1 = batch["rotmats_1"]

        # shape: [B, N]
        res_mask = batch["res_mask"]
        diffuse_mask = batch["diffuse_mask"]
        num_batch, _ = diffuse_mask.shape

        # shape: [B, 1]
        t, s, r = self.sample_t_s_r(num_batch)
        t, s, r = t[:, None], s[:, None], r[:, None]
        batch["t"] = t
        batch["s"] = s
        batch["r"] = r

        if self._cfg.flow_matching_use_new_t:
            fm_t = self.fm_time_sampler(num_batch, self._device)
            batch["flow_matching_t"] = fm_t[:, None]
        else:
            batch["flow_matching_t"] = s

        # Apply corruptions for semigroup loss.
        trans_t = self._corrupt_trans(trans_1, t, res_mask, diffuse_mask)
        if torch.any(torch.isnan(trans_t)):
            raise ValueError("NaN in trans_t during corruption")
        batch["trans_t"] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask, diffuse_mask)
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError("NaN in rotmats_t during corruption")
        batch["rotmats_t"] = rotmats_t

        # Apply corruptions for flow matching.
        trans_t_fm = self._corrupt_trans(
            trans_1, batch["flow_matching_t"], res_mask, diffuse_mask
        )
        if torch.any(torch.isnan(trans_t_fm)):
            raise ValueError("NaN in trans_t_fm during corruption")
        batch["trans_t_fm"] = trans_t_fm

        rotmats_t_fm = self._corrupt_rotmats(
            rotmats_1, batch["flow_matching_t"], res_mask, diffuse_mask
        )

        if torch.any(torch.isnan(rotmats_t_fm)):
            raise ValueError("NaN in rotmats_t_fm during corruption")
        batch["rotmats_t_fm"] = rotmats_t_fm

        return batch


def make_feats(num_batch, num_res, device):
    res_mask = torch.ones(num_batch, num_res, device=device)
    res_idx = torch.arange(num_res, device=device, dtype=torch.float32)[None].repeat(
        num_batch, 1
    )
    feats = {"res_mask": res_mask, "diffuse_mask": res_mask, "res_idx": res_idx}
    return feats
