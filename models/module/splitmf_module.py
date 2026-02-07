from typing import Any, Dict
import torch

from models.module.flowmap_module import FlowMapModuleWithEvalaution, weighted_loss
from models import utils as mu
from data import all_atom
from data import so3_utils
import random


class SplitMeanFlowModule(FlowMapModuleWithEvalaution):

    def training_step(self, batch: Any):
        timer = self.make_timer()

        # Corrupt the batch for semigroup and flow matching losses.
        batch = self.interpolant.corrupt_batch(batch)

        # Log the batch and model state for debugging if NaN ValueError is encountered.
        try:
            batch_losses = self.model_step(batch)
        except Exception as e:
            if isinstance(e, ValueError) and ("NaN" in str(e) or "nan" in str(e)):
                self.log_on_exception(batch)
            raise e

        if self._exp_cfg.debug:
            self.log_on_loss_explosion(batch, batch_losses)

        num_batch = batch_losses["fm_trans_loss"].shape[0]
        self.log_loss(batch, batch_losses)

        # Log training throughput
        self.log_scalar(
            "train/length",
            batch["res_mask"].shape[1],
            prog_bar=False,
            batch_size=num_batch,
        )
        self.log_scalar("train/batch_size", num_batch, prog_bar=False)
        self.log_scalar("train/examples_per_second", num_batch / timer.second())

        # This is the final training objective.
        train_loss = batch_losses["split_meanflow_loss"].mean()
        return train_loss

    def model_step(self, batch: Any):
        loss_mask = batch["res_mask"] * batch["diffuse_mask"]
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError("Empty batch encountered")

        # Flow matching loss.
        if self.training_cfg.inference_fm_loss:
            fm_losses, pred_x1 = self.inference_flow_matching_loss(batch, loss_mask)
        else:
            fm_losses, pred_x1 = self.flow_matching_loss(batch, loss_mask)

        aux_loss = torch.zeros_like(fm_losses["fm_loss"])

        # Auxiliary losses (backbone atom and pairwise distance).

        # Obtain backbone atoms from the model output / data.
        pred_bb_atoms = all_atom.to_atom37(*pred_x1)[:, :, :3]
        gt_bb_atoms = all_atom.to_atom37(batch["trans_1"], batch["rotmats_1"])[:, :, :3]

        # Scale the backbone atoms into nm units.
        pred_bb_atoms *= self.training_cfg.bb_atom_scale
        gt_bb_atoms *= self.training_cfg.bb_atom_scale

        if self.training_cfg.use_bb_loss:
            bb_atom_loss = self.bb_loss(batch, loss_mask, pred_bb_atoms, gt_bb_atoms)
            aux_loss += bb_atom_loss
        else:
            bb_atom_loss = torch.zeros_like(aux_loss)

        if self.training_cfg.use_pair_dist_loss:
            pair_dist_loss = self.pair_loss(loss_mask, pred_bb_atoms, gt_bb_atoms)
            aux_loss += pair_dist_loss
        else:
            pair_dist_loss = torch.zeros_like(aux_loss)

        # Aux loss is only applied when t > t_pass.
        aux_loss *= batch["flow_matching_t"].squeeze(-1) > self.training_cfg.t_pass
        aux_loss *= self.training_cfg.aux_loss_weight
        aux_loss = torch.clamp(aux_loss, max=5)

        # Semigroup loss.
        if not self.training_cfg.only_fm:
            if self.training_cfg.inference_sg_loss:
                sg_losses = self.semigroup_inference_loss(batch, loss_mask)
            else:
                sg_losses = self.semigroup_loss(batch, loss_mask)
        else:
            zeros = torch.zeros_like(fm_losses["fm_trans_loss"])
            sg_losses = {
                "sg_trans_loss": zeros,
                "sg_rot_loss": zeros,
                "weighted_sg_trans_loss": zeros,
                "weighted_sg_rot_loss": zeros,
                "semigroup_loss": zeros,
            }

        unweighted_loss = (
            self.training_cfg.flow_matching_loss_weight * fm_losses["fm_loss"]
            + self.training_cfg.semigroup_loss_weight * sg_losses["semigroup_loss"]
        )

        # Adaptive loss weighting.
        loss = weighted_loss(unweighted_loss, self.training_cfg.loss_p) + aux_loss

        return {
            **fm_losses,
            **sg_losses,
            "bb_atom_loss": bb_atom_loss,
            "pair_dist_loss": pair_dist_loss,
            "unweighted_loss": unweighted_loss,
            "split_meanflow_loss": loss,
        }

    def flow_matching_loss(self, batch, loss_mask):
        trans_t = batch["trans_t_fm"]
        rot_t = batch["rotmats_t_fm"]
        trans_1 = batch["trans_1"]
        rot_1 = batch["rotmats_1"]
        t = batch["flow_matching_t"]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3

        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                trans_sc, _ = self.model(trans_t, rot_t, t, t, batch)
        else:
            trans_sc = None

        t_clip = self.training_cfg.t_normalize_clip
        loss_norm_scale = torch.clamp(1 - t, min=t_clip)[..., None]

        # Model output predictions
        pred_trans_1, pred_rot_1 = self.model(trans_t, rot_t, t, t, batch, trans_sc)

        # Flow matching on translation.
        trans_error = (
            (trans_1 - pred_trans_1) / loss_norm_scale * self.training_cfg.trans_scale
        )
        trans_loss = torch.sum(trans_error**2, dim=(-1, -2))

        # Flow matching on rotation.
        if self._exp_cfg.training.get("euclid_norm_on_so3", False):
            rot_error = (pred_rot_1 - rot_1) / loss_norm_scale[..., None]
            rot_loss = torch.sum(rot_error**2, dim=(-1, -2, -3))
        else:
            gt_rots_vf = so3_utils.calc_rot_vf(rot_t, rot_1)
            pred_rots_vf = so3_utils.calc_rot_vf(rot_t, pred_rot_1)
            rot_error = (gt_rots_vf - pred_rots_vf) / loss_norm_scale
            rot_loss = torch.sum(rot_error**2, dim=(-1, -2))

        # Normalize by the number of residues.
        trans_loss /= loss_denom
        rot_loss /= loss_denom

        weighted_trans_loss = trans_loss * self.training_cfg.translation_loss_weight
        weighted_rot_loss = rot_loss * self.training_cfg.rotation_loss_weight

        # Clamp the loss to avoid exploding case.
        if self.training_cfg.fm_loss_clamp and self.global_step > 10000:
            weighted_trans_loss = torch.clamp(weighted_trans_loss, max=5)
            weighted_rot_loss = torch.clamp(weighted_rot_loss, max=5)

        fm_loss = weighted_trans_loss + weighted_rot_loss
        if torch.any(torch.isnan(fm_loss)):
            raise ValueError("NaN loss encountered")

        loss_dict = {
            "fm_trans_loss": trans_loss,
            "fm_rot_loss": rot_loss,
            "weighted_fm_trans_loss": weighted_trans_loss,
            "weighted_fm_rot_loss": weighted_rot_loss,
            "fm_loss": fm_loss,
        }

        return loss_dict, (pred_trans_1, pred_rot_1)

    def inference_flow_matching_loss(self, batch, loss_mask):
        trans_t = batch["trans_t_fm"]
        rot_t = batch["rotmats_t_fm"]
        trans_1 = batch["trans_1"]
        rot_1 = batch["rotmats_1"]
        t = batch["flow_matching_t"]
        loss_denom = torch.sum(loss_mask, dim=-1) * 3

        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                trans_sc, _ = self.model(trans_t, rot_t, t, t, batch)
        else:
            trans_sc = None

        t_clip = self.training_cfg.t_normalize_clip
        loss_norm_scale = torch.clamp(1 - t, min=t_clip)[..., None]

        # Model output predictions
        pred_trans_1, pred_rot_1 = self.model(trans_t, rot_t, t, t, batch, trans_sc)

        # Flow matching on translation.
        trans_error = (
            (trans_1 - pred_trans_1) / loss_norm_scale * self.training_cfg.trans_scale
        )
        trans_loss = torch.sum(trans_error**2, dim=(-1, -2))

        # Flow matching on rotation.
        if self._exp_cfg.training.get("euclid_norm_on_so3", False):
            rot_error = (pred_rot_1 - rot_1) / loss_norm_scale[..., None]
            rot_loss = torch.sum(rot_error**2, dim=(-1, -2, -3))
        else:
            gt_rots_vf = 10 * (1 - t)[..., None] * so3_utils.calc_rot_vf(rot_t, rot_1)
            pred_rots_vf = so3_utils.calc_rot_vf(rot_t, pred_rot_1)
            rot_error = (gt_rots_vf - pred_rots_vf) / loss_norm_scale
            rot_loss = torch.sum(rot_error**2, dim=(-1, -2))

        # Normalize by the number of residues.
        trans_loss /= loss_denom
        rot_loss /= loss_denom

        weighted_trans_loss = trans_loss * self.training_cfg.translation_loss_weight
        weighted_rot_loss = rot_loss * self.training_cfg.rotation_loss_weight

        # Clamp the loss to avoid exploding case.
        if self.training_cfg.fm_loss_clamp and self.global_step > 10000:
            weighted_trans_loss = torch.clamp(weighted_trans_loss, max=5)
            weighted_rot_loss = torch.clamp(weighted_rot_loss, max=5)

        fm_loss = weighted_trans_loss + weighted_rot_loss
        if torch.any(torch.isnan(fm_loss)):
            raise ValueError("NaN loss encountered")

        loss_dict = {
            "fm_trans_loss": trans_loss,
            "fm_rot_loss": rot_loss,
            "weighted_fm_trans_loss": weighted_trans_loss,
            "weighted_fm_rot_loss": weighted_rot_loss,
            "fm_loss": fm_loss,
        }

        return loss_dict, (pred_trans_1, pred_rot_1)

    def bb_loss(self, batch, loss_mask, pred_bb_atoms, gt_bb_atoms):
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3),
        )

        bb_atom_loss /= loss_denom

        return bb_atom_loss

    def pair_loss(self, loss_mask, pred_bb_atoms, gt_bb_atoms):
        num_batch, num_res = loss_mask.shape
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * 3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * 3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        # Loss mask for the pairwise distance loss.
        # gt_pair_dists, pred_pair_dists: [B, 3 * N, 3 * N]
        # Only consider local loss with gt_pair_dists < 0.6 nm.
        local_loss_mask = gt_pair_dists < 0.6
        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * local_loss_mask, dim=(1, 2)
        )
        dist_mat_loss /= torch.sum(local_loss_mask, dim=(1, 2)) + 1

        if torch.any(torch.isnan(dist_mat_loss)):
            raise ValueError("NaN loss encountered in pair_loss")

        return dist_mat_loss

    def semigroup_loss(self, batch, loss_mask):
        loss_denom = torch.sum(loss_mask, dim=-1) * 3

        # Estimate the xr from the model.
        trans_t = batch["trans_t"]
        rot_t = batch["rotmats_t"]
        t = batch["t"]
        s = batch["s"]
        r = batch["r"]

        with torch.no_grad():
            if self._exp_cfg.use_ema and self.training_cfg.use_ema_on_sg_loss:
                with self.ema_model.average_parameters():
                    hat_xs = self.model.forward_flow(trans_t, rot_t, t, s, batch)
                    hat_xr = self.model.forward_flow(*hat_xs, s, r, batch)
                    trans_r, rot_r = hat_xr
            else:
                hat_xs = self.model.forward_flow(trans_t, rot_t, t, s, batch)
                hat_xr = self.model.forward_flow(*hat_xs, s, r, batch)
                trans_r, rot_r = hat_xr

        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                trans_sc, _ = self.model(trans_t, rot_t, t, r, batch)
        else:
            trans_sc = None

        # Calculate the semigroup loss.
        if self.training_cfg.semigroup_loss_on_velocity:
            s_clip = self.training_cfg.sg_s_clip
            loss_norm_scale = ((1 - s) / torch.clamp(1 - s, min=s_clip))[..., None]

            r_minus_t_clip = self.training_cfg.sg_r_minus_t_clip
            r_minus_t = torch.clamp(r - t, min=r_minus_t_clip)[..., None]
            u_trans_tgt = (trans_r - trans_t) / r_minus_t
            u_rot_tgt = so3_utils.calc_rot_vf(rot_t, rot_r) / r_minus_t

            u_trans, u_rot = self.model.avg_vel(
                trans_t, rot_t, t, r, batch, trans_sc=trans_sc
            )

            trans_error = (
                (u_trans - u_trans_tgt)
                * self.training_cfg.trans_scale
                * loss_norm_scale
            )
            trans_loss = torch.sum(trans_error**2, dim=(-1, -2))

            rot_error = (u_rot - u_rot_tgt) * loss_norm_scale
            rot_loss = torch.sum(rot_error**2, dim=(-1, -2))
        else:
            hat_trans_r, hat_rot_r = self.model.forward_flow(
                trans_t, rot_t, t, r, batch
            )

            loss_scale = (torch.clamp(r - t, min=1e-2) ** 2).squeeze(-1)

            trans_error = (hat_trans_r - trans_r) * self.training_cfg.trans_scale
            trans_loss = torch.sum(trans_error**2, dim=(-1, -2))
            rot_loss = torch.sum(
                so3_utils.rot_squared_dist(hat_rot_r, rot_r),
                dim=-1,
            )

            trans_loss /= loss_scale
            rot_loss /= loss_scale

        # Normalize by the number of residues.
        trans_loss /= loss_denom
        rot_loss /= loss_denom

        weighted_trans_loss = trans_loss * self.training_cfg.translation_loss_weight
        weighted_rot_loss = rot_loss * self.training_cfg.rotation_loss_weight

        # Clamp the loss to avoid exploding case.
        if self.training_cfg.sg_loss_clamp and self.global_step > 10000:
            weighted_trans_loss = torch.clamp(weighted_trans_loss, max=5)
            weighted_rot_loss = torch.clamp(weighted_rot_loss, max=5)

        semigroup_loss = weighted_trans_loss + weighted_rot_loss
        if torch.any(torch.isnan(semigroup_loss)):
            raise ValueError("NaN loss encountered")

        return {
            "sg_trans_loss": trans_loss,
            "sg_rot_loss": rot_loss,
            "weighted_sg_trans_loss": weighted_trans_loss,
            "weighted_sg_rot_loss": weighted_rot_loss,
            "semigroup_loss": semigroup_loss,
        }

    def semigroup_inference_loss(self, batch, loss_mask):
        loss_denom = torch.sum(loss_mask, dim=-1) * 3

        # Estimate the xr from the model.
        trans_t = batch["trans_t"]
        rot_t = batch["rotmats_t"]
        t = batch["t"]
        s = batch["s"]
        r = batch["r"]

        with torch.no_grad():
            hat_xs = self.model.inference_forward_flow(trans_t, rot_t, t, s, batch)
            hat_xr = self.model.inference_forward_flow(*hat_xs, s, r, batch)
            trans_r, rot_r = hat_xr

        # Calculate the semigroup loss.
        if self.training_cfg.semigroup_loss_on_velocity:
            r_minus_t = torch.clamp(r - t, min=1e-4)[..., None]
            u_trans_tgt = (trans_r - trans_t) / r_minus_t
            u_rot_tgt = so3_utils.calc_rot_vf(rot_t, rot_r) / r_minus_t

            u_trans, u_rot = self.model.inference_avg_vel(trans_t, rot_t, t, r, batch)

            trans_error = (u_trans - u_trans_tgt) * self.training_cfg.trans_scale
            trans_loss = torch.sum(trans_error**2, dim=(-1, -2))

            rot_loss = torch.sum((u_rot - u_rot_tgt) ** 2, dim=(-1, -2))
        else:
            hat_trans_r, hat_rot_r = self.model.inference_forward_flow(
                trans_t, rot_t, t, r, batch
            )

            trans_error = (hat_trans_r - trans_r) * self.training_cfg.trans_scale
            trans_loss = torch.sum(trans_error**2, dim=(-1, -2))
            rot_loss = torch.sum(so3_utils.rot_squared_dist(hat_rot_r, rot_r), dim=-1)

        # Normalize by the number of residues.
        trans_loss /= loss_denom
        rot_loss /= loss_denom

        weighted_trans_loss = trans_loss * self.training_cfg.translation_loss_weight
        weighted_rot_loss = rot_loss * self.training_cfg.rotation_loss_weight
        semigroup_loss = weighted_trans_loss + weighted_rot_loss
        if torch.any(torch.isnan(semigroup_loss)):
            raise ValueError("NaN loss encountered")

        return {
            "sg_trans_loss": trans_loss,
            "sg_rot_loss": rot_loss,
            "weighted_sg_trans_loss": weighted_trans_loss,
            "weighted_sg_rot_loss": weighted_rot_loss,
            "semigroup_loss": semigroup_loss,
        }

    def log_loss(self, noisy_batch: Any, batch_losses: Dict[str, torch.Tensor]):
        num_batch = batch_losses["fm_trans_loss"].shape[0]

        total_losses = {k: torch.mean(v) for k, v in batch_losses.items()}
        for k, v in total_losses.items():
            self.log_scalar(
                f"train/{k}",
                v,
                prog_bar=False,
                batch_size=num_batch,
                sync_dist=False,
                rank_zero_only=False,
            )

        # Losses to track. Stratified across t.
        for loss_name, batch_loss in batch_losses.items():
            if "fm" in loss_name:
                batch_t = noisy_batch["flow_matching_t"]
                t_label = "t"
            elif "sg" in loss_name:
                batch_t = noisy_batch["r"] - noisy_batch["t"]
                t_label = "r-t"
            else:
                continue

            stratified_losses = mu.t_stratified_loss(
                batch_t, batch_loss, loss_name=loss_name, t_label=t_label
            )

            for k, v in stratified_losses.items():
                self.log_scalar(
                    f"stratified_loss/{k}",
                    v,
                    prog_bar=False,
                    batch_size=num_batch,
                    sync_dist=False,
                    rank_zero_only=False,
                )

        self.log_scalar(
            "train/loss",
            total_losses["split_meanflow_loss"],
            batch_size=num_batch,
            sync_dist=False,
            rank_zero_only=False,
        )

    def log_on_loss_explosion(self, batch: Any, batch_losses: Dict[str, torch.Tensor]):
        sg_trans_mean = batch_losses["sg_trans_loss"].mean().item()
        sg_rot_mean = batch_losses["sg_rot_loss"].mean().item()

        trans_exploded, median_trans = self._sg_trans_loss_queue.record_and_check(
            sg_trans_mean
        )
        if trans_exploded:
            self.log_on_exception(
                batch,
                msg=(
                    "SG trans loss exploded: "
                    f"{sg_trans_mean} > 100x median({median_trans})"
                ),
            )

        rot_exploded, median_rot = self._sg_rot_loss_queue.record_and_check(sg_rot_mean)
        if rot_exploded:
            self.log_on_exception(
                batch,
                msg=(
                    "SG rot loss exploded: "
                    f"{sg_rot_mean} > 100x median({median_rot})"
                ),
            )
