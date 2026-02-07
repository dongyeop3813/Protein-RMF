from typing import Any, Dict
import torch
import os
import wandb
import pandas as pd
from models.module.lightning_wrapper import LightningModuleWrapper
from analysis import metrics
from analysis import utils as au
from models.architecture.factory import create_meanflow_model
from models import utils as mu
from data.interpolant import SplitMeanFlowInterpolant
from data import all_atom
from data import so3_utils
from data import residue_constants
from pytorch_lightning.loggers.wandb import WandbLogger
import random


class TestTrueRegressionModule(LightningModuleWrapper):

    def __init__(self, cfg):
        super().__init__()
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = create_meanflow_model(cfg.model)

        # Set-up interpolant
        self.interpolant = SplitMeanFlowInterpolant(cfg.interpolant)
        self.model.set_interpolant(self.interpolant)

        self.validation_epoch_metrics = []
        self.validation_epoch_one_step_metrics = []
        self.validation_epoch_samples = []
        self.validation_epoch_one_step_samples = []
        self.save_hyperparameters()

        # Track recent semigroup losses for explosion debugging.
        self._sg_trans_loss_queue = mu.LossQueue()
        self._sg_rot_loss_queue = mu.LossQueue()

    def training_step(self, batch: Any):
        timer = self.make_timer()

        # Corrupt the batch for semigroup and flow matching losses.
        self.interpolant.corrupt_batch(batch)

        # Log the batch and model state for debugging if NaN ValueError is encountered.
        try:
            batch_losses = self.model_step(batch)
        except Exception as e:
            if isinstance(e, ValueError) and ("NaN" in str(e) or "nan" in str(e)):
                self.log_on_exception(batch)
            raise e

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
        fm_losses = self.flow_matching_loss(batch, loss_mask)

        # Semigroup loss.
        sg_losses = self.semigroup_loss(batch, loss_mask)

        unweighted_loss = (
            self.training_cfg.flow_matching_loss_weight * fm_losses["fm_loss"]
            + self.training_cfg.semigroup_loss_weight * sg_losses["semigroup_loss"]
        )

        # Adaptive loss weighting.
        loss = weighted_loss(unweighted_loss, self.training_cfg.loss_p)

        return {
            **fm_losses,
            **sg_losses,
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

        t_clip = self.training_cfg.t_normalize_clip
        loss_norm_scale = torch.clamp(1 - t, min=t_clip)[..., None]

        # Model output predictions
        pred_trans_1, pred_rot_1 = self.model(trans_t, rot_t, t, t, batch)

        # Flow matching on translation.
        trans_error = (
            (trans_1 - pred_trans_1) / loss_norm_scale * self.training_cfg.trans_scale
        )
        trans_loss = torch.sum(trans_error**2 * loss_mask[..., None], dim=(-1, -2))

        # Flow matching on rotation.
        gt_rots_vf = so3_utils.calc_rot_vf(rot_t, rot_1)
        pred_rots_vf = so3_utils.calc_rot_vf(rot_t, pred_rot_1)
        rot_error = (gt_rots_vf - pred_rots_vf) / loss_norm_scale
        rot_loss = torch.sum(rot_error**2 * loss_mask[..., None], dim=(-1, -2))

        # Normalize by the number of residues.
        trans_loss /= loss_denom
        rot_loss /= loss_denom

        weighted_trans_loss = trans_loss * self.training_cfg.translation_loss_weight
        weighted_rot_loss = rot_loss * self.training_cfg.rotation_loss_weight

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

        return loss_dict

    def semigroup_loss(self, batch, loss_mask):
        loss_denom = torch.sum(loss_mask, dim=-1) * 3

        # Estimate the xr from the model.
        trans_t = batch["trans_t"]
        rot_t = batch["rotmats_t"]
        trans_1 = batch["trans_1"]
        rot_1 = batch["rotmats_1"]
        t = batch["t"]
        r = batch["r"]

        hat_trans_1, hat_rot_1 = self.model(trans_t, rot_t, t, r, batch)

        trans_error = (trans_1 - hat_trans_1) * self.training_cfg.trans_scale
        trans_loss = torch.sum(trans_error**2 * loss_mask[..., None], dim=(-1, -2))
        rot_loss = torch.sum(
            so3_utils.rot_squared_dist(hat_rot_1, rot_1),
            dim=-1,
        )

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
            self.log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

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
                )

        self.log_scalar(
            "train/loss", total_losses["split_meanflow_loss"], batch_size=num_batch
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

    def one_step_sample(self, num_batch, num_res, self_condition=False):
        return self.n_step_sample(
            num_batch,
            num_res,
            n_steps=1,
            self_condition=self_condition,
        )

    @torch.no_grad()
    def n_step_sample(self, num_batch, num_res, n_steps, self_condition=False, ts=None):
        samples = self.interpolant.n_step_sample(
            num_batch,
            num_res,
            self.model,
            n_steps,
            ts=ts,
            self_condition=self_condition,
        )

        return samples

    def multi_step_sample(self, num_batch, num_res, num_timesteps=None):
        atom37_traj, _, _ = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            num_timesteps=num_timesteps,
        )
        samples = atom37_traj[-1].numpy()
        return samples

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch["res_mask"]
        num_batch, num_res = res_mask.shape
        self.interpolant.set_device(res_mask.device)
        csv_idx = batch["csv_idx"]

        ####### Original validation code (with integration) #######
        samples = self.multi_step_sample(num_batch, num_res)
        batch_metrics = []
        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                "samples",
                f"sample_{csv_idx[i].item()}_idx_{batch_idx}_len_{num_res}",
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos, os.path.join(sample_dir, "sample.pdb"), no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_idx = residue_constants.atom_order["CA"]
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)

        ####### One step validation code #######
        samples = self.one_step_sample(
            num_batch,
            num_res,
            self_condition=self._interpolant_cfg.self_condition,
        )
        batch_metrics = []
        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                "samples",
                f"one_step_sample_{csv_idx[i].item()}_idx_{batch_idx}_len_{num_res}",
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos, os.path.join(sample_dir, "sample.pdb"), no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_one_step_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_idx = residue_constants.atom_order["CA"]
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_one_step_metrics.append(batch_metrics)

    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key="valid/samples",
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples,
            )
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self.log_scalar(
                f"valid/{metric_name}",
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

        if len(self.validation_epoch_one_step_samples) > 0:
            self.logger.log_table(
                key="valid_one_step/samples",
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_one_step_samples,
            )
            self.validation_epoch_one_step_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_one_step_metrics)
        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self.log_scalar(
                f"valid_one_step/{metric_name}",
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_one_step_metrics.clear()


def weighted_loss(loss, p=0.0):
    """
    Adaptive weighting of losses.

    Args:
        loss: The loss to weight.
        p: The power to weight the loss by.

    Returns:
        The weighted loss.
    """
    weight = 1 / ((loss + 1e-3) ** p)
    return weight.detach() * loss
