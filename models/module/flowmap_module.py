"""
Lightning module for evaluation code for flow map.
"""

from typing import Any, Dict
import torch
from data import utils as dutils
import os
import wandb
import pandas as pd
import numpy as np
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


class FlowMapModuleWithEvalaution(LightningModuleWrapper):

    def __init__(self, cfg):
        super().__init__()
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = create_meanflow_model(cfg.model)
        if cfg.experiment.get("fine_tune_flow_model"):
            self.model.load_state_dict(torch.load(cfg.experiment.flow_model_path))

        self.validation_epoch_metrics = []
        self.validation_epoch_n_step_5_metrics = []
        self.validation_epoch_n_step_2_metrics = []
        self.validation_epoch_one_step_metrics = []

        self.validation_epoch_samples = []
        self.validation_epoch_n_step_5_samples = []
        self.validation_epoch_n_step_2_samples = []
        self.validation_epoch_one_step_samples = []

        self.setup_interpolant()
        self.save_hyperparameters()

        # Track recent semigroup losses for explosion debugging.
        self._sg_trans_loss_queue = mu.LossQueue()
        self._sg_rot_loss_queue = mu.LossQueue()

    def setup_interpolant(self):
        self.interpolant = SplitMeanFlowInterpolant(self._interpolant_cfg)
        self.model.set_interpolant(self.interpolant)
        self.loss_name = "split_meanflow_loss"

    def one_step_sample(self, num_batch, num_res, self_condition=False):
        return self.n_step_sample(
            num_batch,
            num_res,
            n_steps=1,
            self_condition=self_condition,
        )

    @torch.no_grad()
    def n_step_sample(self, num_batch, num_res, n_steps, **kwargs):
        samples = self.interpolant.n_step_sample(
            num_batch,
            num_res,
            self.model,
            n_steps,
            use_inference_vf=self._exp_cfg.training.inference_sg_loss,
            **kwargs,
        )

        return samples

    @torch.no_grad()
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

        ####### 100-step samples with integration #######
        samples = self.multi_step_sample(num_batch, num_res)
        self.save_samples_and_calc_metrics(
            samples, num_batch, num_res, csv_idx, batch_idx, prefix=""
        )

        ####### 5-step validation code #######
        samples = self.n_step_sample(
            num_batch,
            num_res,
            n_steps=5,
            self_condition=self._interpolant_cfg.self_condition,
        )
        self.save_samples_and_calc_metrics(
            samples, num_batch, num_res, csv_idx, batch_idx, prefix="n_step_5_"
        )

        ####### 2-step validation code #######
        samples = self.n_step_sample(
            num_batch,
            num_res,
            n_steps=2,
            self_condition=self._interpolant_cfg.self_condition,
        )
        self.save_samples_and_calc_metrics(
            samples, num_batch, num_res, csv_idx, batch_idx, prefix="n_step_2_"
        )

        ####### One step validation code #######
        samples = self.one_step_sample(
            num_batch,
            num_res,
            self_condition=self._interpolant_cfg.self_condition,
        )
        self.save_samples_and_calc_metrics(
            samples, num_batch, num_res, csv_idx, batch_idx, prefix="one_step_"
        )

    def on_validation_epoch_end(self):
        self.aggregate_metrics_and_log(
            sample_var_name="validation_epoch_samples",
            metric_var_name="validation_epoch_metrics",
            table_key="valid",
        )

        self.aggregate_metrics_and_log(
            sample_var_name="validation_epoch_n_step_5_samples",
            metric_var_name="validation_epoch_n_step_5_metrics",
            table_key="valid_n_step_5",
        )

        self.aggregate_metrics_and_log(
            sample_var_name="validation_epoch_n_step_2_samples",
            metric_var_name="validation_epoch_n_step_2_metrics",
            table_key="valid_n_step_2",
        )

        self.aggregate_metrics_and_log(
            sample_var_name="validation_epoch_one_step_samples",
            metric_var_name="validation_epoch_one_step_metrics",
            table_key="valid_one_step",
        )

        if self._data_cfg.dataset == "scope_overfit_one":
            samples = self.multi_step_sample(500, 60)
            self.log_dict(self.overfit_metric(samples), "valid")

            samples = self.n_step_sample(500, 60, 5)
            self.log_dict(self.overfit_metric(samples), "valid_n_step_5")

            samples = self.n_step_sample(500, 60, 2)
            self.log_dict(self.overfit_metric(samples), "valid_n_step_2")

            samples = self.one_step_sample(500, 60)
            self.log_dict(self.overfit_metric(samples), "valid_one_step")

    def log_dict(self, dictionary: Dict[str, Any], table_key: str):
        for key, value in dictionary.items():
            self.log_scalar(
                f"{table_key}/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    def aggregate_metrics_and_log(self, sample_var_name, metric_var_name, table_key):
        if len(getattr(self, sample_var_name)) > 0:
            self.logger.log_table(
                key=f"{table_key}/samples",
                columns=["sample_path", "global_step", "Protein"],
                data=getattr(self, sample_var_name),
            )
            getattr(self, sample_var_name).clear()
        val_epoch_metrics = pd.concat(getattr(self, metric_var_name))
        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self.log_scalar(
                f"{table_key}/{metric_name}",
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        getattr(self, metric_var_name).clear()

    def save_samples_and_calc_metrics(
        self, samples, num_batch, num_res, csv_idx, batch_idx, prefix=""
    ):
        batch_metrics = []
        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                "samples",
                f"{prefix}sample_{csv_idx[i].item()}_idx_{batch_idx}_len_{num_res}",
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos, os.path.join(sample_dir, "sample.pdb"), no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                getattr(self, f"validation_epoch_{prefix}samples").append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_idx = residue_constants.atom_order["CA"]
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

        batch_metrics = pd.DataFrame(batch_metrics)
        getattr(self, f"validation_epoch_{prefix}metrics").append(batch_metrics)

        return batch_metrics

    def overfit_metric(self, batch: np.ndarray):
        """
        Calculate distance metrics between predicted batch and overfitting ground truth samples.
        Loads overfitting samples directly from the training dataset to ensure all unique samples are included.
        Processes the entire batch at once, computing metrics for each batch sample against all overfit samples.

        Args:
            batch: Batch of data containing the predicted sample structure as atom37 array [B, num_res, 37, 3]

        Returns:
            Dictionary containing distance metrics (averaged over batch)
        """
        ca_idx = residue_constants.atom_order["CA"]

        # Initialize overfit samples on first call.
        if not hasattr(self, "overfit_samples"):
            self.overfit_samples = []

            # Access train dataset through trainer
            train_dataset = self.trainer.datamodule._train_dataset
            overfit_samples_data = train_dataset.get_overfit_samples(max_samples=10)

            # Convert to atom37 format and extract CA atoms
            for sample_data in overfit_samples_data:
                trans_1 = sample_data["trans_1"].unsqueeze(0)  # [1, num_res, 3]
                rotmats_1 = sample_data["rotmats_1"].unsqueeze(0)  # [1, num_res, 3, 3]
                gt_atom37 = all_atom.to_atom37(
                    trans_1, rotmats_1
                )  # [1, num_res, 37, 3]
                gt_atom37_np = gt_atom37[0].detach().cpu().numpy()  # [num_res, 37, 3]

                # Extract CA atoms
                ca_pos = gt_atom37_np[:, ca_idx, :]  # [num_res, 3]
                self.overfit_samples.append(ca_pos)

        batch_size = batch.shape[0]
        pred_ca_pos = batch[:, :, ca_idx, :]  # [B, num_res, 3]

        # Store minimum MSE and RMSD for each batch sample
        batch_min_mse = np.full(batch_size, float("inf"))
        batch_min_rmsd = np.full(batch_size, float("inf"))

        # For each overfit sample, compute distances to all batch samples
        for gt_ca_pos in self.overfit_samples:
            # Broadcast overfit sample to match batch size [B, num_res, 3]
            gt_ca_pos_broadcast = np.tile(
                gt_ca_pos[None, :, :], (batch_size, 1, 1)
            )  # [B, num_res, 3]

            # Center both structures
            pred_centered = pred_ca_pos - pred_ca_pos.mean(
                axis=1, keepdims=True
            )  # [B, num_res, 3]
            gt_centered = gt_ca_pos_broadcast - gt_ca_pos_broadcast.mean(
                axis=1, keepdims=True
            )  # [B, num_res, 3]

            # Align structures (batch processing)
            aligned_pred_trans, aligned_target_trans, _ = dutils.batch_align_structures(
                torch.from_numpy(pred_centered),
                torch.from_numpy(gt_centered),
            )

            # Calculate MSE for each batch sample [B]
            mse_per_sample = np.mean(
                (aligned_pred_trans.numpy() - aligned_target_trans.numpy()) ** 2,
                axis=(1, 2),  # Average over residues and alpha-carbon coordinates
            )  # [B]
            rmsd_per_sample = np.sqrt(mse_per_sample)  # [B]

            # Update minimum MSE and RMSD for each batch sample
            update_mask = mse_per_sample < batch_min_mse
            batch_min_mse[update_mask] = mse_per_sample[update_mask]
            batch_min_rmsd[update_mask] = rmsd_per_sample[update_mask]

        # Return average metrics over the batch
        return {
            "overfit_ca_mse": np.mean(batch_min_mse),
            "overfit_ca_rmsd": np.mean(batch_min_rmsd),
        }


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
