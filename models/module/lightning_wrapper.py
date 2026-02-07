from typing import Any
import math
import torch
import time
import os
import logging
import torch.distributed as dist
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def second(self):
        return time.time() - self.start_time

    def minute(self):
        return (time.time() - self.start_time) / 60.0


class TrainingEpochTimerHook(Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        self.timer = Timer()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = self.timer.minute()
        pl_module.log(
            "train/epoch_time_minutes",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )


class LightningModuleWrapper(LightningModule):
    """
    Lightning module wrapper with utilities functions.

        1. Set the checkpoint and inference directories for distributed training.
        2. Measure the minute per epoch training time.
        3. Provide utilities functions for the underlying module.
    """

    def __init__(self):
        super().__init__()
        self.print_logger = logging.getLogger(__name__)
        self._checkpoint_dir = None
        self._inference_dir = None

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def configure_callbacks(self):
        return [TrainingEpochTimerHook()]

    def log_scalar(
        self,
        key,
        value,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        batch_size=None,
        sync_dist=False,
        rank_zero_only=True,
    ):
        """
        Log function for scalar values.
        It is a wrapper around the LightningModule.log function.
        It prevents the user from accidentally syncing dist when rank_zero_only=True.
        """
        if sync_dist and rank_zero_only:
            raise ValueError("Unable to sync dist when rank_zero_only=True")
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only,
        )

    def make_timer(self):
        return Timer()

    def log_on_exception(self, input: Any, msg: str = "NaN loss encountered"):
        """
        Log on exception. Save the input batch and model state for debugging.
        """
        debug_dir = os.path.join(self.checkpoint_dir, f"debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Try Lightning checkpoint (includes optimizer/scheduler states)
        ckpt_path = os.path.join(
            debug_dir, f"step_{int(self.global_step)}_trainer.ckpt"
        )
        self.trainer.save_checkpoint(ckpt_path)

        # Serialize input batch and model state for debugging
        def _to_cpu(obj):
            if torch.is_tensor(obj):
                return obj.detach().cpu()
            if isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                typ = type(obj)
                return typ(_to_cpu(x) for x in obj)
            return obj

        input_cpu = _to_cpu(input)
        batch_path = os.path.join(
            debug_dir, f"step_{int(self.global_step)}_input_batch.pt"
        )
        model_path = os.path.join(
            debug_dir, f"step_{int(self.global_step)}_model_state.pt"
        )
        torch.save(input_cpu, batch_path)
        torch.save(self.model.state_dict(), model_path)
        self.print_logger.info(
            f"[{int(self.global_step)}] {msg}."
            f"Saved artifacts: input_batch={batch_path}, model={model_path}, checkpoint={ckpt_path}"
        )

    def configure_optimizers(self):
        optim_type = self._exp_cfg.get("optimizer_type", "AdamW")
        if optim_type == "Muon":
            from muon import MuonWithAuxAdam

            param_groups = self.model.model.muon_param_groups()
            param_groups = [
                dict(
                    params=param_groups["use_muon"],
                    use_muon=True,
                    lr=0.02,
                    weight_decay=0.01,
                ),
                dict(
                    params=param_groups["use_adam"],
                    use_muon=False,
                    lr=self._exp_cfg.optimizer.lr,
                    betas=(0.9, 0.95),
                    weight_decay=0.01,
                ),
            ]
            optimizer = MuonWithAuxAdam(param_groups)
        elif optim_type == "Adam":
            optimizer = torch.optim.Adam(
                params=self.model.parameters(), **self._exp_cfg.optimizer
            )
        elif optim_type == "CsAdam":
            from ..cadam import CautiousAdamW

            optimizer = CautiousAdamW(
                params=self.model.parameters(), **self._exp_cfg.optimizer
            )
        else:
            optimizer = torch.optim.AdamW(
                params=self.model.parameters(), **self._exp_cfg.optimizer
            )

        # Learning rate scheduler: Linear warmup + Cosine Annealing (1 cycle)
        warmup_steps = getattr(self._exp_cfg, "lr_warmup_steps", 0)
        max_steps = getattr(self._exp_cfg, "lr_max_steps", 0)
        min_lr_ratio = getattr(self._exp_cfg, "lr_min_ratio", 0.0)

        if max_steps > 0:
            # Linear warmup + Cosine Annealing scheduler

            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup: lr increases from 0 to base_lr
                    return float(step + 1) / float(max(warmup_steps, 1))
                else:
                    # Cosine annealing: lr decreases from base_lr to min_lr
                    progress = float(step - warmup_steps) / float(
                        max(max_steps - warmup_steps, 1)
                    )
                    progress = min(progress, 1.0)  # Clamp to [0, 1]
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval": "step",
                "frequency": 1,
                "name": "lr_warmup_cosine",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif warmup_steps > 0:
            # Linear warmup only (no cosine annealing)

            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                return 1.0

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval": "step",
                "frequency": 1,
                "name": "lr_warmup",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

    def on_before_optimizer_step(self, optimizer):
        # Compute grad norms and rename keys for cleaner logging
        if self._exp_cfg.debug:
            raw_norms = grad_norm(self.model, norm_type=2, group_separator="/")
            if not raw_norms:
                return
            cleaned_norms = {}
            for key, value in raw_norms.items():
                if key.endswith("_total"):
                    cleaned_norms["grad_norm/total"] = value
                else:
                    # Original key example: "grad_2.0_norm/<param_path>"
                    suffix = key.split("/", 1)[1] if "/" in key else key
                    cleaned_norms[f"grad_norm/layers/{suffix}"] = value
            self.log_dict(cleaned_norms)

    @property
    def training_cfg(self):
        return self._exp_cfg.training

    def on_save_checkpoint(self, checkpoint):
        if self._exp_cfg.use_ema:
            checkpoint["ema_model"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self._exp_cfg.use_ema:
            from torch_ema import ExponentialMovingAverage

            self.ema_model = ExponentialMovingAverage(
                self.model.parameters(), decay=self._exp_cfg.ema_decay
            )
            try:
                self.ema_model.load_state_dict(checkpoint["ema_model"])
            except KeyError as e:
                self.print_logger.warning(f"Failed to load EMA model: {e}")
