"""
Script for running inference and evaluation.

Load checkpoint and run sampling for later evaluation (designability).
"""

import os
import time
import json
import shutil
import numpy as np
import pandas as pd
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from analysis import utils as au
from typing import Callable, Optional

from models.module.splitmf_module import SplitMeanFlowModule
from models.module.meanflow_module import MeanFlowModule
from models.inference import DDIMInference

from protein_rewards import ss_rewards, all_atom, motif_rewards
from protein_rewards.motif_rewards import extract_motif_aatype

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)
log = eu.get_pylogger(__name__)



def build_reward_fn_from_config(reward_config: DictConfig):
    """
    Factory function to build a reward function from config.

    Args:
        reward_config: Config with keys:
            - reward_type: "secondary_struct", "motif_rmsd"
            - use_grad_safe: bool, whether to use gradient-safe versions
            - alpha_weight: float, weight for alpha helix (only for secondary_struct)
            - beta_weight: float, weight for beta sheet (only for secondary_struct)

            For motif_rmsd:
            - target_pdb: str, path to reference PDB with target motif
            - motif_position: str, motif specification (e.g., "A1-7/A28-79")
            - atom_type: str, "CA" or "backbone"

    Returns:
        A reward function that takes (atom37_0, atom37_mask, save_path) and returns scores.
    """
    reward_type = reward_config.get("reward_type", "secondary_struct")

    # Handle motif RMSD reward
    if reward_type == "motif_rmsd":
        target_pdb = reward_config.get("target_pdb")
        motif_position = reward_config.get("motif_position")
        atom_type = reward_config.get("atom_type", "CA")

        if target_pdb is None or motif_position is None:
            raise ValueError("motif_rmsd reward requires 'target_pdb' and 'motif_position' in config")

        log.info(f"Built motif RMSD reward: target_pdb={target_pdb}, motif_position={motif_position}, atom_type={atom_type}")
        return motif_rewards.build_motif_reward_fn(
            target_pdb_path=target_pdb,
            motif_position=motif_position,
            atom_type=atom_type,
        )

    use_grad_safe = reward_config.get("use_grad_safe", True)
    alpha_weight = reward_config.get("alpha_weight", 1.0)
    beta_weight = reward_config.get("beta_weight", 1.0)

    # Select the appropriate reward functions
    if use_grad_safe:
        alpha_fn = ss_rewards.alpha_helix_reward_grad_safe
        beta_fn = ss_rewards.beta_sheet_reward_grad_safe
    else:
        alpha_fn = ss_rewards.alpha_helix_reward
        beta_fn = ss_rewards.beta_sheet_reward

    def reward_fn_ss(atom37_0: torch.Tensor, atom37_mask: torch.Tensor, save_path: str | None) -> torch.Tensor:
        ss_rewards.seed_everything(0)
        # Save intermediate structures
        with torch.no_grad():
            for idx in range(len(atom37_0)):
                au.write_prot_to_pdb(atom37_0[idx].detach().cpu().numpy(), save_path.replace("<IDX>", str(idx)), no_indexing=True)

        if len(atom37_0.shape) == 3:
            atom37_0_batched = atom37_0.unsqueeze(0)
        else:
            atom37_0_batched = atom37_0

        # Convert to 4-atom format (N, CA, C, O) for ss_rewards
        B, L, _, D = atom37_0_batched.shape
        prot_coords = torch.zeros((B, L, 4, D), device=atom37_0_batched.device)
        prot_coords[:, :, :3, :] = atom37_0_batched[:, :, :3, :]
        prot_coords[:, :, 3, :] = atom37_0_batched[:, :, 4, :]  # O is at index 4 in atom37

        ss_score = ss_rewards.ss_composition_reward_dense(prot_coords, target_strand=beta_weight, target_helix=alpha_weight)
        ss_score.retain_grad()
        return ss_score


    log.info(f"Built reward function: type={reward_type}, use_grad_safe={use_grad_safe}, "
             f"alpha_weight={alpha_weight}, beta_weight={beta_weight}")
    return reward_fn_ss



class EvalRunner:
    """
        EvalRunner is a class that runs inference for the flow map modules.
        It loads the checkpoint and runs inference for the flow map module.
        It saves the samples to the following directory structure:

        <output_dir>
        |-- <inference_config>
            |-- <length>
               |-- sample_<sample_id>
                  |-- sample.pdb
                  |-- trans_rot.pt (optional)
    """

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        self.cfg = cfg
        self.ckpt_path = cfg.ckpt_path
        self.ckpt_dir = os.path.dirname(self.ckpt_path)
        self.ckpt_cfg = OmegaConf.load(os.path.join(self.ckpt_dir, "config.yaml"))
        self.sample_cfg = cfg.samples
        self.rng = np.random.default_rng(self.cfg.seed)
        ss_rewards.seed_everything(self.cfg.seed)

        self.load_ckpt()
        self.setup_sampling_scheme()
        self.setup_output_dir()
        self.setup_reward_fn()

        log.info(f"Output directory: {self.output_dir}")

    def load_ckpt(self):
        # Read checkpoint and initialize module.
        if self.cfg.module_type == "splitmf":
            self.module = SplitMeanFlowModule.load_from_checkpoint(
                checkpoint_path=self.ckpt_path, cfg=self.ckpt_cfg, map_location="cpu"
            )
        elif self.cfg.module_type == "meanflow":
            self.module = MeanFlowModule.load_from_checkpoint(
                checkpoint_path=self.ckpt_path, cfg=self.ckpt_cfg, map_location="cpu"
            )
        else:
            raise ValueError(f"Unsupported module type: {self.cfg.module_type}")

        log.info(pl.utilities.model_summary.ModelSummary(self.module))
        self.module.eval()
        self.model = self.module.model

        if self.ckpt_cfg.experiment.use_ema:
            self.module.ema_model.store()
            self.module.ema_model.copy_to()
            log.info("EMA model loaded")

    def setup_sampling_scheme(self):
        if self.cfg.do_integration and self.cfg.rot_schedule == "exp":
            # Overwrite the sampling scheme to exp
            self.module.interpolant._rots_cfg.sample_schedule = "exp"
            self.module.interpolant._rots_cfg.exp_rate = self.cfg.rot_schedule_rate

        if self.cfg.do_ddim:
            # Setup DDIM inference
            self.ddim_inference = DDIMInference(
                self.cfg.ddim, self.model, self.module.interpolant
            )

    def setup_output_dir(self):
        if self.cfg.output_dir is None:
            raise ValueError("output_dir is required in the config")

        os.makedirs(self.cfg.output_dir, exist_ok=True)

        sample_config_str = self.sample_config_str()
        self.output_dir = os.path.join(self.cfg.output_dir, sample_config_str)

        os.makedirs(self.output_dir, exist_ok=True)

    def setup_reward_fn(self):
        """Setup reward function from config if reward guidance is enabled."""
        self.motif_aatypes = None
        self.motif_indices = None

        if self.cfg.get("do_reward_guidance", False):
            reward_config = self.cfg.get("reward_config", None)
            if reward_config is not None:
                self.reward_fn = build_reward_fn_from_config(reward_config)

                # Extract motif amino acid types for proper PDB saving
                reward_type = reward_config.get("reward_type", "")
                if reward_type in ["motif_rmsd"]:
                    target_pdb = reward_config.get("target_pdb")
                    motif_position = reward_config.get("motif_position")
                    if target_pdb and motif_position:
                        self.motif_aatypes, self.motif_indices = extract_motif_aatype(
                            target_pdb, motif_position
                        )
                        log.info(f"Extracted {len(self.motif_aatypes)} motif amino acids from {target_pdb}")
            else:
                print("no reward config specified, exiting...."); exit()
        else:
            self.reward_fn = None

    def sample_config_str(self):
        ret = ""
        if self.cfg.get("do_reward_guidance", False):
            guidance_scale = self.cfg.get("guidance_scale", 1.0)
            guidance_scale_trans = self.cfg.get("guidance_scale_trans", None)
            guidance_scale_rot = self.cfg.get("guidance_scale_rot", None)

            # Build scale string: use separate scales if specified, otherwise unified scale
            if guidance_scale_trans is not None or guidance_scale_rot is not None:
                scale_trans = guidance_scale_trans if guidance_scale_trans is not None else guidance_scale
                scale_rot = guidance_scale_rot if guidance_scale_rot is not None else guidance_scale
                scale_str = f"scale_t{scale_trans}_r{scale_rot}"
            else:
                scale_str = f"scale_{guidance_scale}"

            reward_config = self.cfg.get("reward_config", None)
            if reward_config is not None:
                reward_type = reward_config.get("reward_type", "secondary_struct")
                if reward_type == "secondary_struct":
                    alpha_w = reward_config.get("alpha_weight", 1.0)
                    beta_w = reward_config.get("beta_weight", 1.0)
                    ret += f"reward_{reward_type}_a{alpha_w}_b{beta_w}_nsteps_{self.cfg.n_steps}_{scale_str}"
                elif reward_type == "motif_rmsd":
                    motif_pos = reward_config.get("motif_position", "unknown")
                    atom_type = reward_config.get("atom_type", "CA")
                    # Sanitize motif_position for filename (replace / with _)
                    motif_pos_safe = motif_pos.replace("/", "_").replace("-", "to")
                    ret += f"reward_motif_{motif_pos_safe}_{atom_type}_nsteps_{self.cfg.n_steps}_{scale_str}"
                else:
                    ret += f"reward_{reward_type}_nsteps_{self.cfg.n_steps}_{scale_str}"
            else:
                ret += f"reward_guided_nsteps_{self.cfg.n_steps}_{scale_str}"

            setting = self.cfg.get("setting", None)
            if setting is not None:
                ret += f"_{setting}"

            if self.cfg.get("scale_grad_by_t", False):
                ret += "_scale_by_t"

        elif self.cfg.do_integration:
            ret += f"integration_nsteps_{self.cfg.n_steps}"
            if self.cfg.rot_schedule == "exp":
                ret += f"_exp_{self.cfg.rot_schedule_rate}"
        elif self.cfg.do_ddim:
            ret += f"ddim_{self.cfg.ddim.time_steps}_nsteps_{self.cfg.n_steps}"
            if self.cfg.ddim.trans_gamma != 1.0:
                ret += f"_trans_gamma_{self.cfg.ddim.trans_gamma}"
            if self.cfg.ddim.rot_gamma != 1.0:
                ret += f"_rot_gamma_{self.cfg.ddim.rot_gamma}"
        else:
            ret += f"nsteps_{self.cfg.n_steps}"
            if self.cfg.rot_schedule == "linear":
                ret += f"_linear_{self.cfg.rot_schedule_rate}"
            elif self.cfg.rot_schedule == "exp":
                ret += f"_exp_{self.cfg.rot_schedule_rate}"

        if self.cfg.self_condition:
            ret += "_self_condition"
        return ret

    def get_device(self):
        return f"cuda:{eu.get_available_device(1)[0]}"

    def run_sampling(self):
        device = self.get_device()
        self.module.to(device)
        self.module.interpolant.set_device(device)

        log.info(f"Using device: {device}")
        log.info(f"Sampling from flow maps")

        all_sample_lengths = range(
            self.sample_cfg.min_length,
            self.sample_cfg.max_length + 1,
            self.sample_cfg.length_step,
        )
        num_batch = self.sample_cfg.samples_per_length
        

        for length in all_sample_lengths:
            for batch in range(self.sample_cfg.total_batches):
                ss_rewards.seed_everything(batch)
                length_dir = os.path.join(self.output_dir, f"length_{length}")
                self.length_dir = length_dir
                samples = self.sample(num_batch, length)
                if self.cfg.save_trans_rot:
                    trans_batch, rot_batch, _ = samples

                os.makedirs(length_dir, exist_ok=True)
                for sample_id in range(num_batch):
                    sample_id_ = batch * num_batch + sample_id
                    sample_dir = os.path.join(length_dir, f"sample_{sample_id_}")
                    os.makedirs(sample_dir, exist_ok=True)

                    if self.cfg.save_trans_rot:
                        self.save_trans_rot(trans_batch[sample_id], rot_batch[sample_id], sample_dir)
                    else:
                        self.save_samples(samples[sample_id], sample_dir, sample_id_)
                        #self.save_samples(samples[sample_id], sample_dir, sample_id)

            log.info(f"Done writing samples for length {length}")
        log.info(f"Finished sampling")

    def save_trans_rot(self, trans, rot, sample_dir):
        torch.save(
            {"trans": trans, "rot": rot},
            os.path.join(sample_dir, "trans_rot.pt"),
        )

    def save_samples(self, final_pos, sample_dir, sample_id=0):
        pdb_path = os.path.join(sample_dir, f"sample_{sample_id}.pdb")

        # Build aatype array with motif amino acids if available
        aatype = None
        if self.motif_aatypes is not None and self.motif_indices is not None:
            num_res = final_pos.shape[0]
            aatype = np.zeros(num_res, dtype=int)  # Default to ALA (0)
            for idx, aa in zip(self.motif_indices, self.motif_aatypes):
                if idx < num_res:
                    aatype[idx] = aa

        au.write_prot_to_pdb(final_pos, pdb_path, aatype=aatype, no_indexing=True)

    def sample(self, num_batch, num_res):
        # Check if reward guidance is enabled
        do_reward_guidance = self.cfg.get("do_reward_guidance", False)

        if do_reward_guidance:
            samples = self.sample_with_reward_guidance(num_batch, num_res)
        elif self.cfg.do_integration:
            samples = self.module.multi_step_sample(
                num_batch, num_res, self.cfg.n_steps
            )
        elif self.cfg.do_ddim:
            samples = self.ddim_inference.n_step_sample(
                num_batch,
                num_res,
                self.cfg.n_steps,
                last_step=self.cfg.ddim.last_step,
                return_trans_rot=self.cfg.save_trans_rot,
            )
        else:
            samples = self.module.n_step_sample(
                num_batch,
                num_res,
                self.cfg.n_steps,
                self_condition=self.cfg.self_condition,
                rot_schedule=self.cfg.rot_schedule,
                rot_schedule_rate=self.cfg.rot_schedule_rate,
                return_trans_rot=self.cfg.save_trans_rot,
            )
        return samples

    def sample_with_reward_guidance(
        self,
        num_batch: int,
        num_res: int,
        reward_fn: Optional[Callable] = None,
    ):
        """
        Sample with reward guidance.

        Args:
            num_batch: Number of samples to generate
            num_res: Number of residues
            reward_fn: Optional reward function. If None, uses self.reward_fn (from config).

        Returns:
            Samples (atom37 format or trans/rot if save_trans_rot is True)
        """
        if reward_fn is None:
            reward_fn = self.reward_fn
        if reward_fn is None:
            reward_fn = secondary_struct_reward_fn
            log.warning(
                "No reward function configured, using default secondary_struct_reward_fn."
            )

        guidance_scale = self.cfg.get("guidance_scale", 1.0)
        guidance_scale_trans = self.cfg.get("guidance_scale_trans", None)
        guidance_scale_rot = self.cfg.get("guidance_scale_rot", None)
        setting = self.cfg.get("setting", "rX1")  # "rX1" or "rXt"
        scale_grad_by_t = self.cfg.get("scale_grad_by_t", False)

        # Need gradients for reward guidance
        with torch.set_grad_enabled(True):
            result = self.module.interpolant.n_step_sample_with_reward_guidance(
                num_batch=num_batch,
                num_res=num_res,
                model=self.model,
                n_steps=self.cfg.n_steps,
                reward_fn=reward_fn,
                guidance_scale=guidance_scale,
                guidance_scale_trans=guidance_scale_trans,
                guidance_scale_rot=guidance_scale_rot,
                self_condition=self.cfg.self_condition,
                rot_schedule=self.cfg.rot_schedule,
                rot_schedule_rate=self.cfg.rot_schedule_rate,
                return_trans_rot=self.cfg.save_trans_rot,
                output_dir=self.length_dir,
                setting=setting,
                scale_grad_by_t=scale_grad_by_t,
            )

        # Unpack result - now includes reward_info
        if self.cfg.save_trans_rot:
            trans_t, rotmats_t, res_mask, reward_info = result
            samples = (trans_t, rotmats_t, res_mask)
        else:
            samples, reward_info = result

        # Save reward info to output directory
        self.save_reward_info(reward_info, num_res)

        return samples

    def save_reward_info(self, reward_info, num_res):
        """Save reward tracking info to output directory."""
        reward_dir = os.path.join(self.output_dir, f"length_{num_res}")
        os.makedirs(reward_dir, exist_ok=True)

        # Save as JSON
        json_path = os.path.join(reward_dir, "reward_history.json")
        with open(json_path, "w") as f:
            json.dump(reward_info, f, indent=2)
        log.info(f"Saved reward history to {json_path}")

        # Also save as numpy for easy plotting
        np_path = os.path.join(reward_dir, "reward_history.npy")
        np.save(np_path, np.array(reward_info["reward_history"]))
        log.info(f"Saved reward history array to {np_path}")

    def save_motif_info_csv(self):
        """
        Save motif_info.csv for Scaffold-Lab evaluation.

        Creates a CSV with columns: pdb_name, sample_num, contig, redesign_positions, segment_order
        that can be used with scaffold_lab/motif_scaffolding/motif_refolding.py

        Also copies PDB files to a scaffold_lab_pdbs/ directory with the naming convention
        expected by Scaffold-Lab: {pdb_name}_{sample_num}.pdb
        """
        reward_config = self.cfg.get("reward_config", None)
        if reward_config is None:
            log.warning("No reward_config found, skipping motif_info.csv generation")
            return

        reward_type = reward_config.get("reward_type", "")
        if "motif" not in reward_type:
            log.info("Not a motif reward, skipping motif_info.csv generation")
            return

        target_pdb = reward_config.get("target_pdb", "")
        motif_position = reward_config.get("motif_position", "")

        if not motif_position:
            log.warning("No motif_position in config, skipping motif_info.csv")
            return

        # Extract PDB name from target_pdb path
        pdb_name = os.path.splitext(os.path.basename(target_pdb))[0]
        # Remove common suffixes like _motif
        if pdb_name.endswith("_motif"):
            pdb_name = pdb_name[:-6]

        segments = motif_position.split("/")
        num_motif_segments = sum(1 for seg in segments if seg[0].isalpha())
        # Generate chain IDs: A, B, C, ... for each motif segment
        chain_letters = [chr(ord('A') + i) for i in range(num_motif_segments)]
        segment_order = ";".join(chain_letters)

        segments = motif_position.split("/")


        # Collect all generated samples
        all_sample_lengths = range(
            self.sample_cfg.min_length,
            self.sample_cfg.max_length + 1,
            self.sample_cfg.length_step,
        )

        for length in all_sample_lengths:
            rows = []
            length_dir = os.path.join(self.output_dir, f"length_{length}")
            scaffold_lab_pdb_dir = os.path.join(length_dir, "scaffold_lab_pdbs")
            os.makedirs(scaffold_lab_pdb_dir, exist_ok=True)

            if not os.path.exists(length_dir):
                continue

            motif_length = 0
            motif_end = 0
            for seg in segments:
                if seg[0].isalpha():
                    rest = seg[1:]
                    if "-" in rest:
                        start, end = map(int, rest.split("-"))
                        motif_end = max(motif_end, int(end))
                        motif_length += end - start + 1
                    else:
                        motif_end = max(motif_end, int(rest))
                        motif_length += 1

            scaffold_length = length - motif_end

            contig = f"{motif_position}/{scaffold_length}"

            for sample_dir in sorted(os.listdir(length_dir)):
                if not sample_dir.startswith("sample_"):
                    continue
                sample_num = sample_dir.split("_")[1]

                sample_path = os.path.join(length_dir, sample_dir)
                pdb_files = [f for f in os.listdir(sample_path) if f.endswith(".pdb")]

                if pdb_files:
                    src_pdb = os.path.join(sample_path, pdb_files[0])
                    dst_pdb = os.path.join(scaffold_lab_pdb_dir, f"{pdb_name}_{sample_num}.pdb")
                    shutil.copy2(src_pdb, dst_pdb)

                rows.append({
                    "pdb_name": pdb_name,
                    "sample_num": int(sample_num),
                    "contig": contig,
                    "redesign_positions": "",
                    "segment_order": segment_order
                })

            if rows:
                df = pd.DataFrame(rows)
                csv_path = os.path.join(length_dir, "motif_info.csv")
                df.to_csv(csv_path, index=True)
                log.info(f"Saved motif_info.csv to {csv_path} with {len(rows)} samples")
                log.info(f"Copied PDB files to {scaffold_lab_pdb_dir}")
                log.info(f"To run Scaffold-Lab evaluation, use:")
                log.info(f"  inference.backbone_pdb_dir={scaffold_lab_pdb_dir}")
                log.info(f"  inference.motif_csv_path={csv_path}")
                log.info(f"  inference.motif_pdb={target_pdb}")
            else:
                log.warning("No samples found, motif_info.csv not created")


@hydra.main(
    version_base=None,
    config_path="../configs/inference",
    config_name="flowmap_inference",
)
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info(f"Starting inference")
    start_time = time.time()
    sampler = EvalRunner(cfg)
    sampler.run_sampling()

    # Save motif_info.csv for Scaffold-Lab evaluation if using motif reward
    reward_config = cfg.get("reward_config", None)
    if reward_config is not None:
        reward_type = reward_config.get("reward_type", "")
        if reward_type in ["motif_rmsd"]:
            sampler.save_motif_info_csv()

    elapsed_time = time.time() - start_time
    log.info(f"Finished in {elapsed_time:.2f}s")


if __name__ == "__main__":
    run()
