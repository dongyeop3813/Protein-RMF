import abc
import numpy as np
import pandas as pd
import logging
import tree
import torch
import random

from torch.utils.data import Dataset
from data import utils as du
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def _rog_filter(df, quantile):
    y_quant = pd.pivot_table(
        df,
        values="radius_gyration",
        index="modeled_seq_len",
        aggfunc=lambda x: np.quantile(x, quantile),
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    max_len = df.modeled_seq_len.max()
    pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

    row_rog_cutoffs = df.modeled_seq_len.map(lambda x: pred_y[x - 1])
    return df[df.radius_gyration < row_rog_cutoffs]


def _length_filter(data_csv, min_res, max_res):
    return data_csv[
        (data_csv.modeled_seq_len >= min_res) & (data_csv.modeled_seq_len <= max_res)
    ]


def _plddt_percent_filter(data_csv, min_plddt_percent):
    return data_csv[data_csv.num_confident_plddt > min_plddt_percent]


def _max_coil_filter(data_csv, max_coil_percent):
    return data_csv[data_csv.coil_percent <= max_coil_percent]


def _process_csv_row(processed_file_path):
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats)

    # Only take modeled residues.
    modeled_idx = processed_feats["modeled_idx"]
    min_idx = np.min(modeled_idx)
    max_idx = np.max(modeled_idx)
    del processed_feats["modeled_idx"]
    processed_feats = tree.map_structure(
        lambda x: x[min_idx : (max_idx + 1)], processed_feats
    )

    # Run through OpenFold data transforms.
    chain_feats = {
        "aatype": torch.tensor(processed_feats["aatype"]).long(),
        "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
        "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_gt_frames"])[
        :, 0
    ]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()
    res_plddt = processed_feats["b_factors"][:, 1]
    res_mask = torch.tensor(processed_feats["bb_mask"]).int()

    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    chain_idx = processed_feats["chain_index"]
    res_idx = processed_feats["residue_index"]
    new_res_idx = np.zeros_like(res_idx)
    new_chain_idx = np.zeros_like(res_idx)
    all_chain_idx = np.unique(chain_idx).tolist()
    shuffled_chain_idx = (
        np.array(random.sample(all_chain_idx, len(all_chain_idx)))
        - np.min(all_chain_idx)
        + 1
    )
    for i, chain_id in enumerate(all_chain_idx):
        chain_mask = (chain_idx == chain_id).astype(int)
        chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
        new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

        # Shuffle chain_index
        replacement_chain_id = shuffled_chain_idx[i]
        new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask
    if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
        raise ValueError(f"Found NaNs in {processed_file_path}")
    return {
        "res_plddt": res_plddt,
        "aatype": chain_feats["aatype"],
        "rotmats_1": rotmats_1,
        "trans_1": trans_1,
        "res_mask": res_mask,
        "chain_idx": new_chain_idx,
        "res_idx": new_res_idx,
    }


def _add_plddt_mask(feats, plddt_threshold):
    feats["plddt_mask"] = torch.tensor(feats["res_plddt"] > plddt_threshold).int()


def _read_clusters(cluster_path):
    pdb_to_cluster = {}
    with open(cluster_path, "r") as f:
        for i, line in enumerate(f):
            for chain in line.split(" "):
                pdb = chain.split("_")[0]
                pdb_to_cluster[pdb.upper()] = i
    return pdb_to_cluster


class BaseDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_cfg,
        is_training,
        task,
    ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values("modeled_seq_len", ascending=False)
        self._create_split(metadata_csv)
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    def __len__(self):
        return len(self.csv)

    @abc.abstractmethod
    def _filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
        pass

    def _create_split(self, data_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = data_csv
            self._log.info(f"Training: {len(self.csv)} examples")
        else:
            if self._dataset_cfg.max_eval_length is None:
                eval_lengths = data_csv.modeled_seq_len
            else:
                eval_lengths = data_csv.modeled_seq_len[
                    data_csv.modeled_seq_len <= self._dataset_cfg.max_eval_length
                ]
            all_lengths = np.sort(eval_lengths.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.dataset_cfg.num_eval_lengths
            )
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = data_csv[data_csv.modeled_seq_len.isin(eval_lengths)]

            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby("modeled_seq_len").sample(
                self.dataset_cfg.samples_per_eval_length, replace=True, random_state=123
            )
            eval_csv = eval_csv.sort_values("modeled_seq_len", ascending=False)
            self.csv = eval_csv
            self._log.info(
                f"Validation: {len(self.csv)} examples with lengths {eval_lengths}"
            )
        self.csv["index"] = list(range(len(self.csv)))

    def process_csv_row(self, csv_row):
        path = csv_row["processed_path"]
        seq_len = csv_row["modeled_seq_len"]
        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]
        processed_row = _process_csv_row(path)
        if use_cache:
            self._cache[path] = processed_row
        return processed_row

    def _sample_scaffold_mask(self, batch, rng):
        trans_1 = batch["trans_1"]
        num_res = trans_1.shape[0]
        min_motif_size = int(self._dataset_cfg.min_motif_percent * num_res)
        max_motif_size = int(self._dataset_cfg.max_motif_percent * num_res)

        # Sample the total number of residues that will be used as the motif.
        total_motif_size = self._rng.integers(low=min_motif_size, high=max_motif_size)

        # Sample motifs at different locations.
        num_motifs = rng.integers(low=1, high=total_motif_size)

        # Attempt to sample
        attempt = 0
        while attempt < 100:
            # Sample lengths of each motif.
            motif_lengths = np.sort(
                rng.integers(low=1, high=max_motif_size, size=(num_motifs,))
            )

            # Truncate motifs to not go over the motif length.
            cumulative_lengths = np.cumsum(motif_lengths)
            motif_lengths = motif_lengths[cumulative_lengths < total_motif_size]
            if len(motif_lengths) == 0:
                attempt += 1
            else:
                break
        if len(motif_lengths) == 0:
            motif_lengths = [total_motif_size]

        # Sample start location of each motif.
        seed_residues = rng.integers(
            low=0, high=num_res - 1, size=(len(motif_lengths),)
        )

        # Construct the motif mask.
        motif_mask = torch.zeros(num_res)
        for motif_seed, motif_len in zip(seed_residues, motif_lengths):
            motif_mask[motif_seed : min(motif_seed + motif_len, num_res)] = 1.0
        scaffold_mask = 1 - motif_mask
        return scaffold_mask * batch["res_mask"]

    def setup_inpainting(self, feats, rng):
        diffuse_mask = self._sample_scaffold_mask(feats, rng)
        if "plddt_mask" in feats:
            diffuse_mask = diffuse_mask * feats["plddt_mask"]
        if torch.sum(diffuse_mask) < 1:
            # Should only happen rarely.
            diffuse_mask = torch.ones_like(diffuse_mask)
        feats["diffuse_mask"] = diffuse_mask

    def __getitem__(self, row_idx):
        # Process data example.
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row)

        if self._dataset_cfg.add_plddt_mask:
            _add_plddt_mask(feats, self._dataset_cfg.min_plddt_threshold)
        else:
            feats["plddt_mask"] = torch.ones_like(feats["res_mask"])

        if self.task == "hallucination":
            feats["diffuse_mask"] = torch.ones_like(feats["res_mask"]).bool()
        elif self.task == "inpainting":
            if self._dataset_cfg.inpainting_percent < random.random():
                feats["diffuse_mask"] = torch.ones_like(feats["res_mask"])
            else:
                rng = self._rng if self.is_training else np.random.default_rng(seed=123)
                self.setup_inpainting(feats, rng)
                # Center based on motif locations
                motif_mask = 1 - feats["diffuse_mask"]
                trans_1 = feats["trans_1"]
                motif_1 = trans_1 * motif_mask[:, None]
                motif_com = torch.sum(motif_1, dim=0) / (torch.sum(motif_mask) + 1)
                trans_1 -= motif_com[None, :]
                feats["trans_1"] = trans_1
        else:
            raise ValueError(f"Unknown task {self.task}")
        feats["diffuse_mask"] = feats["diffuse_mask"].int()

        # Storing the csv index is helpful for debugging.
        feats["csv_idx"] = torch.ones(1, dtype=torch.long) * row_idx
        return feats


class ScopeDataset(BaseDataset):

    def _filter_metadata(self, raw_csv):
        filter_cfg = self.dataset_cfg.filter
        data_csv = _length_filter(
            raw_csv, filter_cfg.min_num_res, filter_cfg.max_num_res
        )
        # Apply RoG filter to remove outliers (top 4% by default)
        if hasattr(filter_cfg, "rog_quantile") and filter_cfg.rog_quantile is not None:
            data_csv = _rog_filter(data_csv, filter_cfg.rog_quantile)
        data_csv["oligomeric_detail"] = "monomeric"
        return data_csv


class ScopeDatasetOverfitOne(ScopeDataset):
    """
    Dataset that only contains one example.
    It's for debugging purposes with overfitting to one example.
    """

    def _select_len60_examples(self, data_csv, count):
        """Select up to `count` examples with modeled_seq_len == 60."""
        len60 = data_csv[data_csv["modeled_seq_len"] == 60]
        if len60.empty:
            raise ValueError("No length-60 examples available for overfitting.")
        replace = len(len60) < count
        return len60.sample(count, replace=replace, random_state=123)

    def _select_example(self, data_csv):
        """Return a single-row DataFrame containing the overfit example."""
        if len(data_csv) == 0:
            raise ValueError("No examples available after filtering Scope metadata.")

        # Priority 1: explicit processed_path.
        if "overfit_processed_path" in self.dataset_cfg:
            target_path = self.dataset_cfg.overfit_processed_path
            if target_path:
                match = data_csv[data_csv["processed_path"] == target_path]
                if match.empty:
                    raise ValueError(
                        f"`overfit_processed_path`={target_path} not found in metadata."
                    )
                return match.iloc[[0]]

        # Priority 2: explicit pdb identifier.
        if "overfit_pdb_name" in self.dataset_cfg:
            target_pdb = self.dataset_cfg.overfit_pdb_name
            if target_pdb:
                match = data_csv[data_csv["pdb_name"].str.upper() == target_pdb.upper()]
                if match.empty:
                    raise ValueError(
                        f"`overfit_pdb_name`={target_pdb} not found in metadata."
                    )
                return match.iloc[[0]]

        # Priority 3: explicit sequence length (defaults to 60 residues).
        target_len = getattr(self.dataset_cfg, "overfit_seq_len", 60)
        if target_len is not None:
            target_len = int(target_len)
            match = data_csv[data_csv["modeled_seq_len"] == target_len]
            if not match.empty:
                return match.iloc[[0]]
            self._log.warning(
                f"No example with modeled_seq_len={target_len} found; falling back to overfit_idx."
            )

        # Priority 4: stable index (supports negative indices).
        if (
            "overfit_idx" in self.dataset_cfg
            and self.dataset_cfg.overfit_idx is not None
        ):
            idx = int(self.dataset_cfg.overfit_idx)
        else:
            idx = 0
        if idx < 0:
            idx = len(data_csv) + idx
        idx = max(0, min(idx, len(data_csv) - 1))
        return data_csv.iloc[[idx]]

    def _create_split(self, data_csv):
        use_len60_subset = getattr(self.dataset_cfg, "overfit_use_len60_subset", False)
        len60_count = int(getattr(self.dataset_cfg, "overfit_len60_count", 10))

        if use_len60_subset:
            selected = self._select_len60_examples(data_csv, len60_count).copy()
            selected.reset_index(drop=True, inplace=True)
            if self.is_training:
                repeat_count = (
                    getattr(self.dataset_cfg, "overfit_repeat_count", 100) or 100
                )
                repeat_count = max(1, int(repeat_count))
                repeated_examples = pd.concat(
                    [selected] * repeat_count, ignore_index=True
                )
                repeated_examples["index"] = np.arange(
                    len(repeated_examples), dtype=int
                )
                self.csv = repeated_examples
                self._log.info(
                    f"Overfit dataset (len=60 subset): {len(self.csv)} copies from {len(selected)} examples"
                )
            else:
                # For evaluation, always pick the first sample and repeat 10 times.
                first_sample = selected.iloc[[0]].copy()
                repeated_examples = pd.concat([first_sample] * 10, ignore_index=True)
                repeated_examples["index"] = np.arange(
                    len(repeated_examples), dtype=int
                )
                self.csv = repeated_examples
                self._log.info(
                    f"Overfit dataset (len=60 subset, eval): {len(self.csv)} copies from first sample"
                )
            return

        if self.is_training:
            example = self._select_example(data_csv).copy()
            example.reset_index(drop=True, inplace=True)
            repeat_count = (
                getattr(self.dataset_cfg, "overfit_repeat_count", 1000) or 1000
            )
            repeat_count = max(1, int(repeat_count))
            repeated_examples = pd.concat([example] * repeat_count, ignore_index=True)
            repeated_examples["index"] = np.arange(repeat_count, dtype=int)
            self.csv = repeated_examples
            pdb_name = example.iloc[0].get("pdb_name", "unknown")
            seq_len = int(example.iloc[0].get("modeled_seq_len", -1))
            self._log.info(
                f"Overfit dataset: {repeat_count} copies (pdb={pdb_name}, len={seq_len})"
            )
        else:
            eval_lengths = [60]
            eval_csv = data_csv[data_csv.modeled_seq_len.isin(eval_lengths)]

            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.sample(10, replace=True, random_state=123)
            eval_csv = eval_csv.sort_values("modeled_seq_len", ascending=False)
            self.csv = eval_csv
            self._log.info(
                f"Validation: {len(self.csv)} examples with lengths {eval_lengths}"
            )
            self.csv["index"] = list(range(len(self.csv)))

    def __getitem__(self, row_idx):
        # Process data example.
        csv_row = self.csv.iloc[row_idx]
        processed_path = csv_row["processed_path"]

        # Cache key: use processed_path for overfitting dataset since same examples repeat
        cache_key = processed_path

        # Check cache for base features (file I/O is expensive)
        if cache_key not in self._cache:
            feats = self.process_csv_row(csv_row)

            # Add plddt_mask (deterministic, can be cached)
            if self._dataset_cfg.add_plddt_mask:
                _add_plddt_mask(feats, self._dataset_cfg.min_plddt_threshold)
            else:
                feats["plddt_mask"] = torch.ones_like(feats["res_mask"])

            # Cache base features for hallucination task (fully deterministic)
            if self.task == "hallucination":
                feats["diffuse_mask"] = torch.ones_like(feats["res_mask"]).bool()
                feats["diffuse_mask"] = feats["diffuse_mask"].int()
                # Deep copy to avoid sharing tensor references
                self._cache[cache_key] = tree.map_structure(
                    lambda x: (
                        x.clone()
                        if isinstance(x, torch.Tensor)
                        else (
                            x.copy()
                            if hasattr(x, "copy")
                            and not isinstance(x, (str, int, float, bool))
                            else x
                        )
                    ),
                    feats,
                )
            else:
                # For inpainting, cache only base features (diffuse_mask has randomness)
                # Deep copy to avoid sharing tensor references
                self._cache[cache_key] = tree.map_structure(
                    lambda x: (
                        x.clone()
                        if isinstance(x, torch.Tensor)
                        else (
                            x.copy()
                            if hasattr(x, "copy")
                            and not isinstance(x, (str, int, float, bool))
                            else x
                        )
                    ),
                    feats,
                )
        else:
            # Load from cache and make a copy to avoid modifying cached data
            feats = tree.map_structure(
                lambda x: (
                    x.clone()
                    if isinstance(x, torch.Tensor)
                    else (
                        x.copy()
                        if hasattr(x, "copy")
                        and not isinstance(x, (str, int, float, bool))
                        else x
                    )
                ),
                self._cache[cache_key],
            )

        # Handle task-specific masks (inpainting needs fresh computation due to randomness)
        if self.task == "inpainting":
            if self._dataset_cfg.inpainting_percent < random.random():
                feats["diffuse_mask"] = torch.ones_like(feats["res_mask"])
            else:
                rng = self._rng if self.is_training else np.random.default_rng(seed=123)
                self.setup_inpainting(feats, rng)
                # Center based on motif locations
                motif_mask = 1 - feats["diffuse_mask"]
                trans_1 = feats["trans_1"]
                motif_1 = trans_1 * motif_mask[:, None]
                motif_com = torch.sum(motif_1, dim=0) / (torch.sum(motif_mask) + 1)
                trans_1 -= motif_com[None, :]
                feats["trans_1"] = trans_1
            feats["diffuse_mask"] = feats["diffuse_mask"].int()

        # Storing the csv index is helpful for debugging.
        feats["csv_idx"] = torch.ones(1, dtype=torch.long) * row_idx
        return feats

    def get_overfit_samples(self, max_samples=2):
        """
        Get the unique overfitting samples used in this dataset.
        Uses cache if available to avoid redundant file I/O.

        Args:
            max_samples: Maximum number of unique samples to return (default: 2)

        Returns:
            List of sample dictionaries, each containing:
                - trans_1: translation coordinates [num_res, 3]
                - rotmats_1: rotation matrices [num_res, 3, 3]
                - res_mask: residue mask [num_res]
        """
        if not self.is_training:
            # For validation dataset, return empty list
            return []

        # Get unique samples from csv (by processed_path)
        if not hasattr(self, "csv") or len(self.csv) == 0:
            return []

        unique_csv = self.csv.drop_duplicates(subset=["processed_path"])
        num_samples = min(max_samples, len(unique_csv))

        samples = []
        for i in range(num_samples):
            csv_row = unique_csv.iloc[i]
            # Get the first index of this processed_path in the dataset
            row_idx = self.csv[
                self.csv["processed_path"] == csv_row["processed_path"]
            ].index[0]

            # Load sample from dataset (uses cache if available)
            sample_data = self[row_idx]

            # Extract ground truth structures
            samples.append(
                {
                    "trans_1": sample_data["trans_1"],
                    "rotmats_1": sample_data["rotmats_1"],
                    "res_mask": sample_data["res_mask"],
                }
            )

        return samples


class PdbDataset(BaseDataset):

    def __init__(
        self,
        *,
        dataset_cfg,
        is_training,
        task,
    ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

        # Process clusters
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values("modeled_seq_len", ascending=False)

        self._pdb_to_cluster = _read_clusters(self._dataset_cfg.cluster_path)
        self._max_cluster = max(self._pdb_to_cluster.values())
        self._missing_pdbs = 0

        def cluster_lookup(pdb):
            pdb = pdb.upper()
            if pdb not in self._pdb_to_cluster:
                self._pdb_to_cluster[pdb] = self._max_cluster + 1
                self._max_cluster += 1
                self._missing_pdbs += 1
            return self._pdb_to_cluster[pdb]

        metadata_csv["cluster"] = metadata_csv["pdb_name"].map(cluster_lookup)
        self._create_split(metadata_csv)
        self._all_clusters = dict(enumerate(self.csv["cluster"].unique().tolist()))
        self._num_clusters = len(self._all_clusters)

    def _filter_metadata(self, raw_csv):
        """Filter metadata."""
        filter_cfg = self.dataset_cfg.filter
        data_csv = raw_csv[raw_csv.oligomeric_detail.isin(filter_cfg.oligomeric)]
        data_csv = data_csv[data_csv.num_chains.isin(filter_cfg.num_chains)]
        data_csv = _length_filter(
            data_csv, filter_cfg.min_num_res, filter_cfg.max_num_res
        )
        data_csv = _max_coil_filter(data_csv, filter_cfg.max_coil_percent)
        data_csv = _rog_filter(data_csv, filter_cfg.rog_quantile)
        return data_csv
