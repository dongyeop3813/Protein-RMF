## Riemannian MeanFlow (Protein Backbone Experiments)

This repository contains the **protein-backbone experiments** for **Riemannian MeanFlow (RMF)**, a few-step flow-map framework for generative modeling on Riemannian manifolds.

RMF learns **average-velocity flow maps directly on manifolds**, using:
- semigroup-based Riemannian MeanFlow objective,
- x₁-prediction on manifolds (predicting manifold-valued endpoints),
- stabilization techniques tailored to high-dimensional manifolds.

---

### 1. Environment setup

We follow the original FrameFlow setup, using the provided Conda environment file.

```bash
# Create environment with all dependencies.
conda env create -f fm.yml

# Activate environment.
conda activate fm

# (If needed) install torch-scatter for your CUDA/PyTorch version.
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Install this repository as an editable package.
pip install -e .
```

You may adapt the CUDA/PyTorch/torch-scatter versions to your hardware, but the above combination mirrors the original FrameFlow configuration and is known to work for the protein experiments.

---

### 2. Data for protein backbone experiments

All experiments in this repository are **protein-only**:
- training uses protein backbone datasets such as SCOPe / PDB,
- evaluation uses standard designability / novelty / diversity metrics.

Hydra configs under `configs/data/` control data sources and loaders. By default:

- [`configs/base.yaml`](configs/base.yaml) uses `data: scope`,
- which points to [`configs/data/scope.yaml`](configs/data/scope.yaml).

You will need to:
- download or preprocess protein backbone datasets (e.g., SCOPe/PDB) according to your environment,
- set dataset root paths in `configs/data/*.yaml` to your local directories.

For convenience, you can reuse the original FrameFlow datasets:

- preprocessed PDB / SCOPe backbones,
- pretrained weights.

They are hosted on Zenodo (see the original FrameFlow README for the latest links). After downloading, unpack them as:

```bash
tar -xzvf preprocessed_pdb.tar.gz
tar -xzvf weights.tar.gz
tar -xzvf preprocessed_scope.tar.gz
```

Your directory will typically look like:

```bash
├── analysis
├── configs
├── data
├── experiments
├── media
├── models
├── openfold
├── processed_pdb
├── processed_scope
└── weights
```

The relevant code paths are:
- `data/datasets.py` – dataset definitions,
- `data/protein_dataloader.py` – Lightning `DataModule` for proteins.

---

### 3. RMF model sizes (AdaLN_S / AdaLN_M / AdaLN_L)

We provide three RMF model sizes for protein backbones:

- `configs/model/AdaLN_S.yaml` – **small** RMF model (fast, lightweight),
- `configs/model/AdaLN_M.yaml` – **medium** RMF model (~92M params),
- `configs/model/AdaLN_L.yaml` – **large** RMF model (~437M params).

You can switch the model size from the command line using the Hydra `model` config group, e.g.:

```bash
# Small model
python experiments/train_se3_splitmeanflow.py model=AdaLN_S

# Medium model
python experiments/train_se3_splitmeanflow.py model=AdaLN_M

# Large model
python experiments/train_se3_splitmeanflow.py model=AdaLN_L
```

The architecture is built on top of the invariant point attention trunk used in prior protein backbone models, but trained with **Riemannian MeanFlow objectives** and **x₁-prediction** on SE(3) frames.

---

### 4. Training RMF on protein backbones

The main training entrypoint in this repository is:

- `experiments/train_se3_splitmeanflow.py`

which uses the semigroup RMF configuration `configs/semi_mf.yaml` by default.

Example (single-node, few GPUs):

```bash
python -W ignore experiments/train_se3_splitmeanflow.py \
  model=AdaLN_M \
  data.dataset=scope \
  experiment.num_devices=4 \
  experiment.trainer.max_epochs=1000 \
  experiment.wandb.project=rmf-protein
```

Key configuration components:

- `configs/model/AdaLN_{S,M,L}.yaml` – RMF architecture size,
- `configs/base.yaml` – interpolant on SE(3)\u207f, training hyperparameters, logging and checkpointing,
- `configs/base_experiment.yaml` and `configs/base_meanflow.yaml` – alternative training baselines when needed.

Checkpoints and Hydra configs are saved under `ckpt/...` as specified by `experiment.checkpointer.dirpath`.

---

### 5. Few-step inference and reward-guided design

Protein-backbone inference with RMF uses:

- `experiments/inference_se3_flowmaps.py`

and configs under `configs/inference/`.

#### 5.1. Basic protein backbone sampling (RMF)

Use `configs/inference/flowmap_inference.yaml` and set:

- `ckpt_path: "path/to/your_checkpoint.ckpt"` – path to a trained RMF checkpoint,
- `output_dir: ./inference_outputs/flowmaps` – directory where samples will be written,
- `n_steps`, `do_ddim`, `do_integration` – to choose 1, 5, 10, or 100-step regimes.

Run:

```bash
python experiments/inference_se3_flowmaps.py
```

This will:
- load the RMF model from `ckpt_path`,
- generate protein backbone samples for lengths configured in `samples.*`,
- save PDB files (and optionally intermediate SE(3) trajectories) under `output_dir`.

#### 5.2. Reward-guided RMF (protein-only)

RMF supports **reward-guided inference** on protein backbones through manifold reward gradients and **x₁ look-ahead**, as described in the paper.

We provide example configs:

- `configs/inference/reward_guidance_ss.yaml` – guidance toward secondary-structure composition,
- `configs/inference/reward_guidance_motif.yaml` – guidance toward motif RMSD objectives.

You need to set:

- `ckpt_path` to a compatible RMF checkpoint,
- `reward_config` fields:
  - secondary-structure: `alpha_weight`, `beta_weight`,
  - motif: `target_pdb`, `motif_position`, `atom_type`.

Sampling is then run via the same script:

```bash
python experiments/inference_se3_flowmaps.py
```

RMF’s few-step flow maps enable **reward look-ahead**: rewards are evaluated on predicted terminal states rather than noisy intermediates, improving guidance efficiency for protein design.
