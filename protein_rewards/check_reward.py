import torch
import torch.nn as nn
import torch.nn.functional as F

import all_atom
import ss_rewards
from pathlib import Path
from openfold.np import protein as of_protein
import numpy as np


pdb_path="pdb_examples/alpha_helices/5MBA.pdb"
pdb_path="pdb_examples/beta_sheets/108L.pdb"
#pdb_path="pdb_examples/beta_sheets/4R80.pdb"
#pdb_path="pdb_examples/alpha_helices/1IC2.pdb"

chain_id = None  # e.g. "A" to force a chain; None often parses the first/only chain
pdb_str = open(pdb_path, "r").read()

# Parse PDB into OpenFold Protein object (atom37)
prot = of_protein.from_pdb_string(pdb_str, chain_id=chain_id)  # :contentReference[oaicite:0]{index=0}

atom37_0  = prot.atom_positions   # (L, 37, 3) float32
atom37_mask = prot.atom_mask        # (L, 37)    float32 (1 = present)
aatype      = prot.aatype           # (L,)       int (0..20 with UNK)
res_idx     = prot.residue_index    # (L,)       residue numbering from PDB

print(atom37_0.shape, atom37_mask.shape, aatype.shape)


if len(atom37_0.shape) == 3:
    atom37_0 = np.expand_dims(atom37_0, axis=0)
print(atom37_0.shape)
atom37_0 = torch.from_numpy(atom37_0).float().requires_grad_(True)

# need 4 (N, CA, C, O) or 5 (N, CA, C, O, H)
B, L, _, D = atom37_0.shape
prot_coords = torch.zeros((B, L, 4, D))
prot_coords[:, :, :3, :] = atom37_0[:, :, :3, :]
prot_coords[:, :, 3, :] = atom37_0[:, :, 4, :]

print(prot_coords)

ss_rewards.seed_everything(0)
with torch.no_grad():
    print("dssp_bool")
    dssp_ = ss_rewards.assign(prot_coords)
    for i in range(len(dssp_[0])):
        print(f"{i}: {dssp_[0][i].tolist()}")

dssp = ss_rewards.ss_composition_reward_dense(prot_coords, target_strand=1.0, target_helix=0.0)
print(dssp)
