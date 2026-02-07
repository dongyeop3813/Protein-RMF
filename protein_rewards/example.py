import torch
import torch.nn as nn
import torch.nn.functional as F

import all_atom
import ss_rewards
from pathlib import Path
from openfold.np import protein as of_protein

PATH="samples_to_send/n_steps_100/length_100/sample_9/trans_rot.pt"
sample = torch.load(PATH, map_location=torch.device('cpu'))


def save_backbone_atom37_to_pdb(
    atom37_pos: torch.Tensor,      # [L, 37, 3] (or [B, L, 37, 3])
    atom37_mask: torch.Tensor,     # [L, 37]    (or [B, L, 37])
    out_path: str,
    resnames=None,                # optional list length L, e.g. ["ALA", ...]
    chain_id: str = "A",
    model_index: int = 0,         # which batch item if batched
):
    # Handle batch dimension if present
    if atom37_pos.ndim == 4:
        atom37_pos = atom37_pos[model_index]
        atom37_mask = atom37_mask[model_index]

    atom37_pos = atom37_pos.detach().cpu()
    atom37_mask = atom37_mask.detach().cpu()

    L = atom37_pos.shape[0]
    if resnames is None:
        resnames = ["ALA"] * L  # default

    # Indices used by your packing
    idx_to_name = {
        0: "N",
        1: "CA",
        2: "C",
        3: "CB",
        4: "O",
    }

    lines = []
    atom_serial = 1
    for i in range(L):
        resseq = i + 1
        resname = resnames[i]

        for a_idx, atom_name in idx_to_name.items():
            if not bool(atom37_mask[i, a_idx]):
                continue
            x, y, z = atom37_pos[i, a_idx].tolist()

            # PDB ATOM format (columns matter)
            # atom name is right-justified in 4 columns
            line = (
                f"ATOM  {atom_serial:5d} {atom_name:>4s} {resname:>3s} {chain_id}"
                f"{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00           {atom_name[0]:>2s}"
            )
            lines.append(line)
            atom_serial += 1

    lines.append("END")
    Path(out_path).write_text("\n".join(lines) + "\n")

print(sample.keys())
pred_trans = sample['trans'].unsqueeze(0)
pred_rots_mats = sample['rot'].unsqueeze(0)

pred_trans.requires_grad_(True)
pred_rots_mats.requires_grad_(True)

print(pred_trans.dtype, pred_trans.requires_grad)
print(pred_rots_mats.dtype, pred_rots_mats.requires_grad)


#  atom37 bb order ['N', 'CA', 'C', 'CB', 'O']
atom37_0, atom37_mask, aatype = all_atom.to_atom37(pred_trans, pred_rots_mats)
atom37_0.retain_grad()


save_backbone_atom37_to_pdb(atom37_0, atom37_mask, "backbone.pdb")

if len(atom37_0.shape) == 3:
    atom37_0 = atom37_0.unsqueeze(0)
print(atom37_0.shape)
# need 4 (N, CA, C, O) or 5 (N, CA, C, O, H)
B, L, _, D = atom37_0.shape
prot_coords = torch.zeros((B, L, 4, D), device=atom37_0.device)
prot_coords[:, :, :3, :] = atom37_0[:, :, :3, :]
prot_coords[:, :, 3, :] = atom37_0[:, :, 4, :]

print(prot_coords)

ss_rewards.seed_everything(0)
with torch.no_grad():
    print("dssp_bool")
    dssp_ = ss_rewards.assign(prot_coords)
    for i in range(len(dssp_[0])):
        print(f"{i}: {dssp_[0][i].tolist()}")

dssp = ss_rewards.assign_diff(prot_coords)
dssp.retain_grad()
with torch.no_grad():
    print("dssp_soft")
    for i in range(len(dssp[0])):
        print(f"{i}: {dssp[0][i].tolist()}")

assert len(dssp.shape) == 3 and dssp.shape[-1] == 3

tgt = torch.ones(dssp.shape[:-1]) * 2
tgt = tgt.long()
print("tgt", tgt)

loss_tok = F.cross_entropy(dssp.view(-1,3), tgt.view(-1).long(), reduction="none").view_as(tgt)
#loss_tok = F.nll_loss(dssp.view(-1,3), tgt.view(-1).long(), reduction="none").view_as(tgt)

loss = loss_tok.mean()    # scalar

print("loss", loss)
loss.backward()

print("dssp requires_grad:", dssp.requires_grad)
print("dssp grad_fn:", dssp.grad_fn)

print("loss_tok shape:", loss_tok.shape)
print("loss scalar:", loss.item())
print("nonzero tokens:", (loss_tok != 0).sum().item())

print("dssp grad mean:", dssp.grad.abs().mean().item())
print("atom37 grad mean:", atom37_0.grad.abs().mean().item())
print("pred_trans grad mean:", pred_trans.grad.abs().mean().item())

print("trans_grad")
print(pred_trans.grad)

print("rot_grad", pred_rots_mats.grad)

g = pred_trans.grad.abs().sum(-1)[0]
print("fraction nonzero residues:", (g > 1e-12).float().mean().item())
print("max / median:", g.max().item(), g.median().item())
