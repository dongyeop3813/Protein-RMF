import tempfile

import torch
import pandas as pd
import nglview as nv

from analysis import utils as autils
from analysis.metrics import calc_mdtraj_metrics, calc_ca_ca_metrics
from openfold.np import residue_constants
from data import all_atom


def plot(atom37):
    # atom37: (batch, num_res, 37, 3) - keep only first structure if >1 in batch
    atom37_np = atom37.detach().cpu().numpy()  # (batch, num_res, 37, 3)
    batch_idx = 0
    coords = atom37_np[batch_idx]  # (num_res, 37, 3)

    # analysis.utils.write_prot_to_pdb expects a np.ndarray (num_res, 37, 3)
    # Write to a temporary PDB file for visualization

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
        pdb_path = autils.write_prot_to_pdb(
            coords, tf.name, overwrite=True, no_indexing=True
        )

    view = nv.show_file(pdb_path)
    # view.clear_representations()
    view.add_licorice()
    return view


def plot_prot(
    trans,
    rots,
    res_mask,
    clip=None,
    licorice=True,
    ball_and_stick=True,
):
    from data import all_atom

    if clip is not None:
        # Select a sub-structure of the protein
        start, end = clip
        trans = trans[:, start:end, :]
        rots = rots[:, start:end, :, :]
        res_mask = res_mask[:, start:end]

    atom37 = all_atom.atom37_from_trans_rot(trans, rots, res_mask)
    view = plot(atom37)

    if licorice:
        return view
    else:
        view.clear_representations()
        view.add_cartoon(color="residueindex")
    if ball_and_stick:
        view.add_ball_and_stick()
    return view


def metrics(trans, rots):
    num_batch, num_res = trans.shape[:2]
    res_mask = torch.ones(num_batch, num_res, device=trans.device)

    batch_metrics = []

    atom37 = all_atom.atom37_from_trans_rot(trans, rots, res_mask)
    atom37_np = atom37.detach().cpu().numpy()  # (batch, num_res, 37, 3)

    for idx in range(num_batch):
        final_pos = atom37_np[idx]  # (num_res, 37, 3)

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
            pdb_path = autils.write_prot_to_pdb(
                final_pos, tf.name, overwrite=True, no_indexing=True
            )
            mdtraj_metrics = calc_mdtraj_metrics(pdb_path)
            ca_idx = residue_constants.atom_order["CA"]
            ca_ca_metrics = calc_ca_ca_metrics(final_pos[:, ca_idx])
            batch_metrics.append(mdtraj_metrics | ca_ca_metrics)
    return pd.DataFrame(batch_metrics)


def print_dict(cfg):
    import json
    from omegaconf import OmegaConf

    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4))
