"""
Differentiable motif RMSD rewards for reward-guided protein generation.

This module provides differentiable reward functions for evaluating how well
generated protein structures match a target motif. Compatible with the
reward guidance system in data/interpolant.py.
"""

import torch
import torch.nn.functional as F
from typing import Union, Optional
import numpy as np
import biotite.structure.io as strucio
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.application.dssp as dssp

def reference_motif_extract(
    structure_path: Union[str, struc.AtomArray],
    atom_part: Optional[str] = "all-atom",
) -> struc.AtomArray:
    """Extracting motif from input protein structure.

    Args:
        structure_path (Union[str, None]): Input protein structure, can either be a path or an AtomArray.
        atom_part (str, optional): _description_. Defaults to "all".

    Returns:
        motif (biotite.structure.AtomArray): The motif positions extracted by user-specified way (all-atom / CA / backbone)
    """
    if isinstance(structure_path, str):
        array = strucio.load_structure(structure_path, model=1)
    else:
        array = structure_path

    # Get unique chain IDs
    chains = array.chain_id
    unique_chains = sorted(set(chains))

    result = []

    for chain in unique_chains:
        # Get all residues in the current chain
        residues_in_chain = array[array.chain_id == chain]
        residue_ids = residues_in_chain.res_id

        # Get the minimum and maximum residue IDs for this chain
        if len(residue_ids) > 0:
            min_res_id = min(residue_ids)
            max_res_id = max(residue_ids)

            if min_res_id == max_res_id:
                # If there's only one residue, just display the single residue
                result.append(f"{chain}{min_res_id}")
            else:
                # If there's a range, display it in the form chainX-Y
                result.append(f"{chain}{min_res_id}-{max_res_id}")

    # Join the results with slashes as required
    position = "/".join(result)

    return motif_extract(position, structure_path, atom_part)


def motif_extract(
    position: str,
    structure_path: Union[str, struc.AtomArray],
    atom_part: Optional[str] = "all-atom",
    split_char: str = "/"
) -> struc.AtomArray:
    """Extracting motif positions from input protein structure.

    Args:
        position (str): Motif region of input protein. DEMO: "A1-7/A28-79" corresponds defines res1-7 and res28-79 in chain A to be motif.
        structure_path (Union[str, None]): Input protein structure, can either be a path or an AtomArray.
        atom_part (str, optional): _description_. Defaults to "all".
        split_char (str): Spliting character between discontinuous motifs. Defaults to "/".

    Returns:
        motif (biotite.structure.AtomArray): The motif positions extracted by user-specified way (all-atom / CA / backbone)
    """

    position = position.split(split_char)
    if isinstance(structure_path, str):
        array = strucio.load_structure(structure_path, model=1)
    else:
        array = structure_path

    motif_array = []
    for i in position:
        chain_id = i[0]
        i = i.replace(chain_id, "")
        if "-" not in i: # Single-residue motif
            start = end = int(i)
        else:
            start, end = i.split("-")
            start, end = int(start), int(end)
        for res_id in range(start, end + 1):
            mask = (array.chain_id == chain_id) & (array.res_id == res_id)  & (array.hetero==False)
            if atom_part == "CA":
                ca_mask = mask & (array.atom_name == "CA")
                coords = array.coord[ca_mask]
                if len(coords) > 0:
                    motif_array.append(coords[0])
                #motif_array.append(array[(array.chain_id==chain_id) & (array.res_id <= end) & (array.res_id >= start) & (array.hetero==False) & (array.atom_name=="CA")])
            elif atom_part == "backbone":
                #motif_array.append(array[(array.chain_id==chain_id) & (array.res_id <= end) & (array.res_id >= start) & (array.hetero==False) & ((array.atom_name=="N") | (array.atom_name=="CA")| (array.atom_name=="C") | (array.atom_name=="O"))])
                bb_coords = []
                for atom_name in ["N", "CA", "C", "O"]:
                    atom_mask = mask & (array.atom_name == atom_name)
                    c = array.coord[atom_mask]
                    if len(c) > 0:
                        bb_coords.append(c[0])

                if len(bb_coords) == 4:
                    motif_array.append(bb_coords)
                #for id_ in range(start, end+1):
                #    bb_coords = []
                #    for atom_name in ["N", "CA", "C", "O"]:
                #        bb_coords.append(array[(array.chain_id==chain_id) & (array.res_id == id_) & (array.hetero==False) & (array.atom_name==atom_name)])
                #    assert len(bb_coords) == 4, (bb_coords, id_)
                #    motif_array.append(bb_coords)
                

    #motif = motif_array[0]
    #for i in range(len(motif_array) - 1):
    #    motif += motif_array[i + 1]
    return motif_array

def differentiable_kabsch_rmsd(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Differentiable RMSD with Kabsch alignment.

    Computes the optimal rigid alignment between predicted and target coordinates
    using SVD (which is differentiable in PyTorch), then computes RMSD.

    Args:
        pred: (B, N, 3) predicted coordinates
        target: (B, N, 3) or (N, 3) target coordinates
        eps: numerical stability constant

    Returns:
        (B,) RMSD per sample
    """
    if target.dim() == 2:
        target = target.unsqueeze(0).expand(pred.shape[0], -1, -1)

    # Center both point clouds
    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)

    # Kabsch algorithm: find optimal rotation via SVD
    # H = P^T @ Q where P=pred, Q=target
    H = torch.bmm(pred_centered.transpose(1, 2), target_centered)  # (B, 3, 3)
    U, S, Vt = torch.linalg.svd(H)

    # Handle reflection case (ensure proper rotation, not reflection)
    d = torch.sign(torch.linalg.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))))
    D = torch.diag_embed(torch.stack([
        torch.ones(pred.shape[0], device=pred.device),
        torch.ones(pred.shape[0], device=pred.device),
        d
    ], dim=1))

    # Optimal rotation matrix: R = V @ D @ U^T
    R = torch.bmm(torch.bmm(Vt.transpose(1, 2), D), U.transpose(1, 2))  # (B, 3, 3)

    # Apply rotation to predicted coordinates
    pred_aligned = torch.bmm(pred_centered, R)  # (B, N, 3)

    # Compute RMSD
    diff = pred_aligned - target_centered
    rmsd = torch.sqrt(torch.mean(torch.sum(diff**2, dim=-1), dim=-1) + eps)

    return rmsd


def extract_motif_coords(
    atom37: torch.Tensor,
    motif_indices: torch.Tensor,
    atom_type: str = "CA",
) -> torch.Tensor:
    """
    Extract motif coordinates from atom37 format.

    Args:
        atom37: (B, L, 37, 3) atom positions in atom37 format
        motif_indices: (M,) indices of residues in the motif
        atom_type: "CA" for alpha-carbon only, "backbone" for N, CA, C, O

    Returns:
        (B, M, 3) for CA or (B, M*4, 3) for backbone
    """
    B = atom37.shape[0]

    # atom37 format indices: 0=N, 1=CA, 2=C, 3=CB, 4=O
    if atom_type == "CA":
        # Extract CA atoms only
        pred_motif = atom37[:, motif_indices, 1, :]  # (B, M, 3)

    elif atom_type == "backbone":
        # Extract N, CA, C, O atoms
        backbone_idx = [0, 1, 2, 4]  # N, CA, C, O in atom37 format
        pred_motif = atom37[:, motif_indices][:, :, backbone_idx, :]  # (B, M, 4, 3)
        pred_motif = pred_motif.reshape(B, -1, 3)  # (B, M*4, 3)

    else:
        raise ValueError(f"Unknown atom_type: {atom_type}. Use 'CA' or 'backbone'.")

    return pred_motif


def motif_rmsd_reward(
    atom37: torch.Tensor,
    atom37_mask: torch.Tensor,
    target_motif_coords: torch.Tensor,
    motif_indices: list[int],
    save_path: Optional[str] = None,
    atom_type: str = "CA",
) -> torch.Tensor:
    """
    Differentiable motif RMSD reward for reward guidance.

    Computes the RMSD between predicted motif coordinates and target motif
    coordinates after optimal Kabsch alignment. Returns negative RMSD so that
    maximizing the reward minimizes RMSD.

    Args:
        atom37: (B, L, 37, 3) predicted atom positions in atom37 format
        atom37_mask: (B, L) residue mask (1 = valid, 0 = padding)
        target_motif_coords: (M, 3) for CA or (M, 4, 3) for backbone target coordinates
        motif_indices: List of residue indices (0-indexed) that form the motif
        save_path: Optional path template for saving intermediate PDBs (unused here)
        atom_type: "CA" for alpha-carbon only, "backbone" for N, CA, C, O

    Returns:
        Scalar negative RMSD reward (higher = better = lower RMSD)
    """
    B = atom37.shape[0]
    device = atom37.device

    motif_idx_tensor = torch.tensor(motif_indices, device=device, dtype=torch.long)

    # Extract predicted motif coordinates
    pred_motif = extract_motif_coords(atom37, motif_idx_tensor, atom_type)

    # Prepare target coordinates
    if atom_type == "CA":
        if target_motif_coords.dim() == 3:  # (M, 4, 3) backbone format
            target_motif = target_motif_coords[:, 1, :]  # (M, 3) just CA
        else:
            target_motif = target_motif_coords  # (M, 3)

    elif atom_type == "backbone":
        if target_motif_coords.dim() == 2:  # (M, 3) CA only
            raise ValueError("For backbone RMSD, target_motif_coords must be (M, 4, 3)")
        target_motif = target_motif_coords.reshape(-1, 3)  # (M*4, 3)

    # Move target to same device
    target_motif = target_motif.to(device)

    # Compute differentiable RMSD
    rmsd = differentiable_kabsch_rmsd(pred_motif, target_motif)

    # Return negative RMSD as reward (maximize reward = minimize RMSD)
    # Mean over batch for scalar reward
    return -rmsd.mean()


def motif_rmsd_reward_per_sample(
    atom37: torch.Tensor,
    atom37_mask: torch.Tensor,
    target_motif_coords: torch.Tensor,
    motif_indices: list[int],
    save_path: Optional[str] = None,
    atom_type: str = "CA",
) -> torch.Tensor:
    """
    Same as motif_rmsd_reward but returns per-sample rewards instead of mean.

    Returns:
        (B,) negative RMSD per sample
    """
    B = atom37.shape[0]
    device = atom37.device

    motif_idx_tensor = torch.tensor(motif_indices, device=device, dtype=torch.long)
    pred_motif = extract_motif_coords(atom37, motif_idx_tensor, atom_type)

    if atom_type == "CA":
        if target_motif_coords.dim() == 3:
            target_motif = target_motif_coords[:, 1, :]
        else:
            target_motif = target_motif_coords
    elif atom_type == "backbone":
        if target_motif_coords.dim() == 2:
            raise ValueError("For backbone RMSD, target_motif_coords must be (M, 4, 3)")
        target_motif = target_motif_coords.reshape(-1, 3)

    target_motif = target_motif.to(device)
    rmsd = differentiable_kabsch_rmsd(pred_motif, target_motif)

    return -rmsd  # (B,)


def parse_motif_position(position_str: str) -> list[tuple[str, int, int]]:
    """
    Parse a motif position string into segments.

    Args:
        position_str: Motif specification like "A1-7/A28-79" or "A1-7/B10-20"

    Returns:
        List of (chain_id, start, end) tuples
    """
    segments = []
    for segment in position_str.split("/"):
        chain_id = segment[0]
        rest = segment[1:]
        if "-" in rest:
            start, end = map(int, rest.split("-"))
        else:
            start = end = int(rest)
        segments.append((chain_id, start, end))
    return segments


def extract_motif_from_pdb(
    pdb_path: str,
    motif_position: str,
    atom_type: str = "CA",
) -> tuple[torch.Tensor, list[int]]:
    """
    Extract motif coordinates and indices from a reference PDB file.

    Args:
        pdb_path: Path to the reference PDB file
        motif_position: Motif specification string (e.g., "A1-7/A28-79")
        atom_type: "CA" or "backbone"

    Returns:
        target_coords: (M, 3) for CA or (M, 4, 3) for backbone
        motif_indices: List of 0-indexed residue positions in the generated structure
    """
    import biotite.structure.io as strucio

    # Load structure
    #array = strucio.load_structure(pdb_path, model=1)

    # Parse motif position string

    motif_coords = []
    motif_indices = []
    current_idx = 0

    for segment in motif_position.split("/"):
        if segment[0].isalpha():
            chain_id = segment[0]
            rest = segment[1:]
            if "-" in rest:
                start, end = map(int, rest.split("-"))
            else:
                start = end = int(rest)

            print("chain_id", chain_id, start,end)
            for res_id in range(start, end + 1):
                motif_indices.append(current_idx)
                #mask = (array.chain_id == chain_id) & (array.res_id == res_id)
                #if atom_type == "CA":
                #    ca_mask = mask & (array.atom_name == "CA")
                #    coords = array.coord[ca_mask]
                #    if len(coords) > 0:
                #        motif_coords.append(coords[0])
                #        motif_indices.append(current_idx)

                #elif atom_type == "backbone":
                #    bb_coords = []
                #    for atom_name in ["N", "CA", "C", "O"]:
                #        atom_mask = mask & (array.atom_name == atom_name)
                #        c = array.coord[atom_mask]
                #        if len(c) > 0:
                #            bb_coords.append(c[0])

                #    if len(bb_coords) == 4:
                #        motif_coords.append(bb_coords)
                #        motif_indices.append(current_idx)

                current_idx += 1
        else:
            scaffold_length = int(segment)
            current_idx += scaffold_length

    # Convert to tensor
    motif_coords  = reference_motif_extract(
        structure_path = pdb_path,
        atom_part= atom_type)
    #motif_coords = [elem.coord for elem in motif]

    target_coords = torch.tensor(np.array(motif_coords), dtype=torch.float32)
    print(target_coords.shape)

    return target_coords, motif_indices


# Mapping from 3-letter amino acid codes to indices (matching residue_constants order)
AA_3TO_IDX = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'UNK': 20,
}


def extract_motif_aatype(
    pdb_path: str,
    motif_position: str,
) -> tuple[list[int], list[int]]:
    """
    Extract amino acid types from the motif in a reference PDB file.

    Args:
        pdb_path: Path to the reference PDB file
        motif_position: Motif specification string (e.g., "A1-7/20/A28-79")
            - Segments starting with a letter are motif segments (e.g., "A1-7")
            - Segments that are just numbers are scaffold segments (e.g., "20")

    Returns:
        motif_aatypes: List of amino acid type indices for motif residues
        motif_indices: List of 0-indexed residue positions in the generated structure
    """
    import biotite.structure.io as strucio

    # Load structure
    array = strucio.load_structure(pdb_path, model=1)

    motif_aatypes = []
    motif_indices = []
    current_idx = 0

    # Parse motif_position, handling both motif and scaffold segments
    for residue in struc.residue_iter(array):
        res_id   = residue.res_id[0]
        res_name = residue.res_name[0]
        chain_id = residue.chain_id[0]
        aa_idx = AA_3TO_IDX.get(res_name, 20)
        motif_aatypes.append(aa_idx)

        #print(chain_id, res_id, res_name)
    #exit()
    
    for segment in motif_position.split("/"):
        if segment[0].isalpha():
            # Motif segment (e.g., "A1-7" or "B28-79")
            chain_id = segment[0]
            rest = segment[1:]
            if "-" in rest:
                start, end = map(int, rest.split("-"))
            else:
                start = end = int(rest)
            for res_id in range(start, end + 1):
                #mask = (array.chain_id == chain_id) & (array.res_id == res_id)
                #res_atoms = array[mask]

                #if len(res_atoms) > 0:
                #    # Get the residue name (3-letter code)
                #    #res_name = res_atoms.res_name[0]
                #    #aa_idx = AA_3TO_IDX.get(res_name, 20)  # Default to UNK
                #    #motif_aatypes.append(aa_idx)
                #    motif_indices.append(current_idx)

                motif_indices.append(current_idx)
                current_idx += 1
        else:
            # Scaffold segment (e.g., "20") - just advance the index
            scaffold_length = int(segment)
            current_idx += scaffold_length
    return motif_aatypes, motif_indices


def build_motif_reward_fn(
    target_pdb_path: str,
    motif_position: str,
    atom_type: str = "CA",
    return_per_sample: bool = False,
):
    """
    Factory function to create a motif RMSD reward function from a reference PDB.

    This creates a closure that captures the target motif coordinates and indices,
    returning a reward function compatible with n_step_sample_with_reward_guidance.

    Args:
        target_pdb_path: Path to reference PDB with the target motif
        motif_position: Motif specification string (e.g., "A1-7/A28-79")
        atom_type: "CA" or "backbone"
        return_per_sample: If True, return per-sample rewards instead of mean

    Returns:
        Reward function with signature: (atom37, atom37_mask, save_path) -> Tensor

    Example:
        >>> reward_fn = build_motif_reward_fn(
        ...     target_pdb_path="reference.pdb",
        ...     motif_position="A1-7/A28-79",
        ...     atom_type="CA"
        ... )
        >>> # Use in reward guidance
        >>> samples = interpolant.n_step_sample_with_reward_guidance(
        ...     num_batch=8, num_res=100, model=model,
        ...     n_steps=50, reward_fn=reward_fn, guidance_scale=1.0
        ... )
    """
    # Extract target motif at setup time (only done once)
    target_motif_coords, motif_indices = extract_motif_from_pdb(
        target_pdb_path, motif_position, atom_type
    )

    print(f"Built motif reward function:")
    print(f"  Target PDB: {target_pdb_path}")
    print(f"  Motif position: {motif_position}")
    print(f"  Atom type: {atom_type}")
    print(f"  Motif size: {len(motif_indices)} residues")
    print(f"  Motif indices: {motif_indices}")

    reward_func = motif_rmsd_reward_per_sample if return_per_sample else motif_rmsd_reward

    def reward_fn(
        atom37: torch.Tensor,
        atom37_mask: torch.Tensor,
        save_path: str
    ) -> torch.Tensor:
        """
        Compute motif RMSD reward for the current structure.

        Args:
            atom37: (B, L, 37, 3) predicted atom positions
            atom37_mask: (B, L) residue mask
            save_path: Path template for saving intermediate PDBs

        Returns:
            
        """
        # Move target to same device as input
        target = target_motif_coords.to(atom37.device)

        return reward_func(
            atom37=atom37,
            atom37_mask=atom37_mask,
            target_motif_coords=target,
            motif_indices=motif_indices,
            save_path=save_path,
            atom_type=atom_type,
        )

    return reward_fn


def combined_ss_motif_reward_fn(
    target_pdb_path: str,
    motif_position: str,
    atom_type: str = "CA",
    motif_weight: float = 1.0,
    ss_weight: float = 0.0,
    target_helix: float = 0.0,
    target_strand: float = 0.0,
):
    """
    Create a combined reward function with both motif RMSD and secondary structure.

    Args:
        target_pdb_path: Path to reference PDB
        motif_position: Motif specification string
        atom_type: "CA" or "backbone"
        motif_weight: Weight for motif RMSD reward
        ss_weight: Weight for secondary structure composition reward
        target_helix: Target helix fraction (0-1)
        target_strand: Target strand fraction (0-1)

    Returns:
        Combined reward function
    """
    from protein_rewards import ss_rewards

    # Build motif reward
    target_motif_coords, motif_indices = extract_motif_from_pdb(
        target_pdb_path, motif_position, atom_type
    )

    def reward_fn(
        atom37: torch.Tensor,
        atom37_mask: torch.Tensor,
        save_path: str
    ) -> torch.Tensor:
        reward = torch.tensor(0.0, device=atom37.device)

        # Motif RMSD reward
        if motif_weight > 0:
            target = target_motif_coords.to(atom37.device)
            motif_reward = motif_rmsd_reward(
                atom37=atom37,
                atom37_mask=atom37_mask,
                target_motif_coords=target,
                motif_indices=motif_indices,
                save_path=save_path,
                atom_type=atom_type,
            )
            reward = reward + motif_weight * motif_reward

        # Secondary structure reward
        if ss_weight > 0:
            # Convert atom37 to backbone format for SS rewards
            B, L, _, D = atom37.shape
            prot_coords = torch.zeros((B, L, 4, D), device=atom37.device)
            prot_coords[:, :, :3, :] = atom37[:, :, :3, :]
            prot_coords[:, :, 3, :] = atom37[:, :, 4, :]  # O is at index 4

            ss_reward = ss_rewards.ss_composition_reward_dense(
                prot_coords,
                target_strand=target_strand,
                target_helix=target_helix
            )
            reward = reward + ss_weight * ss_reward

        return reward

    return reward_fn


if __name__ == "__main__":
    #motif  = reference_motif_extract(
    #structure_path = "protein_rewards/Scaffold-Lab/demo/motif_scaffolding/native_motifs/2KL8_motif.pdb",
    #atom_part= "CA")
    #print(motif, np.array(motif).shape)

    #motif, indices  = extract_motif_from_pdb(
    #pdb_path = "protein_rewards/Scaffold-Lab/demo/motif_scaffolding/native_motifs/2KL8_motif.pdb",
    #motif_position= "A1-7/20/A28-79",
    #atom_type= "CA",)
    #print(motif.shape)
    #print(indices)
    print(extract_motif_aatype(
        pdb_path= "protein_rewards/Scaffold-Lab/demo/motif_scaffolding/native_motifs/2KL8_motif.pdb",
        motif_position= "A1-7/20/A28-79",
    ))
