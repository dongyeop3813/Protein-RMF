import torch
import torch.nn.functional as F

import pydssp
import random
import numpy as np

from einops import repeat, rearrange




CONST_Q1Q2 = 0.084
CONST_F = 332
DEFAULT_CUTOFF = -0.5
DEFAULT_MARGIN = 1.0

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    
    # For deterministic CuDNN behavior (optional, may slightly impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _check_input(coord):
    org_shape = coord.shape
    assert (len(org_shape)==3) or (len(org_shape)==4), "Shape of input tensor should be [batch, L, atom, xyz] or [L, atom, xyz]"
    coord = repeat(coord, '... -> b ...', b=1) if len(org_shape)==3 else coord
    return coord, org_shape

def _get_hydrogen_atom_position(coord: torch.Tensor) -> torch.Tensor:
    # A little bit lazy (but should be OK) definition of H position here.
    vec_cn = coord[:,1:,0] - coord[:,:-1,2]
    vec_cn = vec_cn / torch.linalg.norm(vec_cn, dim=-1, keepdim=True)
    vec_can = coord[:,1:,0] - coord[:,1:,1]
    vec_can = vec_can / torch.linalg.norm(vec_can, dim=-1, keepdim=True)
    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / torch.linalg.norm(vec_nh, dim=-1, keepdim=True)
    return coord[:,1:,0] + 1.01 * vec_nh


def _get_hydrogen_atom_position_grad_safe(coord: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Gradient-safe version of _get_hydrogen_atom_position.
    Adds epsilon to norm computations to avoid NaN gradients when vectors are near-zero.
    """
    vec_cn = coord[:,1:,0] - coord[:,:-1,2]
    vec_cn = vec_cn / (torch.linalg.norm(vec_cn, dim=-1, keepdim=True) + eps)
    vec_can = coord[:,1:,0] - coord[:,1:,1]
    vec_can = vec_can / (torch.linalg.norm(vec_can, dim=-1, keepdim=True) + eps)
    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / (torch.linalg.norm(vec_nh, dim=-1, keepdim=True) + eps)
    return coord[:,1:,0] + 1.01 * vec_nh


def get_hbond_map(
    coord: torch.Tensor,
    donor_mask: torch.Tensor=None,
    cutoff: float=DEFAULT_CUTOFF,
    margin: float=DEFAULT_MARGIN,
    return_e: bool=False
    ) -> torch.Tensor:
    # check input
    coord, org_shape = _check_input(coord)
    b, l, a, _ = coord.shape
    # add pseudo-H atom if not available
    assert (a==4) or (a==5), "Number of atoms should be 4 (N,CA,C,O) or 5 (N,CA,C,O,H)"
    h = coord[:,1:,4] if a == 5 else _get_hydrogen_atom_position(coord)
    # distance matrix
    nmap = repeat(coord[:,1:,0], '... m c -> ... m n c', n=l-1)
    hmap = repeat(h, '... m c -> ... m n c', n=l-1)
    cmap = repeat(coord[:,0:-1,2], '... n c -> ... m n c', m=l-1)
    omap = repeat(coord[:,0:-1,3], '... n c -> ... m n c', m=l-1)
    d_on = torch.linalg.norm(omap - nmap, dim=-1)
    d_ch = torch.linalg.norm(cmap - hmap, dim=-1)
    d_oh = torch.linalg.norm(omap - hmap, dim=-1)
    d_cn = torch.linalg.norm(cmap - nmap, dim=-1)
    # electrostatic interaction energy
    e = torch.nn.functional.pad(CONST_Q1Q2 * (1./d_on + 1./d_ch - 1./d_oh - 1./d_cn)*CONST_F, [0,1,1,0])
    # mask for local pairs (i,i), (i,i+1), (i,i+2)
    local_mask = ~torch.eye(l, dtype=bool)
    local_mask *= ~torch.diag(torch.ones(l-1, dtype=bool), diagonal=-1)
    local_mask *= ~torch.diag(torch.ones(l-2, dtype=bool), diagonal=-2)
    # mask for donor H absence (Proline)
    if donor_mask is None:
        donor_mask = torch.ones(l, dtype=float)
    else:
        donor_mask = donor_mask.to(float) if torch.is_tensor(donor_mask) else torch.Tensor(donor_mask).to(float)
    donor_mask = repeat(donor_mask, 'l1 -> l1 l2', l2=l)
    if return_e: 
        return e
    # hydrogen bond map (continuous value extension of original definition)
    hbond_map = torch.clamp(cutoff - margin - e, min=-margin, max=margin)
    hbond_map = (torch.sin(hbond_map/margin*torch.pi/2)+1.)/2
    hbond_map = hbond_map * repeat(local_mask.to(hbond_map.device), 'l1 l2 -> b l1 l2', b=b)
    hbond_map = hbond_map * repeat(donor_mask.to(hbond_map.device), 'l1 l2 -> b l1 l2', b=b)
    # return h-bond map
    hbond_map = hbond_map.squeeze(0) if len(org_shape)==3 else hbond_map
    return hbond_map


def get_hbond_map_grad_safe(
    coord: torch.Tensor,
    donor_mask: torch.Tensor=None,
    cutoff: float=DEFAULT_CUTOFF,
    margin: float=DEFAULT_MARGIN,
    return_e: bool=False,
    eps: float=1e-6,
    min_dist: float=0.1,
    ) -> torch.Tensor:
    """
    Gradient-safe version of get_hbond_map.
    Adds epsilon to avoid NaN gradients from division by small distances.

    Args:
        coord: Backbone coordinates [B, L, 4, 3] or [L, 4, 3]
        donor_mask: Optional mask for donor atoms
        cutoff: H-bond energy cutoff
        margin: Margin for soft H-bond map
        return_e: Whether to return raw energies
        eps: Small epsilon for numerical stability in norm
        min_dist: Minimum distance to clamp to avoid 1/0
    """
    coord, org_shape = _check_input(coord)
    b, l, a, _ = coord.shape
    assert (a==4) or (a==5), "Number of atoms should be 4 (N,CA,C,O) or 5 (N,CA,C,O,H)"
    h = coord[:,1:,4] if a == 5 else _get_hydrogen_atom_position_grad_safe(coord, eps=eps)

    # distance matrix with clamping to avoid 1/0
    nmap = repeat(coord[:,1:,0], '... m c -> ... m n c', n=l-1)
    hmap = repeat(h, '... m c -> ... m n c', n=l-1)
    cmap = repeat(coord[:,0:-1,2], '... n c -> ... m n c', m=l-1)
    omap = repeat(coord[:,0:-1,3], '... n c -> ... m n c', m=l-1)

    # Clamp distances to avoid division by zero and exploding gradients
    d_on = torch.clamp(torch.linalg.norm(omap - nmap, dim=-1), min=min_dist)
    d_ch = torch.clamp(torch.linalg.norm(cmap - hmap, dim=-1), min=min_dist)
    d_oh = torch.clamp(torch.linalg.norm(omap - hmap, dim=-1), min=min_dist)
    d_cn = torch.clamp(torch.linalg.norm(cmap - nmap, dim=-1), min=min_dist)

    # electrostatic interaction energy
    e = torch.nn.functional.pad(CONST_Q1Q2 * (1./d_on + 1./d_ch - 1./d_oh - 1./d_cn)*CONST_F, [0,1,1,0])

    # mask for local pairs (i,i), (i,i+1), (i,i+2)
    local_mask = ~torch.eye(l, dtype=bool, device=coord.device)
    local_mask = local_mask & ~torch.diag(torch.ones(l-1, dtype=bool, device=coord.device), diagonal=-1)
    local_mask = local_mask & ~torch.diag(torch.ones(l-2, dtype=bool, device=coord.device), diagonal=-2)

    # mask for donor H absence (Proline)
    if donor_mask is None:
        donor_mask = torch.ones(l, dtype=coord.dtype, device=coord.device)
    else:
        donor_mask = donor_mask.to(dtype=coord.dtype, device=coord.device)
    donor_mask = repeat(donor_mask, 'l1 -> l1 l2', l2=l)

    if return_e:
        return e, donor_mask, local_mask

    # hydrogen bond map (continuous value extension of original definition)
    hbond_map = torch.clamp(cutoff - margin - e, min=-margin, max=margin)
    hbond_map = (torch.sin(hbond_map/margin*torch.pi/2)+1.)/2
    hbond_map = hbond_map * local_mask.unsqueeze(0).float()
    hbond_map = hbond_map * donor_mask.unsqueeze(0)

    hbond_map = hbond_map.squeeze(0) if len(org_shape)==3 else hbond_map
    return hbond_map


def assign(coord: torch.Tensor, donor_mask: torch.Tensor=None) -> torch.Tensor:
    # check input
    coord, org_shape = _check_input(coord)
    # get hydrogen bond map
    hbmap = get_hbond_map(coord, donor_mask=donor_mask)
    hbmap = rearrange(hbmap, '... l1 l2 -> ... l2 l1') # convert into "i:C=O, j:N-H" form
    # identify turn 3, 4, 5
    turn3 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=3) > 0.
    turn4 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=4) > 0.
    turn5 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=5) > 0.
    # assignment of helical sses
    h3 = torch.nn.functional.pad(turn3[:,:-1] * turn3[:,1:], [1,3])
    h4 = torch.nn.functional.pad(turn4[:,:-1] * turn4[:,1:], [1,4])
    h5 = torch.nn.functional.pad(turn5[:,:-1] * turn5[:,1:], [1,5])
    # helix4 first
    helix4 = h4 + torch.roll(h4, 1, 1) + torch.roll(h4, 2, 1) + torch.roll(h4, 3, 1)
    h3 = h3 * ~torch.roll(helix4, -1, 1) * ~helix4 # helix4 is higher prioritized
    h5 = h5 * ~torch.roll(helix4, -1, 1) * ~helix4 # helix4 is higher prioritized
    helix3 = h3 + torch.roll(h3, 1, 1) + torch.roll(h3, 2, 1)
    helix5 = h5 + torch.roll(h5, 1, 1) + torch.roll(h5, 2, 1) + torch.roll(h5, 3, 1) + torch.roll(h5, 4, 1)
    # identify bridge
    unfoldmap = hbmap.unfold(-2, 3, 1).unfold(-2, 3, 1) > 0.
    unfoldmap_rev = unfoldmap.transpose(-4,-3)
    p_bridge = (unfoldmap[:,:,:,0,1] * unfoldmap_rev[:,:,:,1,2]) + (unfoldmap_rev[:,:,:,0,1] * unfoldmap[:,:,:,1,2])
    p_bridge = torch.nn.functional.pad(p_bridge, [1,1,1,1])
    a_bridge = (unfoldmap[:,:,:,1,1] * unfoldmap_rev[:,:,:,1,1]) + (unfoldmap[:,:,:,0,2] * unfoldmap_rev[:,:,:,0,2])
    a_bridge = torch.nn.functional.pad(a_bridge, [1,1,1,1])
    # ladder
    ladder = (p_bridge + a_bridge).sum(-1) > 0
    # H, E, L of C3
    helix = (helix3 + helix4 + helix5) > 0
    strand = ladder
    loop = (~helix * ~strand)
    onehot = torch.stack([loop, helix, strand], dim=-1)
    onehot = onehot.squeeze(0) if len(org_shape)==3 else onehot
    return onehot


def beta_sheet_reward_dense(coord: torch.Tensor, donor_mask: torch.Tensor=None, temperature: float=1.0) -> torch.Tensor:
    """
    Beta sheet reward with dense gradients everywhere.
    Uses sigmoid (no clamping) so gradients never vanish.

    Args:
        coord: Backbone coordinates [B, L, 4, 3] or [L, 4, 3]
        donor_mask: Optional mask for donor atoms
        temperature: Controls sharpness. Lower = sharper, higher = smoother/denser gradients.

    Returns:
        Per-residue beta sheet scores [B, L]
    """
    coord, org_shape = _check_input(coord)
    b, l, a, _ = coord.shape

    # Compute raw H-bond energies (negative = favorable)
    assert (a==4) or (a==5)
    h = coord[:,1:,4] if a == 5 else _get_hydrogen_atom_position(coord)

    nmap = repeat(coord[:,1:,0], '... m c -> ... m n c', n=l-1)
    hmap = repeat(h, '... m c -> ... m n c', n=l-1)
    cmap = repeat(coord[:,0:-1,2], '... n c -> ... m n c', m=l-1)
    omap = repeat(coord[:,0:-1,3], '... n c -> ... m n c', m=l-1)

    d_on = torch.linalg.norm(omap - nmap, dim=-1)
    d_ch = torch.linalg.norm(cmap - hmap, dim=-1)
    d_oh = torch.linalg.norm(omap - hmap, dim=-1)
    d_cn = torch.linalg.norm(cmap - nmap, dim=-1)

    e = CONST_Q1Q2 * (1./d_on + 1./d_ch - 1./d_oh - 1./d_cn) * CONST_F
    e = F.pad(e, [0,1,1,0])

    # Mask local pairs
    local_mask = ~torch.eye(l, dtype=bool, device=coord.device)
    local_mask = local_mask & ~torch.diag(torch.ones(l-1, dtype=bool, device=coord.device), diagonal=-1)
    local_mask = local_mask & ~torch.diag(torch.ones(l-2, dtype=bool, device=coord.device), diagonal=-2)

    # Sigmoid: more negative e -> higher probability (no clamp = dense gradients)
    # Shift by cutoff so the transition is around the H-bond threshold
    hbmap_soft = torch.sigmoid(-(e - DEFAULT_CUTOFF) / temperature)
    hbmap_soft = hbmap_soft * local_mask.unsqueeze(0).float()

    hbmap_soft = rearrange(hbmap_soft, '... l1 l2 -> ... l2 l1')

    # Bridge detection
    unfoldmap = hbmap_soft.unfold(-2, 3, 1).unfold(-2, 3, 1)
    unfoldmap_rev = unfoldmap.transpose(-4,-3)

    # Products of [0,1] sigmoids = soft AND with dense gradients
    p_bridge = (unfoldmap[:,:,:,0,1] * unfoldmap_rev[:,:,:,1,2]) + \
               (unfoldmap_rev[:,:,:,0,1] * unfoldmap[:,:,:,1,2])
    p_bridge = F.pad(p_bridge, [1,1,1,1])

    a_bridge = (unfoldmap[:,:,:,1,1] * unfoldmap_rev[:,:,:,1,1]) + \
               (unfoldmap[:,:,:,0,2] * unfoldmap_rev[:,:,:,0,2])
    a_bridge = F.pad(a_bridge, [1,1,1,1])

    ladder = (p_bridge + a_bridge).sum(-1)

    # Restore original shape if needed
    ladder = ladder.squeeze(0) if len(org_shape)==3 else ladder

    return ladder


# =============================================================================

def get_hbond_ss_scores_dense(
    coords: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
    min_dist: float = 0.1,
) -> dict:
    """
    Extract SS scores from DENSE energy matrix (gradients everywhere).

    Uses sigmoid on raw H-bond energies instead of clamped/sparse map.
    More negative energy = stronger H-bond = higher sigmoid output.

    Unlike get_hbond_ss_scores() which uses clamp+sin (sparse gradients),
    this uses sigmoid on raw energies so gradients flow to ALL residue pairs.

    Args:
        coords: Backbone coordinates [batch, L, 4, 3] with atoms N, CA, C, O
        temperature: Controls sigmoid sharpness. Lower = sharper, higher = denser gradients.
        eps: Epsilon for numerical stability in norms
        min_dist: Minimum distance to avoid 1/0 in energy calculation

    Returns:
        Dictionary with per-residue scores:
        - 'helix4': [batch, L-4] α-helix (i → i+4 H-bonds)
        - 'helix3': [batch, L-3] 3-10 helix (i → i+3 H-bonds)
        - 'helix5': [batch, L-5] π-helix (i → i+5 H-bonds)
        - 'strand': [batch] global β-strand score (long-range H-bonds)
        - 'hbmap': [batch, L, L] full soft H-bond map
    """
    coords, org_shape = _check_input(coords)
    b, l, a, _ = coords.shape

    # Compute H position (gradient-safe)
    h = coords[:,1:,4] if a == 5 else _get_hydrogen_atom_position_grad_safe(coords, eps=eps)

    # Distance matrices with clamping to avoid 1/0
    nmap = repeat(coords[:,1:,0], '... m c -> ... m n c', n=l-1)
    hmap = repeat(h, '... m c -> ... m n c', n=l-1)
    cmap = repeat(coords[:,0:-1,2], '... n c -> ... m n c', m=l-1)
    omap = repeat(coords[:,0:-1,3], '... n c -> ... m n c', m=l-1)

    d_on = torch.clamp(torch.linalg.norm(omap - nmap, dim=-1), min=min_dist)
    d_ch = torch.clamp(torch.linalg.norm(cmap - hmap, dim=-1), min=min_dist)
    d_oh = torch.clamp(torch.linalg.norm(omap - hmap, dim=-1), min=min_dist)
    d_cn = torch.clamp(torch.linalg.norm(cmap - nmap, dim=-1), min=min_dist)

    # Raw electrostatic energy (negative = favorable H-bond)
    e = F.pad(CONST_Q1Q2 * (1./d_on + 1./d_ch - 1./d_oh - 1./d_cn) * CONST_F, [0,1,1,0])

    # Mask local pairs (i,i), (i,i+1), (i,i+2)
    local_mask = ~torch.eye(l, dtype=bool, device=coords.device)
    local_mask = local_mask & ~torch.diag(torch.ones(l-1, dtype=bool, device=coords.device), diagonal=-1)
    local_mask = local_mask & ~torch.diag(torch.ones(l-2, dtype=bool, device=coords.device), diagonal=-2)

    donor_mask = torch.ones(l, dtype=float, device=coords.device)
    donor_mask = repeat(donor_mask, 'l1 -> l1 l2', l2=l)

    hbmap_soft = torch.sigmoid(-(e - DEFAULT_CUTOFF + DEFAULT_MARGIN) / temperature)
    hbmap_soft = hbmap_soft * local_mask.unsqueeze(0).float()
    hbmap_soft = hbmap_soft * donor_mask.unsqueeze(0).float()

    hbmap_original = get_hbond_map(coords, donor_mask=None, return_e=False)

    hbond_map = torch.clamp(DEFAULT_CUTOFF - DEFAULT_MARGIN - e, min=-DEFAULT_MARGIN, max=DEFAULT_MARGIN)
    hbond_map = hbond_map * local_mask.unsqueeze(0).float()
    hbond_map = hbond_map * donor_mask.unsqueeze(0).float()

    # Transpose to "i:C=O accepts from j:N-H" form
    hbmap_soft = hbmap_soft.transpose(-1, -2)

    # Helix scores from diagonals
    helix4_bonds = torch.diagonal(hbmap_soft, offset=4, dim1=-2, dim2=-1)
    helix3_bonds = torch.diagonal(hbmap_soft, offset=3, dim1=-2, dim2=-1)
    helix5_bonds = torch.diagonal(hbmap_soft, offset=5, dim1=-2, dim2=-1)

    # Strand: long-range H-bonds (|i-j| > 5)
    i_idx = torch.arange(l, device=coords.device).unsqueeze(0)
    j_idx = torch.arange(l, device=coords.device).unsqueeze(1)
    long_range_mask = (torch.abs(i_idx - j_idx) > 5).float()
    strand_bonds = (hbmap_soft * long_range_mask).sum(dim=(-1, -2)) / (long_range_mask.sum() + 1e-8)

    if len(org_shape) == 3:
        helix4_bonds = helix4_bonds.squeeze(0)
        helix3_bonds = helix3_bonds.squeeze(0)
        helix5_bonds = helix5_bonds.squeeze(0)
        strand_bonds = strand_bonds.squeeze(0)
        hbmap_soft = hbmap_soft.squeeze(0)

    return {
        'helix4': helix4_bonds,
        'helix3': helix3_bonds,
        'helix5': helix5_bonds,
        'strand': strand_bonds,
        'hbmap': hbmap_soft,
    }


def ss_composition_reward_dense(
    coords: torch.Tensor,
    target_helix: float = 0.0,
    target_strand: float = 0.0,
    temperature: float = 1.0,
    reduction: str="sum"
) -> torch.Tensor:
    """
    SS composition reward using DENSE energy matrix.

    Gradients flow to all residue pairs, not just those near the H-bond threshold.
    Use this instead of ss_composition_reward() when you need denser gradients
    for reward guidance.

    Args:
        coords: Backbone coordinates [batch, L, 4, 3] or [L, 4, 3]
        target_helix: Target helix fraction (0-1)
        target_strand: Target strand fraction (0-1)
        temperature: Sigmoid temperature. Higher = smoother/denser gradients.

    Returns:
        Negative MSE from target composition (maximize this)
    """
    scores = get_hbond_ss_scores_dense(coords, temperature=temperature)
    helix_scores = scores['helix4']
    strand_scores = scores['strand']

    helix_frac = helix_scores.mean(dim=tuple(range(1, helix_scores.ndim)))
    strand_frac = strand_scores.mean(dim=tuple(range(1, strand_scores.ndim))) if strand_scores.dim() > 1 else strand_scores
    

    loss = (helix_frac - target_helix)**2 + (strand_frac - target_strand)**2
    return -loss  # Negative so we can maximize


def helix_content_reward_dense(
    coords: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable reward for α-helix content using dense energy matrix.

    Args:
        coords: Backbone coordinates [batch, L, 4, 3] or [L, 4, 3]
        temperature: Sigmoid temperature

    Returns:
        Scalar reward (mean of i→i+4 H-bond strengths)
    """
    scores = get_hbond_ss_scores_dense(coords, temperature=temperature)
    helix_scores = scores['helix4']
    return helix_scores.mean(dim=tuple(range(1, helix_scores.ndim)))


def strand_content_reward_dense(
    coords: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable reward for β-strand content using dense energy matrix.

    Args:
        coords: Backbone coordinates [batch, L, 4, 3] or [L, 4, 3]
        temperature: Sigmoid temperature

    Returns:
        Scalar reward (mean long-range H-bond strength)
    """
    scores = get_hbond_ss_scores_dense(coords, temperature=temperature)
    strand_scores = scores['strand']

    return strand_scores.mean(dim=tuple(range(1, strand_scores.ndim))) if strand_scores.dim() > 1 else strand_scores

