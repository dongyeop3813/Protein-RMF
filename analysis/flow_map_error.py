import torch
import numpy as np
from analysis import utils as autils
from data import utils as dutils


def rot_dist(rots1, rots2):
    # Geodesic distance on SO(3)
    # rots1, rots2: (..., 3, 3) rotation matrices
    R = torch.matmul(rots1.transpose(-1, -2), rots2)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos_theta)


def align(pred, target):
    pred_trans, pred_rots = pred
    target_trans, target_rots = target

    # Remove the center of mass of the predicted and target structures
    pred_trans = pred_trans - pred_trans.mean(dim=1, keepdim=True)
    target_trans = target_trans - target_trans.mean(dim=1, keepdim=True)

    aligned_pred_trans, _, align_rots = dutils.batch_align_structures(
        pred_trans, target_trans
    )

    aligned_pred_rots = align_rots.transpose(-1, -2) @ pred_rots
    return aligned_pred_trans, aligned_pred_rots


def error(pred, target, use_kabsch=True):
    """Calculate error after Kabsch alignment.

    Args:
        pred: tuple of (pred_trans, pred_rots) where
            pred_trans: (batch, num_res, 3)
            pred_rots: (batch, num_res, 3, 3)
        target: tuple of (target_trans, target_rots) with same shapes
    Returns:
        trans_error: mean translation error after alignment
        rot_error: mean rotation error after alignment
    """
    if use_kabsch:
        pred_trans, pred_rots = align(pred, target)
    else:
        pred_trans, pred_rots = pred

    target_trans, target_rots = target

    # Calculate translation error after alignment
    trans_error = torch.norm(pred_trans - target_trans, dim=-1)
    trans_error = trans_error.mean()

    # Calculate rotation error after alignment
    rot_error = rot_dist(pred_rots, target_rots)
    rot_error = rot_error.mean()

    return trans_error, rot_error


def rot_error(pred, target):
    """Calculate axis and angle error after Kabsch alignment.

    Args:
        pred: tuple of (pred_trans, pred_rots) where
            pred_trans: (batch, num_res, 3)
            pred_rots: (batch, num_res, 3, 3)
        target: tuple of (target_trans, target_rots) with same shapes
    Returns:
        axis_error: mean axis error after alignment
        angle_error: mean angle error after alignment
    """
    from data.so3_utils import rotmat_to_rotvec

    pred_trans, pred_rots = pred
    target_trans, target_rots = target

    # Remove the center of mass of the predicted and target structures
    pred_trans = pred_trans - pred_trans.mean(dim=1, keepdim=True)
    target_trans = target_trans - target_trans.mean(dim=1, keepdim=True)

    # Apply Kabsch alignment to translations
    aligned_pred_trans, aligned_target_trans, align_rots = (
        dutils.batch_align_structures(pred_trans, target_trans)
    )
    # align_rots shape: (batch, 3, 3)

    # Apply alignment rotation to predicted rotations
    # For each residue: R_align @ R_pred
    aligned_pred_rots = (
        align_rots[:, None, :, :].transpose(-1, -2) @ pred_rots
    )  # (batch, num_res, 3, 3)
    aligned_target_rots = align_rots[:, None, :, :].transpose(-1, -2) @ target_rots

    # Calculate rotation error after alignment
    aligned_target_rots = rotmat_to_rotvec(aligned_target_rots)
    target_angle = torch.norm(aligned_target_rots, dim=-1)
    target_axis = aligned_target_rots / target_angle[..., None, None]

    aligned_pred_rots = rotmat_to_rotvec(aligned_pred_rots)
    pred_angle = torch.norm(aligned_pred_rots, dim=-1)
    pred_axis = aligned_pred_rots / pred_angle[..., None, None]

    axis_error = torch.norm(pred_axis - target_axis, dim=-1)
    axis_error = axis_error.mean()

    angle_error = torch.abs(pred_angle - target_angle)
    angle_error = angle_error.mean()

    return axis_error, angle_error


def _project_to_so3(batch_mats: torch.Tensor) -> torch.Tensor:
    """Project a batch of 3x3 matrices onto SO(3) using SVD.

    Args:
        batch_mats: (..., 3, 3) real matrices
    Returns:
        (..., 3, 3) rotation matrices in SO(3)
    """
    U, S, Vh = torch.linalg.svd(batch_mats)
    R = U @ Vh
    # Ensure det(R) = +1 by flipping the last column of U when needed
    det = torch.det(R)
    need_flip = det < 0
    if need_flip.any():
        # Flip sign of last column of U for the problematic entries
        # Build a diag matrix with last diag = -1 for those entries
        shape = R.shape[:-2]
        eye = torch.eye(3, device=R.device, dtype=R.dtype).expand(*shape, 3, 3).clone()
        eye[..., 2, 2] = torch.where(
            need_flip,
            torch.tensor(-1.0, device=R.device, dtype=R.dtype),
            torch.tensor(1.0, device=R.device, dtype=R.dtype),
        )
        R = (U @ eye) @ Vh
    return R


def extract_global_rotation(
    rotmats: torch.Tensor,
    res_mask: torch.Tensor | None = None,
    reference_rotmats: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract a single global rotation from per-residue rotation components.

    The global rotation is estimated as the weighted Procrustes mean on SO(3):
    we average the (relative) rotation matrices across residues and project
    the result back to SO(3) with SVD.

    Args:
        rotmats: (batch, num_res, 3, 3) per-residue rotation matrices.
        res_mask: (batch, num_res) optional mask/weights per residue.
        reference_rotmats: (batch, num_res, 3, 3) optional reference rotations.
            If provided, the relative rotation per residue is R_ref^T @ R.
    Returns:
        global_rot: (batch, 3, 3) estimated global rotation.
    """
    assert rotmats.ndim == 4 and rotmats.shape[-2:] == (
        3,
        3,
    ), "rotmats must be (B, N, 3, 3)"
    B, N, _, _ = rotmats.shape

    if reference_rotmats is not None:
        assert (
            reference_rotmats.shape == rotmats.shape
        ), "reference_rotmats must match rotmats shape"
        rel = reference_rotmats.transpose(-1, -2) @ rotmats  # (B, N, 3, 3)
    else:
        rel = rotmats

    if res_mask is not None:
        assert res_mask.shape[:2] == rotmats.shape[:2], "res_mask must be (B, N)"
        weights = res_mask.to(rotmats.dtype)[..., None, None]  # (B, N, 1, 1)
    else:
        weights = torch.ones((B, N, 1, 1), device=rotmats.device, dtype=rotmats.dtype)

    # Weighted sum then project to SO(3)
    M = (weights * rel).sum(dim=1)  # (B, 3, 3)
    global_rot = _project_to_so3(M)  # (B, 3, 3)
    return global_rot


def remove_global_rotation(
    rotmats: torch.Tensor,
    global_rot: torch.Tensor,
) -> torch.Tensor:
    """Remove a global rotation from per-residue rotations.

    Args:
        rotmats: (batch, num_res, 3, 3)
        global_rot: (batch, 3, 3)
    Returns:
        rotmats_wo_global: (batch, num_res, 3, 3) where each residue is left-multiplied by global_rot^T
    """
    assert rotmats.ndim == 4 and rotmats.shape[-2:] == (3, 3)
    assert global_rot.shape[-2:] == (3, 3)
    # Broadcast global_rot^T over residues
    g_inv = global_rot.transpose(-1, -2)  # (B, 3, 3)
    return g_inv[:, None, :, :] @ rotmats
