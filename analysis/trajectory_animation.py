"""
Trajectory animation utilities for visualizing protein rigid body dynamics.
Provides functions to plot protein structures with cartoon backbone and ball-and-stick sidechains.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
from typing import Optional, List, Tuple, Union
from scipy.interpolate import splprep, splev

from data import all_atom
from openfold.np import residue_constants


def get_rainbow_colors(num_res: int, cmap: str = "rainbow") -> np.ndarray:
    """Generate rainbow colors for residues.

    Args:
        num_res: Number of residues
        cmap: Matplotlib colormap name

    Returns:
        Array of RGBA colors with shape (num_res, 4)
    """
    colormap = plt.cm.get_cmap(cmap)
    colors = colormap(np.linspace(0, 1, num_res))
    return colors


def smooth_backbone_curve(
    ca_positions: np.ndarray, num_points: int = 100
) -> np.ndarray:
    """Create smooth spline curve through CA positions for cartoon representation.

    Args:
        ca_positions: CA atom positions with shape (num_res, 3)
        num_points: Number of interpolation points

    Returns:
        Smoothed curve coordinates with shape (num_points, 3)
    """
    if len(ca_positions) < 4:
        return ca_positions

    try:
        # Fit spline through CA positions
        tck, u = splprep(
            [ca_positions[:, 0], ca_positions[:, 1], ca_positions[:, 2]],
            s=0,
            k=min(3, len(ca_positions) - 1),
        )
        u_new = np.linspace(0, 1, num_points)
        smooth_curve = np.array(splev(u_new, tck)).T
        return smooth_curve
    except:
        # Fallback to linear interpolation
        return ca_positions


def compute_secondary_structure_width(ca_positions: np.ndarray) -> np.ndarray:
    """Compute varying width for cartoon representation based on local curvature.

    Args:
        ca_positions: CA positions with shape (num_res, 3)

    Returns:
        Width values for each residue
    """
    num_res = len(ca_positions)
    widths = np.ones(num_res) * 2.0  # Base width

    if num_res < 3:
        return widths

    # Compute local curvature as indicator of secondary structure
    for i in range(1, num_res - 1):
        v1 = ca_positions[i] - ca_positions[i - 1]
        v2 = ca_positions[i + 1] - ca_positions[i]
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            # Helical regions have more curvature
            if angle > 0.3:  # More curved = helix-like
                widths[i] = 3.5
            else:  # Straight = sheet-like
                widths[i] = 1.5

    return widths


def plot_protein_structure(
    ax: plt.Axes,
    trans: torch.Tensor,
    rots: torch.Tensor,
    res_mask: Optional[torch.Tensor] = None,
    show_backbone: bool = True,
    show_sidechains: bool = True,
    backbone_style: str = "cartoon",  # "cartoon", "tube", "line"
    sidechain_style: str = "ball_and_stick",  # "ball_and_stick", "stick"
    cmap: str = "rainbow",
    alpha: float = 1.0,
    backbone_linewidth: float = 3.0,
    ball_size: float = 20,
) -> None:
    """Plot a single protein structure on a 3D axes.

    Args:
        ax: Matplotlib 3D axes
        trans: Translation tensor (1, num_res, 3) or (num_res, 3)
        rots: Rotation tensor (1, num_res, 3, 3) or (num_res, 3, 3)
        res_mask: Residue mask tensor
        show_backbone: Whether to show backbone
        show_sidechains: Whether to show sidechains
        backbone_style: Style for backbone rendering
        sidechain_style: Style for sidechain rendering
        cmap: Colormap for residue coloring
        alpha: Transparency
        backbone_linewidth: Line width for backbone
        ball_size: Size of atoms in ball-and-stick representation
    """
    # Handle batch dimension
    if trans.dim() == 2:
        trans = trans.unsqueeze(0)
        rots = rots.unsqueeze(0)

    if res_mask is None:
        res_mask = torch.ones(trans.shape[:2], device=trans.device)
    elif res_mask.dim() == 1:
        res_mask = res_mask.unsqueeze(0)

    # Convert to atom37 representation
    atom37 = all_atom.atom37_from_trans_rot(trans, rots, res_mask)
    atom37_np = atom37[0].detach().cpu().numpy()  # (num_res, 37, 3)

    num_res = atom37_np.shape[0]
    colors = get_rainbow_colors(num_res, cmap)

    # Get atom indices
    ca_idx = residue_constants.atom_order["CA"]
    n_idx = residue_constants.atom_order["N"]
    c_idx = residue_constants.atom_order["C"]
    o_idx = residue_constants.atom_order["O"]
    cb_idx = residue_constants.atom_order["CB"]

    # Extract backbone atom positions
    ca_pos = atom37_np[:, ca_idx, :]  # (num_res, 3)
    n_pos = atom37_np[:, n_idx, :]
    c_pos = atom37_np[:, c_idx, :]
    o_pos = atom37_np[:, o_idx, :]
    cb_pos = atom37_np[:, cb_idx, :]

    if show_backbone:
        if backbone_style == "cartoon":
            # Smooth backbone with varying width (cartoon-like)
            smooth_ca = smooth_backbone_curve(ca_pos, num_points=num_res * 10)

            # Create line segments with varying colors
            points = smooth_ca.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Interpolate colors for smooth curve
            segment_colors = get_rainbow_colors(len(segments), cmap)
            segment_colors[:, 3] = alpha

            # Add segments as Line3DCollection
            for i in range(len(segments)):
                ax.plot3D(
                    [segments[i, 0, 0], segments[i, 1, 0]],
                    [segments[i, 0, 1], segments[i, 1, 1]],
                    [segments[i, 0, 2], segments[i, 1, 2]],
                    color=segment_colors[i],
                    linewidth=backbone_linewidth,
                    solid_capstyle="round",
                    alpha=alpha,
                )

        elif backbone_style == "tube":
            # Draw tube-like backbone through CA atoms
            for i in range(num_res - 1):
                ax.plot3D(
                    [ca_pos[i, 0], ca_pos[i + 1, 0]],
                    [ca_pos[i, 1], ca_pos[i + 1, 1]],
                    [ca_pos[i, 2], ca_pos[i + 1, 2]],
                    color=colors[i],
                    linewidth=backbone_linewidth * 1.5,
                    solid_capstyle="round",
                    alpha=alpha,
                )
        else:  # line
            # Simple line through CA atoms
            for i in range(num_res - 1):
                ax.plot3D(
                    [ca_pos[i, 0], ca_pos[i + 1, 0]],
                    [ca_pos[i, 1], ca_pos[i + 1, 1]],
                    [ca_pos[i, 2], ca_pos[i + 1, 2]],
                    color=colors[i],
                    linewidth=backbone_linewidth,
                    alpha=alpha,
                )

    if show_sidechains:
        if sidechain_style == "ball_and_stick":
            # Draw all backbone atoms as balls
            for i in range(num_res):
                color = colors[i]

                # N atoms (blue tint)
                ax.scatter(
                    n_pos[i, 0],
                    n_pos[i, 1],
                    n_pos[i, 2],
                    c=[color],
                    s=ball_size * 0.8,
                    alpha=alpha,
                    edgecolors="none",
                )

                # CA atoms
                ax.scatter(
                    ca_pos[i, 0],
                    ca_pos[i, 1],
                    ca_pos[i, 2],
                    c=[color],
                    s=ball_size,
                    alpha=alpha,
                    edgecolors="none",
                )

                # C atoms
                ax.scatter(
                    c_pos[i, 0],
                    c_pos[i, 1],
                    c_pos[i, 2],
                    c=[color],
                    s=ball_size * 0.8,
                    alpha=alpha,
                    edgecolors="none",
                )

                # O atoms (red tint)
                ax.scatter(
                    o_pos[i, 0],
                    o_pos[i, 1],
                    o_pos[i, 2],
                    c=[color],
                    s=ball_size * 0.7,
                    alpha=alpha,
                    edgecolors="none",
                )

                # CB atoms (if exists)
                if np.any(cb_pos[i] != 0):
                    ax.scatter(
                        cb_pos[i, 0],
                        cb_pos[i, 1],
                        cb_pos[i, 2],
                        c=[color],
                        s=ball_size * 0.9,
                        alpha=alpha,
                        edgecolors="none",
                    )

                    # Draw stick from CA to CB
                    ax.plot3D(
                        [ca_pos[i, 0], cb_pos[i, 0]],
                        [ca_pos[i, 1], cb_pos[i, 1]],
                        [ca_pos[i, 2], cb_pos[i, 2]],
                        color=color,
                        linewidth=1.5,
                        alpha=alpha,
                    )

                # Draw sticks between backbone atoms
                # N-CA
                ax.plot3D(
                    [n_pos[i, 0], ca_pos[i, 0]],
                    [n_pos[i, 1], ca_pos[i, 1]],
                    [n_pos[i, 2], ca_pos[i, 2]],
                    color=color,
                    linewidth=1.5,
                    alpha=alpha,
                )

                # CA-C
                ax.plot3D(
                    [ca_pos[i, 0], c_pos[i, 0]],
                    [ca_pos[i, 1], c_pos[i, 1]],
                    [ca_pos[i, 2], c_pos[i, 2]],
                    color=color,
                    linewidth=1.5,
                    alpha=alpha,
                )

                # C-O
                ax.plot3D(
                    [c_pos[i, 0], o_pos[i, 0]],
                    [c_pos[i, 1], o_pos[i, 1]],
                    [c_pos[i, 2], o_pos[i, 2]],
                    color=color,
                    linewidth=1.5,
                    alpha=alpha,
                )

                # Peptide bond C-N
                if i < num_res - 1:
                    mid_color = (colors[i] + colors[i + 1]) / 2
                    ax.plot3D(
                        [c_pos[i, 0], n_pos[i + 1, 0]],
                        [c_pos[i, 1], n_pos[i + 1, 1]],
                        [c_pos[i, 2], n_pos[i + 1, 2]],
                        color=mid_color,
                        linewidth=1.5,
                        alpha=alpha,
                    )

        else:  # stick only
            for i in range(num_res):
                color = colors[i]
                # Only draw sticks, no balls
                if i < num_res - 1:
                    ax.plot3D(
                        [ca_pos[i, 0], ca_pos[i + 1, 0]],
                        [ca_pos[i, 1], ca_pos[i + 1, 1]],
                        [ca_pos[i, 2], ca_pos[i + 1, 2]],
                        color=color,
                        linewidth=2,
                        alpha=alpha,
                    )


def set_axes_equal(ax: plt.Axes) -> None:
    """Set 3D plot axes to equal scale.

    Makes axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Args:
        ax: 3D matplotlib axes
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def plot_trajectory_grid(
    trans_traj: Union[List[torch.Tensor], torch.Tensor],
    rots_traj: Union[List[torch.Tensor], torch.Tensor],
    res_mask: Optional[torch.Tensor] = None,
    n_frames: int = 6,
    ncols: int = 3,
    figsize: Tuple[float, float] = (15, 10),
    elev: float = 20,
    azim: float = 45,
    show_backbone: bool = True,
    show_sidechains: bool = True,
    backbone_style: str = "cartoon",
    sidechain_style: str = "ball_and_stick",
    cmap: str = "rainbow",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 150,
    backbone_endpoints_only: bool = True,
    **kwargs,
) -> plt.Figure:
    """Plot trajectory frames in a grid layout.

    Args:
        trans_traj: List of translation tensors or stacked tensor (T, B, N, 3)
        rots_traj: List of rotation tensors or stacked tensor (T, B, N, 3, 3)
        res_mask: Residue mask
        n_frames: Number of frames to display
        ncols: Number of columns in grid
        figsize: Figure size
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
        show_backbone: Show backbone
        show_sidechains: Show sidechains
        backbone_style: Backbone rendering style
        sidechain_style: Sidechain rendering style
        cmap: Colormap
        title: Figure title
        save_path: Path to save figure
        dpi: Figure DPI
        backbone_endpoints_only: If True, only show backbone for first and last frames
        **kwargs: Additional arguments for plot_protein_structure

    Returns:
        Matplotlib figure
    """
    # Convert to list if tensor
    if isinstance(trans_traj, torch.Tensor):
        trans_traj = [trans_traj[i] for i in range(trans_traj.shape[0])]
        rots_traj = [rots_traj[i] for i in range(rots_traj.shape[0])]

    total_frames = len(trans_traj)

    # Select frames to display
    if n_frames >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()

    n_frames = len(frame_indices)
    nrows = (n_frames + ncols - 1) // ncols

    fig = plt.figure(figsize=figsize, facecolor="white")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")

        trans = trans_traj[frame_idx]
        rots = rots_traj[frame_idx]

        # Handle batch dimension - take first sample
        if trans.dim() == 3:
            trans = trans[0]
            rots = rots[0]

        # Determine whether to show backbone for this frame
        is_endpoint = idx == 0 or idx == n_frames - 1
        frame_show_backbone = show_backbone and (
            not backbone_endpoints_only or is_endpoint
        )

        plot_protein_structure(
            ax,
            trans,
            rots,
            res_mask,
            show_backbone=frame_show_backbone,
            show_sidechains=show_sidechains,
            backbone_style=backbone_style,
            sidechain_style=sidechain_style,
            cmap=cmap,
            **kwargs,
        )

        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        set_axes_equal(ax)

        # Clean up axes
        ax.set_axis_off()
        ax.set_title(f"t = {frame_idx / (total_frames - 1):.2f}", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )

    return fig


def create_trajectory_animation(
    trans_traj: Union[List[torch.Tensor], torch.Tensor],
    rots_traj: Union[List[torch.Tensor], torch.Tensor],
    res_mask: Optional[torch.Tensor] = None,
    figsize: Tuple[float, float] = (8, 8),
    elev: float = 20,
    azim: float = 45,
    rotate: bool = True,
    rotation_speed: float = 2.0,
    show_backbone: bool = True,
    show_sidechains: bool = True,
    backbone_style: str = "cartoon",
    sidechain_style: str = "ball_and_stick",
    cmap: str = "rainbow",
    interval: int = 50,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    fps: int = 20,
    dpi: int = 150,
    backbone_endpoints_only: bool = True,
    **kwargs,
) -> animation.FuncAnimation:
    """Create animated trajectory visualization.

    Args:
        trans_traj: List of translation tensors or stacked tensor
        rots_traj: List of rotation tensors or stacked tensor
        res_mask: Residue mask
        figsize: Figure size
        elev: Initial elevation angle
        azim: Initial azimuth angle
        rotate: Whether to rotate view during animation
        rotation_speed: Rotation speed in degrees per frame
        show_backbone: Show backbone
        show_sidechains: Show sidechains
        backbone_style: Backbone rendering style
        sidechain_style: Sidechain rendering style
        cmap: Colormap
        interval: Animation interval in milliseconds
        title: Animation title
        save_path: Path to save animation (supports .gif, .mp4)
        fps: Frames per second for saved animation
        dpi: DPI for saved animation
        backbone_endpoints_only: If True, only show backbone for first and last frames
        **kwargs: Additional arguments for plot_protein_structure

    Returns:
        Matplotlib animation object
    """
    # Convert to list if tensor
    if isinstance(trans_traj, torch.Tensor):
        trans_traj = [trans_traj[i] for i in range(trans_traj.shape[0])]
        rots_traj = [rots_traj[i] for i in range(rots_traj.shape[0])]

    n_frames = len(trans_traj)

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # Compute global bounds for consistent view
    all_ca_pos = []
    ca_idx = residue_constants.atom_order["CA"]

    for trans, rots in zip(trans_traj, rots_traj):
        if trans.dim() == 3:
            trans_sample = trans[0]
            rots_sample = rots[0]
        else:
            trans_sample = trans
            rots_sample = rots

        atom37 = all_atom.atom37_from_trans_rot(
            trans_sample.unsqueeze(0), rots_sample.unsqueeze(0), res_mask
        )
        ca_pos = atom37[0, :, ca_idx, :].detach().cpu().numpy()
        all_ca_pos.append(ca_pos)

    all_ca_pos = np.concatenate(all_ca_pos, axis=0)
    center = all_ca_pos.mean(axis=0)
    max_extent = np.max(np.abs(all_ca_pos - center)) * 1.5

    def init():
        ax.clear()
        return []

    def update(frame):
        ax.clear()

        trans = trans_traj[frame]
        rots = rots_traj[frame]

        if trans.dim() == 3:
            trans = trans[0]
            rots = rots[0]

        # Determine whether to show backbone for this frame
        is_endpoint = frame == 0 or frame == n_frames - 1
        frame_show_backbone = show_backbone and (
            not backbone_endpoints_only or is_endpoint
        )

        plot_protein_structure(
            ax,
            trans,
            rots,
            res_mask,
            show_backbone=frame_show_backbone,
            show_sidechains=show_sidechains,
            backbone_style=backbone_style,
            sidechain_style=sidechain_style,
            cmap=cmap,
            **kwargs,
        )

        # Set consistent bounds
        ax.set_xlim([center[0] - max_extent, center[0] + max_extent])
        ax.set_ylim([center[1] - max_extent, center[1] + max_extent])
        ax.set_zlim([center[2] - max_extent, center[2] + max_extent])

        # Rotate view if enabled
        current_azim = azim + (rotation_speed * frame if rotate else 0)
        ax.view_init(elev=elev, azim=current_azim)

        ax.set_axis_off()

        if title:
            ax.set_title(f"{title}\nt = {frame / (n_frames - 1):.2f}", fontsize=12)
        else:
            ax.set_title(f"t = {frame / (n_frames - 1):.2f}", fontsize=12)

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init, interval=interval, blit=False
    )

    if save_path:
        if save_path.endswith(".gif"):
            writer = animation.PillowWriter(fps=fps)
        elif save_path.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=fps)
        else:
            writer = animation.PillowWriter(fps=fps)
            save_path = save_path + ".gif"

        anim.save(save_path, writer=writer, dpi=dpi)
        print(f"Animation saved to {save_path}")

    return anim


def plot_trajectory_comparison(
    trans_traj_list: List[Union[List[torch.Tensor], torch.Tensor]],
    rots_traj_list: List[Union[List[torch.Tensor], torch.Tensor]],
    labels: List[str],
    res_mask: Optional[torch.Tensor] = None,
    n_frames: int = 4,
    figsize: Tuple[float, float] = (16, 8),
    elev: float = 20,
    azim: float = 45,
    cmap: str = "rainbow",
    save_path: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Compare multiple trajectories side by side.

    Args:
        trans_traj_list: List of trajectory translations
        rots_traj_list: List of trajectory rotations
        labels: Labels for each trajectory
        res_mask: Residue mask
        n_frames: Frames per trajectory
        figsize: Figure size
        elev: Elevation angle
        azim: Azimuth angle
        cmap: Colormap
        save_path: Path to save figure
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure
    """
    n_trajs = len(trans_traj_list)

    fig = plt.figure(figsize=figsize, facecolor="white")

    for traj_idx, (trans_traj, rots_traj, label) in enumerate(
        zip(trans_traj_list, rots_traj_list, labels)
    ):
        # Convert to list if tensor
        if isinstance(trans_traj, torch.Tensor):
            trans_traj = [trans_traj[i] for i in range(trans_traj.shape[0])]
            rots_traj = [rots_traj[i] for i in range(rots_traj.shape[0])]

        total_frames = len(trans_traj)
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()

        for frame_plot_idx, frame_idx in enumerate(frame_indices):
            ax_idx = traj_idx * n_frames + frame_plot_idx + 1
            ax = fig.add_subplot(n_trajs, n_frames, ax_idx, projection="3d")

            trans = trans_traj[frame_idx]
            rots = rots_traj[frame_idx]

            if trans.dim() == 3:
                trans = trans[0]
                rots = rots[0]

            plot_protein_structure(ax, trans, rots, res_mask, cmap=cmap, **kwargs)

            ax.view_init(elev=elev, azim=azim)
            set_axes_equal(ax)
            ax.set_axis_off()

            if frame_plot_idx == 0:
                ax.set_ylabel(label, fontsize=10)

            t_value = frame_idx / (total_frames - 1)
            ax.set_title(f"t={t_value:.2f}", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none"
        )

    return fig


# Convenience function for quick plotting
def quick_plot(
    trans: torch.Tensor,
    rots: torch.Tensor,
    res_mask: Optional[torch.Tensor] = None,
    figsize: Tuple[float, float] = (8, 8),
    **kwargs,
) -> plt.Figure:
    """Quick single structure plot.

    Args:
        trans: Translation tensor
        rots: Rotation tensor
        res_mask: Residue mask
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    plot_protein_structure(ax, trans, rots, res_mask, **kwargs)

    set_axes_equal(ax)
    ax.set_axis_off()

    plt.tight_layout()
    return fig
