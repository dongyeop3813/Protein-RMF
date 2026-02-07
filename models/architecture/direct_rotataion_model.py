import torch
from torch import nn

from models.architecture.node_feature_net import *
from models.architecture.edge_feature_net import EdgeFeatureNet
from models.architecture.meanflow_model import MeanFlowModel, AdaLNScale
from models.architecture import ipa_pytorch

import data.utils as du
from data import so3_utils


class DirectRotationModel(MeanFlowModel):
    """
    Velocity prediction model (based on transformer instead of IPA).
    This is non-equivariant.
    """

    def __init__(self, model_conf):
        super(MeanFlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa

        self.ang_to_nm = lambda x: x * du.ANG_TO_NM_SCALE
        self.nm_to_ang = lambda x: x * du.NM_TO_ANG_SCALE

        self.node_feature_net = MeanFlowNodeFeatureNetv2(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f"ipa_{b}"] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer,
                self._ipa_conf.seq_tfmr_num_layers,
                enable_nested_tensor=False,
            )
            self.trunk[f"post_tfmr_{b}"] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final"
            )
            self.trunk[f"node_transition_{b}"] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s
            )
            self.trunk[f"bb_update_{b}"] = ipa_pytorch.Linear(
                self._ipa_conf.c_s,
                9,
                init="final",
            )

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f"edge_transition_{b}"] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

            if self._model_conf.get("strict_time_conditioning", False):
                self.trunk[f"skip_embed_{b}"] = nn.Linear(
                    self._model_conf.node_embed_size, self._ipa_conf.c_s
                )

    def forward(self, trans_t, rotmat_t, t, r, feats, trans_sc=None):
        node_mask = feats["res_mask"]
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = feats["diffuse_mask"]
        res_index = feats["res_idx"]

        so3_t = t
        r3_t = t
        r = r
        trans_t = trans_t
        rotmats_t = rotmat_t

        # Initialize node and edge embeddings
        init_node_embed, time_emb, time_gap_emb = self.node_feature_net(
            so3_t, r3_t, r, node_mask, diffuse_mask, res_index, return_time_emb=True
        )
        time_emb = torch.cat([time_emb, time_gap_emb], dim=-1)

        trans_sc = torch.zeros_like(trans_t)

        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,
            trans_sc,
            edge_mask,
            diffuse_mask,
        )

        # Initial rigids
        # curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        # curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        trans_t = self.ang_to_nm(trans_t)

        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            # Current rigid for this block.
            curr_rigids = du.create_rigid(rotmats_t, trans_t)
            ipa_embed = self.trunk[f"ipa_{b}"](
                node_embed, edge_embed, curr_rigids, node_mask
            )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)
            if self._model_conf.get("strict_time_conditioning", False):
                node_embed = node_embed + self.trunk[f"skip_embed_{b}"](init_node_embed)
            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool)
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f"bb_update_{b}"](
                node_embed * node_mask[..., None]
            )
            # curr_rigids = curr_rigids.compose_q_update_vec(
            #     rigid_update, (node_mask * diffuse_mask)[..., None]
            # )
            trans_t, rotmats_t = self.rigid_update(trans_t, rotmats_t, rigid_update)

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        # pred_trans = curr_rigids.get_trans()
        # pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        pred_trans = trans_t
        pred_rotmats = rotmats_t
        pred_trans = self.nm_to_ang(pred_trans)
        return pred_trans, pred_rotmats

    def rigid_update(
        self, trans_t, rotmats_t, update_vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w1, w2, t_vec = (
            update_vec[..., :3],
            update_vec[..., 3:6],
            update_vec[..., 6:],
        )

        # Orthogonalize w1 and w2
        w1 = safe_normalize(w1)
        # Use batched dot product instead of torch.dot
        w1_dot_w2 = torch.sum(w1 * w2, dim=-1, keepdim=True)
        w2 = w2 - w1_dot_w2 * w1
        w2 = safe_normalize(w2)

        # Get a third vector orthogonal to w1 and w2
        w3 = torch.cross(w1, w2, dim=-1)

        # Get a final rotation matrix concat(w1, w2, w3)
        new_rotmats = torch.stack([w1, w2, w3], dim=-1)
        new_rotmats = so3_utils.rot_mult(rotmats_t, new_rotmats)
        new_trans = trans_t + t_vec
        return new_trans, new_rotmats


def safe_normalize(w, eps=1e-6):
    norm = torch.sqrt(torch.sum(w**2, dim=-1, keepdim=True) + eps)
    return w / norm
