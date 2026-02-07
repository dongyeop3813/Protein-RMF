import torch
from torch import nn

from models.architecture.node_feature_net import *
from models.architecture.edge_feature_net import EdgeFeatureNet
from models.architecture.meanflow_model import MeanFlowModel, AdaLNScale
from models.architecture import ipa_pytorch

import data.utils as du


class VelocityPredictionModel(MeanFlowModel):
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

        node_embed_size = self._model_conf.node_embed_size
        time_emb_size = self._model_conf.node_features.c_timestep_emb
        tfmr_in = self._ipa_conf.c_s

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f"ipa_{b}"] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )

            self.trunk[f"ipa_adaln_{b}"] = AdaLNScale(
                node_embed_size, 2 * time_emb_size
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
            self.trunk[f"bb_update_{b}"] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True
            )

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f"edge_transition_{b}"] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

    def forward(self, trans_t, rotmat_t, t, r, feats):
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
            node_embed = self.trunk[f"ipa_adaln_{b}"](node_embed, time_emb)
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
        w_vec, t_vec = update_vec[..., :3], update_vec[..., 3:]
        new_rotmats = exp(rotmats_t, w_vec)
        new_trans = trans_t + t_vec
        return new_trans, new_rotmats
