import torch
from torch import nn

from models.architecture.node_feature_net import *
from models.architecture.edge_feature_net import EdgeFeatureNet
from models.architecture.parametrization import CondVelocityParametrization
from models.architecture import ipa_pytorch
from data import utils as du

from openfold.utils import rigid_utils as ru
from data.so3_utils import calc_rot_vf, exp, skew_matrix_to_vector


class MeanFlowModel(nn.Module, CondVelocityParametrization):

    def __init__(self, model_conf):
        super(MeanFlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(
            lambda x: x * du.ANG_TO_NM_SCALE
        )
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(
            lambda x: x * du.NM_TO_ANG_SCALE
        )
        if model_conf.node_features.get("version", "v1") == "v2":
            self.node_feature_net = MeanFlowNodeFeatureNetv2(model_conf.node_features)
        elif model_conf.node_features.get("version", "v1") == "v3":
            self.node_feature_net = MeanFlowNodeFeatureNetv3(model_conf.node_features)
        else:
            self.node_feature_net = MeanFlowNodeFeatureNet(model_conf.node_features)
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
        init_node_embed = self.node_feature_net(
            so3_t, r3_t, r, node_mask, diffuse_mask, res_index
        )

        if trans_sc is None:
            trans_sc = torch.zeros_like(trans_t)

        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,
            trans_sc,
            edge_mask,
            diffuse_mask,
        )

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
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
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None]
            )
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        return pred_trans, pred_rotmats


class AdaLNScale(nn.Module):
    def __init__(self, embed_dim: int, time_emb_dim: int, num_scalars: int = 2):
        super(AdaLNScale, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(time_emb_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, num_scalars * embed_dim),
        )
        self.num_scalars = num_scalars

        # AdaLN zero initialization.
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x, time_emb):
        out = self.layers(time_emb)
        scale, shift = out.chunk(2, dim=-1)
        return x * (1 + scale) + shift

    def scalers(self, time_emb):
        out = self.layers(time_emb)
        return out.chunk(self.num_scalars, dim=-1)


def modulate(x, scale, shift):
    return x * (1 + scale) + shift


class MeanFlowModelv2(MeanFlowModel):
    """
    MeanFlowModelv1 + AdaLN time-conditioned skip connection.
    """

    def __init__(self, model_conf):
        super(MeanFlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(
            lambda x: x * du.ANG_TO_NM_SCALE
        )
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(
            lambda x: x * du.NM_TO_ANG_SCALE
        )

        self.node_feature_net = MeanFlowNodeFeatureNetv2(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        node_embed_size = self._model_conf.node_embed_size
        time_emb_size = self._model_conf.node_features.c_timestep_emb
        tfmr_in = self._ipa_conf.c_s

        self.time_gap_layer = nn.Sequential(
            nn.Linear(time_emb_size, 4 * time_emb_size),
            nn.SiLU(),
            nn.Linear(4 * time_emb_size, time_emb_size),
        )
        self.time_emb_layer = nn.Sequential(
            nn.Linear(time_emb_size, 4 * time_emb_size),
            nn.SiLU(),
            nn.Linear(4 * time_emb_size, time_emb_size),
        )

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

        time_emb = self.time_emb_layer(time_emb)
        time_gap_emb = self.time_gap_layer(time_gap_emb)
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
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
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

            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None], stable=True
            )
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        return pred_trans, pred_rotmats


class MeanFlowAdaLNZeroBlock(nn.Module):
    def __init__(
        self,
        node_embed_size: int,
        time_emb_dim: int,
        ipa_conf,
        edge_embed_size: int,
        has_edge_transition: bool,
    ):
        super().__init__()
        self.node_embed_size = node_embed_size
        self.edge_embed_size = edge_embed_size
        self.has_edge_transition = has_edge_transition

        self.modulators = AdaLNScale(node_embed_size, time_emb_dim, num_scalars=7)
        self.ipa = ipa_pytorch.InvariantPointAttention(ipa_conf)
        self.ipa_ln = nn.LayerNorm(ipa_conf.c_s)
        self.tfmr_ln = nn.LayerNorm(ipa_conf.c_s)

        tfmr_layer = torch.nn.TransformerEncoderLayer(
            d_model=ipa_conf.c_s,
            nhead=ipa_conf.seq_tfmr_num_heads,
            dim_feedforward=ipa_conf.c_s,
            batch_first=True,
            dropout=0.0,
            norm_first=False,
        )
        self.seq_tfmr = torch.nn.TransformerEncoder(
            tfmr_layer,
            ipa_conf.seq_tfmr_num_layers,
            enable_nested_tensor=False,
        )
        self.post_tfmr = ipa_pytorch.Linear(ipa_conf.c_s, ipa_conf.c_s, init="final")
        self.node_transition = ipa_pytorch.StructureModuleTransition(c=ipa_conf.c_s)
        self.bb_update = ipa_pytorch.BackboneUpdate(ipa_conf.c_s, use_rot_updates=True)

        if has_edge_transition:
            self.edge_transition = ipa_pytorch.EdgeTransition(
                node_embed_size=ipa_conf.c_s,
                edge_embed_in=edge_embed_size,
                edge_embed_out=edge_embed_size,
            )
        else:
            self.edge_transition = None

        # Project gating scalers to the dimensions required by rigid and edge updates.
        self.gate_rigid_proj = nn.Linear(node_embed_size, 6)
        nn.init.zeros_(self.gate_rigid_proj.weight)
        nn.init.zeros_(self.gate_rigid_proj.bias)

    def forward(
        self,
        node_embed,
        edge_embed,
        curr_rigids,
        node_mask,
        edge_mask,
        diffuse_mask,
        padding_mask,
        time_emb,
    ):
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_tfmr,
            scale_tfmr,
            gate_tfmr,
            gate_rigid,
        ) = self.modulators.scalers(time_emb)

        gate_rigid = self.gate_rigid_proj(gate_rigid)

        gate_attn = 1 + gate_attn  # keep trunk active at init
        gate_tfmr = 1 + gate_tfmr
        gate_rigid = 1 + gate_rigid

        ipa_in = self.ipa_ln(node_embed)
        ipa_in = modulate(ipa_in, scale_attn, shift_attn)

        ipa_out = self.ipa(ipa_in, edge_embed, curr_rigids, node_mask)
        ipa_out *= node_mask[..., None]
        node_embed = node_embed + gate_attn * ipa_out

        tfmr_in = self.tfmr_ln(node_embed)
        tfmr_in = modulate(tfmr_in, scale_tfmr, shift_tfmr)
        seq_tfmr_out = self.seq_tfmr(tfmr_in, src_key_padding_mask=padding_mask)
        seq_tfmr_out = self.post_tfmr(seq_tfmr_out)
        node_embed = node_embed + gate_tfmr * seq_tfmr_out

        node_embed = self.node_transition(node_embed)
        node_embed = node_embed * node_mask[..., None]

        rigid_update = self.bb_update(node_embed)
        rigid_update = gate_rigid * rigid_update
        curr_rigids = curr_rigids.compose_q_update_vec(
            rigid_update, (node_mask * diffuse_mask)[..., None]
        )

        if self.edge_transition is not None:
            edge_embed = self.edge_transition(node_embed, edge_embed)
            edge_embed = edge_embed * edge_mask[..., None]

        return node_embed, edge_embed, curr_rigids


class MeanFlowModelAdaLNZero(MeanFlowModel):
    """
    v3: MeanFlowModel + AdaLN-zero time-conditioning.
    This is similar to v2, but the adaptive scaling is also applied to the blocks before residual connections.
    """

    def __init__(self, model_conf):
        super(MeanFlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(
            lambda x: x * du.ANG_TO_NM_SCALE
        )
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(
            lambda x: x * du.NM_TO_ANG_SCALE
        )

        self.node_feature_net = MeanFlowNodeFeatureNetv2(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        node_embed_size = self._model_conf.node_embed_size
        time_emb_size = self._model_conf.node_features.c_timestep_emb
        time_emb_dim = 2 * time_emb_size

        self.time_gap_layer = nn.Sequential(
            nn.Linear(time_emb_size, 4 * time_emb_size),
            nn.SiLU(),
            nn.Linear(4 * time_emb_size, time_emb_size),
        )
        self.time_emb_layer = nn.Sequential(
            nn.Linear(time_emb_size, 4 * time_emb_size),
            nn.SiLU(),
            nn.Linear(4 * time_emb_size, time_emb_size),
        )

        self.blocks = nn.ModuleList()
        for b in range(self._ipa_conf.num_blocks):
            has_edge_transition = b < self._ipa_conf.num_blocks - 1
            self.blocks.append(
                MeanFlowAdaLNZeroBlock(
                    node_embed_size=node_embed_size,
                    time_emb_dim=time_emb_dim,
                    ipa_conf=self._ipa_conf,
                    edge_embed_size=self._model_conf.edge_embed_size,
                    has_edge_transition=has_edge_transition,
                )
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

        time_emb = self.time_emb_layer(time_emb)
        time_gap_emb = self.time_gap_layer(time_gap_emb)
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
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        padding_mask = (1 - node_mask).to(torch.bool)

        for block in self.blocks:
            node_embed, edge_embed, curr_rigids = block(
                node_embed=node_embed,
                edge_embed=edge_embed,
                curr_rigids=curr_rigids,
                node_mask=node_mask,
                edge_mask=edge_mask,
                diffuse_mask=diffuse_mask,
                padding_mask=padding_mask,
                time_emb=time_emb,
            )

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        return pred_trans, pred_rotmats


class MeanFlowModelv4(MeanFlowModel):
    """
    MeanFlowModelv1 + AdaLN time-conditioned skip connection + Rigid updates are separated from initial rigids.
    """

    def __init__(self, model_conf):
        super(MeanFlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(
            lambda x: x * du.ANG_TO_NM_SCALE
        )
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(
            lambda x: x * du.NM_TO_ANG_SCALE
        )

        self.node_feature_net = MeanFlowNodeFeatureNetv2(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        node_embed_size = self._model_conf.node_embed_size
        time_emb_size = self._model_conf.node_features.c_timestep_emb
        tfmr_in = self._ipa_conf.c_s

        self.time_gap_layer = nn.Sequential(
            nn.Linear(time_emb_size, 4 * time_emb_size),
            nn.SiLU(),
            nn.Linear(4 * time_emb_size, time_emb_size),
        )
        self.time_emb_layer = nn.Sequential(
            nn.Linear(time_emb_size, 4 * time_emb_size),
            nn.SiLU(),
            nn.Linear(4 * time_emb_size, time_emb_size),
        )

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

        time_emb = self.time_emb_layer(time_emb)
        time_gap_emb = self.time_gap_layer(time_gap_emb)
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
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
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
            curr_rigids = self.update_rigids(
                curr_rigids, rigid_update, (node_mask * diffuse_mask)[..., None]
            )
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        return pred_trans, pred_rotmats

    def update_rigids(self, curr_rigids, rigid_update, update_mask):
        q_vec, t_vec = rigid_update[..., :3], rigid_update[..., 3:]
        new_rots = curr_rigids._rots.compose_q_update_vec(
            q_vec, update_mask=update_mask
        )
        t_vec = t_vec * update_mask
        new_translation = curr_rigids._trans + t_vec

        return du.Rigid(new_rots, new_translation)


class MeanFlowModelv4_2(MeanFlowModel):
    """
    MeanFlowModelv1 + AdaLN time-conditioned skip connection + Rigid updates are separated from initial rigids.
    SE3 equivariant version.
    """

    def __init__(self, model_conf):
        super(MeanFlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(
            lambda x: x * du.ANG_TO_NM_SCALE
        )
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(
            lambda x: x * du.NM_TO_ANG_SCALE
        )

        self.node_feature_net = MeanFlowNodeFeatureNetv2(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        node_embed_size = self._model_conf.node_embed_size
        time_emb_size = self._model_conf.node_features.c_timestep_emb
        tfmr_in = self._ipa_conf.c_s

        self.time_gap_layer = nn.Sequential(
            nn.Linear(time_emb_size, 4 * time_emb_size),
            nn.SiLU(),
            nn.Linear(4 * time_emb_size, time_emb_size),
        )
        self.time_emb_layer = nn.Sequential(
            nn.Linear(time_emb_size, 4 * time_emb_size),
            nn.SiLU(),
            nn.Linear(4 * time_emb_size, time_emb_size),
        )

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

        time_emb = self.time_emb_layer(time_emb)
        time_gap_emb = self.time_gap_layer(time_gap_emb)
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
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        init_rots = ru.Rotation(rot_mats=rotmats_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
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
            curr_rigids = self.update_rigids(
                curr_rigids,
                rigid_update,
                (node_mask * diffuse_mask)[..., None],
                init_rots,
            )
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        return pred_trans, pred_rotmats

    def update_rigids(self, curr_rigids, rigid_update, update_mask, init_rots):
        q_vec, t_vec = rigid_update[..., :3], rigid_update[..., 3:]
        new_rots = curr_rigids._rots.compose_q_update_vec(
            q_vec, update_mask=update_mask
        )

        trans_update = init_rots.apply(t_vec)
        trans_update = trans_update * update_mask
        new_translation = curr_rigids._trans + trans_update

        return du.Rigid(new_rots, new_translation)


def project_to_SO3(rotmats: torch.Tensor) -> torch.Tensor:
    """
    Project a tensor to the SO(3) manifold.
    Shape:
        rotmats: (B, 3, 3)
    """
    # Batched SVD-based projection to the nearest rotation matrix (Frobenius norm)
    # Ensures det(R) = 1 by correcting reflections when needed.
    assert (
        rotmats.dim() == 3 and rotmats.size(-2) == 3 and rotmats.size(-1) == 3
    ), "rotmats must be (B, 3, 3)"

    # U @ Vh gives the orthogonal factor from polar decomposition
    U, S, Vh = torch.linalg.svd(rotmats)
    R = U @ Vh

    # If det(R) < 0, flip the last axis to enforce det = 1
    det_R = torch.det(R)
    sign = torch.where(det_R >= 0, torch.ones_like(det_R), -torch.ones_like(det_R))
    D = torch.diag_embed(
        torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], dim=-1)
    )

    R_corrected = U @ D @ Vh
    return R_corrected


def project_to_so3(mat: torch.Tensor) -> torch.Tensor:
    """
    Project a tensor to the space of skew symmetric matrices.
    Shape:
        mat: (B, 3, 3)
    """
    return (mat - mat.transpose(-2, -1)) / 2.0


from data.so3_interpolant import SO3Interpolant


class ToySO3MeanFlowModel(nn.Module):

    def __init__(self, cfg):
        super(ToySO3MeanFlowModel, self).__init__()
        self.cfg = cfg

        self.interpolant = None
        self.num_blocks = cfg.num_blocks
        self.emb_dim = cfg.emb_dim

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                nn.Linear(9 + 2, self.emb_dim),
                nn.SiLU(),
            )
        )
        for _ in range(self.num_blocks - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.emb_dim + 2, self.emb_dim),
                    nn.SiLU(),
                )
            )

        self.predict_head = nn.Linear(self.emb_dim, 9)

    def forward(self, Rt, t, r):
        """
        Shape:
            Rt: (B, 3, 3)
            t: (B,)
            r: (B,)
        """
        x = Rt.reshape(-1, 9)
        for layer in self.layers:
            x = torch.cat([x, t[..., None], r[..., None]], dim=-1)
            x = layer(x)
        x = self.predict_head(x).view(Rt.shape)
        return project_to_SO3(x)

    def set_interpolant(self, interpolant: SO3Interpolant):
        self.interpolant = interpolant

    def avg_vel(self, Rt, t, r):
        if self.cfg.get("cond_vel_parameterization", True):
            R1 = self(Rt, t, r)
            rot_vf = self.interpolant.cond_vf(t, Rt, R1)
        else:
            R1 = self(Rt, t, r)
            rot_vf = calc_rot_vf(Rt, R1)

        return rot_vf

    def forward_flow(self, Rt, t, r):
        rot_vf = self.avg_vel(Rt, t, r)
        Rr = exp(Rt, (r - t)[..., None] * rot_vf)
        return Rr

    def jvp_avg_vel(self, Rt, t, r, tangent):
        """
        tangent: (d_R, d_t, d_r)
        """
        return torch.func.jvp(self.avg_vel, (Rt, t, r), tangent)

    @torch.no_grad()
    def sample(self, x0, num_timesteps=100, verbose=False, return_traj=False):
        ts = torch.linspace(0.01, 1.0, num_timesteps)
        t1 = ts[0]
        traj = [x0]
        num_batch = x0.shape[0]
        device = x0.device

        for i, t2 in enumerate(ts[1:]):
            if verbose:  # and i % 1 == 0:
                print(f"{i=}, t={t1.item():.2f}")

            xt = traj[-1]
            r = t = torch.ones((num_batch,), device=device) * t1
            d_t = t2 - t1

            v_rot = self.avg_vel(xt, t, r)

            # Take reverse step
            xt = self.interpolant.euler_step(d_t, xt, v_rot)

            traj.append(xt)
            t1 = t2

        if return_traj:
            return traj
        else:
            return traj[-1]

    def one_step_sample(self, x0):
        t = torch.zeros((x0.shape[0],), device=x0.device, dtype=torch.float32)
        r = torch.ones((x0.shape[0],), device=x0.device, dtype=torch.float32)
        return self.forward_flow(x0, t, r)


class ToySO3MeanFlowModelv2(ToySO3MeanFlowModel):
    def __init__(self, cfg):
        super(ToySO3MeanFlowModelv2, self).__init__(cfg)

    def forward(self, Rt, t, r):
        """
        Shape:
            Rt: (B, 3, 3)
            t: (B,)
            r: (B,)
        """
        x = Rt.reshape(-1, 9)
        for layer in self.layers:
            x = torch.cat([x, t[..., None], r[..., None]], dim=-1)
            x = layer(x)
        x = self.predict_head(x).view(Rt.shape)
        return project_to_so3(x)

    def avg_vel(self, Rt, t, r):
        v = skew_matrix_to_vector(self(Rt, t, r))
        return v
