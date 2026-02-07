import torch
from torch import nn
from models.utils import (
    get_index_embedding,
    get_time_embedding,
    get_positional_time_embedding,
)


class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb
        self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0], self.c_timestep_emb, max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, so3_t, r3_t, res_mask, diffuse_mask, pos):
        # s: [b]

        b, num_res, device = res_mask.shape[0], res_mask.shape[1], res_mask.device

        # [b, n_res, c_pos_emb]
        # pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask),
        ]
        return self.linear(torch.cat(input_feats, dim=-1))


class MeanFlowNodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(MeanFlowNodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.time_embedding = (
            get_positional_time_embedding
            if self._cfg.time_embedding_type == "positional"
            else get_time_embedding
        )

        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 3 + 1
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb
        self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = self.time_embedding(
            timesteps[:, 0], self.c_timestep_emb, max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, so3_t, r3_t, r, res_mask, diffuse_mask, pos):
        # s: [b]

        b, num_res, device = res_mask.shape[0], res_mask.shape[1], res_mask.device

        # [b, n_res, c_pos_emb]
        # pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask),
            self.embed_t(r - so3_t, res_mask),
        ]
        return self.linear(torch.cat(input_feats, dim=-1))


class MeanFlowNodeFeatureNetv2(nn.Module):
    # 20251103: This is the second version of the node feature net.
    # The difference is that now time gap embedding is passed through a linear layer to be more flexible.

    def __init__(self, module_cfg):
        super(MeanFlowNodeFeatureNetv2, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.time_embedding = (
            get_positional_time_embedding
            if self._cfg.time_embedding_type == "positional"
            else get_time_embedding
        )

        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb

        self.time_gap_layer = nn.Sequential(
            nn.Linear(self.c_timestep_emb, self.c_timestep_emb),
            nn.SiLU(),
            nn.Linear(self.c_timestep_emb, self.c_timestep_emb),
        )

        self.layers = nn.Sequential(
            nn.Linear(embed_size, self.c_s),
            nn.SiLU(),
            nn.Linear(self.c_s, self.c_s),
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = self.time_embedding(
            timesteps[:, 0], self.c_timestep_emb, max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
        self, so3_t, r3_t, r, res_mask, diffuse_mask, pos, return_time_emb=False
    ):
        # s: [b]
        time_gap_emb = self.time_gap_layer(self.embed_t(r - so3_t, res_mask))

        # [b, n_res, c_pos_emb]
        # pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        time_emb = self.embed_t(so3_t, res_mask)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            time_emb,
            time_gap_emb,
        ]
        out = self.layers(torch.cat(input_feats, dim=-1))
        if return_time_emb:
            return out, time_emb, time_gap_emb
        return out


class MeanFlowNodeFeatureNetv3(nn.Module):
    # 20251114: This is the third version of the node feature net.
    # This is used for fine-tuning of the pre-trained flow model to map.

    def __init__(self, module_cfg):
        super(MeanFlowNodeFeatureNetv3, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb
        self.linear = nn.Linear(embed_size, self.c_s)

        self.time_gap_layer = nn.Sequential(
            nn.Linear(self.c_pos_emb + self.c_timestep_emb, self.c_timestep_emb),
            nn.SiLU(),
            nn.Linear(self.c_timestep_emb, self.c_timestep_emb),
            nn.SiLU(),
            nn.Linear(self.c_timestep_emb, embed_size),
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0], self.c_timestep_emb, max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, so3_t, r3_t, r, res_mask, diffuse_mask, pos):
        # s: [b]

        b, num_res, device = res_mask.shape[0], res_mask.shape[1], res_mask.device

        # [b, n_res, c_pos_emb]
        # pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask),
        ]

        time_gap_emb = self.time_gap_layer(
            torch.cat([pos_emb, self.embed_t(r - so3_t, res_mask)], dim=-1)
        )

        return self.linear(torch.cat(input_feats, dim=-1) + time_gap_emb)
