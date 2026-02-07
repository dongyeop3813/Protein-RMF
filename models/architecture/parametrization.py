import torch
from data.so3_utils import calc_rot_vf, exp


class CondVelocityParametrization:
    def set_interpolant(self, interpolant):
        self.interpolant = interpolant

    def avg_vel(
        self,
        trans_t,
        rotmat_t,
        t,
        r,
        feats,
        return_model_output=False,
        trans_sc=None,
    ):
        trans_1, rotmat_1 = self(trans_t, rotmat_t, t, r, feats, trans_sc)
        #trans_1 = 2*trans_t
        #rotmat_1 = rotmat_t
        

        if self._model_conf.get("cond_vel_parameterization", True):
            trans_vf = (trans_1 - trans_t) / torch.clamp(1 - t, min=1e-6)[..., None]
            # trans_vf = self.interpolant.trans_cond_vf(t, trans_t, trans_1)
            rot_vf = self.interpolant.rots_cond_vf(t, rotmat_t, rotmat_1)
        else:
            trans_vf = trans_1 - trans_t
            rot_vf = calc_rot_vf(rotmat_t, rotmat_1)

        if return_model_output:
            return trans_vf, rot_vf, (trans_1, rotmat_1)
        else:
            return trans_vf, rot_vf

    def forward_flow(self, trans_t, rotmat_t, t, r, feats, trans_sc=None, trans_vf_add=None, rot_vf_add=None):
        trans_vf, rot_vf = self.avg_vel(
            trans_t, rotmat_t, t, r, feats, trans_sc=trans_sc
        )

        if trans_vf_add is not None:
            trans_vf = trans_vf + trans_vf_add
        if rot_vf_add is not None:
            rot_vf = rot_vf + rot_vf_add

        trans_r = trans_t + trans_vf * (r - t)[..., None]
        rotmats_r = exp(rotmat_t, (r - t)[..., None] * rot_vf)

        return trans_r, rotmats_r

    def inference_avg_vel(self, trans_t, rotmat_t, t, r, feats):
        trans_1, rotmat_1 = self(trans_t, rotmat_t, t, r, feats)
        trans_vf = (trans_1 - trans_t) / torch.clamp(1 - t, min=1e-6)[..., None]
        rot_vf = self.interpolant.rots_cond_exp_vf(t, rotmat_t, rotmat_1)
        return trans_vf, rot_vf

    def inference_forward_flow(self, trans_t, rotmat_t, t, r, feats, trans_vf_add=None, rot_vf_add=None):
        trans_vf, rot_vf = self.inference_avg_vel(trans_t, rotmat_t, t, r, feats)

        if trans_vf_add is not None:
            trans_vf = trans_vf + trans_vf_add
        if rot_vf_add is not None:
            rot_vf = rot_vf + rot_vf_add

        trans_r = trans_t + trans_vf * (r - t)[..., None]
        rotmats_r = exp(rotmat_t, (r - t)[..., None] * rot_vf)

        return trans_r, rotmats_r

    def jvp_avg_vel(self, trans_t, rotmat_t, t, r, feats, tangent):
        """
        tangent: (d_trans, d_rot, d_t, d_r)
        """

        def u(trans_t, rotmat_t, t, r):
            return self.avg_vel(trans_t, rotmat_t, t, r, feats)

        return torch.func.jvp(u, (trans_t, rotmat_t, t, r), tangent)


class VelocityParametrization:
    def set_interpolant(self, interpolant):
        self.interpolant = interpolant

    def avg_vel(
        self,
        trans_t,
        rotmat_t,
        t,
        r,
        feats,
        return_model_output=False,
        trans_sc=None,
    ):
        trans_vf, rot_vf = self(trans_t, rotmat_t, t, r, feats, trans_sc)
        return trans_vf, rot_vf

    def forward_flow(self, trans_t, rotmat_t, t, r, feats, trans_sc=None, trans_vf_add=None, rot_vf_add=None):
        trans_vf, rot_vf = self.avg_vel(
            trans_t, rotmat_t, t, r, feats, trans_sc=trans_sc
        )

        if trans_vf_add is not None:
            trans_vf = trans_vf + trans_vf_add
        if rot_vf_add is not None:
            rot_vf = rot_vf + rot_vf_add

        trans_r = trans_t + trans_vf * (r - t)[..., None]
        rotmats_r = exp(rotmat_t, (r - t)[..., None] * rot_vf)

        return trans_r, rotmats_r

    def inference_avg_vel(self, trans_t, rotmat_t, t, r, feats):
        trans_vf, rot_vf = self(trans_t, rotmat_t, t, r, feats)
        return trans_vf, rot_vf

    def inference_forward_flow(self, trans_t, rotmat_t, t, r, feats, trans_vf_add=None, rot_vf_add=None):
        trans_vf, rot_vf = self.inference_avg_vel(trans_t, rotmat_t, t, r, feats)

        if trans_vf_add is not None:
            trans_vf = trans_vf + trans_vf_add
        if rot_vf_add is not None:
            rot_vf = rot_vf + rot_vf_add

        trans_r = trans_t + trans_vf * (r - t)[..., None]
        rotmats_r = exp(rotmat_t, (r - t)[..., None] * rot_vf)

        return trans_r, rotmats_r

    def jvp_avg_vel(self, trans_t, rotmat_t, t, r, feats, tangent):
        """
        tangent: (d_trans, d_rot, d_t, d_r)
        """

        def u(trans_t, rotmat_t, t, r):
            return self.avg_vel(trans_t, rotmat_t, t, r, feats)

        return torch.func.jvp(u, (trans_t, rotmat_t, t, r), tangent)
