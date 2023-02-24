import torch
import src.utils.rotation_conversions as geometry

from .smpl import SMPL, JOINTSTYPE_ROOT
from src.models.tools.jointstypes import JOINTSTYPES


class Rotation2xyz:
    def __init__(self, device):
        self.device = device
        self.smpl_model = SMPL().eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, get_rotations_back=False, **kwargs):
        if pose_rep == "xyz":
            return x

        if mask is None:  # rm I guess this mask is used to toughly combine two motions, e.g. out = m1[~mask] + m2[mask]
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")

        if translation:                    # rm x.shape == [#sen, #joints, #6D, #frames]
            x_translations = x[:, -1, :3]  # rm last dimension is translation, 0:3 ==> pos; 3:6 ==> zero
            x_rotations = x[:, :-1]
        else:
            x_rotations = x

        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])  # rm to Input: [#S x #F, joints, 6]
        else:
            raise NotImplementedError("No geometry for this one.")

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0]  # rm [#S x #F, joints, 3, 3], take the first joint as global joint.
            rotations = rotations[:, 1:]  # rm And remain the lasts.

        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],  # rm Use default betas. [#S x #F, 10]
                                dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta
            # import ipdb; ipdb.set_trace()
        out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)
        # rm Out: vertices; vibe; a2m; smpl; a2mpl.
        # get the desirable joints
        joints = out[jointstype]  # rm Use out['smpl'], i.e. smpl position

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()  # rm [#S, #J, 3, #F]

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]

        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz
