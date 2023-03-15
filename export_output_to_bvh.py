import torch
from src.models.smpl import SMPL
from fmbvh.bvh.parser import BVH, JointMotion, JointOffset
from fmbvh.motion_tensor.bvh_casting import write_euler_to_bvh_object, get_positions_from_bvh
from fmbvh.bvh.editor import reorder_bvh
from collections import OrderedDict
from export_smpl_t_pose_to_bvh import gen_smpl_t_pose
from copy import deepcopy
from fmbvh.motion_tensor.rotations import rotation_6d_to_matrix, matrix_to_euler


clip_topology = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3',
                 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
                 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']
clip_p_index = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


smpl_topology = [
    'Pelvis',
    'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot',
    'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot',
    'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
    'L_Collar', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
    'R_Collar', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'
]
smpl_p_index = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]


class Out2BVH:
    def __init__(self, flip_skeleton=False):  # get smpl origin t-pose skeleton
        self.smpl_model = SMPL().eval()
        self.t_pose = gen_smpl_t_pose(self.smpl_model, flip_skeleton=flip_skeleton, place_on_floor=True)
        self.height = self.t_pose.motion_data[self.t_pose.root_name].data[0][1]
        self.mapping = [clip_topology.index(dst) for dst, _ in self.t_pose.dfs()]

    def to_bvh(self, output: torch.Tensor, flip_root=False, place_on_floor=True):
        """
        output: [(B), J, C, T], from MOTION CLIP
        flip_root: Flip root rotation of MOTION CLIP output or not. Should be False if smpl origin t-pose is used.
        """
        if len(output.shape) == 4:
            if output.shape[0] >= 2: print("[Warning] Do not support batch size >= 2.")
            output = output[0]
        output = output[self.mapping, :, :]
        mtx = rotation_6d_to_matrix(output)
        if flip_root:
            mtx[0, 1].neg_()  # joint 0, y-axis
        eul = matrix_to_euler(mtx)
        trs = torch.zeros_like(eul)[[0], ...]
        trs[:, 1, :] = self.height
        obj = deepcopy(self.t_pose)
        write_euler_to_bvh_object(trs, eul, obj)
        return obj


def main():
    generation = torch.load("./output/tmp_save.pth")
    print(generation['output'].shape)  # [B, J, C, T]

    o2b = Out2BVH()
    obj = o2b.to_bvh(generation['output'])
    obj.to_file(f"./output/clip_to_smpl.bvh")


if __name__ == '__main__':
    main()
