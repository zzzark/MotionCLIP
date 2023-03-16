# CMU, SMPL, CLIP
from copy import deepcopy
from fmbvh.motion_tensor.bvh_casting import *
from fmbvh.motion_tensor.motion_process import sample_frames
from fmbvh.motion_tensor.rotations import *
from fmbvh.bvh.editor import rectify_joint
import torch
from src.models.smpl import SMPL
from fmbvh.bvh.parser import BVH, JointMotion, JointOffset
from fmbvh.motion_tensor.bvh_casting import write_euler_to_bvh_object, get_positions_from_bvh
from fmbvh.bvh.editor import reorder_bvh
from collections import OrderedDict


# One instance for three motion types: CMU, SMPL, CLIP
class CSC:
    clip_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3',
                  'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow',
                  'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']
    clip_p_index = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

    smpl_names = [
        'Pelvis',
        'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot',
        'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot',
        'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
        'L_Collar', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
        'R_Collar', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'
    ]
    smpl_p_index = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]

    cmu_names = [
        'Hips',
        'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
        'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
        'Spine', 'Spine1', 'Neck1', 'Head',
        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
        'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'
    ]
    cmu_p_index = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 15, 10, 17, 18, 19]

    cmu_to_smpl = {
        'Hips': 'Pelvis',
        'LeftUpLeg': 'L_Hip', 'LeftLeg': 'L_Knee', 'LeftFoot': 'L_Ankle', 'LeftToeBase': 'L_Foot',
        'RightUpLeg': 'R_Hip', 'RightLeg': 'R_Knee', 'RightFoot': 'R_Ankle', 'RightToeBase': 'R_Foot',
        'Spine': 'Spine1', 'Spine1': 'Spine2', 'Neck1': 'Neck', 'Head': 'Head',
        'LeftShoulder': 'L_Collar', 'LeftArm': 'L_Shoulder', 'LeftForeArm': 'L_Elbow', 'LeftHand': 'L_Wrist',
        'RightShoulder': 'R_Collar', 'RightArm': 'R_Shoulder', 'RightForeArm': 'R_Elbow', 'RightHand': 'R_Wrist',
    }

    clip2smpl = []
    smpl2clip = []
    cmu2smpl_selected_cmu = []
    cmu2smpl = []

    @staticmethod
    def __static_init__():
        CSC.clip2smpl = [CSC.clip_names.index(name) for name in CSC.smpl_names]
        CSC.smpl2clip = [CSC.smpl_names.index(name) for name in CSC.clip_names]
        for cmu_name in CSC.cmu_names:
            if cmu_name in CSC.cmu_to_smpl:
                CSC.cmu2smpl.append(CSC.smpl_names.index(CSC.cmu_to_smpl[cmu_name]))
                CSC.cmu2smpl_selected_cmu.append(CSC.cmu_names.index(cmu_name))

    __t_smpl: BVH = None

    @staticmethod
    def gen_smpl_t_pose(smpl_model: SMPL, scale=10.0, place_on_floor=True, flip_skeleton=False) -> BVH:
        """
        smpl_model: SMPL().eval()
        scale: output scale
        place_on_floor: force to place the t-pose skeleton on floor
        flip_skeleton: flip the skeleton or not
                       [True]: if the root rotation is identity  (standard t-pose skeleton)
                       [False]: if the root rotation is rotated 180 degrees along the x-axis  (MOTION CLIP output)
        """
        if CSC.__t_smpl is not None:
            return deepcopy(CSC.__t_smpl)

        frame = 1
        ide = torch.eye(3)
        glb_rot = ide[None, ...].broadcast_to(frame, -1, -1)
        loc_rot = ide[None, None, ...].broadcast_to(frame, 23, -1, -1)
        betas = torch.zeros(frame, 10)
        output = smpl_model(body_pose=loc_rot, global_orient=glb_rot, betas=betas)
        # vertices = output['vertices']  # torch.Tensor [frame, 6890, 3]
        # faces = smpl_model.faces  # numpy.ndarray [13776, 3]
        pose = output['smpl']  # torch.Tensor [frame, 24, 3]

        # ----------------------------- #
        p = pose[0]  # batch 0
        p *= scale
        if flip_skeleton:
            p.neg_()  # fix upside-down

        # note: joints should be ordered along kinematic chain
        offset = OrderedDict()
        motion = OrderedDict()
        for ith, name in enumerate(CSC.clip_names):
            p_name = '' if CSC.clip_p_index[ith] == -1 else CSC.clip_names[CSC.clip_p_index[ith]]
            c_names = [CSC.clip_names[c_idx] for c_idx, p_idx in enumerate(CSC.clip_p_index) if ith == p_idx]
            off = (p[CSC.clip_p_index[ith]] - p[ith]).tolist() if ith != 0 else [0, 0, 0]  # convert from absolute to relative
            offset[name] = JointOffset(name, p_name, c_names, off,
                                       6 if p_name == '' else 3,
                                       'XYZ' + 'ZYX' if p_name == '' else 'ZYX')
            motion[name] = JointMotion(name, [])

        obj = BVH()
        obj.root_name = CSC.clip_names[0]
        obj.offset_data = offset
        obj.motion_data = motion
        obj.frames = 1
        obj.frame_time = 1 / 30.0

        trs = torch.zeros(1, 3, obj.frames)
        eul = torch.zeros(len(CSC.clip_names), 3, obj.frames)
        write_euler_to_bvh_object(trs, eul, obj)
        reorder_bvh(obj)  # make joints ordered

        if place_on_floor:
            pos = get_positions_from_bvh(obj, True)[..., 0]
            if flip_skeleton:
                y_min = torch.min(pos, dim=0)[0][1].item()
            else:
                y_min = -torch.max(pos, dim=0)[0][1].item()
            trs[:, 1, :] = -y_min
            write_euler_to_bvh_object(trs, eul, obj)

        CSC.__t_smpl = deepcopy(obj)
        return obj

    def __init__(self, obj=None):
        self.cmu: BVH = None
        self.smpl: BVH = None
        self.clip: torch.Tensor = None
        self.t_smpl = self.gen_smpl_t_pose(SMPL().eval())
        self.h_smpl = self.t_smpl.motion_data[self.t_smpl.root_name].data[0][1]

        if isinstance(obj, BVH) and obj.root_name == CSC.cmu_names[0]:
            self.from_cmu(obj)
        elif isinstance(obj, BVH) and obj.root_name == CSC.smpl_names[0]:
            self.from_smpl(obj)
        elif isinstance(obj, torch.Tensor):
            self.from_clip(obj)
        else:
            raise ValueError("Invalid argument `obj`")

    def clear(self):
        self.cmu: BVH = None
        self.smpl: BVH = None
        self.clip: torch.Tensor = None

    def from_cmu(self, cmu_obj: BVH, rectify=True, flip_root=True):
        self.clear()

        if rectify:
            rectify_joint(cmu_obj, 'LeftUpLeg', 'LeftLeg', [+0.05, -0.95, -0.02])
            rectify_joint(cmu_obj, 'RightUpLeg', 'RightLeg', [-0.05, -0.95, -0.02])

        _, bvh_qua = get_quaternion_from_bvh(cmu_obj)
        bvh_qua = sample_frames(bvh_qua, scale_factor=cmu_obj.frame_time / (1.0 / 30.0))
        cmu_mtx = quaternion_to_matrix(bvh_qua)
        smpl_mtx = torch.eye(3)[None, :, :, None].expand(len(CSC.smpl_names), 3, 3, cmu_mtx.shape[-1]).clone()
        smpl_mtx[CSC.cmu2smpl] = cmu_mtx[CSC.cmu2smpl_selected_cmu]
        if flip_root:
            smpl_mtx[0, 1].neg_()

        # --------- SMPL --------- #
        smpl_eul = matrix_to_euler(rotation_6d_to_matrix(matrix_to_rotation_6d(smpl_mtx)))
        smpl_trs = torch.zeros_like(smpl_eul[[0], :, :])
        smpl_trs[:, 1, :] = self.h_smpl
        self.smpl = deepcopy(self.t_smpl)
        self.smpl = write_euler_to_bvh_object(smpl_trs, smpl_eul, self.smpl,
                                              order='ZYX', to_deg=180.0/3.1415926535, frame_time=1/30.0)

        # --------- CLIP --------- #
        smpl_r6d = matrix_to_rotation_6d(smpl_mtx)
        clip_r6d = smpl_r6d[self.smpl2clip]
        self.clip = clip_r6d

    def from_smpl(self, smpl_obj: BVH):
        self.clear()

        self.smpl = smpl_obj
        _, smpl_qua = get_quaternion_from_bvh(self.smpl)
        smpl_mtx = quaternion_to_matrix(smpl_qua)
        smpl_r6d = matrix_to_rotation_6d(smpl_mtx)

        clip_r6d = smpl_r6d[CSC.smpl2clip]
        self.clip = clip_r6d

    def from_clip(self, clip_out: torch.Tensor, flip_root=False):
        """
        clip_out: [(B), J, C, T], from MOTION CLIP
        flip_root: Flip root rotation of MOTION CLIP output or not. Should be False if smpl origin t-pose is used.
        """
        self.clear()

        if len(clip_out.shape) == 4:
            if clip_out.shape[0] >= 2: print("[Warning] Do not support batch size >= 2.")
            clip_out = clip_out[0]
        clip_out = clip_out[self.clip2smpl, :, :]
        mtx = rotation_6d_to_matrix(clip_out)
        if flip_root:
            mtx[0, 1].neg_()  # joint 0, y-axis

        smpl_eul = matrix_to_euler(mtx)
        smpl_trs = torch.zeros_like(smpl_eul)[[0], ...]
        smpl_trs[:, 1, :] = self.h_smpl
        self.smpl = deepcopy(self.t_smpl)
        self.smpl = write_euler_to_bvh_object(smpl_trs, smpl_eul, self.smpl,
                                              order='ZYX', to_deg=180.0/3.1415926535, frame_time=1.0/30.0)
        self.clip = clip_out

    def to_cmu(self) -> BVH:
        raise NotImplementedError

    def to_smpl(self) -> BVH:
        if self.smpl is not None:
            return self.smpl
        else:
            raise Exception("Motion data not given!")

    def to_clip(self) -> torch.Tensor:
        if self.clip is not None:
            return self.clip
        else:
            raise Exception("Motion data not given!")


CSC.__static_init__()


def main():
    # -------- FIRST: MAKE SURE {CLIP -> SMPL} IS CORRECT -------- #
    output = torch.load("./output/tmp_save.pth")['output']
    for i in range(output.shape[0]):
        obj = CSC(output[i])
        obj.smpl.to_file(f"./output/out_{i}.bvh")

    # -------- SECOND: LET CMU -> SMPL, CLIP && TEST THEM -------- #
    obj = CSC(BVH("./assets/cmu21.bvh"))
    obj.smpl.to_file("./output/csc_cmu_to_smpl.bvh")
    torch.save(obj.clip, "./output/csc_cmu_to_clip.pth")

    # CMU -> {CLIP -> SMPL} # direct
    obj = CSC(torch.load("./output/csc_cmu_to_clip.pth"))
    obj.smpl.to_file("./output/cmu_to_clip_to_smpl.bvh")

    # CMU -> SMPL -> {CLIP -> SMPL}  # cycle
    obj = CSC(BVH("./output/csc_cmu_to_smpl.bvh"))
    torch.save(obj.clip, "./output/csc_cmu_to_smpl_to_clip.pth")
    obj = CSC(torch.load("./output/csc_cmu_to_smpl_to_clip.pth"))
    obj.smpl.to_file("./output/csc_cmu_to_smpl_to_clip_to_smpl.bvh")


if __name__ == '__main__':
    main()
