import torch
from fmbvh.bvh.parser import BVH
from fmbvh.motion_tensor.bvh_casting import *
from fmbvh.motion_tensor.motion_process import sample_frames
from fmbvh.motion_tensor.rotations import *
from fmbvh.bvh.editor import rectify_joint


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


cmu_to_smpl = {
    'Hips':            'Pelvis',
    'LeftUpLeg':       'L_Hip',
    'LeftLeg':         'L_Knee',
    'LeftFoot':        'L_Ankle',
    'LeftToeBase':     'L_Foot',
    'RightUpLeg':      'R_Hip',
    'RightLeg':        'R_Knee',
    'RightFoot':       'R_Ankle',
    'RightToeBase':    'R_Foot',
    'Spine':           'Spine1',
    'Spine1':          'Spine2',
    'Neck1':           'Neck',
    'Head':            'Head',
    'LeftShoulder':    'L_Collar',
    'LeftArm':         'L_Shoulder',
    'LeftForeArm':     'L_Elbow',
    'LeftHand':        'L_Wrist',
    'RightShoulder':   'R_Collar',
    'RightArm':        'R_Shoulder',
    'RightForeArm':    'R_Elbow',
    'RightHand':       'R_Wrist',
}
smpl_to_cmu = {v: k for k, v in cmu_to_smpl.items()}


class BVH2Inp:
    def __init__(self):
        pass

    @staticmethod
    def to_input(cmu_obj: BVH, rectify=True, flip_root=True):
        if rectify:
            rectify_joint(cmu_obj, 'LeftUpLeg', 'LeftLeg',   [+0.05, -0.95, -0.02])
            rectify_joint(cmu_obj, 'RightUpLeg', 'RightLeg', [-0.05, -0.95, -0.02])

        _, bvh_qua = get_quaternion_from_bvh(cmu_obj)
        bvh_qua = sample_frames(bvh_qua, scale_factor=cmu_obj.frame_time / (1.0 / 30.0))
        cmu_mtx = quaternion_to_matrix(bvh_qua)
        smpl_mtx = torch.eye(3)[None, :, :, None].expand(len(smpl_topology), 3, 3, cmu_mtx.shape[-1]).clone()

        cmu_topology = [name for name, _ in cmu_obj.dfs()]
        cmu_remain_index = []
        cmu2smpl_mapping = []
        for cmu_name in cmu_topology:
            if cmu_name in cmu_to_smpl:
                cmu2smpl_mapping.append(smpl_topology.index(cmu_to_smpl[cmu_name]))
                cmu_remain_index.append(cmu_topology.index(cmu_name))

        smpl_mtx[cmu2smpl_mapping] = cmu_mtx[cmu_remain_index]

        if flip_root:
            smpl_mtx[0, 1].neg_()
        smpl_r6d = matrix_to_rotation_6d(smpl_mtx)

        smpl2clip_mapping = [smpl_topology.index(name) for name in clip_topology]
        clip_r6d = smpl_r6d[smpl2clip_mapping]
        return clip_r6d


def main():
    from export_output_to_bvh import Out2BVH

    b2i = BVH2Inp()
    o2b = Out2BVH()

    cmu_obj = BVH("./assets/cmu21.bvh")

    input_ = b2i.to_input(cmu_obj)
    output = input_
    smpl_obj = o2b.to_bvh(output)
    smpl_obj.to_file(f"./output/cmu21_to_clip_to_smpl.bvh")

    t, e = get_euler_from_bvh(cmu_obj, 1.0)
    t.zero_()
    e.zero_()
    write_euler_to_bvh_object(t, e, cmu_obj, cmu_obj.rotation_order, 1.0)
    cmu_obj.to_file("./output/t_cmu.bvh")
    o2b.t_pose.to_file("./output/t_smpl.bvh")


if __name__ == '__main__':
    main()
