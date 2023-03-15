import torch
from src.models.smpl import SMPL
from fmbvh.bvh.parser import BVH, JointMotion, JointOffset
from fmbvh.motion_tensor.bvh_casting import write_euler_to_bvh_object, get_positions_from_bvh
from fmbvh.bvh.editor import reorder_bvh
from collections import OrderedDict


clip_topology = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3',
                 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
                 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']
clip_p_index = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def gen_smpl_t_pose(smpl_model: SMPL, scale=10.0, place_on_floor=True, flip_skeleton=False) -> BVH:
    """
    smpl_model: SMPL().eval()
    scale: output scale
    place_on_floor: force to place the t-pose skeleton on floor
    flip_skeleton: flip the skeleton or not
                   [True]: if the root rotation is identity  (standard t-pose skeleton)
                   [False]: if the root rotation is rotated 180 degrees along the x-axis  (MOTION CLIP output)
    """
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
    for ith, name in enumerate(clip_topology):
        p_name = '' if clip_p_index[ith] == -1 else clip_topology[clip_p_index[ith]]
        c_names = [clip_topology[c_idx] for c_idx, p_idx in enumerate(clip_p_index) if ith == p_idx]
        off = (p[clip_p_index[ith]] - p[ith]).tolist() if ith != 0 else [0, 0, 0]  # convert from absolute to relative
        offset[name] = JointOffset(name, p_name, c_names, off,
                                   6 if p_name == '' else 3,
                                   'XYZ' + 'ZYX' if p_name == '' else 'ZYX')
        motion[name] = JointMotion(name, [])

    obj = BVH()
    obj.root_name = clip_topology[0]
    obj.offset_data = offset
    obj.motion_data = motion
    obj.frames = 1
    obj.frame_time = 1 / 30.0

    trs = torch.zeros(1, 3, obj.frames)
    eul = torch.zeros(len(clip_topology), 3, obj.frames)
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

    return obj


def main():
    smpl_model = SMPL().eval()
    scale = 20.0
    obj = gen_smpl_t_pose(smpl_model, scale)
    obj.to_file("./output/smpl.bvh")
    print('smpl_topology: ', [dst for dst, _ in obj.dfs()])
    print('smpl_p_index: ', obj.dfs_parent())


if __name__ == '__main__':
    main()
