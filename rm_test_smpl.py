import torch
from src.models.smpl import SMPL
import trimesh


def test_smpl(device):
    smpl_model = SMPL().eval().to(device)
    frame = 2

    ide = torch.eye(3)
    glb_rot = ide[None, ...].broadcast_to(frame, -1, -1)
    loc_rot = ide[None, None, ...].broadcast_to(frame, 23, -1, -1)
    betas = torch.zeros(frame, 10)
    output = smpl_model(body_pose=loc_rot, global_orient=glb_rot, betas=betas)
    vertices = output['vertices']  # torch.Tensor [frame, 6890, 3]
    faces = smpl_model.faces  # numpy.ndarray [13776, 3]

    return vertices.cpu().numpy(), faces


def main():
    v, f = test_smpl(torch.device("cpu"))
    mesh = trimesh.Trimesh(vertices=v[0], faces=f, process=False)
    mesh.export("./output/out.obj")


if __name__ == '__main__':
    main()
