import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.modeltype.motionclip import MOTIONCLIP
from src.visualize.visualize import viz_clip_text
from src.utils.misc import load_model_wo_clip
import src.utils.fixseed  # noqa
from src.mclip_params import mclip_params

plt.switch_backend('agg')


from src.datasets.get_dataset import get_datasets
from src.models.get_model import get_model as get_gen_model
import clip
from CSC import CSC
from fmbvh.bvh.parser import BVH


def get_mclip_model(parameters):
    # clip_model, preprocess = clip.load("ViT-B/32", device=device)  # Must set jit=False for training
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=parameters['device'], jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16

    for domain in parameters.get('clip_training', '').split('_'):
        clip_num_layers = parameters.get('clip_layers', 12)
        if domain == 'text':
            clip_model.initialize_parameters()
            clip_model.transformer.resblocks = clip_model.transformer.resblocks[:clip_num_layers]
        if domain == 'image':
            clip_model.initialize_parameters()
            clip_model.visual.transformer = clip_model.transformer.resblocks[:clip_num_layers]

    # NO Clip Training ,Freeze CLIP weights
    if parameters.get('clip_training', '') == '':
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

    model = get_gen_model(parameters, clip_model)
    return model


def encode_text(texts, model: MOTIONCLIP):
    """
    :param texts: list of string, np string array or clip tokens
    :param model:
    """
    if (isinstance(texts, list) or isinstance(texts, tuple)) and isinstance(texts[0], str):
        texts = np.array(texts, dtype=str)  # [N, $STR]
        text_tokens = clip.tokenize(texts).to(model.device)
    elif isinstance(texts, np.ndarray):
        text_tokens = clip.tokenize(texts).to(model.device)
    elif isinstance(texts, torch.Tensor):
        text_tokens = texts.to(model.device)
    else:
        raise TypeError(f"unknown type: {type(texts)}")
    clip_features = model.clip_model.encode_text(text_tokens)
    clip_features = clip_features.float()
    return clip_features


def encode_motion(motions, model: MOTIONCLIP):
    """
    :param motions: [(B), J, 6, T]
    :param model:
    :return [(B), C]
    """
    is_batch = True
    if len(motions.shape) == 3:
        motions = motions[None]  # to batch
        is_batch = False

    frames = motions.shape[-1]
    clip_features = model.encoder({
        'x': motions,
        'y': torch.zeros(motions.shape[0], dtype=int, device=motions.device),
        'mask': model.lengths_to_mask(torch.ones(motions.shape[0], dtype=int, device=motions.device) * frames)
    })["mu"]

    return clip_features if is_batch else clip_features[0]


def decode_motion(clip_features, frames, model: MOTIONCLIP):
    """
    :param clip_features: [(B), C]
    :param model:
    :return [(B), J, 6, T]
    """
    is_batch = True
    if len(clip_features.shape) == 1:
        clip_features = clip_features[None]  # to batch
        is_batch = False

    lengths = torch.ones((clip_features.shape[0],), dtype=int).to(clip_features.device) * frames
    mask = MOTIONCLIP.lengths_to_mask(lengths)

    output = model.decoder({
        "z": clip_features,
        "y": clip_features,
        "mask": mask,
        "lengths": lengths
    })['output']

    # we keep the raw output
    # output[:, 0] = torch.tensor([1, 0, 0, 0, -1, 0]).unsqueeze(0).unsqueeze(2)

    return output if is_batch else output[0]


def cosine_similarity(text_features, motion_features, model: MOTIONCLIP):
    # normalized features
    motion_features = motion_features / motion_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logits_scale = model.clip_model.logit_scale.exp().item()
    logits_per_motion = logits_scale * motion_features @ text_features.t()
    return logits_per_motion


def gen_from_text(device='cuda:0'):
    parameters = deepcopy(mclip_params)
    parameters['device'] = device

    # ---- load model ---- #
    model = get_mclip_model(parameters)
    state_dict = torch.load(parameters['checkpointname'], map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)
    model.eval()  # to disable dropout and more ...

    # ----
    with torch.no_grad():
        texts = ["Walking", "Jogging", "Running", "Jumping",
                 "Swimming", "Dribbling", "Dancing", "Boxing",
                 "Fishing", "Painting", "Sitting",
                 "Kill a man with an axe",
                 "Kick a door sadly",
                 "Raise up hands, jump"]
        clip_features = encode_text(texts, model)
        output = decode_motion(clip_features, 60, model)

    for i in range(output.shape[0]):
        obj = CSC(output[i])
        obj.smpl.to_file(f"./output/RM_OUTPUT__{texts[i]}.bvh")


def ae_from_cmu(device='cuda:0'):
    parameters = deepcopy(mclip_params)

    # ---- load model ---- #
    parameters['device'] = device
    model = get_mclip_model(parameters)
    state_dict = torch.load(parameters['checkpointname'], map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)
    model.eval()  # to disable dropout and more ...

    # ----
    with torch.no_grad():
        import glob
        motion_ls = []
        for cmu_file in glob.glob("./assets/cmu/*.bvh"):
            obj = CSC(BVH(cmu_file))
            if obj.clip.shape[-1] >= 60:
                motion_ls.append(obj.clip[:, :, :60])
            else:
                print(f'no enough frames: {cmu_file}')
        motions = torch.stack(motion_ls, dim=0).to(device)

        clip_features = encode_motion(motions, model)
        output = decode_motion(clip_features, 60, model)
        print(output.shape)

    for i in range(output.shape[0]):
        obj = CSC(output[i])
        obj.smpl.to_file(f"./output/RM_REC__{i}.bvh")
        obj = CSC(motion_ls[i])
        obj.smpl.to_file(f"./output/RM_ORG__{i}.bvh")


def calc_cos_sim(device="cuda:0"):
    parameters = deepcopy(mclip_params)

    # ---- load model ---- #
    parameters['device'] = device
    model = get_mclip_model(parameters)
    state_dict = torch.load(parameters['checkpointname'], map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)
    model.eval()  # to disable dropout and more ...

    # ---- load labels ---- #
    text_labels = [
        "angry walk",
        "childlike walk",
        "depressed walk",
        "neutral walk",
        "old walk",
        "proud walk",
        "sexy walk",
        "strutting walk",
    ]
    print(text_labels)

    # ----
    with torch.no_grad():
        import glob
        feature_ls = []
        for i, cmu_file in enumerate(glob.glob("./eval_data/xia/*.bvh")):
            obj = CSC(BVH(cmu_file))
            mo = obj.clip.to(device)
            feature_ls.append(encode_motion(mo, model))
            CSC(obj.clip).smpl.to_file(f"./output/__xia_{i}.bvh")

        motion_features = torch.stack(feature_ls, dim=0)
        text_features = encode_text(text_labels, model)
        sim = cosine_similarity(text_features, motion_features, model)
        sim = sim.cpu().numpy()
        sim_max = np.argmax(sim, axis=1)
        print(sim_max)
        print(np.array(text_labels)[sim_max])
        print()


def action_classify(data_dir='./eval_data/cmu_eval/unlabeled', device="cuda:0"):
    parameters = deepcopy(mclip_params)

    # ---- load model ---- #
    print(parameters['checkpointname'])
    parameters['device'] = device
    model = get_mclip_model(parameters)
    state_dict = torch.load(parameters['checkpointname'], map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)
    model.eval()  # to disable dropout and more ...

    # ---- load labels ---- #
    from src.utils.action_label_to_idx import action_label_to_idx
    action_text_labels = list(action_label_to_idx.keys())
    action_text_labels.sort(key=lambda x: action_label_to_idx[x])
    text_labels = action_text_labels
    print(text_labels)

    # ----
    with torch.no_grad():
        import glob
        motion_ls = []
        for i, cmu_file in enumerate(glob.glob(f"{data_dir}/*.bvh")):
            obj = CSC(BVH(cmu_file))
            assert obj.clip.shape[-1] == 60, f"not 60 frames: {obj.clip.shape[-1]}; {cmu_file}"
            motion_ls.append(obj.clip)

            if not os.path.isfile(f"./output/act_cls_tmp_{i}.bvh"):
                CSC(obj.clip).smpl.to_file(f"./output/act_cls_tmp_{i}.bvh")
        motions = torch.stack(motion_ls, dim=0).to(device)
        motion_features = encode_motion(motions, model)

        text_features = encode_text(text_labels, model)
        sim = cosine_similarity(text_features, motion_features, model)
        sim = sim.cpu().numpy()
        sim_max = np.argpartition(sim, -5, axis=1)[:, -5:]

        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(linewidth=np.inf)
        np_labels = np.array(text_labels)
        for i in range(sim_max.shape[0]):
            print(i, ':', np_labels[sim_max[i]])
