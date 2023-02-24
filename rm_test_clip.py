import clip
import torch
import numpy as np
from PIL import Image


def load_clip(device, use_float=True):
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    if use_float:
        clip_model = clip_model.float()
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    class __CLIP:
        def __init__(self):
            self.device = device
            self.clip_model = clip_model
            self.clip_preprocess = clip_preprocess

        def __call__(self, *args, **kwargs):
            return self.clip_model(*args, **kwargs)

        def cosine_similarity(self, text_features, image_features):
            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logits_scale = self.clip_model.logit_scale.exp().item()
            logits_per_image = logits_scale * image_features @ text_features.t()
            return logits_per_image

        def encode_text(self, *texts):
            texts = np.array(texts, dtype=str)  # [N, $STR]
            text_tokens = clip.tokenize(texts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            return text_features

        def encode_image(self, *images):
            ls = []
            for im in images:
                im = self.clip_preprocess(im).unsqueeze(0).to(self.device)
                ls.append(im)
            images = torch.cat(ls, dim=0)  # [N, C, H, W]
            image_features = self.clip_model.encode_image(images)
            return image_features

    return __CLIP()


def main():
    device = torch.device("cuda:3")
    clip_model = load_clip(device, use_float=True)
    tx_feat = clip_model.encode_text("a football player", "swimming", "a fresh fish", "a dancer")
    im_feat = clip_model.encode_image(Image.open("./assets/soccer.jpg"),
                                      Image.open("./assets/fish.jpg"),
                                      Image.open("./assets/dance.jpg"))

    sim = clip_model.cosine_similarity(tx_feat, im_feat)
    print(sim.dtype)
    print(sim)
    print(sim / torch.max(sim, dim=1, keepdim=True)[0][None, :])


if __name__ == '__main__':
    main()
