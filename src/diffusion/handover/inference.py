import torch
from data import unnormalize_data, normalize_data
from train_eval import create_model_scheduler, read_config
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional
import pdb

def infer_action(model, noise_scheduler, nimage, nstates, config, data_stats):
    device = 'cuda:0'
    with torch.no_grad():
        B = nimage.shape[0]

        # encoder vision features
        image_features = model['vision_encoder'](nimage.flatten(end_dim=1))
        image_features = image_features.reshape(*nimage.shape[:2],-1)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nstates], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        naction = torch.randn((B, config['pred_horizon'], config['action_dim']), device=device)

        for j in noise_scheduler.timesteps[:]:
            # predict the noise residual
            noise_pred = model['noise_pred_net'](sample=naction, timestep=j, global_cond=obs_cond)
            naction = noise_scheduler.step(sample=naction, timestep=j, model_output=noise_pred).prev_sample

        # unnormalize action
        naction = naction.detach().cpu().numpy()
        pred_action = unnormalize_data(pred_action, stats=data_stats['actions'])

        return pred_action

def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            #                         0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)


def run_inference():

    config_path = './config_inference.yaml'
    config = read_config(config_path)

    data_stats_path = "./data.yaml"
    data_stats = read_config(data_stats_path)
    
    device = 'cuda:0'


    model, noise_scheduler = create_model_scheduler(config, device)

    model.eval()

    if config['load_ckpt'] is not None:
        print(f"Loading model: {config['load_ckpt']}")
        state_dict = torch.load(config['load_ckpt'])

        model.load_state_dict(state_dict=state_dict)
    else:
        raise AssertionError("No model checkpoint loaded")

    image_queue = [] # Past 4 observation images
    states = []

    while(len(image_queue) > config['obs_horizon']):
        images = transform_images(image_queue)
        states = normalize_data(states, data_stats['states'])

        action = infer_action(model, noise_scheduler, images, states, config, data_stats)


if __name__ == "__main__":
    run_inference()
