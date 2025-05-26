#@markdown ### **Imports**
# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import cv2
import os
import wandb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data import HandoverDataset, unnormalize_data
from data import *
from model import ConditionalUnet1D, get_resnet, replace_bn_with_gn
import yaml
import pdb
import time
import re

import torch.nn.functional as F


dataset_path = '/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/'

def read_config(config_path):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def create_train_test_datasets(dataset_path, config):
    # parameters
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 1
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 4

    npzFiles = [os.path.join(dirs[0], file) for dirs in os.walk(dataset_path, topdown=True) for file in dirs[2] if file.endswith(".npz")]

    # sorted_paths = sorted(npzFiles, key=lambda s: int(re.search(r'_e(\d+)\.npz$', s).group(1)))
    # npzFiles = [sorted_paths[91], sorted_paths[91]]
    train_files, test_files = train_test_split(npzFiles, test_size=0.1)

    # create dataset from file
    train_dataset = HandoverDataset(
        file_paths=train_files,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon)

    test_dataset = HandoverDataset(
        file_paths=test_files,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon)

    # save training data statistics (min, max) for each dim
    stats = train_dataset.stats

    return train_dataset, test_dataset, stats

def create_model_scheduler(config, device):

    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet(config['resnet'])

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # ResNet18 has output dim of 512
    vision_feature_dim = config['vision_feature_dim']
    # States vector is 14 dimensional
    lowdim_obs_dim = config['lowdim_obs_dim']

    # observation feature has 512 + 14 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = config['action_dim']

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*config['obs_horizon']
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    num_diffusion_iters = config['num_diffusion_iters']
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # device transfer
    nets = nets.to(device)

    return nets, noise_scheduler


def action_reduce(unreduced_loss: torch.Tensor):
    # Reduce over non-batch dimensions to get loss per batch element
    while unreduced_loss.dim() > 1:
        unreduced_loss = unreduced_loss.mean(dim=-1)
    return unreduced_loss.mean()


def plot_predicted_actions_3d(model, noise_scheduler, nimage, 
                           gtaction, nstates, use_wandb, device, dataset_stats):
    model.eval()
    with torch.no_grad():
        B = gtaction.shape[0]

        # encoder vision features
        image_features = model['vision_encoder'](nimage.flatten(end_dim=1))
        image_features = image_features.reshape(*nimage.shape[:2],-1)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nstates], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        naction = torch.randn(gtaction.shape, device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)

        for j in noise_scheduler.timesteps[:]:
            # predict the noise residual
            noise_pred = model['noise_pred_net'](sample=naction, timestep=j, global_cond=obs_cond)
            naction = noise_scheduler.step(sample=naction, timestep=j, model_output=noise_pred).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        # naction = naction[0]
        predaction = naction
        gtaction = gtaction.detach().cpu().numpy()
        # gtaction = unnormalize_data(gtaction, stats=dataset_stats['actions'])

        for i in range(1):        
            fig = plt.figure()
            ax1 = fig.add_subplot(121) # For Image
            ax2 = fig.add_subplot(122, projection='3d')   

            ax1.set_axis_off()         

            ax1.set_title("Image")
            im = nimage[i, -1].permute(1, 2, 0).detach().cpu().numpy()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            ax1.imshow(im)

            # Plot GT and Predicted points in green and red
            ax2.plot(gtaction[i, :, 0], gtaction[i, :, 1], gtaction[i, :, 2], c='green')
            ax2.plot(predaction[i, :, 0], predaction[i, :, 1], predaction[i, :, 2], c='red')

            ax2.scatter(gtaction[i, :, 0], gtaction[i, :, 1], gtaction[i, :, 2], c='green', s=20)
            ax2.scatter(predaction[i, :, 0], predaction[i, :, 1], predaction[i, :, 2], c='red', s=20)
            
            # Draw lines between GT and predicted points to show error
            for pt_idx in range(4):
                ax2.plot([gtaction[i, pt_idx, 0], predaction[i, pt_idx, 0]],
                        [gtaction[i, pt_idx, 1], predaction[i, pt_idx, 1]],
                        [gtaction[i, pt_idx, 2], predaction[i, pt_idx, 2]], 'gray', linewidth=0.5)
            
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')

            if use_wandb:
                wandb.log({"Trajectory Plot": wandb.Image(fig)})
            plt.cla()
            plt.close(fig)


def plot_predicted_actions_2d(model, noise_scheduler, nimage, 
                           gtaction, nstates, use_wandb, device, dataset_stats):
    model.eval()
    with torch.no_grad():
        B = gtaction.shape[0]

        # encoder vision features
        image_features = model['vision_encoder'](nimage.flatten(end_dim=1))
        image_features = image_features.reshape(*nimage.shape[:2],-1)

        # concatenate vision feature and low-dim obs
        obs_features = torch.cat([image_features, nstates], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1)
        # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        naction = torch.randn(gtaction.shape, device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)

        for j in noise_scheduler.timesteps[:]:
            # predict the noise residual
            noise_pred = model['noise_pred_net'](sample=naction, timestep=j, global_cond=obs_cond)
            naction = noise_scheduler.step(sample=naction, timestep=j, model_output=noise_pred).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu')
        # (B, pred_horizon, action_dim)
        # naction = naction[0]
        predaction = naction
        gtaction = gtaction.detach().cpu()
        gtaction = unnormalize_data(gtaction, stats=dataset_stats['actions'])
        predaction = unnormalize_data(predaction, stats=dataset_stats['actions'])

        action_loss = action_reduce(F.mse_loss(predaction, gtaction, reduction="none"))
        action_waypts_cos_similarity = action_reduce(F.cosine_similarity(
            predaction[:, :, :3], gtaction[:, :, :3], dim=-1))
        
        action_angles_cos_similarity = action_reduce(F.cosine_similarity(
            predaction[:, :, 3:6], gtaction[:, :, 3:6], dim=-1))
        
        action_gripper_cos_similarity = action_reduce(F.cosine_similarity(
            predaction[:, :, 6], gtaction[:, :, 6], dim=-1))
        
        if use_wandb:
            wandb.log({"Waypoints Cosine sim": action_waypts_cos_similarity,
                       "Angles Cosine sim": action_angles_cos_similarity,
                       "Gripper state Cosine sim": action_gripper_cos_similarity,
                       "Action Loss": action_loss})
        
        for i in range(1):        
            fig = plt.figure(figsize=(12, 6))
            ax1 = fig.add_subplot(131) # For Image
            ax2 = fig.add_subplot(132) # For x, y
            ax3 = fig.add_subplot(133) # For height

            ax1.set_axis_off()         

            ax1.set_title("Image")
            im = nimage[i, -1].permute(1, 2, 0).detach().cpu().numpy()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            ax1.imshow(im)

            # Plot GT and Predicted points in green and red
            ax2.plot(-gtaction[i, :, 1], gtaction[i, :, 0], c='green', marker='o')
            ax2.plot(-predaction[i, :, 1], predaction[i, :, 0], c='red', marker='o')

            # ax2.scatter(-gtaction[i, :, 1], gtaction[i, :, 0], c='green', s=20)
            # ax2.scatter(-predaction[i, :, 1], predaction[i, :, 0], c='red', s=20)

            x_mean = torch.mean(gtaction[i, :, 0])
            y_mean = torch.mean(gtaction[i, :, 1])

            ax2.set_xlim(-y_mean - 0.2, -y_mean + 0.2)
            ax2.set_ylim(x_mean - 0.2, x_mean + 0.2)

            height_xrange = torch.arange(gtaction.shape[1])
            ax3.plot(height_xrange, gtaction[i, :, 2], c='green', marker='o')
            ax3.plot(height_xrange, predaction[i, :, 2], c='red', marker='o')

            ax3.set_xlabel("Points")
            ax3.set_ylabel("Height (Z)")

            # Draw lines between GT and predicted points to show error
            # for pt_idx in range(4):
            #     ax2.plot([gtaction[i, pt_idx, 0], predaction[i, pt_idx, 0]],
            #             [gtaction[i, pt_idx, 1], predaction[i, pt_idx, 1]], 'black', linewidth=0.5)
            
            ax2.set_xlabel('Y')
            ax2.set_ylabel('X')

            # ax2.set_xlim(-1, 1)
            # ax2.set_ylim(-1, 1)

            if use_wandb:
                wandb.log({"Trajectory Plot": wandb.Image(fig)}, commit=False)

            plt.cla()
            plt.close(fig)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(model, noise_scheduler):
    pass

def train_eval():
    config_path = './config.yaml'
    dataset_path = '/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/'

    config = read_config(config_path=config_path)
    train_dataset, test_dataset, dataset_stats = create_train_test_datasets(dataset_path=dataset_path, 
                                                                          config=config)
    
    # create dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], 
                                               num_workers=2, shuffle=True, pin_memory=True,
                                               persistent_workers=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['eval_batch_size'], 
                                              num_workers=1, shuffle=False, 
                                              pin_memory=True, persistent_workers=True)
    
    device = 'cuda:0'
    model, noise_scheduler = create_model_scheduler(config=config, device=device)
    use_wandb = config['use_wandb']
    num_epochs = config['num_epochs']
    obs_horizon = config['obs_horizon']
    

    if use_wandb:
        wandb.init(project=config['wandb_project_name'])

        config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
        wandb.run.name = config["run_name"]

    ema = EMAModel(model=model, power=0.75)

    optimizer = torch.optim.AdamW(params=model.parameters(), 
                                  lr=float(config['lr']), 
                                  weight_decay=float(config['weight_decay']))
    
    lr_scheduler = get_scheduler(name='cosine',
                                 optimizer=optimizer, 
                                 num_warmup_steps=500,
                                 num_training_steps=len(train_loader) * num_epochs)
    
    if config['load_ckpt'] is not None:
        print(f"Loading model: {config['load_ckpt']}")
        state_dict = torch.load(config['load_ckpt'])

        model.load_state_dict(state_dict=state_dict)
    
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = 0.0
            # batch loop
            with tqdm(train_loader, desc='Train batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch['image'][:,:obs_horizon].to(dtype=torch.float32, device=device)
                    naction = nbatch['actions'].to(dtype=torch.float32, device=device)
                    nstates = nbatch['states'].to(dtype=torch.float32, device=device)
                    B = naction.shape[0]

                    # encoder vision features
                    image_features = model['vision_encoder'](
                        nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(
                        *nimage.shape[:2],-1)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nstates], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = model['noise_pred_net'](
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if use_wandb:
                        wandb.log({'Train Batch Loss': loss, 'current_lr': get_lr(optimizer)})
                    
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(model)

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss += loss_cpu
                    tepoch.set_postfix(loss=loss_cpu)
                torch.save(ema.averaged_model.state_dict(), "latest.pth")
                if use_wandb:
                    wandb.log({"Train Epoch Loss": epoch_loss})
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            if epoch_idx % config['eval_freq'] == 0:
                with torch.no_grad():
                    model.eval()                
                    with tqdm(test_loader, desc='Test batch', leave=False) as tepoch:
                        for nbatch in tepoch:
                            # data normalized in dataset
                            # device transfer
                            nimage = nbatch['image'][:,:obs_horizon].to(dtype=torch.float32, device=device)
                            naction = nbatch['actions'].to(dtype=torch.float32, device=device)
                            nstates = nbatch['states'].to(dtype=torch.float32, device=device)

                            plot_predicted_actions_2d(model, noise_scheduler, nimage, naction, nstates,
                                                      use_wandb, device, dataset_stats)
                            B = naction.shape[0]
            
                            # encoder vision features
                            image_features = model['vision_encoder'](
                                nimage.flatten(end_dim=1))
                            image_features = image_features.reshape(
                                *nimage.shape[:2],-1)
                            # (B,obs_horizon,D)
            
                            # concatenate vision feature and low-dim obs
                            obs_features = torch.cat([image_features, nstates], dim=-1)
                            obs_cond = obs_features.flatten(start_dim=1)
                            # (B, obs_horizon * obs_dim)
            
                            # sample noise to add to actions
                            noise = torch.randn(naction.shape, device=device)
            
                            # sample a diffusion iteration for each data point
                            timesteps = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps,
                                (B,), device=device
                            ).long()
            
                            # add noise to the clean images according to the noise magnitude at each diffusion iteration
                            # (this is the forward diffusion process)
                            noisy_actions = noise_scheduler.add_noise(
                                naction, noise, timesteps)
            
                            # predict the noise residual
                            noise_pred = model['noise_pred_net'](
                                noisy_actions, timesteps, global_cond=obs_cond)
            
                            # L2 loss
                            eval_loss = nn.functional.mse_loss(noise_pred, noise)
                            if use_wandb:
                                wandb.log({"Eval Batch Loss": eval_loss})
            if epoch_idx % config['model_save_freq'] == 0:
                torch.save(model.state_dict(), f"model_epoch_{epoch_idx}")
    # Weights of the EMA model
    # is used for inference
    ema_model = model
    ema_model.copy_to(ema_model.parameters())

    torch.save(ema_model.state_dict(), "ema_model.pth") 

if __name__ == "__main__":
    train_eval()   

    

    

    

    


