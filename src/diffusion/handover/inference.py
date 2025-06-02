import time
import torch
from data import unnormalize_data, normalize_data
from train_eval import create_model_scheduler, read_config
import yaml
# def read_config(config_path):

#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
    
#     return config
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional
import pdb

import cv2
from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile

# RPC CLIENT
import xmlrpc.client

class RPCClient:
    def __init__(self, server_url="http://localhost:8000/RPC2"):
        self.server_url = server_url
        self.proxy = xmlrpc.client.ServerProxy(self.server_url, allow_none=True)
        
    def call(self, method_name, *args):
        """Call a remote method by name with provided arguments"""
        if not hasattr(self.proxy, method_name):
            raise AttributeError(f"Remote server has no method named '{method_name}'")
        
        method = getattr(self.proxy, method_name)
        try:
            result = method(*args)
            return result
        except xmlrpc.client.Fault as err:
            print(f"A fault occurred: {err.faultCode} {err.faultString}")
            raise
        except xmlrpc.client.ProtocolError as err:
            print(f"Protocol error: {err.url} {err.errcode} {err.errmsg}")
            raise
        except Exception as err:
            print(f"Unexpected error: {err}")
            raise
    
    def close(self):
        """Close the connection to the server"""
        if hasattr(self.proxy, '_ServerProxy__close'):
            self.proxy._ServerProxy__close()

def get_image_from_stream(pipeline):
    frames: FrameSet = pipeline.wait_for_frames(100)
    if frames is None:
        return None
    color_frame = frames.get_color_frame()
    if color_frame is None:
        return None
    color_image = color_frame.get_data()

    # import pdb
    # pdb.set_trace()
    color_image = color_image.reshape((1080, 1920, 3))
    color_image = cv2.resize(color_image, (432, 240))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    return color_image


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
        pred_action = unnormalize_data(naction, stats=data_stats['actions'])

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
        # pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_img = torch.unsqueeze(transf_img, 0)  # Add batch dimension
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)


def run_inference():

    config_path = './config_inference.yaml'
    config = read_config(config_path)

    data_stats_path = "./data.yaml"
    data_stats = read_config(data_stats_path)
    
    device = 'cuda:0'

    global robot, depth_data, digitone, digittwo, gripper_State
    robot = RPCClient("http://172.29.4.15:8000/RPC2")

    for i in range(10):
        try:

            camera_config = Config()
            camera_pipeline = Pipeline()

                
            try:
                profile_list = camera_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                try:
                    color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1920, 0, OBFormat.RGB, 15)
                except OBError as e:
                    print(e)
                    color_profile = profile_list.get_default_video_stream_profile()
                    print("color profile: ", color_profile)
                camera_config.enable_stream(color_profile)
            except Exception as e:
                print(e)
                return

            camera_pipeline.start(camera_config)
        
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            if i == 9:
                print("Camera Init failed, exiting...")
                import sys
                sys.exit()
            time.sleep(1)
            continue

    print("Camera stream started")

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

    while True:

        color_image = get_image_from_stream(camera_pipeline)
        if color_image is None:
            continue

        if len(image_queue) == 0:
            image_queue = [color_image for _ in range(config['obs_horizon'])]
        else:
            image_queue.pop(0)
            image_queue.append(color_image)


        joint_states = robot.call("get_joint_position")
        robot_cart_state = robot.call("get_cartesian_position")
        gripper_state = robot.call("get_gripper_width")
        if "error" in robot_cart_state.keys():
            print("Error in getting robot cartesian state:", robot_cart_state['error'])
            continue
        from scipy.spatial.transform import Rotation as R
        robot_cart_state['orientation'] = R.from_matrix(robot_cart_state['orientation']).as_euler('xyz', degrees=False)
        robot_state = [joint_states['joint_positions'], robot_cart_state['position'], robot_cart_state['orientation'], gripper_state['gripper_width']]

        robot_state = np.hstack(robot_state)

        if len(states) == 0:
            states = [robot_state[None] for _ in range(config['obs_horizon'])]
        else:
            states.pop(0)
            states.append(robot_state)
        
        states = np.concatenate(states)

        images = transform_images(image_queue, [240, 432])

        states = normalize_data(states, data_stats['states'])

        states = torch.from_numpy(states).unsqueeze(0).to(dtype=torch.float32, device=device)
        images = images.to(dtype=torch.float32, device=device) 

        action = infer_action(model, noise_scheduler, images, states, config, data_stats)

        translation = action[0, 1, :3]
        rotation = action[0, 1, 3:6]
        gripper_state = action[0, 1, 6] # Float

        print(robot.call("set_cartesian_position", {
            "position": translation,
            "orientation": rotation}))
        
        robot.call("set_gripper_width", gripper_state)


if __name__ == "__main__":
    run_inference()
