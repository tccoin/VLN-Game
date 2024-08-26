#!/usr/bin/env python3

import argparse
import os
import random
import logging

from typing import Dict
import numpy as np
import torch
import cv2
from collections import defaultdict
from tqdm import tqdm, trange
import imageio
import threading
from multiprocessing import Process, Queue
# Gui
import open3d.visualization.gui as gui

import sys
sys.path.append(".")
import time

from habitat import Env, logger
from arguments import get_args
from habitat.config.default import get_config
from utils.shortest_path_follower import ShortestPathFollowerCompat
from utils.task import VLObjectNavEpisode
from utils.vis_gui import ReconstructionWindow
from agents.explored_agent import Explored_Agent
from habitat.utils.visualizations import maps
import quaternion

def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

# def generate_point_cloud(window):
def main(args, send_queue, receive_queue):
    args.exp_name = "vlobjectnav-"+ args.vln_mode

    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=log_dir + "eval.log",
        level=logging.INFO)

    args.task_config = "vlobjectnav_replica.yaml"
    config = get_config(config_paths=["configs/"+ args.task_config])

    logging.info(args)
    # logging.info(config)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.set_grad_enabled(False)

    config.defrost()
    # config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    config.freeze()

    env = Env(config=config)
    obs = env.reset()
    follower = ShortestPathFollowerCompat(
        env._sim, 0.3, False
    )

    agent = Explored_Agent(args, follower)
    # agent = Operate_Agent(args)


    num_episodes = len(env.episodes)
    
    scene_id = env._sim.curr_scene_name
    old_scene_id = '' 

    count_episodes = 0
    # for count_episodes in trange(num_episodes):
    start = time.time()
    
    while count_episodes < num_episodes:
        while scene_id == old_scene_id:
            obs = env.reset()
            scene_id = env._sim.curr_scene_name
            count_episodes += 1
                
        
        agent.reset()
        # print("Instrcution: ", obs["instruction"]['text'])

        image = transform_rgb_bgr(obs["rgb"])  # 224*224*3
        image_rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_BGR2RGB) 
        # cv2.imshow("RGB0", obs["rgb"])
        agent_state = env.sim.get_agent_state()
        init_agent_position = agent_state.position
        init_sim_rotation = quaternion.as_rotation_matrix(agent_state.sensor_states["depth"].rotation)
        
        count_steps = 0
        start_ep = time.time()
        while not env.episode_over:

            agent_state = env.sim.get_agent_state()
            action = agent.act(obs, agent_state, send_queue, receive_queue)


            if action == None:
                continue
            obs = env.step(action)
            

            count_steps += 1
            
        count_episodes += 1
          
        scene_path = "{}{}/".format(args.path_npz, scene_id)
        print('Saving model to {}...'.format(scene_path))
        if not os.path.exists(scene_path):
            os.makedirs(scene_path)
        np.savez(scene_path + 'arrays.npz', objects=agent.objects.to_serializable(), init_agent_position=init_agent_position, init_sim_rotation=init_sim_rotation)
        print('Finished.')

        old_scene_id = scene_id

def visualization_thread(send_queue, receive_queue):
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    app_win = ReconstructionWindow(args, mono, send_queue, receive_queue)
    app.run()


if __name__ == "__main__":
    args = get_args()

    send_queue = Queue()
    receive_queue = Queue()

    if args.visualize:
        # Create a thread for the Open3D visualization
        visualization = threading.Thread(target=visualization_thread, args=(send_queue, receive_queue,))
        visualization.start()

    # Run ROS code in the main thread
    main(args, send_queue, receive_queue)