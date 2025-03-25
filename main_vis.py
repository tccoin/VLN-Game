#!/usr/bin/env python3

import argparse
import os
import random
import logging

import json
from typing import Dict
import numpy as np
import torch
import habitat
from habitat import Env, logger, make_dataset
from arguments import get_args
from habitat.config.default import get_config

from agents.objnav_agent import ObjectNav_Agent
import cv2
from collections import defaultdict
from tqdm import tqdm, trange
import imageio

import threading
from multiprocessing import Process, Queue

import sys
sys.path.append(".")
import time
import re

from constants import category_to_id

from utils.shortest_path_follower import ShortestPathFollowerCompat
from utils.task import PreciseTurn
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)

# Gui
import open3d.visualization.gui as gui

from utils.vis_gui import ReconstructionWindow


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def get_scene_id(scene_path):
    # Get the scene_id from the path
    scene_id = scene_path.split("/")[-1]
    scene_id = re.sub(r'\.basis\.glb$', '', scene_id)
    return scene_id

# def generate_point_cloud(window):
def main(args, send_queue, receive_queue):
    
    args.exp_name = args.exp_name + "-" + args.detector

    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)
    
    video_save_dir = '{}/{}/episodes_video'.format(
                args.dump_location, args.exp_name)
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=log_dir + "eval.log",
        level=logging.INFO)

    config = get_config(config_paths=["configs/"+ args.task_config])

    logging.info(args)
    # logging.info(config)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.set_grad_enabled(False)
 

    config.defrost()
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    config.freeze()

    # print(config)
    env = Env(config=config)

    follower = ShortestPathFollowerCompat(
        env._sim, 0.1, False
    )

    args.turn_angle = config.SIMULATOR.TURN_ANGLE
    agent = ObjectNav_Agent(args, follower)

    agg_metrics: Dict = defaultdict(float)

    num_episodes = len(env.episodes)
    if args.episode_count > -1:
        num_episodes = min(args.episode_count, len(env.episodes))
    print("num_episodes: ", num_episodes)

    per_episode_error = []
    # read all_info.log
    if os.path.exists(log_dir + "all_info.log"):
        with open(log_dir + "all_info.log", 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.strip()
                if line != "":
                    per_episode_error.append(json.loads(line))
    episode_done_list = [x["episode"] for x in per_episode_error]
    print("{} episode info from all_info.log is loaded".format(len(per_episode_error)))

    fail_case = {}
    fail_case['collision'] = 0
    fail_case['success'] = 0
    fail_case['detection'] = 0
    fail_case['exploration'] = 0
    last_log = per_episode_error[-1]
    for i, result_status in enumerate(['success', 'exploration', 'collision', 'detection']):  
        fail_case[result_status] = last_log[result_status]

    if "*" in config.DATASET.CONTENT_SCENES:
        dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
        scenes = dataset.get_scenes_to_load(config.DATASET)
    scenes = config.DATASET.CONTENT_SCENES
    print("Loaded scenes: ", scenes)
    count_episodes = 0
    # for count_episodes in trange(num_episodes):
    start = time.time()
    
    # per episode error log
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
    config.defrost()
    config.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config.freeze()

    print(f"Check existing videos...")
    n_episode_done = 0
    n_episode_left = 0
    for i in range(len(env.episodes)):
        scene_id = get_scene_id(env.episodes[i].scene_id)
        episode_id = env.episodes[i].episode_id
        episode_label = f'{scene_id}_{episode_id}'
        video_save_path = '{}/{}/episodes_video/eps_{}_vis.mp4'.format(args.dump_location, args.exp_name, episode_label)
        if os.path.exists(video_save_path):
            if episode_label in episode_done_list:
                print(f"Episode {episode_label} found in all_info.log.")
                n_episode_done += 1
            else:
                print(f"Episode {episode_label} missing in all_info.log. Video file removed.")
                os.remove(video_save_path)
                n_episode_left += 1
        else:
            print(f"Episode {episode_label} has no video file.")
            n_episode_left += 1
    sum_done = n_episode_done + n_episode_left
    print(f"Number of episodes done: {n_episode_done}, Number of episodes left: {n_episode_left}")  
    print(f"Total number of episodes: {sum_done}")
    print(f"Check with all_info.log: {len(per_episode_error)}=={sum_done}")


    while count_episodes < num_episodes:
        obs = env.reset()

        scene_id = get_scene_id(env.current_episode.scene_id)
        episode_id = env.current_episode.episode_id
        episode_label = f'{scene_id}_{episode_id}'
        print("Running episode: ", episode_label)
        
        agent.reset(episode_label)
        # print("Instrcution: "+ obs["instruction"]['text'])

        image = transform_rgb_bgr(obs["rgb"])  # 224*224*3
        image_rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_BGR2RGB) 
        # cv2.imshow("RGB0", obs["rgb"])

        video_save_path = '{}/{}/episodes_video/eps_{}_vis.mp4'.format(
            args.dump_location, args.exp_name, episode_label)
        frames = []

        # skip if video_save_path exists
        if os.path.exists(video_save_path):
            print(f"Skiping... Video already exists: {video_save_path}")
            count_episodes += 1
            continue

        count_steps = 0
        start_ep = time.time()
        while not env.episode_over:
            # dd_s_time = time.time()
            if count_episodes < args.skip_frames:
                action = 0 # NOTE multy: debug specific episode
            else:
                agent_state = env.sim.get_agent_state()
                
                if not args.keyboard_actor:
                    action = agent.act(obs, agent_state, send_queue, receive_queue)
                else:
                    action = agent.keyboard_actor(obs, agent_state, send_queue, receive_queue)

            if action == None:
                continue # NOTE multy: why skip action?
            obs = env.step(action)

            count_steps += 1
            
            # dd_e_time = time.time()
            # print(' time:%.3fs\n'%(dd_e_time - dd_s_time)) 


        if (
            action == 0 and 
            env.get_metrics()["spl"]
        ):
            # print("you successfully navigated to destination point")
            fail_case['success'] += 1
        else:
            # print("your navigation was not successful")
            if count_steps >= config.ENVIRONMENT.MAX_EPISODE_STEPS - 1:
                fail_case['exploration'] += 1
            elif agent.replan_count > 20:
                fail_case['collision'] += 1
            else:
                fail_case['detection'] += 1
                    
        count_episodes += 1

        end = time.time()
        time_elapsed = time.gmtime(end - start)
        log = " ".join([
            "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
            "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
            "num timesteps {},".format(count_steps),
            "FPS {},".format(int(count_steps / (end - start_ep)))
        ]) + '\n'

        log += "Failed Case: collision/exploration/detection/success/total:"
        log += " {:.0f}/{:.0f}/{:.0f}/{:.0f}({:.0f}),".format(
            np.sum(fail_case['collision']),
            np.sum(fail_case['exploration']),
            np.sum(fail_case['detection']),
            np.sum(fail_case['success']),
            count_episodes) + '\n'
        
        metrics = env.get_metrics()
        for m, v in metrics.items():
            if isinstance(v, dict):
                for sub_m, sub_v in v.items():
                    agg_metrics[m + "/" + str(sub_m)] += sub_v
            else:
                agg_metrics[m] += v
        case_summary = {}
        case_summary["episode"] = episode_label
        case_summary["habitat_success"] = env.get_metrics()["success"]
        case_summary['distance_to_goal'] = metrics['distance_to_goal']
        case_summary['spl'] = metrics['spl']
        case_summary.update(fail_case.copy())
        case_summary["upstair_flag"] = agent.upstair_flag
        case_summary["downstair_flag"] = agent.downstair_flag
        case_summary["count_steps"] = count_steps
        case_summary["target"] = category_to_id[obs['objectgoal'][0]]
        per_episode_error.append(case_summary)
        with open(log_dir + "all_info.log", 'w') as fp:
            for item in per_episode_error:
                # write each item on a new line
                fp.write(json.dumps(item) + "\n")

        log += "Metrics: "
        log += ", ".join(k + ": {:.3f}".format(v / count_episodes) for k, v in agg_metrics.items()) + " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)

        print(log)
        logging.info(log)

        if args.save_video:
            # imageio.mimsave(video_save_path, frames, fps=2)
            imageio.mimsave(video_save_path, agent.vis_frames, fps=2)
            print(f"Video saved to {video_save_path}")
     
        

    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    for stat_key in avg_metrics.keys():
        logger.info("{}: {:.3f}".format(stat_key, avg_metrics[stat_key]))

    return


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