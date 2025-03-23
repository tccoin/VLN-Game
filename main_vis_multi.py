#!/usr/bin/env python3

import argparse
import os
import sys
sys.path.append(".")
import time
import random
import logging
import re
import json

from typing import Dict
import numpy as np
import torch

import habitat
from habitat import Env, logger
from arguments import get_args
from habitat.config.default import get_config
from habitat import make_dataset

from agents.objnav_agent import ObjectNav_Agent
import cv2
from collections import defaultdict
from tqdm import tqdm, trange
import imageio

import threading
from multiprocessing import Process, Queue
import torch.multiprocessing as mp
from utils.task import PreciseTurn
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)

from utils.shortest_path_follower import ShortestPathFollowerCompat

from constants import category_to_id

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

def VLNav_env(args, config, rank, dataset, send_queue, receive_queue, gui_queue=None):

    args.rank = rank
    random.seed(config.SEED+rank)
    np.random.seed(config.SEED+rank)
    torch.manual_seed(config.SEED+rank)
    torch.set_grad_enabled(False)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.cuda.set_device(args.gpu_id)
    
    env = Env(config, dataset)
    num_episodes = len(env.episodes)
    print("total number of episodes: ", num_episodes)
    print([ep.episode_id for ep in env.episodes])
    
    
    follower = ShortestPathFollowerCompat(
        env._sim, 0.1, False
    )
    
    agent = ObjectNav_Agent(args, follower)

    count_episodes = 0
    # for count_episodes in trange(num_episodes):
    
    while count_episodes < num_episodes:
        try:
            obs = env.reset()

            scene_id = get_scene_id(env.current_episode.scene_id)
            episode_id = env.current_episode.episode_id
            episode_label = f'{scene_id}_{episode_id}'
            print("Running episode: ", episode_label)

            agent.reset(episode_label)

            video_save_path = '{}/{}/episodes_video/eps_{}_vis.mp4'.format(
                args.dump_location, args.exp_name, episode_label)
            frames = []

            # skip if video_save_path exists
            if os.path.exists(video_save_path):
                print(f"Skiping {episode_label}... Video already exists: {video_save_path}")
                continue
        
            count_steps = 0
            start_ep = time.time()
            while not env.episode_over:

                dd_s_time = time.time()

                agent_state = env.sim.get_agent_state()
                action = agent.act(obs, agent_state, send_queue, gui_queue)
                
                # cv2.imshow("Thread {}".format(rank), vis_image)
                # cv2.waitKey(1)

                # cv2.imshow("RGB0", annotated_image)
                    
                if action == None:
                    continue
                obs = env.step(action)

                count_steps += 1
                
                dd_e_time = time.time()
                # print(' time:%.3fs\n'%(dd_e_time - dd_s_time)) 

            infos = 0
            if (
                action == 0 and 
                env.get_metrics()["spl"]
            ):
                # print("you successfully navigated to destination point")
                infos = 1 #success
            else:
                # print("your navigation was not successful")
                if count_steps >= config.ENVIRONMENT.MAX_EPISODE_STEPS - 1:
                    infos = 2 # exploration
                elif agent.replan_count > 20:
                    infos = 3 # collision
                else:
                    infos = 4 # detection

            count_episodes += 1

            metrics = env.get_metrics()

            extra_info = {
                'episode_label': episode_label,
                'objectgoal': obs['objectgoal'][0],
                'upstair_flag': agent.upstair_flag,
                'downstair_flag': agent.downstair_flag,
            }

            if args.save_video:
                # imageio.mimsave(video_save_path, frames, fps=2)
                imageio.mimsave(video_save_path, agent.vis_frames, fps=2)
                print(f"Video saved to {video_save_path}")
            
            receive_queue.put([metrics, infos, count_steps, extra_info])
        except Exception as e:
            print("Error in episode {}: {}".format(count_episodes, e))

    return


def visualization_thread(args, send_queue, gui_queue):
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    app_win = ReconstructionWindow(args, mono, send_queue, gui_queue)
    app.run()

def main():
    
    args = get_args()
    
    args.exp_name = args.exp_name + "-" + args.detector
    
    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    video_save_dir = '{}/{}/episodes_video'.format(
                args.dump_location, args.exp_name)
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)

    logging.basicConfig(
        filename=log_dir + "eval.log",
        level=logging.INFO)
    
    
    logging.info(args)
    mp_ctx = mp.get_context("forkserver")
    send_queue = mp_ctx.Queue() # query from the gui
    receive_queue = mp_ctx.Queue() # query for gathering metrics
    gui_queue = mp_ctx.Queue() # data for visualization

    config_env = get_config(config_paths=["configs/" + args.task_config])

    config_env.defrost()
    config_env.DATASET.SPLIT = args.split
    config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    config_env.freeze()

    scenes = config_env.DATASET.CONTENT_SCENES
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    if "*" in config_env.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config_env.DATASET)
    
    n_gpu = torch.cuda.device_count()
    args.num_processes = args.num_processes * n_gpu
    print("Number of all threads: ", args.num_processes)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1


    args.num_processes = int(args.num_processes/n_gpu)
    print("Number of GPUs: ", n_gpu)
    print("Number of processes on each GPU: ", args.num_processes)
    print("Scenes per thread:")
    for gpu_id in range(n_gpu):
        for i in range(args.num_processes):
            n_scene = sum(scene_split_sizes[:i*n_gpu+gpu_id+1])-sum(scene_split_sizes[:i*n_gpu+gpu_id])
            print(f'gpu_id: {gpu_id}, process: {i}, n_scene: {n_scene}')

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

    n_episode_done_list = []
    num_episode_left_list = []
    processes = []

    if args.visualize:
        # Create a thread for the Open3D visualization
        print("start visualization thread")
        visualization = threading.Thread(target=visualization_thread, args=(args, send_queue, gui_queue))
        visualization.start()

    for gpu_id in range(n_gpu):
        for i in range(args.num_processes):
            proc_config = config_env.clone()
            proc_config.defrost()
            proc_config.DATASET.SPLIT = args.split

            args.gpu_id = gpu_id
            proc_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

            if len(scenes) > 0:
                proc_config.DATASET.CONTENT_SCENES = scenes[
                    sum(scene_split_sizes[:i*n_gpu+gpu_id]):
                    sum(scene_split_sizes[:i*n_gpu+gpu_id+1])
                ]
                print("GPU {},Thread {}: {}".format(gpu_id, i, proc_config.DATASET.CONTENT_SCENES))

            dataset = make_dataset(proc_config.DATASET.TYPE, config=proc_config.DATASET)
            proc_config.SIMULATOR.SCENE = dataset.episodes[0].scene_id
            proc_config.freeze()

            # thread = threading.Thread(
            #         target=VLNav_env,
            #         args=(
            #             args,
            #             proc_config,
            #             i,
            #             dataset,
            #             receive_queue,
            #         ),
            #     )
            
            # thread.start()
            if i==0 and gpu_id==0:
                # send the actual send_queue
                proc = mp_ctx.Process(target=VLNav_env, args=(args, proc_config, i, dataset, send_queue, receive_queue, gui_queue))
            else:
                # send a dummy send queue
                send_queue_dummy = mp_ctx.Queue()
                proc = mp_ctx.Process(target=VLNav_env, args=(args, proc_config, i, dataset, send_queue_dummy, receive_queue))
            processes.append(proc)

            start_success = False
            while not start_success:
                try:
                    proc.start()
                    start_success = True
                except Exception as e:
                    print(f"Error starting process {i} on GPU {gpu_id}: {e}, retrying...")
                    time.sleep(1)
                    

            print(f"Process {i} on GPU {gpu_id} started")
            
            # compare video number with all_info.log
            # remove video if not in all_info.log
            print(f"Check existing videos...")
            env = Env(proc_config, dataset)
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
            print(f"Number of episodes done: {n_episode_done}, Number of episodes left: {n_episode_left}")  

            n_episode_done_list.append(n_episode_done)
            num_episode_left_list.append(n_episode_left)
            env.close()

    print("All processes started!")
    sum_done = sum(n_episode_done_list)
    sum_left = sum(num_episode_left_list)
    num_episodes = sum_done + sum_left
    print(f"Total number of episodes: {num_episodes}")
    print(f"Number of episodes done: {sum_done}")
    print(f"Number of episodes left: {sum_left}")
    print(f"Check with all_info.log: {len(per_episode_error)}=={sum_done}")
    logging.info(num_episodes)
    count_episodes = 0
    agg_metrics: Dict = defaultdict(float)
    total_fail = []
    last_log = per_episode_error[-1]
    for i, result_status in enumerate(['success', 'exploration', 'collision', 'detection']):  
        for j in range(last_log[result_status]):
            total_fail.append(i+1)
    total_steps = 0
    start = time.time()
    while count_episodes < num_episodes:
        if not receive_queue.empty():
            print("received")
            count_episodes += 1
            metrics, infos, count_steps, extra_info = receive_queue.get()
            episode_label = extra_info['episode_label']

            total_steps += count_steps
            
            for m, v in metrics.items():
                agg_metrics[m] += v
                
            if infos > 0:
                total_fail.append(infos) 

            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Episode: {}".format(episode_label),
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(total_steps ),
                "FPS {},".format(int(total_steps  / (end - start)))
            ]) + '\n'
            
            log += "Failed Case: collision/exploration/detection/success/total:"
            log += " {:.0f}/{:.0f}/{:.0f}/{:.0f}({:.0f}),".format(
                total_fail.count(3),
                total_fail.count(2),
                total_fail.count(4),
                total_fail.count(1),
                len(total_fail)) + '\n'
            
            log += "Metrics: "
            log += ", ".join(k + ": {:.3f}".format(v / count_episodes) for k, v in agg_metrics.items()) + " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)

            print(log)
            logging.info(log)

            case_summary = {}
            case_summary["episode"] = episode_label
            case_summary["habitat_success"] = metrics["success"]
            case_summary['distance_to_goal'] = metrics['distance_to_goal']
            case_summary['spl'] = metrics['spl']
            case_summary['success'] = total_fail.count(1)
            case_summary['exploration'] = total_fail.count(2)
            case_summary['collision'] = total_fail.count(3)
            case_summary['detection'] = total_fail.count(4)
            case_summary["upstair_flag"] = extra_info['upstair_flag']
            case_summary["downstair_flag"] = extra_info['downstair_flag']
            case_summary["count_steps"] = count_steps
            case_summary["target"] = category_to_id[extra_info['objectgoal']]
            per_episode_error.append(case_summary)
            with open(log_dir + "all_info.log", 'w') as fp:
                for item in per_episode_error:
                    # write each item on a new line
                    fp.write(json.dumps(item) + "\n")
            

if __name__ == "__main__":
    main()
    
    



    