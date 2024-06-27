#!/usr/bin/env python3

import argparse
import os
import random
import logging
import sys
sys.path.append(".")
import time

from typing import Dict
import numpy as np
import torch
from habitat import Env, logger
from arguments import get_args
from habitat.config.default import get_config
from habitat import make_dataset

from agents.vlnav_agent import Mapping_Agent
import cv2
from collections import defaultdict
import torch.multiprocessing as mp

from utils.shortest_path_follower import ShortestPathFollowerCompat
from utils.task import VLObjectNavEpisode




def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def VLNav_env(args, config, rank, dataset, receive_queue):
    args.rank = rank

    random.seed(config.SEED+rank)
    np.random.seed(config.SEED+rank)
    torch.manual_seed(config.SEED+rank)
    torch.set_grad_enabled(False)

    env = Env(config, dataset)

    num_episodes = len(env.episodes)
    print("num_episodes: ", num_episodes)
    receive_queue.put(num_episodes)
    
    
    follower = ShortestPathFollowerCompat(
        env._sim, 0.3, False
    )
    
    agent = Mapping_Agent(args, follower)

    count_episodes = 0
    # for count_episodes in trange(num_episodes):
    
    while count_episodes < num_episodes:
        obs = env.reset()
        agent.reset()
        text_queries = obs["instruction"]['text']
        

        # image = transform_rgb_bgr(obs["rgb"])  # 224*224*3
        # image_rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_BGR2RGB) 
        # cv2.imshow("RGB0", obs["rgb"])

        count_steps = 0
        start_ep = time.time()
        while not env.episode_over:

            dd_s_time = time.time()

            if count_episodes < 1:
                action = 0
            else:
                agent_state = env.sim.get_agent_state()
                action = agent.mapping(obs, agent_state)
                
            if action == None:
                continue
            obs = env.step(action)

            count_steps += 1
            
            dd_e_time = time.time()
            # print(' time:%.3fs\n'%(dd_e_time - dd_s_time)) 

        infos = {}
        infos['failed'] = 0
        if (
            action == 0 and 
            env.get_metrics()["spl"]
        ):
            # print("you successfully navigated to destination point")
            infos['failed'] = 1 #success
        else:
            # print("your navigation was not successful")
            if count_steps >= config.ENVIRONMENT.MAX_EPISODE_STEPS - 1:
                infos['failed'] = 2 # exploration
            elif agent.replan_count > 20:
                infos['failed'] = 3 # collision
            else:
                infos['failed'] = 4 # detection

        count_episodes += 1
        
        print(env.current_episode.scene_id)
        
        infos['count_steps'] = count_steps
        # infos['total_usage'] = sum(agent.total_usage)

        metrics = env.get_metrics()
        
        receive_queue.put([metrics, infos])

    return


def main():
    
    args = get_args()
    
    args.exp_name = "vlobjectnav-"+ args.vln_mode
    
    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=log_dir + "eval.log",
        level=logging.INFO)


    mp_ctx = mp.get_context("forkserver")
    receive_queue = mp_ctx.Queue() 
    
    config_env = get_config(config_paths=["configs/" + args.task_config])
    config_env.defrost()
    config_env.DATASET.SPLIT = args.split
    config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    config_env.freeze()

    scenes = config_env.DATASET.CONTENT_SCENES
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    if "*" in config_env.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config_env.DATASET)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    print("Scenes per thread:")
    num_episode = []
    processes = []
    for i in range(args.num_processes):
        proc_config = config_env.clone()
        proc_config.defrost()
        proc_config.DATASET.SPLIT = args.split

        if len(scenes) > 0:
            proc_config.DATASET.CONTENT_SCENES = scenes[
                sum(scene_split_sizes[:i]):
                sum(scene_split_sizes[:i + 1])
            ]
            print("Thread {}: {}".format(i, proc_config.DATASET.CONTENT_SCENES))

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
        
        proc = mp_ctx.Process(target=VLNav_env, args=(args, proc_config, i, dataset, receive_queue))
        processes.append(proc)
        proc.start()
        
        num_episode.append(receive_queue.get())


    num_episodes = sum(num_episode)
    print("received num_episodes: ", num_episodes)
    logging.info(num_episodes)
    count_episodes = 0
    agg_metrics: Dict = defaultdict(float)
    total_fail = []
    total_usage = []
    total_steps = 0
    start = time.time()
    while count_episodes < num_episodes:
        
        if not receive_queue.empty():
            print("received")
            count_episodes += 1
            metrics, infos = receive_queue.get()
            
            total_steps += infos['count_steps']
            
            for m, v in metrics.items():
                agg_metrics[m] += v
                
            if infos['failed'] > 0:
                total_fail.append(infos['failed']) 
                
            # if infos['total_usage'] > 0:
            #     total_usage.append(infos['total_usage'])

            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
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

            # log += " Total usage: " + str(sum(total_usage)) + ", average usage: " + str(np.mean(total_usage))
            
            print(log)
            logging.info(log)
            

if __name__ == "__main__":
    main()
    
    



    