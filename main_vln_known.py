#!/usr/bin/env python3

import argparse
import os
import random
import logging
import glob
import math
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
import re
import sys
sys.path.append(".")
import time

import open_clip
from constants import category_to_id
from habitat import Env, logger
from arguments import get_args
from habitat.config.default import get_config
from visualize_results import ReconstructionWindow
from habitat.utils.visualizations import maps
from habitat import make_dataset

import ast
from utils.chat_utils import chat_with_gpt
from utils.equ_ranking import Equilibrium_Ranking
from agents.system_prompt import Instruction_system_prompt
from utils.task import VLObjectNavEpisode
from utils.model_utils import compute_clip_features
from utils.slam_classes import DetectionList, MapObjectList
from utils.compute_similarities import (
    color_by_clip_sim,
    cal_clip_sim
)


def Bbox_prompt(candidate_objects, candidate_landmarks):
    # Target object        
    # target_bbox = [{"centroid": [round(ele, 1) for ele in obj['bbox'].get_center()], 
    #                 "extent": [round(ele, 1) for ele in obj['bbox'].get_extent()]} 
    #                 for obj in candidate_objects ]
    
    target_bbox = [{"centroid": [round(obj['bbox'].get_center()[0], 1), 
                            round(obj['bbox'].get_center()[2], 1), 
                            round(obj['bbox'].get_center()[1], 1)], 
            "extent": [round(obj['bbox'].extent[0], 1), 
                        round(obj['bbox'].extent[2], 1), 
                        round(obj['bbox'].extent[1], 1)]} 
            for obj in candidate_objects]


    # Landmark objects
    landmark_bbox = {}
    for k,candidate_landmark in candidate_landmarks.items():
        landmark_bbox[k] = [{"centroid": [round(ldmk['bbox'].get_center()[0], 1), 
                                round(ldmk['bbox'].get_center()[2], 1), 
                                round(ldmk['bbox'].get_center()[1], 1)],  
                            "extent": [round(ldmk['bbox'].extent[0], 1), 
                                round(ldmk['bbox'].extent[2], 1), 
                                round(ldmk['bbox'].extent[1], 1)]}  
                            for ldmk in candidate_landmark]
        
    # inference the target if all the objects are found
    evaluation = {}
    if len(target_bbox) > 0 and all(landmark_bbox[key] for key in landmark_bbox.keys()) > 0:
        evaluation = {
            "Target Candidate BBox ('centroid': [cx, cy, cz], 'extent': [dx, dy, dz])": {
                str(i): bbox for i, bbox in enumerate(target_bbox)
            },
            "Target Candidate BBox Volume (meter^3)": {
                str(i): round(bbox["extent"][0] * bbox["extent"][1] * bbox["extent"][2], 3)
                for i, bbox in enumerate(target_bbox)
            },
        }

        evaluation["Targe Candidate to nearest Landmark Distance (meter)"] = {
            str(i): { key: min([round(
                math.sqrt(
                    (bbox["centroid"][0] - landmark["centroid"][0]) ** 2
                    # + (bbox["centroid"][1] - landmark_location_centroid[1]) ** 2
                    + (bbox["centroid"][2] - landmark["centroid"][2]) ** 2
                ),
                3,
            ) 
                for landmark in landmarks])
                for key, landmarks in landmark_bbox.items()}
            for i, bbox in enumerate(target_bbox)
        }
        evaluation["Landmark Location: (cx, cy, cz)"] = {
                phrase: {str(i): [round(ele, 1) for ele in landmark["centroid"]]
                for i, landmark in enumerate(landmarks)}
                for phrase, landmarks in landmark_bbox.items()
            },
        
        
        print(evaluation)
        return str(evaluation)
    else:
        return None
        

# def generate_point_cloud(window):
def main(args, send_queue, receive_queue):
    args.exp_name = "vlobjectnav-"+ args.vln_mode

    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=log_dir + "eval.log",
        level=logging.INFO)

    args.task_config = "vlobjectnav_hm3d.yaml"
    config = get_config(config_paths=["configs/"+ args.task_config])

    logging.info(args)
    # logging.info(config)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.set_grad_enabled(False)

    config.defrost()
    config.DATASET.SPLIT = args.split
    # config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    config.freeze()

    # env = Env(config=config)

    # num_episodes = len(env.episodes)
    # print(num_episodes)
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
    num_episodes = len(dataset.episodes)
    print(num_episodes)

    count_episodes = 0
    count_success = 0

    # Initialize the CLIP model
    device = "cuda:{}".format(args.gpu_id)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(device).half()
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    

    for episode in dataset.episodes:
        pattern = re.compile(r'-(.*?)/')
        match = pattern.search(episode.scene_id)
        scene_key = match.group(1)
        scene_path = "./saved_pcd1/" + scene_key + "/arrays.npz"
        if not os.path.exists(scene_path):
            count_episodes += 1
            continue
        data = np.load(scene_path, allow_pickle=True)
        objects = MapObjectList(device=device)
        objects.load_serializable(data['objects'])
        point_sum_points = data['full_points']
        point_sum_colors = data['full_colors']
        init_agent_position = data['init_agent_position']
        init_sim_rotation = data['init_sim_rotation']
        print(init_agent_position)
        
        points_2d = point_sum_points[:, [0, 2]]
        x_min, y_min = np.min(points_2d, axis=0)
        x_max, y_max = np.max(points_2d, axis=0)

        scale = 800  # Size of the image in pixels
        points_2d_normalized = (points_2d - [x_min, y_min]) / ([x_max - x_min, y_max - y_min])
        points_2d_scaled = (points_2d_normalized * scale).astype(np.int32)
        

        # Create an empty black image
        image = np.zeros((scale, scale, 3), dtype=np.uint8)

        # Convert colors from [0, 1] to [0, 255] and to BGR format for OpenCV
        colors_scaled = (point_sum_colors * 255).astype(np.uint8)
        colors_bgr = colors_scaled[:, [2, 1, 0]]  # RGB to BGR

        # Draw each point with color
        for (x, z), color in zip(points_2d_scaled, colors_bgr):
            cv2.circle(image, (x, z), 1, color.tolist(), -1)  # Draw colored points
      

        cv2.imshow('Top View Projection with Color', image)
        cv2.waitKey(0) 

        # print("Instrcution: ", obs["instruction"]['text'])
        # text_queries = obs["instruction"]['text']
        # object_category = category_to_id[obs['objectgoal'][0]]
        print("Instrcution: ", episode.instruction_text)
        text_queries = episode.instruction_text
        object_category = episode.object_category
        print("object_category: ", object_category)
        all_objects = []
        landmark_data = []       
        
        goal_position = episode.goals[0].position
        
        if args.vln_mode == "clip":
            target_data = text_queries
        else:
            if args.vln_mode == "llm":

                chat_history_for_llm = []
                
                chat_history_for_llm.append({"role": "system", "content": Instruction_system_prompt})
                
                chat_history_for_llm.append({"role": "user", "content": text_queries})
                response_message = chat_with_gpt(chat_history_for_llm, 2)
                chat_history_for_llm.append({"role": "assistant", "content": response_message})
                
                generated_text = ast.literal_eval(response_message)
                ground_json = generated_text["command"]["args"]["ground_json"]
                target_data = ground_json["target"]["phrase"]
                print("target_data: ", target_data)
                if "landmark" in ground_json:
                    landmark_data.append(ground_json["landmark"]["phrase"])
                    all_objects.append(ground_json["landmark"]["phrase"])
                    print("landmark_data: ", landmark_data)
                
            elif args.vln_mode == "llm_game":
                equ_ranking = Equilibrium_Ranking(text_queries, object_category)
                response_message = equ_ranking.response_message

                ground_json = ast.literal_eval(response_message)
                target_data = ground_json["target"]
                print("target_data: ", target_data)
                if "landmark" in ground_json:
                    for landmark in ground_json["landmark"]:
                        landmark_data.append(landmark)
                        all_objects.append(landmark)
                    print("landmark_data: ", landmark_data)
                
        all_objects.append(target_data)
        # for key, value in self.landmark_data.items():
        #     self.all_objects.append(value["phrase"])
            
        print("self.all_objects: ", all_objects)
        
        
        total_candidate_objects = []
        total_similarities = []
        candidate_target = []
        candidate_landmarks = {}
        candidate_id = []
        similarity_threshold = 0.27
        det_threshold = 0.8
        
      
        if len(objects) > 0:
            objects, target_similarities = color_by_clip_sim("looks like a " + target_data, 
                                                        objects, 
                                                        clip_model, 
                                                        clip_tokenizer)
            
            target_similarities = target_similarities.cpu().numpy()
            
            candidate_target = [objects[i] for i in range(len(objects)) 
                                if (target_similarities[i] > similarity_threshold \
                                    or \
                                    max(objects[i]['conf']) > det_threshold) ]
            
            candidate_id = [i for i in range(len(objects)) 
                            if (target_similarities[i] > similarity_threshold \
                                    or \
                                    max(objects[i]['conf']) > det_threshold) ]
            
            if len(candidate_target) == 0:
                candidate_id = [np.argmax(target_similarities)]
                
                candidate_target = [objects[np.argmax(target_similarities)] ]
                

            if len(candidate_target) > 0:
                print("find targets: " + target_data, len(candidate_target), " ", max(target_similarities))
     
            
            if len(landmark_data) == 0:
                candidate_id = [np.argmax(target_similarities)]
                candidate_objects = [objects[np.argmax(target_similarities)] ]
            else:
                for landmark in landmark_data:
                    objects, landmark_similarities = color_by_clip_sim("looks like a " + landmark, 
                                                                            objects, 
                                                                            clip_model, 
                                                                            clip_tokenizer, 
                                                                            color_set = False)
    
                    candidate_landmarks[landmark] = [objects[i] for i in range(len(objects)) 
                                    if (landmark_similarities[i] > similarity_threshold \
                                    or \
                                    max(objects[i]['conf']) > det_threshold) ]
                    candidate_id.extend([i for i in range(len(objects)) 
                                    if (landmark_similarities[i] > similarity_threshold \
                                    or \
                                    max(objects[i]['conf']) > det_threshold) ])
                    
                    if len(candidate_landmarks) == 0:
                        candidate_id = [np.argmax(landmark_similarities)]
                
                        candidate_landmarks = [objects[candidate_id[0]] ]
                        
                    # if len(candidate_landmarks[landmark]) > 0:
                    print("find ", landmark , ": ", len(candidate_landmarks[landmark]), " ", max(landmark_similarities.cpu().numpy()))
                    total_candidate_objects.extend(candidate_landmarks.values())
                    total_similarities.append(landmark_similarities)


                # found all objects after filting and candidate objects increase
                if len(candidate_target) > 0 and all(candidate_landmarks[key] for key in candidate_landmarks.keys()) > 0  :
                    user_prompt = Bbox_prompt(candidate_target, candidate_landmarks)

                    if user_prompt != None:
                        if args.vln_mode == "llm":
                            chat_history_for_llm.append({"role": "user", "content": user_prompt})
                            response_message = chat_with_gpt(chat_history_for_llm)
                            chat_history_for_llm.append({"role": "assistant", "content": response_message})
                            
                            ground_json = ast.literal_eval(response_message)
                            if ground_json["command"]["name"] == "finish_grounding":
                                candidate_objects = [candidate_target[int(ground_json["command"]["args"][
                                "top_1_object_id"
                            ])]]     
                                print("Found candidate_objects!")  
                                
                        elif args.vln_mode == "llm_game":
                            # candidate_index = self.equ_ranking.generator_search(user_prompt)
                            candidate_index = equ_ranking.equilibrium_search(user_prompt, len(candidate_target))
                            if candidate_index == len(candidate_target) : # -1: not sure
                                print("reject all candidates")
                            else:
                                candidate_objects = [candidate_target[candidate_index]]
                                print("Found candidate_objects!")  
                
        Open3d_goal_pose = candidate_objects[0]['pcd'].get_center()
        print("Open3d_goal_pose: ", Open3d_goal_pose)
        Rx = np.array([[0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0]])
        R_habitat2open3d = init_sim_rotation @ Rx.T
        habitat_goal_pose = np.dot(R_habitat2open3d, Open3d_goal_pose) + init_agent_position
        habitat_final_pose = habitat_goal_pose.astype(np.float32)

        euclid_dist = np.power(np.power(np.array(habitat_final_pose) - np.array(goal_position), 2).sum(0), 0.5)
        print(euclid_dist)
        Ground_goal_position = np.dot(R_habitat2open3d.T, (goal_position - init_agent_position).T).T
        print("Ground_goal_position: ", Ground_goal_position)
        print("goal_position: ", goal_position)
        
        if euclid_dist < 0.5:
            count_success += 1
        
        count_episodes += 1
        log = "Metrics: "
        log += "{:.0f}/{:.0f}/{:.0f}".format(count_success, count_episodes, num_episodes)    
        print(log)
        
        if args.visualize:
            receive_queue.put([objects.to_serializable(), 
                                np.asarray(point_sum_points), 
                                np.asarray(point_sum_colors), 
                                Ground_goal_position,
                                candidate_id])
            while send_queue.empty():
                time.sleep(1)
                


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