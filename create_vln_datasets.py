#!/usr/bin/env python3

import os
import glob
import random
import json
import pandas as pd
import numpy as np
import gzip
from arguments import get_args
from habitat.config.default import get_config
from utils.task import VLObjectNavEpisode

import habitat
from habitat import make_dataset
from collections import defaultdict
from tqdm import tqdm, trange

import sys
sys.path.append(".")


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]



def main():
    args = get_args()

    config = get_config(config_paths=["configs/"+ args.task_config])

    config.defrost()
    config.DATASET.SPLIT = args.split
    config.freeze()
    
    anno_path = 'data/anno.json'
    with open(anno_path, 'r') as file:
        annos = json.load(file)
    
        
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
    scenes = config.DATASET.CONTENT_SCENES
    if "*" in config.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.DATASET)
        
    for i in range(len(scenes)):
        proc_config = config.clone()
        proc_config.defrost()
        proc_config.DATASET.CONTENT_SCENES = [scenes[i]]
        dataset = make_dataset(proc_config.DATASET.TYPE, config=proc_config.DATASET)
        
        # get the mapping
        root_path = dataset.scene_ids[0].rsplit('/', 1)[0] 
        with open(root_path + "/obj_id2tgt_id.json", 'r') as file:
            obj_id2tar_id = json.load(file)
            
        # append the current scene object id
        curr_anno = {}
        for anno in annos:
            if scenes[i] in anno["item_id"]:
                curr_anno[anno["target_id"]] = anno["utterance"]
        
        
        new_dataset = make_dataset("VLObjectNav-v1")
        
        used_objects = []
        count_episode = 0
        for episode in dataset.episodes:
            if len(episode.goals) > 1:
                for goal in episode.goals:
                    object_id = episode.info['closest_goal_object_id']
                    if goal.object_id != object_id:
                        continue
                    target_id = obj_id2tar_id[str(object_id)]
                    if str(target_id) in curr_anno and object_id not in used_objects:
                        matching_utterances = curr_anno[str(target_id)]
                        print(matching_utterances)
                        
                        new_episode = VLObjectNavEpisode(
                            episode_id=str(count_episode),
                            goals=[goal],
                            scene_id=episode.scene_id,
                            scene_dataset_config=episode.scene_dataset_config,
                            start_position=episode.start_position,
                            start_rotation=episode.start_rotation,
                            instruction_text=matching_utterances,
                        )
                        new_dataset.episodes.append(new_episode)
                        
                        count_episode += 1
                        used_objects.append(object_id)
                        break # only pick one object for each episode
        
        # scene_key = os.path.basename(root_path)
        out_file = f"./data/datasets/vlobjectnav_hm3d_v2/val/content/{scenes[i]}.json.gz"
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with gzip.open(out_file, "wt") as f:
            f.write(new_dataset.to_json())
            
        print("load new scene: ", scenes[i])
                    
        
        
        
        
        
        
        
        
    
        results = dataset.to_json()
    
        print(results)



if __name__ == "__main__":
    main()
