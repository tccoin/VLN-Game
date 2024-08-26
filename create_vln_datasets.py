#!/usr/bin/env python3

import os
import glob
import random
import json
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import gzip
from arguments import get_args
from habitat.config.default import get_config
from utils.task import VLObjectNavEpisode

import habitat
import habitat_sim
from habitat import make_dataset
from collections import defaultdict
from tqdm import tqdm, trange

import sys
sys.path.append(".")

ISLAND_RADIUS_LIMIT = 1.5

task_category = {
    "chair": 0,
    "sofa": 1,
    "tv": 2,
    "table": 3,
}

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.

    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(
    s, t, id, sim, near_dist, far_dist, geodesic_to_euclid_ratio
):
    # check height difference to assure s and  tar are from same floor
    if np.abs(s[1] - t[1]) > 0.5:
        return False, 0, 0
    
    euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)

    d_separation = sim.geodesic_distance(s, t)
    if d_separation == np.inf:
        return False, 0, 0
    if not near_dist <= d_separation <= far_dist:
        return False, 0, 0
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return False, 0, 0
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 0, 0
    
    return True, d_separation, euclid_dist



def main():
    args = get_args()

    # args.task_config = "vlobjectnav_replica.yaml"
    # config = get_config(config_paths=["configs/"+ args.task_config])
 
    scenes = glob.glob("./data/scene_datasets/replica_v1/*/habitat/mesh_semantic.ply")
    task_category_dataset_static = {}
    for scene in scenes:
        proc_config = habitat.get_config()
        proc_config.defrost()
        proc_config.SIMULATOR.SCENE = scene
        proc_config.SIMULATOR.AGENT_0.SENSORS = []
        proc_config.freeze()
        sim = habitat.sims.make_sim("Sim-v0", config=proc_config.SIMULATOR)
        # proc_config = habitat_sim.SimulatorConfiguration()
        # proc_config.scene_id = scene
        # agent_config = habitat_sim.AgentConfiguration()
        # sim = habitat_sim.simulator.Simulator(habitat_sim.Configuration(proc_config, [agent_config]))

        # print(scenes[i])
        
        semantic_scene = sim.semantic_annotations()
        if(len(semantic_scene.objects) == 0): 
            return

        # # new_dataset = make_dataset("VLObjectNav-v1")
        # dset = habitat.datasets.make_dataset("ObjectNav-v1")
        # # print(vars(dset))
        # # print(num_episodes_per_scene)
        # dset.goals_by_category = dict(
        #     generate_objectnav_goals_by_category(
        #         sim,
        #         task_category
        #     )
        # )
        # # print(len(dset.goals_by_category))

        # dset.episodes = list(
        #     generate_objectnav_episode(
        #         sim, task_category, num_episodes_per_scene, is_gen_shortest_path=True
        #     )
        # )
        
        # dset.category_to_task_category_id = generate_objectnav_task_category_id(sim, task_category)
        # # dset.category_to_task_category_id = task_category
        # dset.category_to_scene_annotation_category_id = dset.category_to_task_category_id

        
        # out_file = f"./data/datasets/objectnav_replica/v1/val/content/{scene}.json.gz"
        # os.makedirs(os.path.dirname(out_file), exist_ok=True)
        # with gzip.open(out_file, "wt") as f:
        #     f.write(dset.to_json())
        # count_scene=count_scene+1
        # print(count_scene)
        # print("episode finish!")
    
    
    
        ######################################################################
        # 统计数据集中的类别
        #####################################################################
        # for obj in semantic_scene.objects:
        #     if obj is not None and obj.category is not None:
        #         if obj.category.name() in task_category_dataset_static.keys():
        #             task_category_dataset_static[obj.category.name()] += 1
        #         else:
        #             task_category_dataset_static[obj.category.name()] = 1

        # #         print(
        # #             f"Object id:{obj.id}, category:{obj.category.name()}, index:{obj.category.index()}"
        # #             f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
        # #         )
        # sim.close()
        # print(task_category_dataset_static)
        # plt.tick_params(axis='x', labelsize=8)    # 设置x轴标签大小
        # plt.tick_params(axis='y', labelsize=4)    # 设置x轴标签大小
        # g = sorted(task_category_dataset_static.items(), key=lambda item:item[1])
        # # print(g)
        # # print(type(g[0][1]), g[0][1])
        # blist=[]
        # clist=[]
        # for key in g:
        #     blist.append(key[0])
        #     clist.append(key[1])
        # plt.barh(range(len(blist)), clist, tick_label=blist, align="center", color="c")
        # #添加图形属性
        # plt.ylabel('Category')
        # plt.xlabel('Number')
        # plt.title('Replica Dataset Category Statistics')
        # plt.grid()
        # plt.savefig("Replica.jpg", dpi=600)
        ######################################################################
        
        
         
        
        
        # used_objects = []
        # count_episode = 0
        # for category_list in dataset.goals_by_category.values():
        #     if len(category_list) > 1 and category_list[0].object_category in object_category_list:
        #         for goal in category_list:
        #             object_id = goal.object_id

        #             target_id = obj_id2tar_id[str(object_id)]
        #             if str(target_id) in curr_anno and object_id not in used_objects:
        #                 matching_utterances = curr_anno[str(target_id)]
        #                 print(matching_utterances)
                        
        #                 for retry in range(100):
        #                     source_position = sim.sample_navigable_point()
        #                     # source_position[1] = High

        #                     is_compatible, dist, euclid = is_compatible_episode(
        #                         source_position,
        #                         goal.position,
        #                         target_id,
        #                         sim,
        #                         near_dist=2,
        #                         far_dist=8,
        #                         geodesic_to_euclid_ratio=1.1,
        #                     )

        #                     if is_compatible:
        #                         break
                        
        #                 if is_compatible:
        #                     angle = np.random.uniform(0, 2 * np.pi)
        #                     source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        #                     new_episode = VLObjectNavEpisode(
        #                         episode_id=str(count_episode),
        #                         goals=[goal],
        #                         scene_id=scenes[i],
        #                         scene_dataset_config='./' + sim.active_dataset,
        #                         start_position=source_position,
        #                         start_rotation=source_rotation,
        #                         instruction_text=matching_utterances,
        #                     )
        #                     new_dataset.episodes.append(new_episode)
                            
        #                     count_episode += 1
        #                     used_objects.append(object_id)
        #                     break # only pick one object for each episode
            
        # # scene_key = os.path.basename(root_path)
        # out_file = f"./data/datasets/vlobjectnav_hm3d_v2/val1/content/{scene}.json.gz"
        # os.makedirs(os.path.dirname(out_file), exist_ok=True)
        # with gzip.open(out_file, "wt") as f:
        #     f.write(new_dataset.to_json())
            
        # print("load new scene: ", scene)
                    
     



if __name__ == "__main__":
    main()
