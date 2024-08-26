import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
from collections import defaultdict
import random
import argparse
import numpy as np
import torchvision
import skimage.transform
import torch

from habitat.utils.visualizations import maps

from habitat.config.default import get_config
from utils.task import VLObjectNavEpisode

from arguments import get_args

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
UP_KEY="q"
DOWN_KEY="e"
FINISH="f"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )
    
def main():
    args = get_args()

    # config=habitat.get_config("envs/habitat/configs/tasks/objectnav_gibson.yaml")
    config = get_config(config_paths=["configs/vlobjectnav_replica.yaml"])
    # config = get_config(config_paths=["configs/"+ args.task_config])
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.freeze()
    
    env = habitat.Env(
        config=config
    )
    env.seed(10000)
    
    num_episodes = len(env.episodes)
    print("num_episodes: ", num_episodes)

    agg_metrics: Dict = defaultdict(float)

    observations = env.reset()

    # num_episodes = 20
    count_episodes = 0

    while count_episodes < num_episodes:

        observations = env.reset()
       
        # print("Instrcution: "+ observations["instruction"]['text'])
        print(env.current_episode.goals[0].object_category)
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
        # top_down_map = draw_top_down_map(env.get_metrics(), observations["rgb"].shape[0])
        # cv2.imshow("top_down_map", top_down_map)
        print("Agent stepping around inside environment.")

        # print("Agent Position: ", env.get_sim_location())


        count_steps = 0
        while not env.episode_over:
            keystroke = cv2.waitKey(0)

            if keystroke == ord(FORWARD_KEY):
                action = HabitatSimActions.MOVE_FORWARD
                print("action: FORWARD")
            elif keystroke == ord(LEFT_KEY):
                action = HabitatSimActions.TURN_LEFT
                print("action: LEFT")
            elif keystroke == ord(RIGHT_KEY):
                action = HabitatSimActions.TURN_RIGHT
                print("action: RIGHT")
            elif keystroke == ord(FINISH):
                action = HabitatSimActions.STOP
                print("action: LOOK_UP")
            elif keystroke == ord(UP_KEY):
                action = HabitatSimActions.LOOK_UP
                print("action: LOOK_DOWN")
            elif keystroke == ord(DOWN_KEY):
                action = HabitatSimActions.LOOK_DOWN
                print("action: FINISH")
            else:
                print("INVALID KEY")
                continue

            observations = env.step(action)
            count_steps += 1

            cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
            top_down_map = draw_top_down_map(env.get_metrics(), observations["rgb"].shape[0])
            cv2.imshow("top_down_map", top_down_map)
            
            metrics = env.get_metrics()
            print(metrics["distance_to_goal"])
            

        print("Episode finished after {} steps.".format(count_steps))

        if (
            action == HabitatSimActions.STOP and
           
            env.get_metrics()["spl"]
        ): 
            print("you successfully navigated to destination point")
        else:
            print("your navigation was not successful")

        metrics = env.get_metrics()
        print(metrics["distance_to_goal"])


        # for m, v in metrics.items():
        #     agg_metrics[m] += v
        count_episodes += 1

    # avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    # for k, v in avg_metrics.items():
    #     print("{}: {:.3f}".format(k, v))





if __name__ == "__main__":
    main()