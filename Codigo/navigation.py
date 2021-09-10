# REACTIVE NAVIGATION - MAIN CODE
# Developed by: Luna Jimenez Fernandez
#
# This file contains the main structure of the reactive navigation system developed,
# where agents can be trained and evaluated
#
# This file contains code to do the following actions:
#   * Train a RL-capable agent during a number of episodes
#   * Evaluate the performance of a trained agent using a validation set
#   * Generate a video of the agent's performance
#
# The Reactive Navigation Agent implementation itself is divided among several classes:
#   * Main model (neural network, act logic): ReactiveNavigationModel (models/reactive_navigation)
#   * Experience replay logic: ExperienceReplay (models/experience_replay)
#   * Reward computing logic: ReactiveNavigationEnv (envs/reactive_navigation_env)
#   * DQL training implementation: ReactiveNavigationTrainer (trainers/reactive_navigation_env)
#
# In addition, other utils are made available:
#   * Log creating and handling: LogManager (utils/log_manager.py)
#   * Agent implementation compatible with benchmarking tools: ReactiveNavigationAgent
#     (agents/reactive_navigation_agent)
#   * Script to clean the GPU cache and launch profiling tools: prepare_launch.sh
#
# The proposed agent uses a PyTorch architecture. However, the files for a previous
# Keras based architecture are provided (with name DEPRECATED_*)
# These files are not updated and should not be used, but they are kept for documentation sake
#
# Structure of the file:
#   1 - Imports
#   2 - Program variables
#   3 - User-defined variables
#   4 - Main code
#       4A - Training
#       4B - Benchmark
#       4C - Video
#   5 - Init code
#       5A - Argument declaration
#       5B - Argument parsing
#       5C - Argument processing and program launch

###############
# 1 - IMPORTS #
###############

# General imports
import argparse
import os
import random
import sys
import textwrap
import numpy as np
import torch

# Habitat imports
import habitat
from habitat import Benchmark
from habitat_baselines.common.environments import NavRLEnv

from habitat import get_config as get_task_config
from habitat_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.utils.common import generate_video

# Reactive Navigation imports

# Even if these imports seem unused, it's necessary to pre-import them
# to ensure that the decorator (function used to register
# them in the baseline registry) is run
from trainers.reactive_navigation_trainer import ReactiveNavigationTrainer
from envs.reactive_navigation_env import ReactiveNavigationEnv
# from trainers.reactive_navigation_trainer_keras import ReactiveNavigationTrainerKeras

# Agent imports
from agents.reactive_navigation_agent import ReactiveNavigationAgent, InformedReactiveNavigationAgent
from habitat_baselines.agents.simple_agents import RandomAgent, RandomForwardAgent, GoalFollower
from habitat_baselines.agents.ppo_agents import PPOAgent

#########################
# 2 - PROGRAM VARIABLES #
#########################

# These variables refer mostly to static values, such as paths to files.
# They are declared here (instead of being hard-coded in the code) for ease of use
# and for easier modification of the code by third parties

# Config file paths - Training
# Each agent has its own specific training config file
# Note that a config file can be specified via argument, overloading this value
config_paths_training = {
    "ppo": "./configs/ppo_pointnav_train.yaml",
    "reactive": "./configs/standard/reactive_pointnav_train_contour.yaml"
}

# Benchmark config file
# All agents share the same config file to be used during the benchmarking process
config_path_benchmark = "./configs/benchmarks/benchmark_gibson.yaml"

# Video config file
# All agents share the same config file to be used during video generation
config_path_video = "./configs/video_config.yaml"

# Dataset paths
# If an extra dataset was to be added, the path can be specified as a new value
dataset_paths = {
    "matterport": "./data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz",
    "gibson": "./data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
}

##############################
# 3 - USER-DEFINED VARIABLES #
##############################

# These variables are defined via arguments when the script is launched
# The values written below are considered the DEFAULT values, and will be the values
# used for the script unless specified otherwise
# Note that most parameters for the agents are actually specified within their config files

# agent_type - Specifies the agent type to be used
# Possible agent types:
# BENCHMARKS:
#   - random            - A completely random agent, to be used as a benchmark
#   - random_forward    - An agent that randomly moves, with bias towards
#                         moving forward, to be used as a benchmark
#   - goal_follower     - An agent that always tries to move towards the goal,
#                         to be used as a benchmark
# GOLD STANDARD:
#   - ppo               - A more advanced, reinforcement learning based agent using Proximal Policy Optimization (ppo)
#                         provided by habitat-lab
# PROPOSED:
#   - reactive          - The developed agent, using reactive navigation and deep reinforcement learning
#                         Note that there are several variations of the proposed algorithm
#   - informed_reactive - A variation of the reactive navigation agent that uses additional information
#                         during evaluation (using the distance sensor to automatically stop if the goal is reached)
agent_type = "reactive"

# dataset - Specifies the dataset to be used
# Possible values:
#   - matterport
#   - gibson
dataset = "gibson"

# mode - Specifies the mode in which the simulator will be launched
# Possible values:
#   - training - Trains the agent via reinforcement learning with a training set
#   - benchmark - Evaluates the performance of the agent using the provided Habitat Lab benchmark tool
#   - video - Generates a video of the trained agent performance in an episode
# Note that benchmark agents cannot run in Training mode
mode = "training"

# These variables can be used by Reinforcement Learning agents

# weights - For trainable agents, specifies the pre-trained weights for the neural network
# If specified, the program will be launched into "play" mode instead (showcasing the performance)
# If not specified, randomly initialized weights will be used instead
weights = None

# config - If specified, uses a different config file (overriding any other choice) for the agent
# If not specified, uses the default config file for the agent
config = None


#######################
# 4 - MAIN CODE LOOPS #
#######################

def training_main(config_path, training_dataset):
    """
    Main method for agent training

    Trains the specified agent using the specified environment, configuration
    and dataset

    Note that both the agent trainer and the environment
    are actually specified in the config file.

    :param config_path: Path to the config file
    :type config_path: str
    :param training_dataset: Name of the dataset to use
    :type training_dataset: str
    """

    # Initial message
    print("Starting program in training mode...")

    # Instantiate the Config from the config file
    training_config = get_config(config_path)

    # Add the dataset info to the config file
    training_config.defrost()
    training_config.TASK_CONFIG.DATASET.DATA_PATH = dataset_paths[training_dataset]
    training_config.TASK_CONFIG.DATASET.SPLIT = "train"
    training_config.TASK_CONFIG.DATASET.NAME = training_dataset
    training_config.freeze()

    # Get the appropriate trainer from the config file and instantiate it using the Baseline Registry
    trainer_name = training_config.TRAINER_NAME
    trainer_init = baseline_registry.get_trainer(trainer_name)
    trainer = trainer_init(training_config)

    # Start the training process
    trainer.train()


def benchmark_main(config_path, pretrained_weights=None):
    """
    Main method for agent evaluation

    Evaluates the performance of the agent using the specified environment, configuration
    and dataset

    The following metrics are considered for evaluation:
        * Final distance to the goal
        * Whether the episode was a success or not
        * SPL (Success weighted by path length)
        * Soft SPL
        * Collision count

    :param config_path: Path to the config file
    :type config_path: str
    :param pretrained_weights: (OPTIONAL) Path to the pre-trained weights used by the agents
    :type pretrained_weights: str
    """

    # Initial message
    print("Starting program in benchmark mode...")

    # Instantiate the Config from the config file
    benchmark_config = get_task_config(config_path)

    # Add the path to the pre-trained weights (if applicable) to the config file
    if pretrained_weights:
        benchmark_config.defrost()
        benchmark_config.MODEL_PATH = pretrained_weights
        benchmark_config.freeze()

    # If the seed is specified, instantiate it
    if benchmark_config.SEED:
        random.seed(benchmark_config.SEED)
        np.random.seed(benchmark_config.SEED)
        torch.manual_seed(benchmark_config.SEED)

    # Instantiate the appropriate agent and the benchmark
    if agent_type == "random":
        # Random agent
        agent = RandomAgent(benchmark_config.TASK.SUCCESS_DISTANCE,
                            benchmark_config.TASK.GOAL_SENSOR_UUID)
    elif agent_type == "random_forward":
        # Random forward agent
        agent = RandomForwardAgent(benchmark_config.TASK.SUCCESS_DISTANCE,
                                   benchmark_config.TASK.GOAL_SENSOR_UUID)
    elif agent_type == "goal_follower":
        # Goal follower
        agent = GoalFollower(benchmark_config.TASK.SUCCESS_DISTANCE,
                             benchmark_config.TASK.GOAL_SENSOR_UUID)
    elif agent_type == "ppo":
        # PPO agent

        # Specify the necessary info in the config file
        benchmark_config.defrost()
        benchmark_config.INPUT_TYPE = "depth"
        benchmark_config.RESOLUTION = 256
        benchmark_config.PTH_GPU_ID = 0
        benchmark_config.HIDDEN_SIZE = 512
        benchmark_config.RANDOM_SEED = 0
        benchmark_config.freeze()

        agent = PPOAgent(benchmark_config)

    elif agent_type == "informed_reactive":
        # Informed reactive agent
        agent = InformedReactiveNavigationAgent(benchmark_config,
                                                benchmark_config.TASK.SUCCESS_DISTANCE,
                                                weights=pretrained_weights)
    else:
        # Reactive agent (default)
        agent = ReactiveNavigationAgent(benchmark_config,
                                        weights=pretrained_weights)

    # Note that the CONFIG PATH is passed to the benchmark
    # (instead of an already instantiated Config object)
    # Thus, all info must be directly contained within the file
    benchmark = Benchmark(config_path)

    # Evaluate the agent and print the metrics
    episode_count = 250
    metrics = benchmark.evaluate(agent, episode_count)

    print("= EVALUATION METRICS =\n")
    print("Mean metrics over {} episodes:".format(episode_count))
    print("\t* Mean distance to goal: {}".format(metrics["distance_to_goal"]))
    print("\t* Mean success rate: {}".format(metrics["success"]))
    print("\t* Mean SPL: {}".format(metrics["spl"]))
    print("\t* Mean soft SPL: {}".format(metrics["softspl"]))
    print("\t* Mean collision count: {}".format(metrics["collisions/count"]))


def video_main(config_path, video_dataset, pretrained_weights=None):
    """
    Method used for video generation

    Using the tools provided by Habitat, simulates an episode using the first
    episode of the dataset and generates a video of the agent performance from it

    :param config_path: Path to the config file
    :type config_path: str
    :param video_dataset: Name of the dataset to use
    :type video_dataset: str
    :param pretrained_weights: (OPTIONAL) Path to the pre-trained weights used by the agents
    :type pretrained_weights: str
    """

    # Initial message
    print("Starting program in video mode...")

    # Instantiate the Config from the config file
    video_config = get_config(config_path)

    # Add the dataset
    video_config.defrost()
    video_config.TASK_CONFIG.DATASET.DATA_PATH = dataset_paths[video_dataset]
    video_config.TASK_CONFIG.DATASET.SPLIT = "val"
    video_config.TASK_CONFIG.DATASET.NAME = video_dataset

    # Add extra metrics to the config file
    # (top down map and collisions)
    video_config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    video_config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")

    # Add the path to the pre-trained weights (if applicable) to the config file
    if pretrained_weights:
        video_config.MODEL_PATH = pretrained_weights

    video_config.freeze()

    # Instantiate the appropriate agent and the benchmark
    if agent_type == "random":
        # Random agent
        agent = RandomAgent(video_config.TASK_CONFIG.TASK.SUCCESS_DISTANCE,
                            video_config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID)
    elif agent_type == "random_forward":
        # Random forward agent
        agent = RandomForwardAgent(video_config.TASK_CONFIG.TASK.SUCCESS_DISTANCE,
                                   video_config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID)
    elif agent_type == "goal_follower":
        # Goal follower
        agent = GoalFollower(video_config.TASK_CONFIG.TASK.SUCCESS_DISTANCE,
                             video_config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID)
    elif agent_type == "ppo":
        # PPO agent

        # Specify the necessary info in the config file
        video_config.defrost()
        video_config.INPUT_TYPE = "depth"
        video_config.RESOLUTION = 256
        video_config.PTH_GPU_ID = 0
        video_config.HIDDEN_SIZE = 512
        video_config.RANDOM_SEED = 0
        video_config.freeze()

        agent = PPOAgent(video_config)

    elif agent_type == "informed_reactive":
        # Informed reactive agent
        agent = InformedReactiveNavigationAgent(video_config,
                                                video_config.TASK_CONFIG.TASK.SUCCESS_DISTANCE,
                                                weights=pretrained_weights)
    else:
        # Reactive agent (default)
        agent = ReactiveNavigationAgent(video_config,
                                        pretrained_weights)

    # Create the folder to store the video
    os.makedirs(video_config.VIDEO_DIR, exist_ok=True)

    # Instantiate the environment using the config file
    env = NavRLEnv(video_config)

    # If the seed is specified, instantiate it
    if video_config.SEED:
        random.seed(video_config.SEED)
        np.random.seed(video_config.SEED)
        torch.manual_seed(video_config.SEED)
        env.seed(video_config.SEED)

    # Store all the images to convert into video into a list
    processed_images = []

    # Perform the episode while generating the appropriate images
    observations = env.reset()
    agent.reset()
    print("Starting episode...")

    while not env.get_done(observations):
        action = agent.act(observations)
        observations, _, _, info = env.step(action=action)
        processed_images.append(observations_to_image(observations, info))

    print("Episode finished, generating video...")

    # Process the metrics to remove offending metrics
    # (top down map)
    metrics = env.get_info(observations)
    metrics.pop('top_down_map', None)
    collisions = metrics.pop('collisions', None)
    metrics["collisions"] = collisions["count"]

    # With all images generated, create the appropriate video
    # (no Tensorboard writer is used)
    # Note that having FFMPEG and libx264-dev installed is required
    # (install using conda-forge to ensure the driver is correctly installed)
    generate_video(video_option=video_config.VIDEO_OPTION,
                   video_dir=video_config.VIDEO_DIR,
                   images=processed_images,
                   episode_id=1,
                   checkpoint_idx=1,
                   metrics=metrics,
                   tb_writer=None)

    # Close the environment
    env.close()
    print("Video generated successfully")


#################
# 5 - MAIN CODE #
#################

# Execute this code only when this script is run directly
if __name__ == "__main__":

    # Message is printed due to the long loading time of the dependencies,
    # to show the user that the program has not crashed
    print("All dependencies have been successfully loaded.")

    # 5A - ARGUMENT DECLARATION #

    # All arguments have been specified in section 2 (USER-DEFINED VARIABLES)

    # Arguments are declared using argparse
    parser = argparse.ArgumentParser(description=textwrap.dedent("""\
    Runs a simulation of an embodied agent navigating within an indoor environment using Habitat-Lab.
    
    The agents perform a point navigation task (navigating to a specific point.)
    In addition, the script can be run to train the agents or evaluate their performance."""),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog="Note that most agent parameters can be configured in the appropriate "
                                            "config file (.yaml)")
    # agent_type
    parser.add_argument("-ag",
                        "--agent_type",
                        choices=["random", "random_forward", "goal_follower", "ppo", "reactive", "informed_reactive"],
                        help=textwrap.dedent("""\
                        Agent type used by the simulator. Agent types are as follows:
                            * random: A completely random agent, to be used as a benchmark
                            * random_forward: An agent that randomly moves, with bias towards
                                              moving forward, to be used as a benchmark
                            * goal_follower: An agent that always tries to move towards the goal,
                                             to be used as a benchmark
                            * ppo: A more advanced reinforcement learning based agent using Proximal Policy Optimization 
                                   (ppo) provided by habitat-lab
                            * reactive: The developed agent, using reactive navigation and deep reinforcement learning
                                        Note that there are several variations of the proposed algorithm
                            * informed_reactive: A variation of the reactive navigation agent that 
                                                 uses additional information during evaluation (using the 
                                                 distance sensor to automatically stop if the goal is reached)
                        DEFAULT: {}""".format(agent_type)))

    # dataset
    parser.add_argument("-ds",
                        "--dataset",
                        choices=["matterport", "gibson"],
                        help=textwrap.dedent("""\
                        Dataset used by the simulator to train and evaluate the agents. 
                        DEFAULT: {}""".format(dataset)))

    # mode
    parser.add_argument("-m",
                        "--mode",
                        choices=["training", "benchmark", "video"],
                        help=textwrap.dedent("""\
                        Execution mode of the program. The program can run in the following modes:
                            * training: Trains the agent via reinforcement learning with a training set.
                            * benchmark: Evaluates the performance of the agent in a validation set, using the
                                         benchmarking tool provided by Habitat Lab
                            * video: Generates a video of the agents performance in an specific episode
                        Note that only the RL-capable agents (ppo and reactive) are able to be used in training mode
                        DEFAULT: {}""".format(mode)))

    # weights
    parser.add_argument('-w',
                        '--weights',
                        help="Path to the file containing pretrained weights for the agent. "
                             "If not specified, random initial weights will be used.")

    # config
    parser.add_argument('-c',
                        '--config',
                        help="Path to a config file to use during the program execution. "
                             "If not specified, a default config file will be used for each agent.")

    # 5B - ARGUMENT PARSING #

    # Parse the arguments for their use and checks that the arguments are valid

    arguments = vars(parser.parse_args())

    if arguments["agent_type"] is not None:
        agent_type = arguments["agent_type"]

    if arguments["dataset"] is not None:
        dataset = arguments["dataset"]

    if arguments["mode"] is not None:
        mode = arguments["mode"]

    if arguments["weights"] is not None:
        weights = arguments["weights"]

    if arguments["config"] is not None:
        config = arguments["config"]

    # 5C - ARGUMENT PROCESSING #

    # Ensure that the combination of agent type / mode is appropriate
    if mode == "training" and agent_type not in ["ppo", "reactive"]:
        # Print an error message and abort the execution
        print("ERROR: Only pure RL-capable agents (ppo and reactive) can be used in training mode")
        sys.exit()

    # Choose the appropriate config file to be used
    if config:
        # User has specified a config file
        config_file = config
    else:
        # Config file depends on the selected mode
        if mode == "training":
            # Choose the appropriate config file for the agent
            config_file = config_paths_training[agent_type]
        elif mode == "benchmark":
            # Choose the generic benchmarking config file
            config_file = config_path_benchmark
        else:
            # Choose the generic video config file
            config_file = config_path_video

    # Depending on the execution mode, run the appropriate main code
    if mode == "training":
        # TRAINING MODE
        training_main(config_file,
                      dataset)
    elif mode == "benchmark":
        # BENCHMARK MODE
        benchmark_main(config_file,
                       weights)
    elif mode == "video":
        # VIDEO MODE
        video_main(config_file,
                   dataset,
                   weights)
