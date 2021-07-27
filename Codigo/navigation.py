# REACTIVE NAVIGATION - MAIN CODE
# Developed by: Luna Jimenez Fernandez
#
# This file contains the main structure of the reactive navigation system developed
# (including the arguments, parameters for the simulator, data display...)
#
#
# Structure of the file:
#   1 - Imports
#   2 - Program variables
#   3 - User-defined variables
#   4 - Main code loops
#       4A - Training loop
#       4
#   5 - Init code
#       5A - Argument declaration
#       5B - Argument parsing
#       5C - Argument processing and program launch

# NOTA: EL TIPO DE COMENTARIO SE LLAMA SPHINX MARKUP
# (garantiza compatibilidad con el mayor numero de sistemas posibles)

###############
# 1 - IMPORTS #
###############

import argparse
import textwrap
import sys

import cv2
import habitat
import numpy as np

from utils.log_manager import LogManager

#########################
# 2 - PROGRAM VARIABLES #
#########################

# These variables refer mostly to static values, such as paths to files.
# They are declared here (instead of being hard-coded in the code) for ease of use
# and for easier modification of the code by third parties

# Config file paths
# If an extra config file was to be added, the path can be specified as a new value
config_path = {
    "training": "./configs/navigation_train_config.yaml",
    "evaluation": "./configs/navigation_eval_config.yaml",
    "showcase": "./configs/navigation_eval_config.yaml"
}

# Dataset paths
# If an extra dataset was to be added, the path can be specified as a new value
dataset_paths = {
    "matterport": "data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz",
    "gibson": "data/datasets/pointnav/gibson/v1/{split}/{split}.json.gz"
}

##############################
# 3 - USER-DEFINED VARIABLES #
##############################

# These variables are defined via arguments when the script is launched
# The values written below are considered the DEFAULT values, and will be the values
# used for the scrip unless specified otherwise

# SIMULATOR RELATED #

# agent_type - Specifies the agent type to be used
# Possible agent types:
#   - manual - A basic agent controlled by the user. Not to be used as a benchmark, provided for showcase.
#   - slam - A basic agent using SLAM provided by habitat-lab (BENCHMARK)
#   - ppo - A more advanced, reinforcement learning based agent using Proximal Policy Optimization (ppo)
#           provided by habitat-lab (BENCHMARK)
#   - neural_slam - A version of SLAM implementing Deep Reinforcement Learning to improve its performance
#   - reactive - The developed agent, using reactive navigation and deep reinforcement learning
agent_type = "reactive"

# dataset - Specifies the dataset to be used
# Possible values:
#   - matterport
#   - gibson
dataset = "matterport"

# mode - Specifies the mode in which the simulator will be launched
# Possible values:
#   - training - Trains the agent via reinforcement learning with a training set
#   - evaluation - Evaluates the performance of the agent in a validation set
#   - showcase - Shows and stores the performance of the agent in a randomly selected scenario
# Note that not all agents may be able to run in all modes
mode = "training"


# TRAINING RELATED #
# These variables are used by Reinforcement Learning agents

# seed - Specifies a seed to be used for all random choices
# If not specified, a random seed will be used instead
seed = None

# weights - For trainable agents, specifies the pre-trained weights for the neural network
# If specified, the program will be launched into "play" mode instead (showcasing the performance)
# If not specified, randomly initialized weights will be used instead
weights = None

# total_episodes - Total amount of episodes during which the agent will be trained
# The dataset will be evenly split among these episodes
total_episodes = 10000

# max_steps - Maximum steps made by the agent during each episode
# If this value is exceeded (the goal is not reached within this time), the episode will be considered failed
max_steps = 10000

# goal_radius - Size of the goal (in meters)
# The agent must be within this distance of the goal for it to be considered reached
goal_radius = 0.2

# learning_rate - Learning rate used by the neural network
learning_rate = 0.001

# DEEP REINFORCEMENT LEARNING RELATED #
# These variables are specifically used by Deep Reinforcement Learning agents

# er_size - Maximum size of the Experience Replay. Once the ER is full, the oldest experiences will be overridden
er_size = 20000

# batch_size - Size of the batches used to sample the Experience Replay
batch_size = 64

# gamma - Gamma value (learning rate of Deep Q-Learning) used by the agent
gamma = 0.99

# epsilon - Epsilon value (initial chance to perform a random action due to exploration-exploitation) used by the agent
epsilon = 1.0

# min_epsilon - Minimum epsilon value achieved after a percentage of epochs (specified below) have been completed
min_epsilon = 0.05

# min_epsilon_percentage - Percentage of epochs (from 0 to 1) after which the value of epsilon will reach min_epsilon
# epsilon decreases linearly from epsilon to min_epsilon epoch by epoch
min_epsilon_percentage = 0.75

#######################
# 4 - MAIN CODE LOOPS #
#######################

def training_main():
    """
    Main loop for agent training.

    """
    pass

def evaluation_main():
    pass

def showcase_main():
    pass

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
    parser = argparse.ArgumentParser(description="Runs a simulation of an embodied agent in an indoor environment "
                                                 "for training.",
                                     formatter_class=argparse.RawTextHelpFormatter)

    # agent_type
    parser.add_argument("-ag",
                        "--agent_type",
                        choices=["manual", "slam", "ppo", "neural_slam", "reactive"],
                        help=textwrap.dedent("""\
                        Agent type used by the simulator. Agent types are as follows:
                            * manual: A basic agent controlled by the user. Not to be used as a benchmark provided for showcase.
                            * slam: A basic agent using SLAM provided by habitat-lab. (BENCHMARK)
                            * ppo: A more advanced, reinforcement learning based agent using Proximal Policy Optimization (ppo) provided by habitat-lab. (BENCHMARK)
                            * neural_slam: A version of SLAM implementing Deep Reinforcement Learning to improve its performance.
                            * reactive: The developed agent, using reactive navigation and deep reinforcement learning.
                        DEFAULT: {}""".format(agent_type)))

    # dataset
    parser.add_argument("-ds",
                        "--dataset",
                        choices=["matterport", "gibson"],
                        help=textwrap.dedent("""\
                        Dataset used to train the agents. 
                        DEFAULT: {}""".format(dataset)))

    # mode
    parser.add_argument("-m",
                        "--mode",
                        choices=["training", "evaluation", "showcase"],
                        help=textwrap.dedent("""\
                        Agent type used by the simulator. Agent types are as follows:
                            * training: Trains the agent via reinforcement learning with a training set.
                            * evaluation: Evaluates the performance of the agent in a validation set.
                            * showcase: Shows and stores the performance of the agent in a randomly selected scenario.
                    
                        Note that not some agents may be unable to be used in some of these modes.g.
                        DEFAULT: {}""".format(mode)))

    # seed
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        help="Seed used for all random events. If not specified, a random seed will be used. "
                             "Note that reproducibility with a seed is not fully guaranteed due to parallelization.")

    # weights
    parser.add_argument('-w',
                        '--weights',
                        help="Path to the file containing pretrained weights for the agent. "
                             "If not specified, random initial weights will be used.")

    # total_episodes
    parser.add_argument('-te',
                        '--total_episodes',
                        type=int,
                        help=textwrap.dedent("""\
                        Total amount of episodes used by the agent to train. Value must be a positive integer.
                        DEFAULT: {}""".format(total_episodes)))

    # max_steps
    parser.add_argument('-ms',
                        '--max_steps',
                        type=int,
                        help=textwrap.dedent("""\
                            Maximum number of steps the can be performed by an agent during an episode.
                            If this value is exceeded (the agent does not reach the goal within max_steps), the episode is considered as failed.
                            Value must be a positive integer.
                            DEFAULT: {}""".format(max_steps)))

    # goal_radius
    parser.add_argument('-gr',
                        '--goal_radius',
                        type=float,
                        help=textwrap.dedent("""\
                                Size of the goal (in meters). The agent must be within this distance of the goal for the episode to be completed. 
                                Value must be a positive number.
                                DEFAULT: {}""".format(goal_radius)))

    # learning_rate
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        help=textwrap.dedent("""\
                            Learning rate for the neural networks. Value must be between 0.0 and 1.0.
                            DEFAULT: {}""".format(learning_rate)))

    # er_size
    parser.add_argument('-er',
                        '--er_size',
                        type=int,
                        help=textwrap.dedent("""\
                            Maximum size of the Experience Replay. Value must be a positive integer.
                            DEFAULT: {}""".format(er_size)))

    # batch_size
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        help=textwrap.dedent("""\
                                Batch size used when sampling the Experience Replay. Value must be a positive integer.
                                DEFAULT: {}""".format(batch_size)))

    # gamma
    parser.add_argument('-g',
                        '--gamma',
                        type=float,
                        help=textwrap.dedent("""\
                            Gamma value (learning rate of Deep Q-Learning). Value must be between 0.0 and 1.0.
                            DEFAULT: {}""".format(gamma)))

    # epsilon
    parser.add_argument('-e',
                        '--epsilon',
                        type=float,
                        help=textwrap.dedent("""\
                                Epsilon value (initial chance to perform a random action due to exploration-exploitation). Value must be between 0.0 and 1.0.
                                DEFAULT: {}""".format(epsilon)))

    # min_epsilon
    parser.add_argument('-me',
                        '--min_epsilon',
                        type=float,
                        help=textwrap.dedent("""\
                                    Minimum epsilon value, achieved after a percentage of epochs (specified by min_epsilon_percentage). Value must be between 0.0 and 1.0.
                                    DEFAULT: {}""".format(min_epsilon)))

    # min_epsilon_percentage
    parser.add_argument('-mep',
                        '--min_epsilon_percentage',
                        type=float,
                        help=textwrap.dedent("""\
                                        Percentage of epochs after which the value of epsilon will reach min_epsilon
                                        Epsilon will decrease linearly from epsilon to min_epsilon. Value must be between 0.0 and 1.0.
                                        DEFAULT: {}""".format(min_epsilon_percentage)))

    # 5B - ARGUMENT PARSING #

    # Parse the arguments for their use and checks that the arguments are valid
    # The program will exit if any invalid arguments are detected

    arguments = vars(parser.parse_args())
    bad_arguments = False

    if arguments["agent_type"] is not None:
        agent_type = arguments["agent_type"]

    if arguments["dataset"] is not None:
        dataset = arguments["dataset"]

    if arguments["mode"] is not None:
        mode = arguments["mode"]

    if arguments["seed"] is not None:
        seed = arguments["seed"]

    if arguments["weights"] is not None:
        weights = arguments["weights"]

    if arguments["total_episodes"] is not None:
        total_episodes = arguments["total_episodes"]
        if total_episodes < 0:
            bad_arguments = True
            print("ERROR: Total episodes must be a positive integer. Value provided: {}".format(total_episodes))

    if arguments["max_steps"] is not None:
        max_steps = arguments["max_steps"]
        if max_steps < 0:
            bad_arguments = True
            print("ERROR: Max steps must be a positive integer. Value provided: {}".format(max_steps))

    if arguments["goal_radius"] is not None:
        goal_radius = arguments["goal_radius"]
        if goal_radius < 0.0:
            bad_arguments = True
            print("ERROR: Goal radius must be a positive number. Value provided: {}".format(goal_radius))

    if arguments["learning_rate"] is not None:
        learning_rate = arguments["learning_rate"]
        if learning_rate < 0.0 or learning_rate > 1.0:
            bad_arguments = True
            print("ERROR: Learning rate must be between 0.0 and 1.0. Value provided: {}".format(learning_rate))

    if arguments["er_size"] is not None:
        er_size = arguments["er_size"]
        if er_size < 0:
            bad_arguments = True
            print("ERROR: Experience Replay size must be a positive integer. Value provided: {}".format(er_size))

    if arguments["batch_size"] is not None:
        batch_size = arguments["batch_size"]
        if batch_size < 0:
            bad_arguments = True
            print("ERROR: Batch size must be a positive integer. Value provided: {}".format(batch_size))

    if arguments["gamma"] is not None:
        gamma = arguments["gamma"]
        if gamma < 0.0 or gamma > 1.0:
            bad_arguments = True
            print("ERROR: Gamma must be between 0.0 and 1.0. Value provided: {}".format(gamma))

    if arguments["epsilon"] is not None:
        epsilon = arguments["epsilon"]
        if epsilon < 0.0 or epsilon > 1.0:
            bad_arguments = True
            print("ERROR: Epsilon must be between 0.0 and 1.0. Value provided: {}".format(epsilon))

    if arguments["min_epsilon"] is not None:
        min_epsilon = arguments["min_epsilon"]
        if min_epsilon < 0.0 or min_epsilon > 1.0:
            bad_arguments = True
            print("ERROR: Minimum epsilon must be between 0.0 and 1.0. Value provided: {}".format(min_epsilon))

    if arguments["min_epsilon_percentage"] is not None:
        min_epsilon_percentage = arguments["min_epsilon_percentage"]
        if min_epsilon_percentage < 0.0 or min_epsilon_percentage > 1.0:
            bad_arguments = True
            print("ERROR: Minimum epsilon percentage must be between 0.0 and 1.0. Value provided: {}".format(min_epsilon_percentage))

    # If the bad arguments flag was raised, exit the program
    if bad_arguments:
        print("Bad arguments have been detected, program will shut down.")
        sys.exit()

    # 5C - ARGUMENT PROCESSING #

