# REACTIVE NAVIGATION - MAIN CODE
# Developed by: Luna Jimenez Fernandez
#
# This file contains the main structure of the reactive navigation system developed
# (including the arguments, parameters for the simulator, data display...)
#
# The developed agents are contained in the "agents" folder, in separate files.
#
# Structure of the file:
#   1 - Imports
#   2 - User-defined variables
#   3 - Main code
#       3A - Argument declaration
#       3B - Argument parsing
#       3C - Main loop launch

# NOTA: EL TIPO DE COMENTARIO SE LLAMA SPHINX MARKUP
# (garantiza compatibilidad con el mayor numero de sistemas posibles)

###############
# 1 - IMPORTS #
###############

import argparse

import cv2
import habitat
import numpy as np

from utils.log_manager import LogManager

##############################
# 2 - USER-DEFINED VARIABLES #
##############################

# TRAINING RELATED #

# agent_type - Specifies the agent type to be used
# Possible agent types:
#   - slam - A basic agent using SLAM provided by habitat-lab (BENCHMARK)
#   - ppo - A more advanced, reinforcement learning based agent using Proximal Policy Optimization (ppo)
#           provided by habitat-lab (BENCHMARK)
#   - neural_slam - A version of SLAM implementing Deep Reinforcement Learning to improve its performance
#   - reactive - The developed agent, using reactive navigation and deep reinforcement learning
# DEFAULT: reactive
agent_type = "reactive"

# dataset - Specifies the dataset to be used
# Possible values:
#   - matterport
#   - gibson
# DEFAULT: matterport
dataset = "matterport"

# REINFORCEMENT LEARNING RELATED #
# These variables are used specifically by Reinforcement Learning agents

# seed - Specifies a seed to be used for all random choices
# If not specified, a random seed will be used instead
seed = None

# weights - For trainable agents, specifies the pre-trained weights for the neural network
# If specified, the program will be launched into "play" mode instead (showcasing the performance)
# If not specified, randomly initialized weights will be used instead
weights = None

habitat.RLEnv()



#################
# 3 - MAIN CODE #
#################

# Execute this code only when this script is run directly
if __name__ == "__main__":

    # ARGUMENT DECLARATION #
    # Arguments are declared using argparse
    parser = argparse.ArgumentParser(description="Runs a simulation of an embodied agent in an indoor environment.")

    # All arguments have been specified in section 2 (USER-DEFINED VARIABLES)
    # dataset
    parser.add_argument("-ds",
                        "--dataset",
                        choices=["matterport", "gibson"],
                        help="Dataset used to train the agents. DEFAULT: {}".format(dataset))

    # ARGUMENT PARSING
