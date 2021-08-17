# REACTIVE NAVIGATION - MAIN CODE
# Developed by: Luna Jimenez Fernandez
#
# This file contains the main structure of the reactive navigation system developed
# TODO: ACABAR ESTO
#
#
# Structure of the file:
#   1 - Imports
#   2 - Program variables
#   3 - User-defined variables
#   4 - Main code
#       4A - Training
#       4B - Evaluation
#       4C - Benchmark
#   5 - Init code
#       5A - Argument declaration
#       5B - Argument parsing
#       5C - Argument processing and program launch

# TODO: EL TIPO DE COMENTARIO SE LLAMA SPHINX MARKUP
# (garantiza compatibilidad con el mayor numero de sistemas posibles)

###############
# 1 - IMPORTS #
###############

import argparse
import textwrap
import tensorflow as tf

# Habitat imports

import habitat
from habitat_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry

# Reactive navigation imports
from trainers.reactive_navigation_trainer import ReactiveNavigationTrainer
from envs.reactive_navigation_env import ReactiveNavigationEnv


#########################
# 2 - PROGRAM VARIABLES #
#########################

# These variables refer mostly to static values, such as paths to files.
# They are declared here (instead of being hard-coded in the code) for ease of use
# and for easier modification of the code by third parties

# Config file paths
# Each agent has its own specific config file, and a base config file shared by all of them
# Note that a config file can be specified via argument, overloading this value

# TODO - Faltan los configs del resto de agentes
config_paths = {
    "base": "./configs/base_config.yaml",
    "slam": None,
    "ppo": None,
    "neural_slam": None,
    "reactive": "./configs/reactive_pointnav_train_contour.yaml"
}

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
#   - benchmark - Evaluates the performance of the agent using the provided Habitat Lab benchmark tool
# Note that not all agents may be able to run in all modes
mode = "training"

# These variables are used by Reinforcement Learning agents

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

    Trains the specified agent using the specified environment and configuration.
    Note that both the agent trainer and the environment are actually specified in the config file.

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


def evaluation_main():
    pass


def benchmark_main():
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
                        choices=["slam", "ppo", "neural_slam", "reactive"],
                        help=textwrap.dedent("""\
                        Agent type used by the simulator. Agent types are as follows:
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
                        Dataset used by the simulator to train and evaluate the agents. 
                        DEFAULT: {}""".format(dataset)))

    # mode
    parser.add_argument("-m",
                        "--mode",
                        choices=["training", "evaluation", "benchmark"],
                        help=textwrap.dedent("""\
                        Execution mode of the program. The program can run in the following modes:
                            * training: Trains the agent via reinforcement learning with a training set.
                            * evaluation: Evaluates the performance of the agent in a validation set.
                            * showcase: Shows and stores the performance of the agent in a randomly selected scenario.
                    
                        Note that not some agents may be unable to be used in some of these modes.
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

    # Choose the appropriate config file to be used
    if config:
        # User has specified a config file
        config_file = config
    else:
        # Use the default config file
        config_file = config_paths[agent_type]

    # Limit the memory usage of TensorFlow
    # This is done to avoid OutOfMemory errors during execution
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Depending on the execution mode, run the appropriate main code
    if mode == "training":
        # TRAINING MODE
        training_main(config_file, dataset)
    elif mode == "evaluation":
        # EVALUATION MODE
        evaluation_main()


