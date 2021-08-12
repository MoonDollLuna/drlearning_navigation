# REACTIVE NAVIGATION - TRAINER
# Developed by: Luna Jimenez Fernandez
#
# This file contains the definition for aReactive Navigation trainer,
# used to train the Reactive Navigation Agent (using our proposed algorithm)
#
# Deep Q-Learning is used as the method of training, with the possibility of using
# Prioritized Experience Replay as an option
#
# Due to the size and complexity of the file, the class definition has been defined
# with the following structure:
#   1 - IMPORTS
#   2 - CLASS ATTRIBUTES
#   3 - CONSTRUCTOR
#   4 - AUXILIARY METHODS
#   5 - MAIN (INHERITED) METHODS
#

###############
# 1 - IMPORTS #
###############

import copy
import os
import time

from typing import Dict

# Habitat (Baselines)
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

# Reactive navigation
from models.reactive_navigation import ReactiveNavigationModel
from models.experience_replay import ExperienceReplay, PrioritizedExperienceReplay


@baseline_registry.register_trainer(name="reactive")
class ReactiveNavigationTrainer(BaseRLTrainer):
    """

    """

    ########################
    # 2 - CLASS ATTRIBUTES #
    ########################

    # Note that some attributes are contained in the superclass (BaseRLTrainer)

    # NETWORKS
    # Q Network used by the trainer. This network is updated after each action taken
    _q_network: ReactiveNavigationModel
    # Target network used by the trainer. This network is used to obtain the target Q Values,
    # and is updated by copying the Q Network at the end of each epoch
    _target_network: ReactiveNavigationModel

    # EXPERIENCE REPLAY
    # Flag to specify whether standard or prioritized Deep Q-Learning is to be used
    _prioritized: bool
    # Experience Replay used by the trainer, to store all experiences that happened during training
    # Can be either standard or prioritized Experience Replay
    _experience_replay: ExperienceReplay

    # ENVIRONMENT PARAMETERS
    # Size of the image
    _image_size: int
    # List of actions available for the agent
    # This is stored as a list of strings
    _agent_actions: list

    # DQL PARAMETERS
    # Learning rate of the neural network
    _learning_rate: float
    # Maximum size of the Experience Replay
    _er_size: int
    # Batch size to use when sampling the Experience Replay
    _batch_size: int
    # Gamma value (learning rate of DQL)
    _gamma: float
    # Epsilon value (initial chance to perform a random action due to exploration-exploitation)
    _epsilon: float
    # Minimum epsilon value, achieved after a percentage of epochs (min_epsilon_percentage)
    _min_epsilon: float
    # Percentage of epochs (between 0 and 1) after which epsilon will reach min_epsilon.
    # The value of epsilon will decrease linearly from epsilon to min_epsilon
    _min_epsilon_percentage: float
    # (PRIORITIZED ONLY) Alpha value (priority degree)
    # The higher alpha is, the higher the probability of choosing higher error experiences is
    _prioritized_alpha: float
    # (PRIORITIZED ONLY) Beta value (bias degree)
    # The higher the value is, the less weight variations have (to avoid big oscillations)
    _prioritized_beta: float

    # DOCUMENTATION PARAMETERS
    # Location of the checkpoints folder
    _checkpoint_folder: str
    # Location of the log folder
    _log_folder: str
    # Initial time when the agent started training
    _start_time: float

    ###################
    # 3 - CONSTRUCTOR #
    ###################

    def __init__(self, config=None):
        """
        Constructor of the trainer. All parameters are passed using a config file

        This environment uses Deep Q-Learning to train the Reactive Navigation agent, using the
        (also designed ad-hoc for this purpose) Reactive Navigation environment

        This constructor can work in two different methods:
            * Standard
            * Prioritized

        :param config: Config class, containing the necessary parameters for this environment.
        :type config: Config
        """

        # Initialize the parent class
        super().__init__(config)

        # Store all config parameters
        _rl_config = config.RL
        _dql_config = _rl_config.DQL

        self._image_size = config.SIMULATOR.DEPTH_SENSOR.WIDTH
        self._agent_actions = config.TASK.POSSIBLE_ACTIONS

        self._prioritized = _dql_config.prioritized
        self._learning_rate = _dql_config.learning_rate
        self._er_size = _dql_config.er_size
        self._batch_size = _dql_config.batch_size
        self._gamma = _dql_config.gamma
        self._epsilon = _dql_config.epsilon
        self._min_epsilon = _dql_config.min_epsilon
        self._min_epsilon_percentage = _dql_config.min_epsilon_percentage
        self._prioritized_alpha = _dql_config.prioritized_alpha
        self._prioritized_beta = _dql_config.prioritized_beta

        self._checkpoint_folder = config.CHECKPOINT_FOLDER
        self._log_folder = config.LOG_FOLDER

    #########################
    # 4 - AUXILIARY METHODS #
    #########################

    # PRIVATE METHODS
    def _init_train(self):
        """
        Initializes the following:
            * Neural networks are initialized (both the Q Network and the Target Network)
            * The Experience Replay is instantiated (either standard or prioritized)
            * Initial training time is stored
            * Necessary folders for the log and checkpoints are created

        This allows the training process to start properly
        """
        # Initialize the neural networks
        self._q_network = ReactiveNavigationModel(self._image_size,
                                                  self._agent_actions,
                                                  learning_rate=self._learning_rate)
        self._target_network = copy.deepcopy(self._q_network)

        # Initialize the Experience Replay (depending on the type of DQL to be applied)
        if self._prioritized:
            self._experience_replay = PrioritizedExperienceReplay(self._er_size,
                                                                  self._prioritized_alpha,
                                                                  self._prioritized_beta)
        else:
            self._experience_replay = ExperienceReplay(self._er_size)

        # Store the initial time
        self._start_time = time.time()

        # Create the folder structure for both log and checkpoints
        os.makedirs(self._checkpoint_folder)
        os.makedirs(self._log_folder)

    ####################
    # 5 - MAIN METHODS #
    ####################

    # PRIVATE METHODS #

    def _eval_checkpoint(self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0) -> None:
        pass

    # PUBLIC METHODS #

    def save_checkpoint(self, file_name):
        """
        Saves a checkpoint (the weights of the neural network) using the specified
        filename

        :param file_name: Filename to be used when storing the checkpoint
        :type file_name: str
        """
        pass

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        pass

    def train(self):
        """
        Main method. Trains a Reactive Navigation agent using the proposed rewards systems
        using Deep Q-Learning (either standard or prioritized)
        """

        # Do all the necessary pre-steps
        self._init_train()



