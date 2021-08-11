# REACTIVE NAVIGATION - TRAINER
# Developed by: Luna Jimenez Fernandez
#
# This file contains the definition for aReactive Navigation trainer,
# used to train the Reactive Navigation Agent (using our proposed algorithm)
#
# Deep Q-Learning is used as the method of training, with the possibility of using
# Prioritized Experience Replay as an option

# IMPORTS #

import copy

from typing import Dict
from keras.models import Model

# Habitat Baselines
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

    # ATTRIBUTES #
    # Note that some attributes are contained in the superclass (BaseRLTrainer)

    # NETWORKS
    # Q Network used by the trainer. This network is updated after each action taken
    _q_network: Model
    # Target network used by the trainer. This network is used to obtain the target Q Values,
    # and is updated by copying the Q Network at the end of each epoch
    _target_network: Model

    # EXPERIENCE REPLAY
    # Flag to specify whether standard or prioritized Deep Q-Learning is to be used
    _prioritized: bool
    # Experience Replay used by the trainer, to store all experiences that happened during training
    # Can be either standard or prioritized Experience Replay
    _experience_replay: ExperienceReplay

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

    # CONSTRUCTOR #
    def __init__(self, config=None):
        """

        :param config:
        """

    # PRIVATE METHODS #

    def _eval_checkpoint(self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0) -> None:
        pass

    # PUBLIC METHODS #

    def save_checkpoint(self, file_name) -> None:
        pass

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        pass

    # MAIN METHODS #

    def train(self) -> None:
        pass



