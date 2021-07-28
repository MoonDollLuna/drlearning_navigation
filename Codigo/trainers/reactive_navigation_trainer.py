# REACTIVE NAVIGATION - TRAINER
# Developed by: Luna Jimenez Fernandez
#
# This file contains the definition for aReactive Navigation trainer,
# used to train the Reactive Navigation Agent (using our proposed algorithm)
#
# Deep Q-Learning is used as the method of training, with the possibility of using
# Prioritized Experience Replay as an option

# IMPORTS #

from typing import Dict

# Habitat Baselines
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

# Reactive navigation
from models.reactive_navigation import ReactiveNavigationModel


@baseline_registry.register_trainer(name="reactive")
class ReactiveNavigationTrainer(BaseRLTrainer):
    """

    """

    # ATTRIBUTES #
    # Note that some attributes are contained in the superclass (BaseRLTrainer)


    # CONSTRUCTOR #
    def __init__(self, config=None):
        pass

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



