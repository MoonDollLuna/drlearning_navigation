# REACTIVE NAVIGATION - AGENT
# Developed by: Luna Jmenez Fernandez
#
# This file contains the definition of a Reactive Navigation Agent, to be used
# along the benchmarking tools provided by Habitat-Lab in order to
# measure the performance of the trained agent
#
# This agent will be benchmarked on the following attributes:
#
#
#

# IMPORTS #

import torch

# Habitat Lab
from habitat.config import Config
from habitat.core.agent import Agent
from habitat.core.simulator import Observations

# Reactive Navigation
from models.reactive_navigation import ReactiveNavigationModel
from models.experience_replay import State


class ReactiveNavigationAgent(Agent):
    """
    Embodied agent using the proposed Reactive Navigation algorithm to traverse through
    an indoors environment to reach a goal (Point Navigation task)

    This agent is designed to be used with the Benchmark tool provided by Habitat Lab,
    in order to measure its trained performance against other agents
    """

    # ATTRIBUTES

    # Reactive Navigation model to be used by the agent
    _model: ReactiveNavigationModel

    def __init__(self, config, weights=None):
        """
        Creates an instance of the Reactive Navigation Agent when given a config file

        :param config: Configuration class, containing the necessary parameters for this agent creation
        :type config: Config
        :weights: (OPTIONAL) Path to the file containing the pre-trained weights of the CNN
        :type weights: str
        """

        # Creates the device, using CUDA if possible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract the necessary parameters from the config
        # The agent must be able to accept any kind of config file
        # (either a basic one for benchmarks or a RL one for video generation)
        # Use EAFP (Easier to Ask Forgiveness than Permission) to check it
        if hasattr(config, "TASK_CONFIG"):
            image_size = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
            action_list = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        else:
            image_size = config.SIMULATOR.DEPTH_SENSOR.WIDTH
            action_list = config.TASK.POSSIBLE_ACTIONS

        # Instantiate the model
        self._model = ReactiveNavigationModel(image_size,
                                              action_list,
                                              device,
                                              weights=weights).to(device)

    # PUBLIC (INHERITED) METHODS #

    def reset(self):
        """
        Prepares the agent for the start of an episode.

        Called before the environment is reset by the benchmark.
        """

        # The agent doesn't need any special preparations on episode start
        pass

    def act(self, observations):
        """
        Given the observations from the environment (the state), choose the next action

        Called before "step" is called by the benchmark.

        :param observations: Observations of the environment perceived by the agent
        :type observations: Observations
        :return: A dictionary with the shape {"action": <chosen action>}
        :rtype: dict
        """

        # Extract the state from the environment
        state = State.get_state(observations)

        # Return the chosen action by the agent
        return {"action": self._model.act(state)}
