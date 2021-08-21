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

import os
import random
import time
import datetime

import numpy as np
import torch

# Habitat (Baselines)
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

# Reactive navigation
from models.reactive_navigation import ReactiveNavigationModel
from envs.reactive_navigation_env import ReactiveNavigationEnv
from models.experience_replay import State, Experience, ExperienceReplay, PrioritizedExperienceReplay
from utils.log_manager import LogManager


@baseline_registry.register_trainer(name="reactive")
class ReactiveNavigationTrainer(BaseRLTrainer):
    """
    A implementation of the trainer used by the Reactive Navigation agent, in order to train it
    using Deep Q-Learning

    The trainer is developed specifically for this agent, and can train:
        * Standard or Prioritized Deep Q-Learning
        * Deep Q-Learning using Contours or Column based rewards (based on the environment)
        * Standard or Weighted action choosing
    """

    ########################
    # 2 - CLASS ATTRIBUTES #
    ########################

    # Note that some attributes are contained in the superclass (BaseRLTrainer)

    # ENVIRONMENT PARAMETERS
    # Identifier for the environment, used to grab it from the Habitat Baseline Registry
    _env_name: str
    # Handle for the environment to be used during training
    _env: ReactiveNavigationEnv
    # Name of the dataset
    _dataset_name: str
    # Size of the image
    _image_size: int
    # List of actions available for the agent
    # This is stored as a list of strings
    _agent_actions: list
    # Dictionary of actions (key) and their associated indexes (value)
    _action_to_index: dict
    # Distance to the goal at which it is considered successful
    _goal_distance: float
    # Total number of steps to perform during training (-1 means not limited)
    _total_number_steps: int
    # Total number of updates to perform during training (-1 means not limited)
    _number_updates: int

    # NETWORKS AND EXPERIENCE REPLAY PARAMETERS
    # Q Network used by the trainer. This network is updated after each action taken
    _q_network: ReactiveNavigationModel
    # Target network used by the trainer. This network is used to obtain the target Q Values,
    # and is updated by copying the Q Network at the end of each epoch
    _target_network: ReactiveNavigationModel
    # Flag to specify whether standard or prioritized Deep Q-Learning is to be used
    _prioritized: bool
    # Experience Replay used by the trainer, to store all experiences that happened during training
    # Can be either standard or prioritized Experience Replay
    _experience_replay: ExperienceReplay

    # DQL PARAMETERS
    # Seed for all experiments. May be None to use a random seed
    _seed: int
    # Learning rate of the neural network
    _learning_rate: float
    # Maximum size of the Experience Replay
    _er_size: int
    # Batch size to use when sampling the Experience Replay
    _batch_size: int
    # When training the network, batches of experiences are divided into sub-batches of this size
    # Improves memory usage at the cost ot training time
    _training_batch_size: int
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
    # Count of checkpoints
    _checkpoint_count: int
    # Location of the log folder during training
    _training_log_folder: str
    # Location of the log folder during evaluation
    _evaluation_log_folder: str
    # Flag to indicate whether the log is silent (doesn't output messages to the screen, TRUE) or not (FALSE)
    _silent: bool
    # Initial time when the agent started training
    _start_time: float
    # Rewards method used
    _rewards_method: str

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

        :param config: Config class, containing the necessary parameters for this training script.
        :type config: Config
        """

        # Initialize the parent class
        super().__init__(config)

        # Store all config parameters
        _rl_config = config.RL
        _dql_config = _rl_config.DQL

        self._env_name = config.ENV_NAME

        self._dataset_name = config.TASK_CONFIG.DATASET.NAME
        self._image_size = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        self._agent_actions = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        self._goal_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
        self._total_number_steps = config.TOTAL_NUM_STEPS
        self._number_updates = config.NUM_UPDATES

        self._seed = _rl_config.seed if _rl_config.seed else None
        self._prioritized = _dql_config.prioritized
        self._learning_rate = _dql_config.learning_rate
        self._er_size = _dql_config.er_size
        self._batch_size = _dql_config.batch_size
        self._training_batch_size = _dql_config.training_batch_size
        self._gamma = _dql_config.gamma
        self._epsilon = _dql_config.epsilon
        self._min_epsilon = _dql_config.min_epsilon
        self._min_epsilon_percentage = _dql_config.min_epsilon_percentage
        self._prioritized_alpha = _dql_config.prioritized_alpha
        self._prioritized_beta = _dql_config.prioritized_beta

        self._checkpoint_folder = config.CHECKPOINT_FOLDER
        self._training_log_folder = config.TRAINING_LOG_FOLDER
        self._evaluation_log_folder = config.EVALUATION_LOG_FOLDER
        self._silent = config.LOG_SILENT
        self._rewards_method = _rl_config.REWARD.reward_method

        # Create the dictionary
        self._action_to_index = {}
        for index, action in enumerate(self._agent_actions):
            self._action_to_index[action] = index

    #########################
    # 4 - AUXILIARY METHODS #
    #########################

    # PRIVATE METHODS
    def _init_train(self):
        """
        Initializes the following:
            * Creates the device to be used during training
            * Neural networks are initialized (both the Q Network and the Target Network)
            * The Experience Replay is instantiated (either standard or prioritized)
            * Initial training time is stored
            * All necessary seeds are initialized
            * Necessary folders for the log and checkpoints are created
            * Instantiates the environment
            * Creates and returns the Log Manager object

        This allows the training process to start properly

        :return: Log Manager, used to write the necessary log information / device used by pyTorch
        :rtype: tuple
        """

        # Creates the device, using CUDA if possible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the neural networks
        self._q_network = ReactiveNavigationModel(self._image_size,
                                                  self._agent_actions,
                                                  device,
                                                  learning_rate=self._learning_rate).to(device)
        self._target_network = ReactiveNavigationModel(self._image_size,
                                                       self._agent_actions,
                                                       device,
                                                       learning_rate=self._learning_rate).to(device)

        # Copy the weight from the Q Network to the Target network
        self._target_network.load_state_dict(self._q_network.state_dict())

        # Initialize the Experience Replay (depending on the type of DQL to be applied)
        if self._prioritized:
            self._experience_replay = PrioritizedExperienceReplay(self._er_size,
                                                                  self._prioritized_alpha,
                                                                  self._prioritized_beta)
        else:
            self._experience_replay = ExperienceReplay(self._er_size)

        # Store the initial time
        self._start_time = time.time()

        # Create the folder structure and variables for both log and checkpoints
        os.makedirs(self._checkpoint_folder, exist_ok=True)
        os.makedirs(self._training_log_folder, exist_ok=True)
        self._checkpoint_count = 0

        # Instantiate the environment
        env_init = baseline_registry.get_env(self._env_name)
        self._env = env_init(self.config)

        # Initialize the seeds if necessary
        if self._seed:
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)
            self._env.seed(self._seed)

        # Compute the approximate length of the training, for the log
        if self._total_number_steps != -1:
            training_length = "{} steps".format(self._total_number_steps)
        else:
            training_length = "{} episodes".format(self._number_updates)

        # Create and return the appropriate log manager, along the device
        return LogManager(self._training_log_folder,
                          "reactive",
                          self._dataset_name,
                          training_length,
                          self._start_time,
                          self._silent,
                          epoch_parameters=["average_reward"],
                          reward_method=self._rewards_method), device

    def _train_network_standard(self):
        """
        Auxiliary method used to train the Q-Network when using Standard Prioritized Replay

        This method does the following:
            * Sample the ER to obtain the experiences
            * Unwraps the experiences into usable forms
            * Computes the Q values for the current and next state
            * Computes the UPDATED Q values from these previous Q values
        """

        # Sample the ER to obtain the experiences
        sampled_experiences = self._experience_replay.sample_memory(self._batch_size)

        # Unwrap the experiences
        (sampled_initial_states,
         sampled_actions,
         sampled_rewards,
         sampled_next_states,
         sampled_finals) = Experience.unwrap_experiences(sampled_experiences)

        # Compute the Q values for the current state (using the Q network)
        # and the Q values for the next state (using the target network)
        current_state_values = self._q_network.predict(sampled_initial_states, self._training_batch_size)
        next_state_predictions = self._target_network.predict(sampled_next_states, self._training_batch_size)

        # Store the updated values in a different list
        updated_current_state_values = current_state_values.copy()

        # Compute the updated Q values
        for index in range(len(sampled_experiences)):
            # Check if the experience is a final one or not
            if sampled_finals[index]:
                # Final experience: the Q Value of the state is simply the reward / penalty
                # q_value(t) = reward
                updated_current_state_values[index][self._action_to_index[sampled_actions[index]]] = sampled_rewards[index]
            else:
                # Not a final experience: the Q value is based on the obtained reward
                # and the max Q value for the next state
                # q_value(t) = reward + gamma * next_state_best_q
                updated_current_state_values[index][self._action_to_index[sampled_actions[index]]] = sampled_rewards[index] + self._gamma * np.amax(next_state_predictions[index])

        # With the updated predictions, fit the Q network
        self._q_network.fit_model(sampled_initial_states,
                                  current_state_values,
                                  updated_current_state_values,
                                  self._training_batch_size)

    def _train_network_prioritized(self):
        """
        Auxiliary method used to train the Q-Network when using Prioritized Prioritized Replay

        This method does the following:
            * Sample the ER to obtain the experiences
            * Unwraps the experiences into usable forms
            * Computes the Q values for the current and next state
            * Computes the UPDATED Q values from these previous Q values (using weights)
            * Update the errors in the Experience Replay
        :return:
        """

        # Sample the ER to obtain the experiences AND the experiences IDs
        # (to be used later, to update errors)
        (sampled_experiences, sampled_errors), sampled_ids = self._experience_replay.sample_memory(self._batch_size)

        # Unwrap the experiences
        (sampled_initial_states,
         sampled_actions,
         sampled_rewards,
         sampled_next_states,
         sampled_finals) = Experience.unwrap_experiences(sampled_experiences)

        # Compute the Q values for the current state (using the Q network)
        # and the Q values for the next state (using the target network)
        current_state_values = self._q_network.predict(sampled_initial_states, self._training_batch_size)
        next_state_predictions = self._target_network.predict(sampled_next_states, self._training_batch_size)

        # Store the updated values in a different list
        updated_current_state_values = current_state_values.copy()

        # WEIGHT COMPUTING

        # Compute the probabilities of the full Experience Replay
        ranks = [(1 / x) for x in range(1, len(self._experience_replay.experience_replay) + 1)]
        # Compute the divisor of the function (the addition of all probabilities to the power of alpha)
        divisor = sum([x ** self._prioritized_alpha for x in ranks])
        # Compute the actual probability of every element when normalized
        probabilities = [(x ** self._prioritized_alpha) / divisor for x in ranks]

        # Compute the weight of all experiences
        weights = [((1/len(self._experience_replay.experience_replay)) * (1/probability)) ** self._prioritized_beta for
                   probability in probabilities]
        # Normalize all weights (dividing them by max_weight)
        weights = [weight/max(weights) for weight in weights]

        # Create a list to store the errors (to update them later)
        error_list = []

        # Compute the new Q value and error of each experience
        for index in range(len(sampled_experiences)):

            # Compute the new Q value.
            # Note that all Q values must be normalized by their weight due to Prioritized Experience Replay

            # Check if the experience is a final one or not
            if sampled_finals[index]:
                # Final experience: the Q Value of the state is simply the reward / penalty
                # q_value(t) = reward * weight
                new_q = sampled_rewards[index] * weights[sampled_ids[index]]
                updated_current_state_values[index][self._action_to_index[sampled_actions[index]]] = new_q
            else:
                # Not a final experience: the Q value is based on the obtained reward
                # and the max Q value for the next state
                # q_value(t) = (reward + gamma * next_state_best_q) * weight
                new_q = (sampled_rewards[index] + self._gamma * np.amax(next_state_predictions[index])) * weights[sampled_ids[index]]
                updated_current_state_values[index][self._action_to_index[sampled_actions[index]]] = new_q

            # Compute and append the error of the experience
            # The error is the quadratic error between the new Q value and the actually obtained one
            # error = (predicted_q - updated_q) ^ 2
            error = (current_state_values[index][self._action_to_index[sampled_actions[index]]] - new_q) ** 2
            error_list.append(error)

        # With the updated predictions, fit the Q network and update the errors in the experience replay
        self._q_network.fit_model(sampled_initial_states,
                                  current_state_values,
                                  updated_current_state_values,
                                  self._training_batch_size)
        self._experience_replay.update_errors(error_list, sampled_ids)

    ####################
    # 5 - MAIN METHODS #
    ####################

    # PRIVATE METHODS #

    def _eval_checkpoint(self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0) -> None:
        """
        Since this method is not used by this specific implementation,
        it remains unimplemented
        """

        raise NotImplementedError

    # PUBLIC METHODS #

    def save_checkpoint(self, file_name):
        """
        Saves a checkpoint (the weights of the neural network) using the specified
        filename

        :param file_name: Filename to be used when storing the checkpoint
        :type file_name: str
        """

        self._target_network.save_weights_file(self._checkpoint_folder, file_name)

    def load_checkpoint(self, checkpoint_path, *args, **kwargs):
        """
        Loads a checkpoint (the pre-trained weights of the neural network) using
        the specified filename

        NOTE: Trainers are supposed to return a Dict object (to be used with Torch),
        but None is returned in this case (since the Reactive Navigation agent uses Keras)

        :param checkpoint_path: Path to the checkpoint (pre-trained weights)
        :type checkpoint_path: str
        :param args: Positional arguments
        :param kwargs: Keyword arguments
        :return: None (for compatibility reasons)
        :rtype: None
        """

        # Load the weights for both networks
        self._q_network.load_weights_file(checkpoint_path)
        self._target_network.load_weights_file(checkpoint_path)

    def train(self):
        """
        Main method. Trains a Reactive Navigation agent using the proposed rewards systems
        using Deep Q-Learning (either standard or prioritized)

        Note that the training process used by Habitat Lab is the following:

        while training is not over (the agent hasn't reached the max number of steps or updates):
            * execute the steps in the environment (and increase the step counter for each step)
            * after the steps, update the agent (and increase the update counter once)
            * log all necessary information

        Deep Q-Learning uses the following structure for training:

        for each epoch, perceive the initial state and while the epoch is not finished:
            * choose an action randomly (exploration) or greedily (exploitation)
            * apply the action to the state and perceive the next state and reward
            * store the experience <s, a, r, s', f> into the experience replay
            * sample the experience replay
            * fit the Q network using the experiences
        """

        # Message to ensure that everything has loaded properly
        print("Starting reactive navigation training")

        # Do all the necessary pre-steps and create the log manager and the Torch device
        log_manager, device = self._init_train()

        # Store the current epsilon (chance for random action) value
        # and the linear decrease between epsilons
        current_epsilon = self._epsilon

        # Keep track of the starting training time
        starting_time = time.time()

        # Loop while the training process is not finished yet
        while not self.is_done():

            # Reset the environment to the start of the episode and get the initial state
            observations = self._env.reset()
            current_state = State.get_state(observations)

            # Track the needed variables for the log
            train_time = time.time()
            actions_taken = 0
            average_reward = 0

            # Act until the episode is finished
            # Note that the agent is trained after each step
            while not self._env.get_done(observations):

                # DEEP Q-LEARNING

                # 1 - Act according to the state
                # Note that, due to exploration-exploitation, there is a chance for a random action
                # to be performed

                # Compute the random value
                random_value = random.random()

                # Act accordingly
                if random_value < current_epsilon:
                    # Random action (exploration)
                    action = random.choice(self._agent_actions)
                else:
                    # Greedy action (exploitation)
                    action = self._q_network.act(current_state)

                # Step using the action (apply the action to the environment)
                observations, reward, done, info = self._env.step(action=action)
                new_state = State.get_state(observations)

                # 2 - Store the new information into the experience replay
                self._experience_replay.insert_experience(current_state,
                                                          action,
                                                          reward,
                                                          new_state,
                                                          done)
                average_reward += reward

                # 3 - Set the new current state
                current_state = new_state

                # 4 - Sample from the Experience Replay and train the network with said samples
                # Note that the sampling process differs between Standard and Prioritized Experience Replay
                # (and thus, will be treated separately)

                # NOTE: The sampling will only be done if the size of the Experience Replay is already
                # bigger than the batch size. This is done to speed up the training (to avoid retracing calls
                # as the sample size grows up to the batch size)

                if self._experience_replay.experience_replay_size() >= self._batch_size:
                    if not self._prioritized:
                        # Standard ER
                        self._train_network_standard()
                    else:
                        # Prioritized ER
                        self._train_network_prioritized()

                # 5 - Update the trainer step counter
                self.num_steps_done += 1
                actions_taken += 1

            # Episode is over, update the target network
            self._target_network.load_state_dict(self._q_network.state_dict())

            # Compute the time needed to finish the episode
            train_time = time.time() - train_time

            # Update the value of epsilon
            # The value of epsilon decreases linearly with the training progress
            # This formula comes from plotting a straight line between the initial and final point
            current_epsilon = abs((((self._epsilon - self._min_epsilon) / self._min_epsilon_percentage)
                                   * self.percent_done()) - self._epsilon)
            # Clamp the current epsilon to the minimum
            if current_epsilon < self._min_epsilon:
                current_epsilon = self._min_epsilon
            # If it is necessary, store a checkpoint
            if self.should_checkpoint():

                # Generate a timestamp for the checkpoint
                timestamp = datetime.datetime.fromtimestamp(self._start_time).strftime('%Y-%m-%d_%H:%M:%S')

                # Store the checkpoint and increase the counter
                self.save_checkpoint("reactive_weight_{}_{}".format(timestamp, self._checkpoint_count + 1))
                self._checkpoint_count += 1

                # Print a message to notify it
                print("Checkpoint written")

            # Track the necessary info in the log manager
            log_manager.write_episode(self.num_updates_done + 1,
                                      train_time,
                                      actions_taken,
                                      observations["pointgoal_with_gps_compass"][0],
                                      observations["pointgoal_with_gps_compass"][0] < self._goal_distance,
                                      extra_parameters=[average_reward/actions_taken])

            # Increase the update counter
            self.num_updates_done += 1

        # Compute the final training time
        starting_time = time.time() - starting_time

        # After the training process is over, close the environment and the log manager
        log_manager.close(starting_time)
        self._env.close()
