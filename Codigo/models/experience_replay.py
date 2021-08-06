# REACTIVE NAVIGATION - EXPERIENCE REPLAY
# Developed by: Luna Jimenez Fernandez
#
# This file contains all classes used to create and manage the Experience Replay
# (a queue of experiences) used by Deep Q-Learning during the agent training

# IMPORTS #
from numpy import ndarray
from collections import deque
from heapq import heappush, heappop, heapify


class State:
    """
    State represents a state of the world as understood by the Reactive Navigation agent.

    A state contains the following elements:
        * The distance to the goal (float)
        * The angle to the goal in radians (float)
        * An image perceived by the depth camera (ndarray)
    """

    # ATTRIBUTES #

    # Distance to the goal (in simulator units)
    _distance: float
    # Angle to the goal (in radians)
    _angle: float
    # View of the world (as perceived by the depth camera)
    _image: ndarray

    # CONSTRUCTOR #

    def __init__(self, distance, angle, image):
        """
        Creates a state (representing the current knowledge of the world as understood by the agent)

        :param distance: Distance to the goal (in simulator units)
        :type distance: float
        :param angle: Angle to the goal (in radians)
        :type angle: float
        :param image: Image obtained from the depth camera (without any processing)
        :type image: ndarray
        """

        # Store all values
        self._distance = distance
        self._angle = angle
        self._image = image

    # PUBLIC METHODS #

    def unwrap_state(self):
        """
        Unwraps the state into a form that can be directly used to train the neural networks

        The state is returned with shape [image, [distance, angle]]

        :return: A list containing all elements of the state, as described above
        :rtype: list
        """

        return [self._image, [self._distance, self._angle]]



class Experience:
    """
    An Experience is a memory that an agent stores into the Experience Replay, containing
    the following information:
        * The original state of the agent (s)
        * The action taken by the agent at state s (a)
        * The reward obtained by performing action a at state s (r)
        * The state reached by applying action a at state s (s')

    Thus, an experience has the structure <s, a, r, s'>
    """

    # ATTRIBUTES #

    # Initial state of the agent (s)
    _initial_state: State
    # Action performed by the agent in state s (a)
    _action: str
    # Reward obtained by the agent after performing an action a in state s (r)
    _reward: float
    # State reached after performing an action a in state s (s')
    _next_state: State

    # CONSTRUCTOR

    def __init__(self, initial_state, action, reward, next_state):
        """
        Creates an experience with structure <initial state, action, reward, next state> to be
        stored in the Experience replay

        :param initial_state: Initial state of the agent
        :type initial_state: State
        :param action: Action taken by the agent in initial_state
        :type action: str
        :param reward: Reward obtained by the agent when applying action in initial_state
        :type reward: float
        :param next_state: State reached after applying action in initial_tate
        :type next_state: State
        """

        # Store the parameters
        self._initial_state = initial_state
        self._action = action
        self._reward = reward
        self._next_state = next_state


class ExperienceReplay:
    pass


class PrioritizedExperienceReplay:
    pass
