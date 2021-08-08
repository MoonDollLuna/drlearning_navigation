# REACTIVE NAVIGATION - EXPERIENCE REPLAY
# Developed by: Luna Jimenez Fernandez
#
# This file contains all classes used to create and manage the Experience Replay
# (a queue of experiences) used by Deep Q-Learning during the agent training
#
# More specifically, it contains:
#   * STATE: A definition of a state as understood by our system (distance, angle and image)
#   * EXPERIENCE: A wrapper for the experiences stored within the Experience Replay
#   * EXPERIENCE REPLAY (ER): The structure used to control the Experience Replay memory, including
#                             insertions, samplings...
#   * PRIORITIZED EXPERIENCE REPLAY (PER): A variation of ER that uses prioritized sampling, to
#                                          improve performance by sampling more useful experiences

# IMPORTS #
import random
import math
import numpy as np

from numpy import ndarray
from collections import deque


class State:
    """
    State represents the knowledge the Reactive Navigation agent has about the world in an specific instant.

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

        A state contains the following elements:
            * The distance to the goal (float)
            * The angle to the goal in radians (float)
            * An image perceived by the depth camera (ndarray)

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

    In addition, an experience stores information about whether a reached state is final or not (f)

    Thus, an experience has the structure <s, a, r, s', f>
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
    # Indicates whether a reached state is final (f)
    _final: bool

    # CONSTRUCTOR

    def __init__(self, initial_state, action, reward, next_state, final):
        """
        Creates an experience with structure <initial state, action, reward, next state, final> to be
        stored in the Experience replay

        :param initial_state: Initial state of the agent
        :type initial_state: State
        :param action: Action taken by the agent in initial_state
        :type action: str
        :param reward: Reward obtained by the agent when applying action in initial_state
        :type reward: float
        :param next_state: State reached after applying action in initial_tate
        :type next_state: State
        :param final: Whether the next state is final (TRUE) or not (FALSE)
        :type final: bool
        """

        # Store the parameters
        self._initial_state = initial_state
        self._action = action
        self._reward = reward
        self._next_state = next_state
        self._final = final


class ExperienceReplay:
    """
    Experience Replay is a queue-based data structure where Deep Q-Learning stores
    the experiences made by an agent during training. From these memory, experiences
    are sampled to train the agent

    The main data structure behind the Experience Replay is a FIFO queue with a maximum
    size (specified during construction), where the experiences are stored. Once the queue
    fills, the older experiences will be removed

    In addition, experiences are sampled randomly without replacement, using a uniform distribution
    (all experiences have the same probability of being chosen)
    """

    # ATTRIBUTES #
    # Queue where the experiences of the agent are stored
    # This queue has a maximum size specified during construction
    _experience_replay: deque

    # CONSTRUCTOR #
    def __init__(self, max_size=None):
        """
        Initializes the Experience Replay with the specified maximum size

        If not specified, the queue will have no maximum size (not recommended)

        :param max_size: Maximum size of the queue. The oldest elements will be removed from the queue
                         if it is filled
        :type max_size: int
        """
        
        # Construct the queue
        self._experience_replay = deque(maxlen=max_size)

    # PUBLIC METHODS #
    def insert_experience(self, initial_state, action, reward, next_state, final):
        """
        Creates and inserts a new experience into the Experience Replay

        :param initial_state: Initial state of the agent
        :type initial_state: State
        :param action: Action taken by the agent in initial_state
        :type action: str
        :param reward: Reward obtained by the agent when applying action in initial_state
        :type reward: float
        :param next_state: State reached after applying action in initial_tate
        :type next_state: State
        :param final: Whether the next state is final (TRUE) or not (FALSE)
        :type final: bool
        """

        # Create the experience and enqueue it
        experience = Experience(initial_state, action, reward, next_state, final)
        self._experience_replay.append(experience)

    def sample_memory(self, sample_size):
        """
        Takes a sample from the Experience Replay of size sample_size

        This sample is taken without replacement, using simple random sampling
        (all experiences have the same probability of being sampled)

        :param sample_size: Size of the sample to be taken
        :type sample_size: int
        :return: A list of Experience items
        :rtype: list
        """

        # Ensure that the size of the sample is not larger than the current memory
        if len(self._experience_replay) < sample_size:
            size = len(self._experience_replay)
        else:
            size = sample_size

        # Generate and return the memories
        return random.sample(self._experience_replay, size)


class PrioritizedExperienceReplay:
    """
    Prioritized Experience Replay is a proposed improvement of the original Experience Replay,
    with the following key differences:
        * Experiences are stored with an ERROR (the difference between the obtained and the expected Q-value)
        * Experiences with more error have a higher probability of being sampled
        * ALL experiences can still be sampled (to avoid overfitting)

    In this case, while Experiences follow the structure <initial state, action, reward, next state, final>
    as usual, entries in the Experience Replay are tuples with the following values:
        * The Experience (with <s, a, r, s', f> structure)
        * The quadratic error of said experience (difference between the expected and the actual value)

    The probability of an Experience being sampled depends on its error (the higher the error is, the higher the
    probability is). This is also regulated via two variables:
        * Alpha: Priority degree. The higher alpha is the higher the probability of choosing higher error experiences is
        * Beta: Bias degree. The higher the value is the less weight variations have (to avoid big oscillations)

    This variant offers better results, since agents will learn more from more important experiences
    (instead of just randomly sampling experiences which may or may not be relevant for the agent)
    """

    # ATTRIBUTES #

    # Queue where the experiences of the agent are stored
    # This queue has a maximum size specified during construction
    #
    # Note that a Priority Queue is not used since we want to conserve the original functionality
    # (experiences keep their order, and once the ER is full the oldest experiences are removed)
    _experience_replay: deque

    # Alpha value (priority degree). The higher alpha is, the higher the probability of choosing
    # higher error experiences is
    _alpha: float

    # Beta value (bias degree). The higher the value is, the less weight variations have
    # (to avoid big oscillations)
    _beta: float

    # CONSTRUCTOR #

    def __init__(self, max_size, alpha, beta):
        """
        Initializes the Experience Replay with the specified maximum size and degrees

        If not specified, the queue will have no maximum size (not recommended)

        :param max_size: Maximum size of the queue. The oldest elements will be removed from the queue
                         if it is filled
        :type max_size: int
        :param alpha: Alpha value (priority degree). The higher alpha is, the higher the probability of choosing
                    higher error experiences is
        :type alpha: float
        :param beta: Beta value (bias degree). The higher the value is, the less weight variations have
                     (to avoid big oscillations)
        :type beta: float
        """

        # Initialize the queue
        self._experience_replay = deque(maxlen=max_size)

        # Store the parameters
        self._alpha = alpha
        self._beta = beta

    # PUBLIC METHODS #

    def insert_experience(self, initial_state, action, reward, next_state, final):
        """
        Creates and inserts a new experience into the Experience Replay

        Note that stored experiences have the following structure:
            * The experience <s, a, r, s', f>
            * The quadratic error of said experience

        :param initial_state: Initial state of the agent
        :type initial_state: State
        :param action: Action taken by the agent in initial_state
        :type action: str
        :param reward: Reward obtained by the agent when applying action in initial_state
        :type reward: float
        :param next_state: State reached after applying action in initial_tate
        :type next_state: State
        :param final: Whether the next state is final (TRUE) or not (FALSE)
        :type final: bool
        """

        # Create and store the structure for the experience
        # Experiences are initialized with infinite error
        experience = (Experience(initial_state, action, reward, next_state, final), math.inf)
        self._experience_replay.append(experience)

    def sample_memory(self, sample_size):
        """
        Takes a sample from the Experience Replay of size sample_size

        This sample is taken without replacement, using rank based prioritized sampling.
        The following formula is used:

        P(i) = p(i)^alpha / summatory of all p(i)^alpha

        where p(i) = 1 / rank(i); the rank of experience i when ordered by error

        :param sample_size: Size of the sample to be taken
        :type sample_size: int
        :return: A tuple with two lists, a list of Experience / Error items and a list of the indexes of said items
        :rtype: tuple
        """

        # PROBABILITIES

        # Compute the probability of each element based on its rank
        # Since experiences will be ordered (from the highest to the lowest error) it can be computed directly
        ranks = [(1/x) for x in range(1, len(self._experience_replay) + 1)]

        # Compute the divisor of the function (the addition of all probabilities to the power of alpha)
        divisor = sum([x ** self._alpha for x in ranks])

        # Compute the actual probability of every element when normalized
        probabilities = [(x ** self._alpha) / divisor for x in ranks]

        # SAMPLING

        # Ensure that the size of the sample is not larger than the current memory
        if len(self._experience_replay) < sample_size:
            size = len(self._experience_replay)
        else:
            size = sample_size

        # Sample the ids and return both experiences and ids
        id_samples = np.random.choice(np.arange(0, len(self._experience_replay)),
                                      size,
                                      False,
                                      probabilities)
        # The deque is temporarily converted into a list to simplify the process
        deque_list = np.array(list(self._experience_replay))

        return deque_list[id_samples].tolist(), id_samples.tolist()

    def update_errors(self, errors):
        pass
        # TODO: ACABA ESTE METODO CUANDO SE NECESITE
