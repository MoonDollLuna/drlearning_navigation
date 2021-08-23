# REACTIVE NAVIGATION - ENVIRONMENT
# Developed by: Luna Jimenez Fernandez
#
# This file contains the definition for an environment specific for the training
# of the Reactive Navigation Agent (using our proposed algorithm)
#
# Note that this environment is a variation of the already existing NavRLEnv
# (provided by the Habitat Baselines), that is adapted to use the proposed reward systems
#
# There are two rewards systems provided, both based on attractive and repulsive fields:
#   * Contour - Based on the original paper, all detected contours in the image are considered as obstacles
#               towards the repulsive field value
#   * Column  - The image is divided into several columns of equal size, with each column being a possible
#               obstacle. Each column is individually evaluated to check if an obstacle is there.
#
# Both algorithms are explained in more detail below

# IMPORTS #
import math
import cv2
import numpy as np

from numpy import ndarray
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import NavRLEnv


@baseline_registry.register_env(name="ReactiveNavEnv")
class ReactiveNavigationEnv(NavRLEnv):
    """
    This class contains the specification of the environment to be used by the Reactive Navigation Agent.

    This is essentially a sub-class of NavRLEnv that has been modified to:
        * Use the config files designed for the agent
        * Use the proposed reward systems when training the agent

    The agent has also been configured to work within the Habitat Lab mainframe, being registered within the
    baseline registry.
    """

    # ATTRIBUTES #

    # IMAGE PROCESSING ATTRIBUTES
    # Pixels to be trimmed from the bottom of the image (the image provided by the depth camera)
    # This parameter is a heuristic used since our robot is an embodied agent that will always see the floor
    # at a constant height (so it can be trimmed to avoid detecting false obstacles)
    _bottom_trim: int
    # Maximum color threshold for obstacles within the image. Objects with a color below the threshold are considered
    # obstacles, while objects with a color above the threshold are not.
    # Note that the color goes from 0.0 (pure black) to 1.0 (pure white).
    _obstacle_threshold: float
    # Minimum area (in pixels) for contours / columns. Contours / columns with less than this amount of pixels
    # will be ignored (to avoid false obstacles)
    _min_contour_area: int
    # (ONLY FOR COLUMN BASED REWARDS)
    # Number of columns to divide the image into. All columns will have equal widths, while the height will remain.
    _reward_columns: int

    # REWARD COMPUTING ATTRIBUTES
    # Reward methods. As described above, two rewards method exist:
    #   * contour: Contour based approach that translates the original paper proposal (using a laser array)
    #              into a depth camera based approach.
    #   * column : Approach that divides the image into several columns, considering each column a potential
    #              obstacle. This was proposed as a way to give bigger obstacles more weight (as they would
    #              fill several columns instead of counting as a single contour)
    _reward_method: str
    # Approximate distance (in simulator units) at which an obstacle is when exactly at _obstacle_threshold distance
    _obstacle_distance: float

    # Obstacle mercy steps. The agent will ignore obstacle checks for this amount of steps
    # This solves the problem of agents spawning right next to obstacles immediately ending episodes
    _obstacle_mercy_steps: int
    # Counter for the mercy steps
    _mercy_counter: int

    # Positive gain applied to the attractive field (used to increase its weight)
    _attraction_gain: float
    # Positive gain applied to the repulsive field (used to increase its weight)
    _repulsive_gain: float
    # Limit applied to the repulsive field's weight. The bigger it is, the smaller the influence is
    _repulsive_limit: float
    # Percentage (between 0.0 and 1.0) of _obstacle_distance. When the agent's distance to the goal is smaller
    # than (_repulsive_goal_influence * _obstacle_distance), the influence of the repulsive field is reduced
    _repulsive_goal_influence: float
    # Reward given for a successful episode. All positive rewards will be clipped to this value
    _success_reward: float
    # Penalty given for a failed episode. All negative rewards will be clipped to this value
    _failure_penalty: float
    # Goal distance (goal at which the agent is considered to be at the goal)
    _goal_distance: float

    # REWARD ATTRIBUTES (NOT PROVIDED BY THE CONFIG FILE)
    # Value of the previous shaping, used to compute a reward for each step
    _previous_shaping: float

    # CONSTRUCTOR #

    def __init__(self, config, dataset=None):
        """
        Constructor of the environment. All attributes are passed as a config file.

        This environment is specifically designed to be used with the Reactive Navigation Agent,
        in order to provide the necessary reward calculations.

        :param config: Config class, containing the necessary parameters for this environment.
        :type config: Config
        :param dataset: (OPTIONAL) Dataset to use the environment on.
        :type dataset: Dataset
        """

        # Store the necessary values from the Config
        _rl_config = config.RL
        _image_config = _rl_config.IMAGE
        _reward_config = _rl_config.REWARD

        self._bottom_trim = _image_config.bottom_trim
        self._obstacle_threshold = _image_config.obstacle_threshold
        self._min_contour_area = _image_config.min_contour_area
        self._reward_columns = _image_config.reward_columns

        self._reward_method = _reward_config.reward_method
        self._obstacle_distance = _reward_config.obstacle_distance

        # NOTE: Obstacle mercy steps are stored as one extra than indicated
        # This is since mercy steps are decreased BEFORE the actual step
        # (so, for example, 6 mercy steps would translate for 5 actual steps of mercy)
        self._obstacle_mercy_steps = _reward_config.obstacle_mercy_steps + 1

        self._attraction_gain = _reward_config.attraction_gain
        self._repulsive_gain = _reward_config.repulsive_gain
        self._repulsive_limit = _reward_config.repulsive_limit
        self._repulsive_goal_influence = _reward_config.repulsive_goal_influence
        self._success_reward = _reward_config.success_reward
        self._failure_penalty = _reward_config.failure_penalty
        self._goal_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE

        # Construct the super parent
        # Parent needs to be constructed AFTER attribute declaration to avoid null references
        super().__init__(config, dataset)

    # PRIVATE METHODS #

    def _process_image(self, depth_image):
        """
        Given an image (obtained from the depth camera), process it to obtain a binary image
        containing exclusively the detected obstacles by the camera

        The initial, trimmed depth image is also returned to be used by other methods

        :param depth_image: Image provided by the depth camera (with each pixel ranging
                            from 0.0 - black to 1.0 - white)
        :type depth_image: ndarray
        :return: A vector of two images: A binary image processed to only contain the obstacles closer to the agent /
                 The original depth image, trimmed
        :rtype: vector
        """

        original_image = np.copy(depth_image)

        # STEP 1 - Normalize the image to the range of [0, 255] to properly work with OpenCV
        original_image = original_image * 255
        original_image = original_image.astype(np.uint8)

        # STEP 2 - Trim the bottom of the image (to avoid the floor interfering)
        trimmed_image = original_image[0:255 - self._bottom_trim, :]

        # STEP 3 - Fill pure black values (0) with pure white (255) values
        # This is done to avoid visual glitches, since pure black typically correlates with
        # bugs in the simulator
        filled_image = trimmed_image
        filled_image[filled_image == 0] = 255

        # STEP 4 - Threshold the image
        # All pixels with a value lower than (_obstacle_threshold * 255) will be kept with a value of 1 (OBSTACLES),
        # while all other pixels will be changed to 0s. This way, only obstacles are kept.
        # This is done using an inverse binary threshold
        thresholded_image = cv2.threshold(filled_image,
                                          int(self._obstacle_threshold * 255),
                                          255, cv2.THRESH_BINARY_INV)[1]

        # STEP 5 - Remove noise, using morphological transformation.
        # Specifically, an opening morphological transformation (erosion, followed by dilation) is used.
        # The kernel to use is a rectangle of shape (3, 3)
        cleaned_image = cv2.morphologyEx(thresholded_image,
                                         cv2.MORPH_OPEN,
                                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        # STEP 6 - Increase the size of the objects in the image (to remove possible holes).
        # This is done with a dilation morphological transformation
        dilated_image = cv2.dilate(cleaned_image,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        return dilated_image, trimmed_image

    def _identify_obstacle_distances_contour(self, depth_image, processed_image):
        """
        Given a pre-processed image (see _process_image for more info),
        identifies the obstacles in the image and provides the distance to each one.

        For the "contour" reward method, a contouring technique is used to identify all objects
        in the image above a size. For each of these objects, the closest non-zero distance
        is used to identify how close it is to the camera.

        Note that non-zero values are used since pure-zero values (pitch black) usually signify
        visual glitches in the image.

        :param depth_image: Original (trimmed) image from the depth camera
        :type depth_image: ndarray
        :param processed_image: A pre-processed image from the depth camera
        :type processed_image: ndarray
        :return: An array of the distance to each obstacle in the image
        :rtype: collections.Iterable
        """

        distances = []

        # Identify the contours in the image
        # In this case, RETR_LIST is used (since hierarchies are not relevant)
        # Not all points are stored (CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(processed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # All contours are evaluated separately
        for cont in contours:

            # Ignore contours that do not have a minimum size
            if cv2.contourArea(cont) > self._min_contour_area:

                # Create a mask based on the contour, to isolate the obstacle in the image
                mask = np.zeros(processed_image.shape, np.uint8)
                mask = cv2.drawContours(mask, [cont], -1, (255, 255, 255), -1)
                masked_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)

                # From that image, grab the closest distance and convert it from [0, 255] to
                # a distance used by the simulator
                # Note that pure 0 values are ignored (since they're usually image glitches)
                distance = np.min(masked_image[np.nonzero(masked_image)])                   # In [0, 255]
                distance = distance / 256.0                                                 # In [0.0, 1.0]
                distance = (distance * self._obstacle_distance) / self._obstacle_threshold  # In simulator units

                # Ignore contours that are further than the threshold
                if distance < self._obstacle_distance:
                    distances.append(distance)

        return distances

    def _identify_obstacle_distances_columns(self, depth_image, processed_image):
        """
        Given a pre-processed image (see _process_image for more info),
        identifies the obstacles in the image and provides the distance to each one.

        For the "column" reward method, the image is split into a number of columns
        (sub-images, with each column having the original image height and all images having equal width).
        Instead of identifying contours on the whole image, the number of obstacle pixels (pixels with value 1)
        is counted, and if a big enough number is reached, the whole column is considered an obstacle.

        The idea is to avoid image processing techniques (to save time), and to give bigger obstacles a
        bigger weight (since very big obstacles may count as a single contour, while they may fill several columns)

        :param depth_image: Original (trimmed) image from the depth camera
        :type depth_image: ndarray
        :param processed_image: A pre-processed image from the depth camera
        :type processed_image: ndarray
        :return: An array of the distance to each obstacle column in the image
        :rtype: array
        """

        # Split both images into columns of equal width
        depth_columns = np.array_split(depth_image, self._reward_columns, axis=1)
        processed_columns = np.array_split(processed_image, self._reward_columns, axis=1)

        distances = []

        # Process each column independently
        for index, column in enumerate(processed_columns):

            # Ignore columns that do not have a minimum amount of "obstacle pixels"
            # (values different from 0)
            if np.count_nonzero(column) > self._min_contour_area:

                # From that column, grab the closest distance and convert it from [0, 255] to
                # a distance used by the simulator
                # Note that pure 0 values are ignored (since they're usually image glitches)
                depth_column = depth_columns[index]
                distance = np.min(depth_column[np.nonzero(depth_column)])                       # In [0, 255]
                distance = distance / 256.0                                                     # In [0.0, 1.0]
                distance = (distance * self._obstacle_distance) / self._obstacle_threshold      # In simulator units

                # Ignore columns that are further than the threshold
                if distance < self._obstacle_distance:
                    distances.append(distance)

        return distances

    def _compute_shaping_value(self, goal_distance, depth_image):
        """
        Given the the distance to the goal provided by the GPS and
        an image provided by the depth camera for the current step, compute a shaping value
        (a value assigned to the state).

        This value is computed based on two fields:
            * Attractive field: A force exerted by the goal, that drives the agent TOWARDS the goal
            * Repulsive field: A force exerted by all obstacles, that drives the agent AWAY from the obstacles.

        The shaping is computed as:
        shaping = -attractive_field - repulsive_field

        The reward is computed as:
        reward = shaping (current state) - shaping (previous state)

        :param goal_distance: Distance to the goal (provided by the GPS)
        :type goal_distance: float
        :param depth_image: Image provided by the depth camera (with each pixel ranging
                            from 0.0 - black to 1.0 - white)
        :type depth_image: ndarray
        :return: The shaping value for the current state
        :rtype: float
        """

        # ATTRACTIVE
        # Compute the ATTRACTIVE field force, with the following formula:
        # attractive_field = attraction_gain * distance_to_goal
        attractive_field = self._attraction_gain * goal_distance

        # REPULSIVE
        # Pre-process the depth image to obtain a thresholded and cleaned image
        processed_image, trimmed_image = self._process_image(depth_image)

        # Compute the distances to the obstacles
        # This is done differently depending on the rewards method
        # CONTOUR
        if self._reward_method == "contour":
            distances = self._identify_obstacle_distances_contour(trimmed_image, processed_image)
        # COLUMN
        else:
            distances = self._identify_obstacle_distances_columns(trimmed_image, processed_image)

        # Compute the value of beta, according to the distance to the goal
        # beta = repulsive_gain when goal_distance > repulsive_goal_influence * obstacle_distance
        # beta = repulsive_gain / e^(4 * ((repulsive_goal_influence * obstacle_distance) - goal_distance))
        # otherwise
        goal_influence = self._repulsive_goal_influence * self._obstacle_distance

        if goal_distance > goal_influence:
            beta = self._repulsive_gain
        else:
            beta = self._repulsive_gain / math.exp(4 * (goal_influence - goal_distance))

        # Compute the value of the REPULSIVE field with the following formula
        # repulsive_field = beta * (SUMMATORY FOR EACH DISTANCE TO AN OBSTACLE)
        # ((1 / (repulsive_limit + distance_to_obstacle)) - (1 / (repulsive_limit + obstacle_distance)))
        repulsive_field = 0
        for distance in distances:
            repulsive_field += ((1 / (self._repulsive_limit + distance)) -
                                (1 / (self._repulsive_limit + self._obstacle_distance)))

        repulsive_field = repulsive_field * beta

        # SHAPING
        # Compute and return the final shaping
        # shaping = -attractive_field - repulsive_field
        shaping = -attractive_field - repulsive_field

        return shaping

    def _check_obstacle_collision(self, observations):
        """
        Determine whether the agent is "colliding" (too close) to an obstacle

        An agent is considered to be colliding when its distance to the closest obstacle
        is less than goal_distance

        If the agent collides with an obstacle, the episode is immediately finished

        Episodes have a "mercy period": Obstacle collisions are ignored for the obstacle_mercy
        steps

        :param observations: Observations from the environment taken by the agent
        :type observations: Observations
        :return: TRUE if the agent is colliding, false otherwise
        :rtype: bool
        """

        # Check if the mercy steps are still in effect
        if self._mercy_counter > 0:
            # Mercy active: collisions are not checked
            return False

        # Extract the depth view from the observations
        depth_view = observations["depth"]

        # Find the closest non-zero distance in the image
        try:
            closest_distance = np.min(depth_view[np.nonzero(depth_view)])
        except ValueError:
            # Sanity check: if all values are 0, assume a collision
            return True

        # Convert the distance from [0, 1] to actual distance
        distance = (closest_distance * self._obstacle_distance) / self._obstacle_threshold

        # Check if the distance is smaller than goal_distance
        return distance < self._goal_distance

    # PUBLIC METHODS #

    def reset(self):
        """
        Sets up the agent and the environment at the start of an episode.

        In addition to the standard reset procedures, this also computes the initial shaping value

        :return: A dictionary containing all initial observations
        :rtype: Observations
        """

        # Reset the mercy counter
        self._mercy_counter = self._obstacle_mercy_steps

        # Set up everything (provided by the superclass NavRLEnv)
        # In addition, get the initial observations
        initial_observations = super().reset()

        # Compute the initial shaping value
        self._previous_shaping = self._compute_shaping_value(initial_observations["pointgoal_with_gps_compass"][0],
                                                             initial_observations["depth"])

        return initial_observations

    def step(self, *args, **kwargs):
        """
        Steps through the environment

        Overridden method to decrease the mercy counter BEFORE each step.
        If the mercy counter was overridden after the step, there could be situations
        where the agent finishes the episode but doesn't know until later
        """

        # Decrease the mercy counter (and ensure it doesn't go below 0)
        self._mercy_counter -= 1
        if self._mercy_counter < 0:
            self._mercy_counter = 0

        # Perform the normal step process
        return super().step(*args, **kwargs)

    def get_done(self, observations):
        """
        Checks if the episode has finished already

        Method overridden to add a third additional ending condition: obstacles

        :param observations: Observations from the environment taken by the agent
        :type observations: Observations
        :return: TRUE if the episode is over, FALSE otherwise
        :rtype: bool
        """

        return super().get_done(observations) or self._check_obstacle_collision(observations)

    def get_reward_range(self):
        """
        Returns the range of possible rewards as a tuple (lowest_value, highest_value)

        :return: Tuple containing (lowest_reward, highest_reward)
        :rtype: vector
        """

        return self._failure_penalty, self._success_reward

    def get_reward(self, observations):
        """
        Returns a reward based on the state achieved. Rewards work as follow:

            * Episode ended successfully: success_reward
            * Episode ended in a failure: failure_penalty
            * Otherwise: shaping(current) - shaping(previous)

        Rewards will be clamped to the range [failure_penalty, success_reward]

        :param observations: Observations from the environment
        :type: dict
        :return: A reward for the state-action
        :rtype: float
        """

        # Check if the episode is over
        if self.get_done(observations):
            # Check if the episode was a success
            if self._env.get_metrics()[self._success_measure_name]:
                # SUCCESS
                return self._success_reward
            else:
                # FAILURE
                return self._failure_penalty

        # If the episode is not over, the new shaping needs to be computed
        shaping = self._compute_shaping_value(observations["pointgoal_with_gps_compass"][0],
                                              observations["depth"])

        # Compute and clamp the reward
        reward = shaping - self._previous_shaping
        reward = max(min(reward, self._success_reward), self._failure_penalty)

        # Update the shaping value
        self._previous_shaping = shaping

        return reward
