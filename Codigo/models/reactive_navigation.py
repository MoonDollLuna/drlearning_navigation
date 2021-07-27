# REACTIVE NAVIGATION - MODEL
# Developed by: Luna Jimenez Fernandez
#
# This file contains the Reactive Navigation main model, used by both the
# reactive navigation trainer (for Reinforcement Learning) and the
# reactive navigation agent (for evaluation using a Benchmark)
#
# The model consists of a CNN with the following structure:
#
# TODO: [ STRUCTURE ]
# This structure has been developed ad-hoc
#
# The input of the CNN is as follows:
#   * A 256x256 (shape can be specified) grayscale image (obtained from the Depth Camera)
#   * A pair of scalar values, Distance and Angle to the goal (obtained from the GPS)
#
# The output of the CNN is four neurons, each representing the Q-Value of a pair of state-action:
#   * STOP
#   * MOVE FORWARD
#   * TURN RIGHT
#   * TURN LEFT
#

# IMPORTS #

# Keras
from keras import Input
from keras.models import Model

# Layers
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate


class ReactiveNavigationModel:
    """
    The proposed Reactive Navigation model, using a CNN (trained using Deep Q-Learning)
    to allow an embodied agent to travel through an interior environment (such as a house).
    """
    # TODO: ACABA

    _image_size: int
    _action_list: list
    _cnn: Model

    # CONSTRUCTOR #

    def __init__(self, image_size, action_list):
        """
        Constructor method

        :param image_size: Size of the image used by the CNN
        :type image_size: int
        :param action_list: List of available actions
        :type action_list: list
        """

        # Store the given values
        self._image_size = image_size
        self._action_list = action_list

        # Prepare the CNN
        self._initialize_cnn(self._image_size, len(self._action_list))

    # INTERNAL METHODS #

    def _initialize_cnn(self, image_size, action_size):
        """
        Initializes the Convolutional Neural Network (CNN) used by the model.

        The CNN has the following structure (in order):
            * Input of the image (in grayscale, with shape image_size x image_size)
            * First layer of convolution:
                * Two convolutional layers with 32 3x3 filters (2D, using ReLU)
                * A pooling layer (Max-Pool of size 3)
            * Second layer of convolution:
                * Two convolutional layers with 64 3x3 filters (2D, using ReLU)
                * A pooling layer (Max-Pool of size 2)
            * A Merge layer where the result of the previous convolution is joined with
              the scalar inputs (Distance and Angle to the goal)
            * Two fully connected layers (ReLU) of 1024 neurons
            * action_size neurons (Linear)

        The CNN uses the following hyperparameters:
            * Optimizer: Adam
            * Error: Mean Squared Error

        Note that, since the CNN will be used for reinforcement learning, no dropout will be used

        :param image_size: Size of the image. The shape of the image will be (image_size x image_size)
        :type image_size: int
        :param action_size: Number of actions available for the agent
        :type action_size: int
        """

        # All layers are randomly initialized using Glorot initializer

        # Create the Inputs of the Neural Network
        image_input = Input(shape=(image_size, image_size))
        scalar_input = Input(shape=(2,))

        # Create the first layers of convolution
        # (Convolution: 32 3x3 filters, ReLU)
        # (Pooling: Max, 3x3)
        conv1_1 = Conv2D(filters=32,
                         kernel_size=3,
                         activation="relu",
                         kernel_initializer="glorot_uniform")(image_input)
        conv1_2 = Conv2D(filters=32,
                         kernel_size=3,
                         activation="relu",
                         kernel_initializer="glorot_uniform")(conv1_1)
        pool1 = MaxPooling2D(pool_size=3)(conv1_2)

        # Create the second layers of convolution
        # (Convolution: 64 3x3 filters, ReLU)
        # (Pooling: Max, 2x2)
        conv2_1 = Conv2D(filters=64,
                         kernel_size=3,
                         activation="relu",
                         kernel_initializer="glorot_uniform")(pool1)
        conv2_2 = Conv2D(filters=64,
                         kernel_size=3,
                         activation="relu",
                         kernel_initializer="glorot_uniform")(conv2_1)
        pool2 = MaxPooling2D(pool_size=2)(conv2_2)

        # Flatten the input, so it can be used with dense layers
        flatten = Flatten()(pool2)

        # Merge the results of the convolutional layers with the scalar input
        merge = concatenate([flatten, scalar_input])

        # Create the dense layers
        # (1024 neurons, ReLU)
        dense1 = Dense(1024,
                       activation="relu",
                       kernel_initializer="glorot_uniform")(merge)
        dense2 = Dense(1024,
                       activation="relu",
                       kernel_initializer="glorot_uniform")(dense1)

        # Create the output layer (action_size outputs, Lineal)
        # Note that the output MUST be lineal (instead of the typical sigmoid function)
        # for Deep Reinforcement Learning
        output = Dense(action_size,
                       activation="lineal",
                       kernel_initializer="glorot_uniform")(dense2)

        # Create and compile the model of the full CNN (Adam optimizer, MSE)
        # Mean Square Error is used (instead of more typical cross-entropy values) due to Deep Reinforcement Learning
        # (since MSE is the value trying to be minimized)
        model = Model(inputs=[image_input, scalar_input],
                      outputs=output)
        model.summary()
        model.compile(optimizer="adam",
                      loss="mse")

        # Store the model
        self._cnn = model

    # PUBLIC METHODS
    # TODO: ACABA
    def train_model(self):
        pass
