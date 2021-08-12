# REACTIVE NAVIGATION - MODEL
# Developed by: Luna Jimenez Fernandez
#
# This file contains the Reactive Navigation main model, used by both the
# reactive navigation trainer (for Reinforcement Learning) and the
# reactive navigation agent (for evaluation using a Benchmark)
#
# The model consists of a CNN with the following structure:
#
#     * Input of the image (in grayscale, with shape image_size x image_size)
#     * First layer of convolution:
#         * Two convolutional layers with 32 3x3 filters (2D, using ReLU)
#         * A pooling layer (Max-Pool of size 3)
#     * Second layer of convolution:
#         * Two convolutional layers with 64 3x3 filters (2D, using ReLU)
#         * A pooling layer (Max-Pool of size 2)
#     * A Merge layer where the result of the previous convolution is joined with
#       the scalar inputs (Distance and Angle to the goal)
#     * Two fully connected layers (ReLU) of 1024 neurons
#     * Output neurons (Linear) equal to the number of actions
#
# This structure has been developed ad-hoc
#
# The inputs of the CNN are as follows:
#   * A 256x256 (shape can be specified) grayscale image (obtained from the Depth Camera)
#   * A pair of scalar values, Distance and Angle to the goal (obtained from the GPS)
#
# The output of the CNN is four neurons, each representing the Q-Value of a pair of state-action:
#   * STOP
#   * MOVE FORWARD
#   * TURN RIGHT
#   * TURN LEFT

# IMPORTS #
from os.path import join

# Keras
from keras import Input
from keras.models import Model
from keras.optimizers import Adam

# Layers
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate


class ReactiveNavigationModel:
    """
    The proposed Reactive Navigation model, using a CNN (trained using Deep Q-Learning)
    to allow an embodied agent to travel through an interior environment (such as a house).

    The model takes a grayscale image (taken from a depth camera) and the distance and angle
    to the goal (provided by the GPS) as inputs, outputting the actual action that needs to be
    performed.

    In addition, the model is prepared for batch training.
    """

    # ATTRIBUTES #

    # Size of the image. Images have a shape of (_image_size x _image_size)
    _image_size: int
    # Dictionary containing each action (str, key) with its assigned neuron (int, value)
    # Example: "stop" => 0
    _action_to_int_dict: dict
    # Dictionary containing each neuron (int, key) with its assigned action (str, value)
    # Example: 0 => "stop"
    _int_to_action_dict: dict
    # CNN contained by the model, to be trained and used
    _cnn_model: Model

    # CONSTRUCTOR #

    def __init__(self, image_size, action_list, learning_rate=None, weights=None):
        """
        Constructor method

        :param image_size: Size of the image used by the CNN
        :type image_size: int
        :param action_list: List of available actions
        :type action_list: list
        :param learning_rate: (OPTIONAL) Learning rate of the neural network
        :type learning_rate: float
        :param weights: (OPTIONAL) Path to the file containing the pre-trained weights of the CNN
        :type weights: str
        """

        # Store the given image size
        self._image_size = image_size

        # Generate the dictionaries from the action list
        self._action_to_int_dict, self._int_to_action_dict = self._initialize_dicts(action_list)

        # Prepare the CNN and, if available, load the weights
        self._cnn_model = self._initialize_cnn(self._image_size, len(action_list), learning_rate)
        if weights is not None:
            self._cnn_model.load_weights(weights)

    # INTERNAL METHODS #

    @staticmethod
    def _initialize_cnn(image_size, action_size, learning_rate=None):
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
        :return: A CNN with the specified structure, with random initial weights
        :rtype: Model
        :param learning_rate: (OPTIONAL) Learning rate of the neural network
        :type learning_rate: float
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
        # TODO ESTO ES DEBUG
        model.summary()
        model.compile(optimizer=Adam(learning_rate=learning_rate) if learning_rate else "adam",
                      loss="mse")

        return model

    @staticmethod
    def _initialize_dicts(action_list):
        """
        Generates the appropriate dictionaries from the provided action list

        :param action_list: List of available actions
        :type action_list: list
        :return: Tuple containing (in order):
            Dictionary containing each action (str, key) with its assigned neuron (int, value) /
            Dictionary containing each neuron (int, key) with its assigned action (str, value)
        :rtype: tuple
        """

        act_to_int = {}
        int_to_act = {}

        # Loop through the list and store it in the dictionaries
        for i, action in enumerate(action_list):
            act_to_int[action] = i
            int_to_act[i] = action

        return act_to_int, int_to_act

    # PUBLIC METHODS

    # TODO: ACABA
    def act(self):
        pass

    def train_model(self):
        pass

    def save_weights(self, file_path, file_name):
        """
        Stores the weights of the CNN in the specified location, using the specified name

        :param file_path: Path where the weights will be stored (without the file name)
        :type file_path: str
        :param file_name: Name of the file storing the weights (WITHOUT THE EXTENSION)
        :type file_name: str
        """

        # Join the path with the file name and append the extension (h5)
        path = join(file_path, file_name)
        path = path + ".h5"

        # Store the weights
        self._cnn_model.save_weights(path)
