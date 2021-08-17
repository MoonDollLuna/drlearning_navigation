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
#     * Three layers of Convolution (16 filters, kernel sizes 5/3/3) -
#                       Pooling (sizes 3/3/2)
#     * A Merge layer where the result of the previous convolution is joined with
#       the scalar inputs (Distance and Angle to the goal)
#     * Two fully connected layers (ReLU) of 256 neurons
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
import numpy as np
from keras import Input
from keras.models import Model
from keras.optimizers import Adam

# Layers
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate

# Reactive Navigation
from models.experience_replay import State


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
            self.load_weights_file(weights)

    # INTERNAL METHODS #

    @staticmethod
    def _initialize_cnn(image_size, action_size, learning_rate=None):
        """
        Initializes the Convolutional Neural Network (CNN) used by the model.

        The CNN has the following structure (in order):
            * Input of the image (in grayscale, with shape image_size x image_size)
            * Three layers of Convolution (16 filters, kernel sizes 5/3/3) - Pooling (sizes 3/3/2)
            * A Merge layer where the result of the previous convolution is joined with
              the scalar inputs (Distance and Angle to the goal)
            * Two fully connected layers (ReLU) of 256 neurons
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
        image_input = Input(shape=(image_size, image_size, 1))
        scalar_input = Input(shape=(2,))

        # Create the first layer of convolution
        conv1 = Conv2D(filters=16,
                       kernel_size=5,
                       activation="relu")(image_input)

        pool1 = MaxPooling2D(pool_size=3)(conv1)

        # Create the second layer of convolution
        conv2 = Conv2D(filters=16,
                       kernel_size=3,
                       activation="relu")(pool1)
        pool2 = MaxPooling2D(pool_size=3)(conv2)

        # Create the third layer of convolution
        conv3 = Conv2D(filters=16,
                       kernel_size=3,
                       activation="relu")(pool2)
        pool3 = MaxPooling2D(pool_size=2)(conv3)

        # Flatten the input, so it can be used with dense layers
        flatten = Flatten()(pool3)

        # Merge the results of the convolutional layers with the scalar input
        merge = concatenate([flatten, scalar_input])

        # Create the dense layers
        # (256 neurons, ReLU)
        dense1 = Dense(256,
                       activation="relu",
                       kernel_initializer="glorot_uniform")(merge)
        dense2 = Dense(256,
                       activation="relu",
                       kernel_initializer="glorot_uniform")(dense1)

        # Create the output layer (action_size outputs, Lineal)
        # Note that the output MUST be lineal (instead of the typical sigmoid function)
        # for Deep Reinforcement Learning
        output = Dense(action_size,
                       activation="linear",
                       kernel_initializer="glorot_uniform")(dense2)

        # Create and compile the model of the full CNN (Adam optimizer, MSE)
        # Mean Square Error is used (instead of more typical cross-entropy values) due to Deep Reinforcement Learning
        # (since MSE is the value trying to be minimized)
        model = Model(inputs=[image_input, scalar_input],
                      outputs=output)
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

    def get_weights(self):
        """
        Returns the weights of the neural network

        :return: Weights of the neural network
        """

        return self._cnn_model.get_weights()

    def set_weights(self, weights):
        """
        Loads the weights of the neural network

        :param weights: Weights of the neural network
        """

        self._cnn_model.set_weights(weights)

    def load_weights_file(self, file_path):
        """
        Load pre-trained weights for the CNN from the specified path

        :param file_path: Location of the pre-trained weights (in .h5 format)
        :type file_path: str
        """

        # Load the weights
        self._cnn_model.load_weights(file_path)

    def save_weights_file(self, file_path, file_name):
        """
        Stores the weights of the CNN in the specified location, using the specified name

        :param file_path: Path where the weights will be stored (without the file name)
        :type file_path: str
        :param file_name: Name of the file storing the weights (WITHOUT THE EXTENSION)
        :type file_name: str
        """

        # Join the path with the file name and append the extension (h5)
        path = join(file_path, "{}.h5".format(file_name))

        # Store the weights
        self._cnn_model.save_weights(path)

    def predict(self, states):
        """
        Batch predicts the Q-Values of an array of States

        :param states: List of States
        :type states: list
        :return: List of Q-Values for each of the states
        :rtype: list
        """

        # Prepare the list of states using the network format
        unwrapped_states = [state.unwrap_state() for state in states]

        # Extract the list of images and scalar values, and convert them to np format
        images = np.asarray([state[0] for state in unwrapped_states])
        scalars = np.asarray([state[1] for state in unwrapped_states])

        # Process and return the Q values
        return self._cnn_model.predict([images, scalars],
                                       batch_size=len(unwrapped_states))

    def act(self, state):
        """
        Given a state, returns the optimal action (according to the model policy)
        to be taken.

        :param state: Current state perceived by the agent
        :type state: State
        :return: Action to be performed and Q-Value for the chosen action
        :rtype: tuple
        """

        # Prepare the state for the neural network
        state_image, state_scalars = state.unwrap_state()

        # Predict the action using the CNN
        predicted_actions = self._cnn_model.predict([np.asarray([state_image]),
                                                     np.asarray([state_scalars])])

        # Return the best action
        best_action_index = np.argmax(predicted_actions[0])
        return self._int_to_action_dict[best_action_index]

    def fit_model(self, states, predictions):
        """
        Given a list of states and their updated predictions, fit the CNN to learn weights for these new values

        :param states: List of current states
        :type states: list
        :param predictions: List of predicted Q-Values for each pair state-action
        :type predictions: list
        """

        # Prepare the list of states using the network format
        unwrapped_states = [state.unwrap_state() for state in states]

        # Extract the list of images and scalar values, and convert them to np format
        images = np.asarray([state[0] for state in unwrapped_states])
        scalars = np.asarray([state[1] for state in unwrapped_states])

        # Fit the network
        self._cnn_model.fit([images, scalars],
                            np.array(predictions),
                            batch_size=len(states),
                            epochs=1, verbose=0)
