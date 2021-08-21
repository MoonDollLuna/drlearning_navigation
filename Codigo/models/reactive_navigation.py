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
# This structure has been developed ad-hoc, based on AlexNet
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
import numpy as np

# PyTorch
import torch

from torch import Tensor, device
from torch.nn import Module
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten
from torch.nn import SmoothL1Loss
from torch.nn.functional import relu
from torch.optim import Adam

# Reactive Navigation
from models.experience_replay import State


class ReactiveNavigationModel(Module):
    """
    The proposed Reactive Navigation model, using a CNN (trained using Deep Q-Learning)
    to allow an embodied agent to travel through an interior environment (such as a house).

    The model takes a grayscale image (taken from a depth camera) and the distance and angle
    to the goal (provided by the GPS) as inputs, outputting the actual action that needs to be
    performed.

    In addition, the model is prepared for batch training.

    Due to memory leaks when working with TensorFlow, the model has been implemented
    using PyTorch
    """

    # ATTRIBUTES #

    # Reference to the device to be used for training
    _device = device
    # Optimizer to use while training the network
    # Adam is used
    _optimizer = Adam
    # Loss function to use while training the network
    # Huber loss (an improved approach to MSE) is used
    _loss = SmoothL1Loss

    # Size of the image. Images have a shape of (_image_size x _image_size)
    _image_size: int
    # Dictionary containing each action (str, key) with its assigned neuron (int, value)
    # Example: "stop" => 0
    _action_to_int_dict: dict
    # Dictionary containing each neuron (int, key) with its assigned action (str, value)
    # Example: 0 => "stop"
    _int_to_action_dict: dict

    # CONSTRUCTOR #

    def __init__(self, image_size, action_list, device_used, learning_rate=None, weights=None):
        """
        Constructor method

        :param image_size: Size of the image used by the CNN
        :type image_size: int
        :param action_list: List of available actions
        :type action_list: list
        :param device_used: Device in which to perform all operations
        :type device_used: device
        :param learning_rate: (OPTIONAL) Learning rate of the neural network
        :type learning_rate: float
        :param weights: (OPTIONAL) Path to the file containing the pre-trained weights of the CNN
        :type weights: str
        """

        # Construct the parent
        super().__init__()

        # Store the given image size
        self._image_size = image_size

        # Generate the dictionaries from the action list
        self._action_to_int_dict, self._int_to_action_dict = self._initialize_dicts(action_list)

        # Store a handle to the device
        self._device = device_used

        # Create the optimizer and the loss function
        self._optimizer = Adam(self.parameters(), lr=learning_rate) if learning_rate else Adam(self.parameters())
        self._loss = SmoothL1Loss()

        # Create the CNN structure and, if available, load the weights
        self._prepare_cnn(self._image_size, len(action_list))
        if weights is not None:
            checkpoint_dict = torch.load(weights)
            self.load_state_dict(checkpoint_dict["state_dict"])

    # INTERNAL METHODS #

    @staticmethod
    def _conv_output_size(image_size, conv_kernel_sizes, pool_kernel_sizes):
        """
        Computes the actual number of neurons outputted by the convolution steps

        Used to compute the exact input size for dense layers connected to Conv2 layers
        Note that it is assumed that all Conv2d layers are followed by MaxPool2d layers

        All convolution layers are assumed to have:
            * padding = 0
            * dilation = 1
            * stride = 1

        All pooling layers are assumed to have:
            * padding = 0
            * dilation = 1
            * stride = equal to kernel size

        :param conv_kernel_sizes: List of the kernel sizes of the convolutional layers
        :type conv_kernel_sizes: list
        :param pool_kernel_sizes: List of the kernel sizes of the pooling layers
        :type pool_kernel_sizes: list
        :return: Total size of the resulting image
        :rtype: int
        """

        # Store the initial size
        current_size = image_size

        # For each kernel size
        for conv_kernel, pool_kernel in zip(conv_kernel_sizes, pool_kernel_sizes):

            # Compute the size of the convolution layer
            current_size = np.floor((current_size + 2*0 - 1*(conv_kernel - 1) - 1) + 1)
            # Compute the size of the pooling layer
            current_size = np.floor(((current_size + 2*0 - 1*(pool_kernel - 1) - 1) // pool_kernel) + 1)

        # Return the final output
        return current_size

    def _prepare_cnn(self, image_size, action_size):
        """
        Initializes the Convolutional Neural Network (CNN) used by the model.

        The CNN has the following structure (in order):
            * Input of the image (in grayscale, with shape image_size x image_size)
            * Three layers of Convolution (16 filters, kernel sizes 5/3/3) - Pooling (sizes 3/3/2)
            * A Merge layer where the result of the previous convolution is joined with
              the scalar inputs (Distance and Angle to the goal)
            * A connected layer (ReLU)
            * action_size neurons (Linear)

        The CNN will use the following hyperparameters:
            * Optimizer: Adam
            * Error: Mean Squared Error

        Note that, since the CNN will be used for reinforcement learning, no dropout will be used

        :param image_size: Size of the image. The shape of the image will be (image_size x image_size)
        :type image_size: int
        :param action_size: Number of actions available for the agent
        :type action_size: int
        """

        # Create the first layer of convolution / pooling
        # Input: one image of 1 layer
        # Output: 16 filters of kernel size 5
        # Pooling: kernel size 3
        self._conv1 = Conv2d(in_channels=1,
                             out_channels=16,
                             kernel_size=5)
        self._pool1 = MaxPool2d(kernel_size=3)

        # Create the second layer of convolution / pooling
        # Input: 16 filters
        # Output: 32 filters of kernel size 3
        # Pooling: kernel size 3
        self._conv2 = Conv2d(in_channels=16,
                             out_channels=32,
                             kernel_size=3)
        self._pool2 = MaxPool2d(kernel_size=3)

        # Create the third layer of convolution / pooling
        # Input: 32 filters
        # Output: 64 filters of kernel size 3
        # Pooling: kernel size 2
        self._conv3 = Conv2d(in_channels=32,
                             out_channels=64,
                             kernel_size=3)
        self._pool3 = MaxPool2d(kernel_size=2)

        # Find the size of the CNN layers output
        cnn_size = self._conv_output_size(image_size,
                                          [5, 3, 3],
                                          [3, 3, 2])

        # Create a flatten layer, to flatten the output
        # for the dense layer
        self._flatten = Flatten()

        # Compute the number of neurons on the next layer
        # Note that two extra neurons are added, for the extra inputs
        dense_size = (cnn_size ** 2) * 32 + 2

        # Create the final dense output layer
        # Inputs: dense_size (computed previously)
        # Outputs: action_size
        self._dense = Linear(in_features=dense_size,
                             out_features=action_size)

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

    def forward(self, image, scalars):
        """
        Main method of PyTorch, specifies how the neural network will process its inputs

        :param image: Tensor of the depth image
        :type image: Tensor
        :param scalars: Tensor of the scalar values to be used
        :type scalars: Tensor
        :return: Processed tensor, after going through the whole network
        :rtype: Tensor
        """

        # Make sure all data is processed through the appropriate device
        image = image.to(self._device)
        scalars = scalars.to(self._device)

        # First layer of CNN:
        # CNN -> ReLU -> Pooling
        image = self._conv1(image)
        image = relu(image)
        image = self._pool1(image)

        # Second layer of CNN:
        # CNN -> ReLU -> Pooling
        image = self._conv2(image)
        image = relu(image)
        image = self._pool2(image)

        # Third layer of CNN:
        # CNN -> ReLU -> Pooling
        image = self._conv3(image)
        image = relu(image)
        image = self._pool3(image)

        # Flatten the CNN output
        image = self._flatten(image)

        # Join the two tensor inputs
        joined = torch.cat((image, scalars), dim=1)

        # Pass the output through the dense layer (linear function) and return it
        return self._dense(joined)

    def predict(self, states, chunk_size):
        """
        Batch predicts the Q-Values of a batch of States

        :param states: List of States
        :type states: list
        :param chunk_size: The batch of states is divided into chunks of this size
        :type chunk_size: int
        :return: List of Q-Values for each of the states
        :rtype: list
        """

        # These predictions must be done in eval mode
        self.eval()

        # Prepare the list of states using the network format
        unwrapped_states = [state.unwrap_state() for state in states]

        # Split the list of states into chunks of chunk_size
        number_of_lists = len(unwrapped_states) / chunk_size
        chunked_states = np.array_split(unwrapped_states, number_of_lists)

        # Store the results of the process in a list
        predicted_values = []

        # Process each chunk independently
        for chunk in chunked_states:
            # Extract the list of images and scalar values, and convert them to tensor format
            images = torch.tensor([state[0] for state in chunk],
                                  device=self._device)
            scalars = torch.tensor([state[1] for state in chunk],
                                   device=self._device)

            # Process and add the Q values (without gradient)
            with torch.no_grad():
                predicted_q_values = self.__call__(images, scalars)
                for q_values in predicted_q_values:
                    predicted_values.append(q_values.tolist())

        return predicted_values

    def act(self, state):
        """
        Given a state, returns the optimal action (according to the model policy)
        to be taken.

        :param state: Current state perceived by the agent
        :type state: State
        :return: Action to be performed
        :rtype: str
        """

        # Prepare the state for the neural network
        state_image, state_scalars = state.unwrap_state()

        # Prepare tensors for both image and scalar values
        image_tensor = torch.as_tensor([state_image],
                                       device=self._device)

        scalar_tensor = torch.tensor([state_scalars],
                                     device=self._device)

        # Obtain the prediction by calling the module
        # Note that this must be done in eval mode and without gradients
        # (with no_grad active)
        self.eval()
        with torch.no_grad():
            predicted_q_values = self.__call__(image_tensor, scalar_tensor)

        # Return the best action
        # (max returns a tuple (value, index))
        best_action_index = torch.max(predicted_q_values, 1)[1]
        return self._int_to_action_dict[best_action_index]

    def fit_model(self, values, predictions, chunk_size):
        """
        Given a list of values and their actual predictions, optimize the CNN weights

        :param values: List of obtained Q-Values for each pair state-action
        :type values: list
        :param predictions: List of predicted Q-Values for each pair state-action
        :type predictions: list
        :param chunk_size: The batch of states is divided into chunks of this size
        :type chunk_size: int
        """

        # The fitting must be done in train mode
        self.train()

        # Split the list values and predictions into chunks of chunk_size
        number_of_lists = len(values) / chunk_size

        chunked_values = np.array_split(values, number_of_lists)
        chunked_predictions = np.array_split(predictions, number_of_lists)

        # Process each chunk independently
        for chunk_values, chunk_predictions in zip(chunked_values, chunked_predictions):

            # Convert the list of obtained and predicted values to Tensor format
            obtained_values = torch.tensor(chunk_values,
                                           device=self._device)
            predicted_values = torch.tensor(chunk_predictions,
                                            device=self._device)

            # Compute the loss
            loss = self._loss(obtained_values, predicted_values)

            # Optimize the model
            self._optimizer.zero_grad()
            loss.backwards()
            self._optimizer.step()
