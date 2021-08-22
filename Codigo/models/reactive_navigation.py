# REACTIVE NAVIGATION - MODEL
# Developed by: Luna Jimenez Fernandez
#
# This file contains the Reactive Navigation main model, used by both the
# reactive navigation trainer (for Reinforcement Learning) and the
# reactive navigation agent (for evaluation using a Benchmark)
#
# The model consists of a hybrid Neural Network using the following structure:
# CNN:
#   * Input: One grayscale image of size (image_size x image_size)
#   * Layer of Convolution (16 filters, kernel size 5) / ReLU / Max Pooling (pool size 3)
#   * Layer of Convolution (32 filters, kernel size 3) / ReLU / Max Pooling (pool size 3)
#   * Layer of Convolution (16 filters, kernel size 3) / ReLU / Max Pooling (pool size 2)
#   * Hidden layer of 64 neurons (ReLU)
#   * Output layer of 3 neurons (Linear)
# MLP:
#   * Input: 2 scalar values (distance and angle to the goal)
#   * Hidden layer of 10 neurons (ReLU)
#   * Hidden layer of 10 neurons (ReLU)
#   * Output: 3 neurons (Linear)
#
# HYBRID:
#   * Input: 6 neurons (3 from the CNN, 3 from the MLP)
#   * Hidden layer of 32 neurons (ReLU)
#   * Hidden layer of 32 neurons (ReLU)
#   * Output: action_size neurons (Linear)
#
# This structure has been developed ad-hoc, based on AlexNet
#
# The inputs of the hybrid network are as follows
#   * CNN: A 256x256 (shape can be specified) grayscale image (obtained from the Depth Camera)
#   * MLP: A pair of scalar values, Distance and Angle to the goal (obtained from the GPS)
#
# The output of the CNN is four neurons, each representing the Q-Value of a pair of state-action:
#   * STOP
#   * MOVE FORWARD
#   * TURN RIGHT
#   * TURN LEFT

# IMPORTS #
import numpy as np
from numpy import ndarray

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

        # Create the CNN structure and, if available, load the weights
        self._prepare_cnn(self._image_size, len(action_list))

        if weights is not None:
            checkpoint_dict = torch.load(weights)
            self.load_state_dict(checkpoint_dict["state_dict"])

        # Start the model in eval mode
        self.eval()

        # Create the optimizer
        self._optimizer = Adam(self.parameters(), lr=learning_rate) if learning_rate else Adam(self.parameters())

    # STATIC METHODS #

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

    @staticmethod
    def _image_to_tensor(image, target_device):
        """
        Converts a numpy image into a tensor with the appropriate structure

        Numpy arrays have the shape: [height, width, channels]
        Pytorch tensors have the shape: [batch_size, channels, height, width]

        :param image: Grayscale image in format (image_size, image_size)
        :type image: ndarray
        :param target_device: Device where the image will be stored
        :type target_device: device
        :return: Tensor of the image in the appropriate scale
        :rtype: Tensor
        """

        # Create the tensor from the image
        image_tensor = torch.as_tensor(image,
                                       device=target_device)

        # Reorder the channels of the tensor
        image_tensor = image_tensor.permute(2, 0, 1)

        # Unsqueeze the tensor to gain the extra batch size channel
        image_tensor = image_tensor.unsqueeze(dim=0)

        return image_tensor

    @staticmethod
    def _split_into_chunks(initial_list, chunk_size):
        """
        Splits a list into chunks of size N

        :param initial_list: List to be chunked
        :type initial_list: list
        :param chunk_size: Size of the chunks
        :type chunk_size: int
        :return: List containing the chunks (smaller lists of length n)
        :rtype: list
        """

        # Use list comprehension to chunk the list
        return [initial_list[i:i + chunk_size] for i in range(0, len(initial_list), chunk_size)]

    @staticmethod
    def _clean_image_input(input_image):
        """
        Clean the input image to remove graphical glitches

        Habitat-Lab sometimes has wrong renderings (graphical glitches)
        that are seen as depth 0.0 (pitch black, closest to the camera)
        by the depth camera

        This method converts those 0.0 values into 1.0 values (pitch white,
        furthest from the camera)

        :param input_image: Image received by the depth camera
        :type input_image: ndarray
        :return: Cleaned image without pitch black (0.0) values
        :rtype: ndarray
        """

        # Since the image matrix is made of float values,
        # check for values smaller than 0.001 instead (floats can have inaccuracies)
        processed_image = input_image
        processed_image[processed_image <= 0.001] = 1.0

        return processed_image

    # PRIVATE METHODS #

    def _prepare_cnn(self, image_size, action_size):
        """
        Initializes hybrid neural network (CNN + MLP) used by the model.

        The hybrid network has the following structure:

        CNN:
            * Input: One grayscale image of size (image_size x image_size)
            * Layer of Convolution (16 filters, kernel size 5) / ReLU / Max Pooling (pool size 3)
            * Layer of Convolution (32 filters, kernel size 3) / ReLU / Max Pooling (pool size 3)
            * Layer of Convolution (16 filters, kernel size 3) / ReLU / Max Pooling (pool size 2)
            * Hidden layer of 64 neurons (ReLU)
            * Output layer of 3 neurons (Linear)

        MLP:
            * Input: 2 scalar values (distance and angle to the goal)
            * Hidden layer of 10 neurons (ReLU)
            * Hidden layer of 10 neurons (ReLU)
            * Output: 3 neurons (Linear)

        HYBRID:
            * Input: 6 neurons (3 from the CNN, 3 from the MLP)
            * Hidden layer of 32 neurons (ReLU)
            * Hidden layer of 32 neurons (ReLU)
            * Output: action_size neurons (Linear)

        This network structure has been developed ad-hoc for this task

        The network will use the following hyperparameters:
            * Optimizer: Adam
            * Error: Mean Squared Error

        Note that, since the network will be used for reinforcement learning, no dropout will be used

        :param image_size: Size of the image. The shape of the image will be (image_size x image_size)
        :type image_size: int
        :param action_size: Number of actions available for the agent
        :type action_size: int
        """

        # CONVOLUTIONAL NEURAL NETWORK #

        # Create the first layer of convolution / pooling
        # Input: one image of 1 layer
        # Output: 16 filters of kernel size 5
        # Pooling: kernel size 3
        self._cnn_conv1 = Conv2d(in_channels=1,
                                 out_channels=16,
                                 kernel_size=5)
        self._cnn_pool1 = MaxPool2d(kernel_size=3)

        # Create the second layer of convolution / pooling
        # Input: 16 filters
        # Output: 32 filters of kernel size 3
        # Pooling: kernel size 3
        self._cnn_conv2 = Conv2d(in_channels=3,
                                 out_channels=32,
                                 kernel_size=3)
        self._cnn_pool2 = MaxPool2d(kernel_size=3)

        # Create the third layer of convolution / pooling
        # Input: 32 filters
        # Output: 16 filters of kernel size 3
        # Pooling: kernel size 2
        self._cnn_conv3 = Conv2d(in_channels=32,
                                 out_channels=16,
                                 kernel_size=3)
        self._cnn_pool3 = MaxPool2d(kernel_size=2)

        # Create a flatten layer, to flatten the output for the concatenation
        self._cnn_flatten = Flatten()

        # Find the size of the CNN layers output
        cnn_size = self._conv_output_size(image_size,
                                          [5, 3, 3],
                                          [3, 3, 2])
        # CNN output actual size is cnn_size (side of the output) squared,
        # multiplied by the number of filters
        cnn_size = (cnn_size ** 2) * 16

        # Create a hidden layer (ReLU)
        # Input: cnn_size neurons
        # Output: 64 neurons
        self._cnn_hidden = Linear(cnn_size, 64)

        # Create an output layer (Linear)
        # Input: 64 neurons
        # Output: 3 neurons
        self._cnn_output = Linear(64, 3)

        # MULTI LAYERED PERCEPTRON #

        # Create the first hidden dense layer (ReLU)
        # Input: 2 scalar values
        # Output: 10 neurons - (nn_input (2 values) + nn_output (1 neuron)) * 2
        self._mlp_hidden1 = Linear(2, 10)

        # Create the second hidden dense layer (ReLU)
        # Input: 10 neurons
        # Output: 10 neurons
        self._mlp_hidden2 = Linear(10, 10)

        # Create the output layer (Linear)
        # Input: 10 neurons
        # Output: 3 neurons
        self._mlp_output = Linear(10, 3)

        # MERGE #
        # Output concatenation is done in the Forward method

        # Create the first hidden dense layer (ReLU)
        # Input: 6 neurons
        # Output: 32
        self._merged_hidden1 = Linear(6, 32)

        # Create the second hidden dense layer (ReLU)
        # Input: 32 neurons
        # Output: 32 neurons
        self._merged_hidden2 = Linear(32, 32)

        # Create a final output layer
        # Input: 32 neurons
        # Output: action_size neurons
        self._merged_output = Linear(32, int(action_size))

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

        # CNN

        # First layer of convolution:
        # CNN -> ReLU -> Pooling
        image = relu(self._conv1(image))
        image = self._pool1(image)

        # Second layer of convolution:
        # CNN -> ReLU -> Pooling
        image = relu(self._conv2(image))
        image = self._pool2(image)

        # Third layer of convolution:
        # CNN -> ReLU -> Pooling
        image = relu(self._conv3(image))
        image = self._pool3(image)

        # Flatten the CNN output
        image = self._flatten(image)

        # Hidden layer (ReLU)
        image = relu(self._cnn_hidden(image))

        # Output layer (linear)
        image = self._cnn_output(image)

        # MLP

        # First hidden layer (ReLU)
        scalars = relu(self._mlp_hidden1(scalars))

        # Second hidden layer (ReLU)
        scalars = relu(self._mlp_hidden2(scalars))

        # Output layer (linear)
        scalars = self._mlp_output(scalars)

        # HYBRID

        # Join the output of the CNN (3 neurons) and the MLP (3 neurons)
        hybrid = torch.cat((image, scalars), dim=1)

        # Hidden layer 1 (ReLU)
        hybrid = relu(self._merged_hidden1(hybrid))

        # Hidden layer 2 (ReLU)
        hybrid = relu(self._merged_hidden2(hybrid))

        # Final output (linear)
        return self._merged_output(hybrid)

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
        number_of_lists = len(unwrapped_states) // chunk_size
        chunked_states = self._split_into_chunks(unwrapped_states, number_of_lists)

        # Store the results of the process in a list
        predicted_values = []

        # Process each chunk independently
        for chunk in chunked_states:

            # Extract the list of images and scalars
            images = [self._image_to_tensor(self._clean_image_input(state[0]), self._device) for state in chunk]
            scalars = [state[1] for state in chunk]

            # Convert both lists to tensors
            image_tensors = torch.cat(images)
            scalar_tensors = torch.tensor(scalars,
                                          dtype=torch.float,
                                          device=self._device)

            # Process and add the Q values (without gradient)
            with torch.no_grad():
                predicted_q_values = self.__call__(image_tensors,
                                                   scalar_tensors)
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
        image_tensor = self._image_to_tensor(self._clean_image_input(state_image),
                                             self._device)
        scalar_tensor = torch.tensor(state_scalars,
                                     device=self._device)

        # Image tensor already has 4 dimensions
        # Unsqueeze the scalar tensor to create the extra batch size dimension
        scalar_tensor = scalar_tensor.unsqueeze(0)

        # Obtain the prediction by calling the module
        # Note that this must be done in eval mode and without gradients
        # (with no_grad active)
        self.eval()
        with torch.no_grad():
            predicted_q_values = self.__call__(image_tensor, scalar_tensor)

        # Compute the actual action index
        # (max returns a tensor (value, index))
        best_action_index = torch.max(predicted_q_values, 1)[1]
        # (item returns the actual value contained in a tensor)
        best_action_index = best_action_index.item()
        return self._int_to_action_dict[best_action_index]

    def fit_model(self, states, expected_q_values, chunk_size):
        """
        Given a list of input states for the CNNs (images and scalars) and their actual values,
        does a pass through the network to fit the weights

        The following hyperparameters are used:
            * Optimizer: Adam
            * Loss: Huber loss (Smooth L1 loss)

        :param states: List of states (images and scalar values) to be passed through the network
        :type states: list
        :param expected_q_values: List of expected Q values for each pair of (image, scalars)
        :type expected_q_values: list
        :param chunk_size: The batch of states is divided into chunks of this size
        :type chunk_size: int
        """

        # The fitting must be done in train mode
        self.train()

        # Prepare the list of states using the network format
        unwrapped_states = [state.unwrap_state() for state in states]

        # Split the list values and predictions into chunks of chunk_size
        number_of_lists = len(unwrapped_states) // chunk_size

        chunked_states = self._split_into_chunks(unwrapped_states, number_of_lists)
        chunked_q_values = self._split_into_chunks(expected_q_values, number_of_lists)

        # Process each chunk independently
        for chunk_states, chunk_q_values in zip(chunked_states,
                                                chunked_q_values):

            # Extract the list of images and scalars
            images = [self._image_to_tensor(self._clean_image_input(state[0]), self._device) for state in chunk_states]
            scalars = [state[1] for state in chunk_states]

            # Convert both list to tensors
            image_tensors = torch.cat(images)
            scalar_tensors = torch.tensor(scalars,
                                          dtype=torch.float,
                                          device=self._device)

            # Convert the list of expected values to a tensor
            expected_q_tensor = torch.tensor(chunk_q_values,
                                             dtype=torch.float,
                                             device=self._device)

            # Instantiate and compute the loss using the model
            criterion = SmoothL1Loss()
            loss = criterion(self.__call__(image_tensors, scalar_tensors),
                             expected_q_tensor)

            # Optimize the model using the computed loss
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
