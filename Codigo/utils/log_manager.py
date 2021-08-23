# REACTIVE NAVIGATION - LOG MANAGER
# Developed by: Luna Jimenez Fernandez
#
# This file contains a class used to simplify the logging process used by the program
# A log is automatically created within a new folder, in order to store the performance of
# the agent during the episodes
#
# The following metrics are measured during training:
#   * Episode ID
#   * Time taken to finish the episode
#   * Actions taken to finish the episode
#   * Initial distance to the goal
#   * Final distance to the goal
#   * Whether the episode was successful or not
#
# In addition, Reactive Navigation also tracks the following metrics:
#   * Average reward per episode
#


# IMPORTS #
import csv
import datetime

from typing import TextIO, Any


# CLASS DEFINITION #

class LogManager:
    """
    Class used to handle the logging process used by the agents during training and evaluation

    This class automatically handles the locking and writing of text files,
    as well as the creation of folders to store the data created.

    The log manager keeps track of the following attributes for each episode / checkpoint:
        * Episode ID
        * Time taken to finish the episode
        * Actions taken to finish the episode
        * Initial distance to the goal
        * Final distance to the goal
        * Distance "shortened" by the agent (how closer the agent is to the goal)
        * Whether the episode was a success or not

    In addition, other parameters may be specified
    """

    # ATTRIBUTES #

    # Name of the file
    _filename: str
    # Handle of the internal file, used to write on it
    _file: TextIO
    # Handle of the CSV writer, used to write the necessary CSV data
    # Note that the _writer class is hidden, so it cannot be directly specified
    _csv_writer: Any
    # Flag for the log manager. If TRUE, the log will not be output to the screen
    _silent: bool
    # List containing the extra parameters specified when constructed
    _extra_parameters: list

    # CONSTRUCTOR #

    def __init__(self, file_path, agent_type, dataset, training_length,
                 timestamp, silent=False, episode_parameters=None, **header_parameters):
        """
        A basic instance of the LogManager class.

        :param file_path: Path to the file
        :type file_path: str
        :param agent_type: Agent type, used to name the log file
        :type agent_type: str
        :param dataset: Dataset used to train the agent, used to name the log file
        :type dataset: str
        :param training_length: Expected length of the training (in frames or episodes).
                                Must be a full string in the shape "X frames" or "X episodes"
        :type training_length: str
        :param timestamp: Timestamp when the training process was created, used for the filename
        :type timestamp: float
        :param silent: If TRUE, logs will not be output to the screen
        :type silent: bool
        :param episode_parameters: (OPTIONAL) List of extra parameters to be included in each episode
        :type episode_parameters: list
        :param header_parameters: (OPTIONAL) Extra parameters to be specified on the header of the file
        :type header_parameters: str
        """

        # Compute a date from the timestamp
        date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H:%M:%S')

        # Create and store the handle of the log file
        self._filename = "{}/{}-{}-{}.csv".format(file_path, agent_type, dataset, date)
        self._file = open(self._filename, "w")

        # Create and store the handle for the CSV writer
        self._csv_writer = csv.writer(self._file)

        # Store the silent flag
        self._silent = silent

        # Store the additional parameters
        self._extra_parameters = episode_parameters

        # Write the header of the log file and the column names
        self._write_header(agent_type, dataset, training_length, date, **header_parameters)
        self._write_column_titles(["episode", "time_taken", "actions_taken",
                                   "initial_goal_distance", "final_goal_distance",
                                   "distance_travelled", "successful"],
                                  self._extra_parameters)

    # PRIVATE METHODS #
    def _write_comment(self, comment):
        """
        Writes and (if the silent flag is FALSE) prints a comment

        This method uses the standard Python writer

        :param comment: Comment to be written into the log
        :type comment: str
        """

        # Write the comment into the log
        self._file.write(comment + "\n")

        # If appropriate, print the comment into the screen
        if not self._silent:
            print(comment)

        # Flush the file
        self._file.flush()

    def _write_header(self, agent_type, dataset, training_length, date, **parameters):
        """
        Writes the header for the log file, containing all initial data:
            * Type of agent
            * Dataset used
            * Length of the training (either in frames or episodes)
            * Date of the training (in YY/MM/DD HH:MM:SS format)
            * If additional parameters are specified, they are also included here

        The header is written using comments, so the resulting file is still a valid CSV file

        :param agent_type: Type of agent that has been used for training
        :type agent_type: str
        :param dataset: Dataset that has been used for training
        :type dataset: str
        :param training_length: Expected length of the training (in frames or episodes).
                                Must be a full string in the shape "X frames" or "X episodes"
        :type training_length: str
        :param date: Date of the start of training (with YY/MM/DD HH:MM:SS format)
        :type date: str
        :param parameters: Additional parameters of the agent, passed as a dictionary
        :type parameters: dict
        """

        # Write and print the main header
        self._write_comment("# =*= AGENT TRAINING =*=")
        self._write_comment("#")
        self._write_comment("# = TRAINING PARAMETERS =")
        self._write_comment("#  * Agent type: {}".format(agent_type))
        self._write_comment("#  * Dataset used: {}".format(dataset))
        self._write_comment("#  * Training length: {}".format(training_length))
        self._write_comment("#  * Date / hour: {}".format(date))

        # If there are additional parameters, print them below the main parameters
        if parameters:
            self._write_comment("# = ADDITIONAL PARAMETERS =")
            for parameter in parameters:
                self._write_comment("#  * {}: {}".format(parameter, parameters[parameter]))
            self._write_comment("#")

        # Print the title for the episode values
        self._write_comment("# = EPISODE TRAINING RESULTS\n")

    def _write_column_titles(self, column_names, extra_parameters):
        """
        Writes the column names, including any extra parameter managed by the log manager

        :param column_names: List containing all of the common column names
        :type column_names: list
        :param extra_parameters: List containing the column name for the additional epoch parameters
        :type extra_parameters: list
        """

        # Concatenate the lists
        columns = column_names + extra_parameters

        # Write the column titles
        self._csv_writer.writerow(columns)

    # PUBLIC METHODS #
    def write_episode(self, episode_id, time_taken, actions_taken,
                      initial_goal_distance, final_goal_distance,
                      successful, extra_parameters=None):
        """
        Writes an episode using the following parameters:
            * Episode ID
            * Time taken
            * Actions taken
            * Initial distance to the goal
            * Final distance to the goal
            * Distance traversed by the agent
            * Successful

        In addition, will take a list with extra parameters that must be added in the same order they were declared
        when creating the log manager

        :param episode_id: ID of the current episode
        :type episode_id: int
        :param time_taken: Time taken (in seconds) to train the current episode
        :type time_taken: float
        :param actions_taken: Actions needed to complete the current episode
        :type actions_taken: int
        :param initial_goal_distance: Distance to the goal at the start of the episode
        :type initial_goal_distance: float
        :param final_goal_distance: Distance to the goal when the episode was finished
        :type final_goal_distance: float
        :param successful: Flag to specify whether the episode was successful (True) or not (False)
        :type successful: bool
        :param extra_parameters: (OPTIONAL) List containing the extra parameters specified during construction
        :type extra_parameters: list
        """

        # Assert the length of the extra parameters (must be equal to the number of extra parameters)
        assert len(extra_parameters) == len(self._extra_parameters), \
            "Number of additional parameters must be equal to the parameters declared when creating this manager"

        # Compute the distance travelled towards the goal by the agent
        travelled_distance = initial_goal_distance - final_goal_distance

        # Create and write the list of parameters to the file
        parameter_list = [episode_id, time_taken, actions_taken,
                          initial_goal_distance, final_goal_distance,
                          travelled_distance, successful] + extra_parameters
        self._csv_writer.writerow(parameter_list)

        # If the log is not silent, print the parameters to the screen
        if not self._silent:
            # Create the message
            message = "Episode: {} / Time taken: {}s / Actions taken: {} / " \
                      "Initial distance to the goal: {} / Final distance to the goal: {} / " \
                      "Distance travelled towards the goal: {} / Successful: {}".format(episode_id,
                                                                                        time_taken,
                                                                                        actions_taken,
                                                                                        initial_goal_distance,
                                                                                        final_goal_distance,
                                                                                        travelled_distance,
                                                                                        successful)
            # Append the additional parameters
            for title, value in zip(self._extra_parameters, extra_parameters):
                message = message + " / {}: {}".format(title, value)

            # Print the message
            print(message)

        # Flush the file every 100 episodes, to keep progress in case of an error
        if episode_id % 100 == 0:
            self._file.flush()

    def close(self, total_time):
        """
        Write the final training details, flush the data and close the writers

        :param total_time: Total time taken for the agent training (in seconds)
        :type total_time: float
        """

        # Print the final training information
        self._write_comment("\n")
        self._write_comment("# = FINAL TRAINING RESULTS =")
        self._write_comment("#  * Total training time: {} s".format(total_time))

        # Close the file to release the handle
        self._file.close()
