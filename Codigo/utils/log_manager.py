# REACTIVE NAVIGATION - LOG MANAGER
# Developed by: Luna Jimenez Fernandez
#
# This file contains a class used to simplify the logging process used by the program
# A log is automatically created within a new folder, in order to store the performance of
# the agent during the episodes
#
# Metrics measured:
#

# IMPORTS #
import csv
import datetime

from typing import TextIO, Any


# CLASS DEFINITION #

class LogManager:
    """
    Class used to handle the logging process used by the agents

    This class automatically handles the locking and writing of text files,
    as well as the creation of folders to store the data created.

    The log manager keeps track of the following attributes for each episode:
        *

    """
    # TODO ACABAR

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

    # TODO: Recompensa media por episodio

    # CONSTRUCTOR #

    def __init__(self, agent_type, dataset, timestamp, silent=False, **parameters):
        """
        A basic instance of the LogManager class.

        :param agent_type: Agent type, used to name the log file
        :type agent_type: str
        :param dataset: Dataset used to train the agent, used to name the log file
        :type dataset: str
        :param timestamp: Timestamp when the training process was created, used for the filename
        :type timestamp: float
        :param silent: If TRUE, logs will not be output to the screen
        :type silent: bool
        :param parameters: (OPTIONAL) Extra parameters to be specified on the header of the file
        """

        # Compute a date from the timestamp
        date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        # Create and store the handle of the log file
        self._filename = "{}-{}-{}.csv".format(agent_type, dataset, date)
        self._file = open(self._filename, "w")

        # Create and store the handle for the CSV writer
        self._csv_writer = csv.writer(self._file)

        # Store the silent flag
        self._silent = silent

        # Write the header of the log file
        self._write_header(agent_type, dataset, date, **parameters)

    # PRIVATE METHODS #
    def _write_comment(self, comment):
        """
        Writes and (if the silent flag is FALSE) prints a comment

        This method uses the standard Python writer

        :param comment: Comment to be written into the log
        :type comment: str
        """

        # Write the comment into the log
        self._file.write(comment)

        # If appropriate, print the comment into the screen
        if not self._silent:
            print(comment)

    def _write_header(self, agent_type, dataset, date, **parameters):
        """
        Writes the header for the log file, containing all initial data:
            * Type of agent
            * Dataset used
            * Date of the training (in YY/MM/DD HH:MM:SS format)
            * If additional parameters are specified, they are also included here

        The header is written using comments, so the resulting file is still a valid CSV file

        :param agent_type: Type of agent that has been used for training
        :type agent_type: str
        :param dataset: Dataset that has been used for training
        :type dataset: str
        :param date: Date of the start of training (with YY/MM/DD HH:MM:SS format)
        :type date: str
        :param parameters: Additional parameters of the agent, passed as a dictionary
        :type parameters: dict
        """

        # Write and print the main header
        header = """# =*= AGENT TRAINING =*=
        #
        # = AGENT PARAMETERS =
        #   * Agent type: {}
        #   * Dataset used: {}
        #   * Date: {}
        #""".format(agent_type, dataset, date)
        self._write_comment(header)

        # If there are additional parameters, print them below the main parameters
        if parameters:
            self._write_comment("# = ADDITIONAL PARAMETERS =")
            for parameter in parameters:
                self._write_comment("#  * {}: {}".format(parameter, parameters[parameter]))
            self._write_comment("#")

        # Print the title for the epoch values
        self._write_comment("# = EPOCH TRAINING RESULTS\n")

    # PUBLIC METHODS #
    def write_epoch(self):
        pass
