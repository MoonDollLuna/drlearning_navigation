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

from os import makedirs
from os.path import join


# CLASS DEFINITION #

class LogManager:
    """
    Class used to handle the logging process used by the agents

    This class automatically handles the locking and writing of text files,
    as well as the creation of folders to store the data created.
    """

    # CONSTRUCTOR #

    def __init__(self, agent_name, dataset):
        """
        A basic instance of the LogManager class.

        :param agent_name: Name of the agent to be logged.
        :type agent_name: str
        :param dataset: Dataset used to train the agent.
        :type dataset: str
        """

        # Store the values provided
        self._agent_name = agent_name
        self._dataset = dataset

        # Prepare the folder structure
        # The structure followed is "Resultados" -> [Agent Name] -> [Dataset] -> Log File.
        path = join("Resultados", self._agent_name, self._dataset)
        makedirs(path)

        # Generate a timestamp to identify the file

        # Create and store the handle of the log file
        # If the log file already exists, it will be emptied to avoid overlapping executions

        # TODO: TIMESTAMP, CAMBIAR NOMBRE TEST
        self._file = open(join(path, "test.txt"), "w")

    # PUBLIC METHODS #
