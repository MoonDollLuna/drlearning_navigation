# REACTIVE NAVIGATION - LOG MANAGER
# Developed by: Luna Jimenez Fernandez
#
# This file contains a class used to simplify the logging process used by the program
# A log is automatically created within a new folder, in order to store the performance of
# the agent during the episodes
#
# Metrics measured:
#

class LogManager:
    """
    Class used to handle the logging process used by the agents

    This class automatically handles the locking and writing of text files, as well as
    the creation of folders to store the data created.
    """

    def __init__(self):
        pass