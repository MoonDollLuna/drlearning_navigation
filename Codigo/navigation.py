# REACTIVE NAVIGATION - MAIN CODE
# Developed by: Luna Jimenez Fernandez
#
# This file contains the main structure of the reactive navigation system developed
# (including the arguments, parameters for the simulator, data display...)
#
# The developed agents are contained in the "agents" folder, in separate files.
#
# Structure of the code:
#   1 - Imports
#   2 - User-defined variables
#   3 - Main code
#       3A - Argument declaration
#       3B - Argument parsing
#       3C -

# NOTA: EL TIPO DE COMENTARIO SE LLAMA SPHINX MARKUP
# (garantiza compatibilidad con el mayor numero de sistemas posibles)

###############
# 1 - IMPORTS #
###############

import argparse

import cv2
import habitat
import numpy as np

from utils.log_manager import LogManager

##############################
# 2 - USER-DEFINED VARIABLES #
##############################

# dataset - Specifies the dataset to be used
# Possible values:
#   - matterport
#   - gibson
# DEFAULT: matterport
dataset = "matterport"




#################
# 3 - MAIN CODE #
#################

# The code is only executed when this script is run directly
if __name__ == "__main__":

    # ARGUMENT DECLARATION #
    # Arguments are declared using argparse
    parser = argparse.ArgumentParser(description="")
