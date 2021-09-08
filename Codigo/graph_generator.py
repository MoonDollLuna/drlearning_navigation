# REACTIVE NAVIGATION - MAIN CODE
# Developed by: Luna Jimenez Fernandez
#
# This file contains the script used to generate all the necessary graphs
# from the log files (.csv) created during agent training
#
# The script expects the following inputs:
#   * Episode ID
#   * Time taken to finish the episode (in seconds)
#   * Actions taken to finish the episode
#   * Initial distance to the goal
#   * Final distance to the goal
#   * Distance "shortened" by the agent (how much closer the agent is to the goal
#     at the end of the episode)
#   * Whether the episode was successful (TRUE) or not (FALSE)
#   * Average reward for the episode
#
# It will output the following graphs:
#  * Time taken per episode
#  * Time taken per episode (cumulative)
#  * Actions taken per episode
#  * Percentage of distance travelled by the agent per episode
#  * Total rate of successful episodes per agent type
#  * Average reward per episode
#
# All generated graphs offer a standard and smoothed variant

# IMPORTS #

import argparse
import os
import sys
import csv
import textwrap
from os.path import splitext
from itertools import accumulate
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

# USER DEFINED VARIABLES #

# Number of average episodes used when denoising the graphs
# This is used to reduce the noise in graphs, changing plot points to averages over the last
# average_episodes episodes
average_episodes = 50

# If True, EPS graphs are generated along the png files
eps = False

# If specified (not None), all data will be cut short at the specified number of episodes
episode_limit = None


# UTILITY METHODS #

def load_from_file(file_path, episode_limit=None):
    """
    Given a file path, loads and returns all the relevant information

    :param file_path: Path to the .csv file to be loaded
    :type file_path: str
    :param episode_limit: (OPTIONAL) If specified, files are read up to the given limit
    :type episode_limit: int
    :return: A vector with the following info:
                * List of episode IDs
                * List of episode times
                * List of episode actions
                * List of initial distances to the goal
                * List of final distances to the goal
                * List of bool values (whether the episode was successful or not)
                * List of average rewards
    :rtype: vector
    """

    # Store the information of all rows
    episode_ids = []
    episode_times = []
    episode_actions = []
    episode_initial_distances = []
    episode_final_distances = []
    episode_successful = []
    episode_rewards = []

    # Open the file using the default Python reader
    with open(file_path, "r") as file:

        # Use a CSV reader to read the file
        # Comments and whitespace lines are filtered
        read_rows = csv.reader(filter(lambda filter_row: filter_row[0] != "\n" and filter_row[0] != '#', file),
                               delimiter=",")

        # Ignore the first row (column titles)
        next(read_rows, None)

        # Read the row information
        for row in read_rows:
            episode_ids.append(int(row[0]))
            episode_times.append(float(row[1]))
            episode_actions.append(int(row[2]))
            episode_initial_distances.append(float(row[3]))
            episode_final_distances.append(float(row[4]))
            episode_successful.append(row[6] == "True")
            episode_rewards.append(float(row[7]))

    # Trim the logs if a limit is specified
    if episode_limit:
        episode_ids = episode_ids[0:episode_limit]
        episode_times = episode_times[0:episode_limit]
        episode_actions = episode_actions[0:episode_limit]
        episode_initial_distances = episode_initial_distances[0:episode_limit]
        episode_final_distances = episode_final_distances[0:episode_limit]
        episode_successful = episode_successful[0:episode_limit]
        episode_rewards = episode_rewards[0:episode_limit]

    return episode_ids, episode_times, episode_actions, episode_initial_distances, \
           episode_final_distances, episode_successful, episode_rewards


def cumulative_from_list(number_list):
    """Generates a list of cumulative sums from an initial numbered list"""

    return list(accumulate(number_list))


def percentages_from_list(initial_distances, final_distances, goal_size):
    """
    Generates a list containing the percentage of the total distance travelled by the agent

    As an example, if the agent started at 10m and travelled 5m, the percentage would approximately be 0.5
    Note that goal radius is taken into account

    All values must be smaller than +1 (reached the goal), but values can be negative

    :param initial_distances: List containing the initial distance to the goal for each episode
    :type initial_distances: list
    :param final_distances: List containing the final distance to the goal for each episode
    :type final_distances: list
    :param goal_size: Radius of the goal
    :type goal_size: float
    :return: List containing the percentage of the total distance travelled by each agent per episode
    :rtype: list
    """

    percentages_list = []

    # Each pair of distances is processed separately
    for initial, final in zip(initial_distances, final_distances):

        # Compute the percentage of the distance travelled (where goal_size is 100%)
        # This is done using a line equation, clamped to 1 (100%)
        percentage = (-final + initial) / (initial - goal_size)
        percentage = min(percentage, 1.0)

        # Store the percentage
        percentages_list.append(percentage)

    return percentages_list


def smooth_list(unsmoothed_list, smooth_factor):
    """
    Given a list, smoothes the list by using averages

    This replaces each group of smooth_factor items in the list by a single item
    equal to the average of those elements

    :param unsmoothed_list: Original list to be smoothed
    :type unsmoothed_list: list
    :param smooth_factor: Number of elements to be used for averages
    :type smooth_factor: int
    :return: Smoothed list
    :rtype: list
    """

    # Small hack: store the first element of the list at the beginning
    # so the list starts at the right place
    smoothed_list = [unsmoothed_list[0]]

    # Average each chunk
    for i in range(0, len(unsmoothed_list), smooth_factor):
        # Grab the chunk
        chunk = unsmoothed_list[i:i + smooth_factor]

        # Append the average to the list
        smoothed_list.append(mean(chunk))

    return smoothed_list


def extract_key_from_dict(dictionary, key):
    """
    Extracts the elements of a specified key from the dictionary

    Extracted elements are contained a tuple of lists ([legends], [elements]), where:
        * [legends] contains the legend name of all logs
        * [elements] contains the elements of the key for all logs in the dictionary

    :param dictionary: Dictionary to use
    :type dictionary: dict
    :param key: Name of the key to be extracted
    :type key: str
    :return: Tuple of lists, ([legends], [elements])
    :rtype: tuple
    """

    # Get the keys of the dictionary
    keys = dictionary.keys()

    # Store the results in a list
    extracted_legends = []
    extracted_values = []

    # Get the info for each key
    for dict_key in keys:
        extracted_legends.append(dict_key)
        extracted_values.append(dictionary[dict_key][key])

    return extracted_legends, extracted_values


# GRAPH METHODS #

def graph_plot(x_axis, y_axis_list, legend_titles,
               x_ax_title, y_ax_title, fix_y_axis, file_name, eps, top_y_lim=None):
    """
    Generates and saves a graph from the specified info

    :param x_axis: List containing the values of the X axis
    :type x_axis: list
    :param y_axis_list: List containing lists of the Y values (one per legend)
    :type y_axis_list: list
    :param legend_titles: Titles to be used for the legend
    :type legend_titles: list
    :param x_ax_title: Title of the X axis
    :type x_ax_title: str
    :param y_ax_title: Title of the Y axis
    :type y_ax_title: str
    :param fix_y_axis: If TRUE, the y axis will be fixed to 0
    :type fix_y_axis: bool
    :param file_name: Name to be used for the filename. Folder name is constant
    :type file_name: str
    :param eps: If TRUE, the graph will also be stored in EPS format
    :type eps: bool
    :param top_y_lim: If specified, the Y axis is limited to the specified value on top
    :type top_y_lim: float
    """

    # Create the figure
    plt.figure(figsize=(10, 6))

    # For each legend element, plot it
    for y_axis, legend in zip(y_axis_list, legend_titles):
        plt.plot(x_axis, y_axis, label=legend)

    # Add the axes titles
    plt.xlabel(x_ax_title)
    plt.ylabel(y_ax_title)

    # If necessary, fix the Y axis
    if fix_y_axis:
        plt.ylim(bottom=0)

    # If necessary, limit the Y axis on top
    if top_y_lim:
        plt.ylim(top=top_y_lim)

    # Specify the number of ticks for the X axis (16 ticks)
    plt.locator_params(axis='x', nbins=16)

    # Show the legend
    plt.legend()

    # Ensure that a folder for the graph is created
    os.makedirs("Graphs", exist_ok=True)

    # Store the graph as a PNG
    plt.savefig(os.path.join("Graphs", "{}.png".format(file_name)),
                bbox_inches="tight",
                dpi=1200)

    # If necessary, store the graph as an EPS
    if eps:
        plt.savefig(os.path.join("Graphs", "{}.eps".format(file_name)),
                    bbox_inches="tight",
                    dpi=1200,
                    format="eps")

    # Close the figure to save memory
    plt.close()


def assisted_graph_plot(x_axis, dictionary, key,
                        x_ax_title, y_ax_title, fix_y_axis, file_name, eps, top_y_lim=None):
    """Helper function for graph_plot that automatically handles the info extracting process"""

    # Get the legend names and y axis values
    legends, y_axis = extract_key_from_dict(dictionary, key)

    # Generate the graph
    graph_plot(x_axis, y_axis, legends, x_ax_title, y_ax_title, fix_y_axis, file_name, eps, top_y_lim)


def bar_plot(dictionary, key, y_ax_title, file_name, eps):
    """
    Generates and saves a bar from the specified info

    :param dictionary: Dictionary containing all the processed info
    :type dictionary: dict
    :param key: Key to be used to extract the info from the dictionary
    :type key: str
    :param y_ax_title: Title of the Y axis
    :type y_ax_title: str
    :param file_name: Name to be used for the filename. Folder name is constant
    :type file_name: str
    :param eps: If TRUE, the graph will also be stored in EPS format
    :type eps: bool
    """

    # Create the figure
    plt.figure(figsize=(12, 6))

    # Extract the info from the dictionary
    legend_names, values = extract_key_from_dict(dictionary, key)

    # Plot the info
    plt.bar(legend_names, values)

    # Add the ax title
    plt.ylabel(y_ax_title)

    # Ensure that a folder for the graph is created
    os.makedirs("Graphs", exist_ok=True)

    # Store the graph as a PNG
    plt.savefig(os.path.join("Graphs", "{}.png".format(file_name)),
                bbox_inches="tight",
                dpi=1200)

    # If necessary, store the graph as an EPS
    if eps:
        plt.savefig(os.path.join("Graphs", "{}.eps".format(file_name)),
                    bbox_inches="tight",
                    dpi=1200,
                    format="eps")

    # Close the figure to save memory
    plt.close()


# MAIN METHOD #

def graph_generation(files_loaded_list, average_episodes_used, eps_used, episode_limit=None):
    """
    From a list of loaded files, generates the appropriate graphs

    This generates both noisy and normalized graphs
    (using averages over the last average_episodes_used episodes
    instead of the original data points)

    :param files_loaded_list: List of .csv files loaded, possibly with legend names
    :type files_loaded_list: list
    :param average_episodes_used: Number of episodes to be used when smoothing the graphs
    :type average_episodes_used: int
    :param eps_used: If TRUE, graphs will be stored in EPS in addition to PNG
    :type eps_used: bool
    :param episode_limit: (OPTIONAL) If specified, log files are read up to the given episode
    :type episode_limit: int
    """

    # Loaded info is stored in a dictionary with the following structure
    # <LEGEND_NAME>:
    #       <attribute>: [list of elements for that attribute]

    loaded_info = {}

    print("Loading files...")

    # FILE PROCESSING #
    for file_info in files_loaded_list:

        # Extract the filename and, if possible, the legend name
        file_path = file_info[0]
        legend_name = splitext(file_path)[0]
        if len(file_info) > 1:
            legend_name = file_info[1]

        # Read the file
        ids, times, actions, initial_distances, final_distances, successful, rewards = load_from_file(file_path,
                                                                                                      episode_limit)

        # Process the additional info necessary for graph generations
        # Cumulative time
        cumulative_times = cumulative_from_list(times)

        # Percentage of distance travelled
        percentage_distances = percentages_from_list(initial_distances, final_distances, 0.3)

        # Total number of successful episodes
        total_successful_episodes = sum(successful)

        # Compute the smoothed variants of all lists
        smoothed_times = smooth_list(times, average_episodes_used)
        smoothed_cumulative_times = smooth_list(cumulative_times, average_episodes_used)
        smoothed_actions = smooth_list(actions, average_episodes_used)
        smoothed_distances = smooth_list(percentage_distances, average_episodes_used)
        smoothed_rewards = smooth_list(rewards, average_episodes_used)

        # Compute the ids for the smoothed graphs
        smoothed_ids = list(range(0, len(ids) + 1, average_episodes_used))
        # (first ID must be 1 instead of 0)
        smoothed_ids[0] = 1

        # Store the information in a dictionary
        info = {
            "ids": ids,
            "smoothed_ids": smoothed_ids,
            "times": times,
            "smoothed_times": smoothed_times,
            "cumulative_times": cumulative_times,
            "cumulative_smoothed_times": smoothed_cumulative_times,
            "actions": actions,
            "smoothed_actions": smoothed_actions,
            "distances": percentage_distances,
            "smoothed_distances": smoothed_distances,
            "successful": total_successful_episodes,
            "rewards": rewards,
            "smoothed_rewards": smoothed_rewards
        }

        # Store the dictionary in the global dictionary
        loaded_info[legend_name] = info

    # GRAPH PLOTTING
    print("Creating figures...")

    # Extract the IDs and smoothed IDs (shared by all graphs)
    general_ids = extract_key_from_dict(loaded_info, "ids")[1][0]
    general_smoothed_ids = extract_key_from_dict(loaded_info, "smoothed_ids")[1][0]

    # Generate all graphs (original and smoothed)
    # Time taken
    assisted_graph_plot(general_ids, loaded_info, "times",
                        "Episodios completados", "Duración del episodio (segs)", True,
                        "times", eps_used)
    assisted_graph_plot(general_smoothed_ids, loaded_info, "smoothed_times",
                        "Episodios completados", "Duración del episodio (segs)", True,
                        "smoothed_times", eps_used)

    # Cumulative time taken
    assisted_graph_plot(general_ids, loaded_info, "cumulative_times",
                        "Episodios completados", "Duración total (segs)", True,
                        "cumulative_times", eps_used)
    assisted_graph_plot(general_smoothed_ids, loaded_info, "cumulative_smoothed_times",
                        "Episodios completados", "Duración total (segs)", True,
                        "cumulative_smoothed_times", eps_used)

    # Actions taken
    assisted_graph_plot(general_ids, loaded_info, "actions",
                        "Episodios completados", "Acciones realizadas", True,
                        "actions", eps_used)
    assisted_graph_plot(general_smoothed_ids, loaded_info, "smoothed_actions",
                        "Episodios completados", "Acciones realizadas", True,
                        "smoothed_actions", eps_used)

    # Distance travelled
    assisted_graph_plot(general_ids, loaded_info, "distances",
                        "Episodios completados", "Porcentaje de distancia hasta la meta recorrida", True,
                        "distances", eps_used, top_y_lim=1.0)
    assisted_graph_plot(general_smoothed_ids, loaded_info, "smoothed_distances",
                        "Episodios completados", "Porcentaje de distancia hasta la meta recorrida", True,
                        "smoothed_distances", eps_used, top_y_lim=1.0)

    # Rewards
    assisted_graph_plot(general_ids, loaded_info, "rewards",
                        "Episodios completados", "Recompensa media", False,
                        "rewards", eps_used)
    assisted_graph_plot(general_smoothed_ids, loaded_info, "smoothed_rewards",
                        "Episodios completados", "Recompensa media", False,
                        "smoothed_rewards", eps_used)

    # Generate a bar plot graph for the successful episodes amount
    bar_plot(loaded_info, "successful", "Episodios completados con éxito", "success", eps_used)

    print("Graphs generated successfully")


# MAIN CODE #

# Execute the code only if the script is run directly
if __name__ == "__main__":

    # ARGUMENT DECLARATION

    # Arguments are declared using argparse
    parser = argparse.ArgumentParser(description=textwrap.dedent("""\
    Creates graphs from the information contained in the specified .csv log files.
    These logs contain the results of the agent training process.
    
    The following graphs are created:
        * Time taken per episode
        * Cumulative time taken per episode
        * Actions taken per episode
        * Percentage of distance travelled by the agent per episode
        * Total rate of successful episodes per agent type
        * Average reward per episode 
        
    All generated graphs offer both a standard and a smoothed version."""),
                                     formatter_class=argparse.RawTextHelpFormatter)

    # FILE (-f or --file) - Loads a file into the plot.
    # Usage: --file filename <legend name> WHERE:
    #       * filename: Name of the .cvs file containing the data (must include the .csv extension)
    #       * legend_name: OPTIONAL. If specified, this name will be used as the name for this
    #                      data line when plotting the graph. Otherwise, the file name will be used by default
    # This argument can be called several times, to load several files at once.
    parser.add_argument('-f',
                        '--file',
                        action='append',
                        nargs='+',
                        metavar=('filename', 'legend_name'),
                        help=textwrap.dedent("""\
                        Information about the file to be loaded to be used during plot generation.
                        This argument contains the following sub-arguments:
                            * filename: Path to the .csv file containing the data
                            * legend_name: (OPTIONAL) If specified, this name will be used to identify the data
                                           in the graphs' legends. Otherwise, filename will be used.
                        
                        This argument can be specified several times, to load several log files."""))

    # AVERAGE_EPISODES (-avg or --average_episodes)
    # Number of episodes to be used by averages when smoothing
    parser.add_argument('-avg',
                        '--average_episodes',
                        type=int,
                        help="Number of episodes to be used while smoothing the graphs. \n"
                             "Smoothed graphs will substitute the data points by average points every "
                             "average_episodes episodes. Number must be positive.\n"
                             "DEFAULT: {}".format(average_episodes))

    # EPS (-eps or --eps)
    # If TRUE, graphs will be stored in .eps format in addition to .png format
    parser.add_argument('-eps',
                        '--eps',
                        action='store_true',
                        help="If specified, graphs will be stored in EPS format in addition to PNG format.")

    # EPISODE_LIMIT (-el or --episode_limit)
    # If specified, files will be read up to the given episode.
    # Useful when logs have different number of episodes
    parser.add_argument('-el',
                        "--episode_limit",
                        type=int,
                        help="If specified, log files will be read up to the given episode. Must be a positive integer.")

    # ARGUMENT PARSING #

    # Parse the arguments and check that they are valid
    arguments = vars(parser.parse_args())

    if arguments["file"] is not None:
        files_loaded = arguments["file"]
    else:
        print("ERROR: At least one file must be loaded.")
        sys.exit()

    if arguments["average_episodes"] is not None:
        average_episodes = arguments["average_episodes"]
        if average_episodes <= 0:
            print("ERROR: Average episode count must be a positive integer.")
            sys.exit()

    if arguments["eps"] is not None:
        eps = True

    if arguments["episode_limit"] is not None:
        episode_limit = arguments["episode_limit"]
        if episode_limit <= 0:
            print("ERROR: Episode limit must be a positive integer")

    # Execute the main code
    graph_generation(files_loaded,
                     average_episodes,
                     eps,
                     episode_limit)
