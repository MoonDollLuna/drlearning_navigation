#!/bin/sh

# Made by Luna Jimenez Fernandez
# This simple bash script cleans the GPU cache and launches several profiling tools
# This script is useful to monitor PC performance during training
# (to avoid memory leaks, excessive memory usages...)

# Clean the GPU cache
rm -r /home/luna/.nv/*

# Open utilities to monitor the GPU, CPU, RAM and Swap usage
gnome-terminal -- watch -d -n 1 nvidia-smi
gnome-terminal -- htop
