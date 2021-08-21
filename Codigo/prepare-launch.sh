#!/bin/sh

# Clean the GPU cache
rm -r /home/luna/.nv/*

# Open utilities to monitor the GPU, CPU, RAM and Swap usage
gnome-terminal -- watch -d -n 1 nvidia-smi
gnome-terminal -- htop
