#!/bin/bash

# Get the IP address of the machine
IP_ADDRESS=$(hostname -I | awk '{print $1}')

# Set the OLLAMA_HOST environment variable
export OLLAMA_HOST="$IP_ADDRESS:11434"

# Run the ollama serve command with a timeout of 3600 seconds (1 hour)
timeout 21600 ~/binaries/ollama serve