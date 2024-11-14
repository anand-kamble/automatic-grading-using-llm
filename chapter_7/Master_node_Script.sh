#!/bin/bash

# List of machines
machines=("class01" "class02" "class03" "class04" "class05" "class06" "class07" "class08" "class09" "class10" "class11" "class12" "class13" "class14" "class15" "class16" "class17" "class18" "class19")

# List of colors
# This makes it easy to identify the outputs from different machines
colors=(
    "\033[31m" # Red
    "\033[32m" # Green
    "\033[33m" # Yellow
    "\033[34m" # Blue
    "\033[35m" # Magenta
    "\033[36m" # Cyan
    "\033[37m" # White
    "\033[91m" # Bright Red
    "\033[92m" # Bright Green
    "\033[93m" # Bright Yellow
    "\033[94m" # Bright Blue
    "\033[95m" # Bright Magenta
    "\033[96m" # Bright Cyan
    "\033[97m" # Bright White
)

# Get the current machine's hostname
current_machine=$(hostname)

# Function to run the command on each machine and color the output
run_command() {
    local machine=$1
    local color=$2
    ssh -o "StrictHostKeyChecking no" "$machine" 'bash -s' < ./ollama_script.sh | while IFS= read -r line; do
        echo -e "${color}${machine}: ${line}\033[0m"
    done
}

# Loop over each machine and run the command in the background
for i in "${!machines[@]}"; do
    machine=${machines[$i]}
    color=${colors[$((i % ${#colors[@]}))]}
    if [ "$machine" != "$current_machine" ]; then
        run_command "$machine" "$color" &
    fi
done

# Wait for all background processes to finish
wait