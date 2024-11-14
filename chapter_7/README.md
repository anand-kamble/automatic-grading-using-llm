# Chapter 7: Distributed Processing and Scalability

This repository contains the scripts and tools used for developing a distributed grading system, allowing load balancing and scalability across multiple machines. The setup involves parallelized query processing, load balancing with Nginx, and task scheduling across classroom machines using Ollama servers.

## Repository Structure

- **Master_node_Script.sh**: Bash script to initialize and manage Ollama servers across multiple classroom machines. This script uses SSH to start the Ollama service on each machine and coordinates the distribution of queries.
- **PerfCounterTimer.py**: Python script for capturing and recording the time taken by each task during distributed processing. It uses the `PerfCounterTimer` class to log task durations in a structured format, which is helpful for performance analysis.
- **TaskScheduler.py**: Python class that implements a round-robin scheduling algorithm to distribute query tasks evenly across available machines. It handles task creation, scheduling, execution, and logs any failed tasks.
- **ollama_script.sh**: Script for setting up and running the Ollama servers on individual classroom machines. It configures each server with the required parameters, enabling them to process queries independently.

## Setup and Usage

### Prerequisites

Ensure the following are installed on each machine in the network:
- Docker and Docker Compose (for running Nginx if needed)
- SSH access configured between the master node and classroom machines
- Python 3.x environment with `pandas` installed for logging and data handling

### Starting the Ollama Servers

1. **Master Node Initialization**  
   Run the `Master_node_Script.sh` on the master node to start the Ollama servers across classroom machines. This script will connect to each machine via SSH and initialize the Ollama service.

   ```bash
   ./Master_node_Script.sh
   ```

2. **Running the test**
    Run the `main.py` file which will require the database and will start queries across all the machines.
    ```bash
    python main.py
    ```


## Results

Experiment logs and timing data are saved in output files specified within each script, allowing for analysis of system scalability and performance.
