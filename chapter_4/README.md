# Chapter 4: Initial Experiments with LLMs

This repository contains the setup and experiment code for hosting and evaluating large language models (LLMs) using Docker, LangChain, and Ngrok. This environment supports automated grading through a RetrievalQA chain and is set up for efficient remote access and testing.

## Repository Structure

- **Dockerfile**: Builds the Docker image required to host the Llama.cpp environment. This file includes steps to install necessary dependencies, copy files, and set permissions for executables.
- **docker-compose.yaml**: Defines the Docker Compose configuration for easier container orchestration. This file simplifies the setup of the container, including port mapping and GPU allocation.
- **langchain_experiment.py**: Contains the code for running LangChain experiments using the RetrievalQA chain with Llama.cpp. This script initializes the model, loads documents, and performs document retrieval and response generation.
- **run_server.sh**: A bash script to automate the startup of the Llama.cpp server with Ngrok for WAN access. It handles Ngrok configuration, including setting up the authentication token.
- **utils.py**: Utility functions for document loading, embedding creation, and vector storage. This file includes helper methods used in `langchain_experiment.py`.

## Setup and Installation

### Prerequisites

Ensure Docker and Docker Compose are installed. Also, install Ngrok and add the authentication token if you plan to use remote access.

### Build the Docker Image

To build the Docker container with the Llama.cpp environment:

```bash
docker build -t llama_experiment .
```

### Run the Container

You can start the container using Docker Compose:

```bash
docker-compose up
```

This will create and start the container with GPU support (if available) and expose the server on port 8080, allowing remote access through Ngrok.

### Running Experiments

Use the following command to run the experiment script:

```bash
python langchain_experiment.py
```

This script will:
1. Load documents into a Chroma vector store.
2. Initialize Llama.cpp with specified configurations.
3. Run the RetrievalQA chain on sample questions and log responses and ratings.

## Ngrok Configuration

To configure Ngrok, edit `run_server.sh` with your Ngrok authentication token. Start the server by running:

```bash
./run_server.sh
```

