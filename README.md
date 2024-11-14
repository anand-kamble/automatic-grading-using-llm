# Automatic Grading Using LLMs

This repository contains the code and configurations for a project exploring the use of Large Language Models (LLMs) in automating grading in educational contexts. The project leverages distributed processing, retrieval-augmented generation (RAG), and scalability techniques to assess model accuracy and performance across a range of tasks.

## Repository Structure

- **LICENSE**: License for the repository.
- **README.md**: Documentation and instructions for the repository.
- **chapter_4/**: Code for initial experiments with LangChain and Llama.cpp, including Docker configuration.
  - **Dockerfile**: Docker setup for deploying the Llama.cpp environment.
  - **README.md**: Detailed instructions for Chapter 4.
  - **docker-compose.yaml**: Docker Compose file for container management.
  - **langchain_experiment.py**: Experimentation script using LangChain.
  - **run_server.sh**: Script to run the Llama.cpp server.
  - **utils.py**: Utility functions for the experiments.

- **chapter_6/**: Evaluation techniques and tools used in assessing system performance.
  - **Evaluation-Implementation.py**: Implementation for evaluating the grading model.
  - **Langsmith-Tracing.py**: Tracing implementation using Langsmith.
  - **Phoenix-Evaluation.py**: Evaluation script for Phoenix RAG framework.
  - **README.md**: Documentation for Chapter 6.
  - **Testset-Generation.py**: Script for generating synthetic test data with Ragas.
  - **Trulens-Evaluation.py**: Evaluation script using TruLens.

- **chapter_7/**: Distributed processing and scalability implementation for handling high query volumes.
  - **Master_node_Script.sh**: Script to initialize distributed Ollama servers across classroom machines.
  - **PerfCounterTimer.py**: Timer utility to measure task execution performance.
  - **README.md**: Documentation for Chapter 7.
  - **TaskScheduler.py**: Task scheduler for distributing tasks using round-robin scheduling.
  - **main.py**: Main script for executing distributed query processing.
  - **ollama_script.sh**: Script for setting up Ollama servers on remote machines.

- **logo/**: Contains logos or images used in the project.

## Project Overview

### Objective
This project aims to explore the feasibility of LLMs in automated grading by setting up a scalable, efficient, and accurate grading system using distributed LLM instances across multiple machines.

### Key Features
- **Scalability**: Distributed processing setup to handle high volumes of grading queries.
- **Evaluation Techniques**: Assessment of grading accuracy using tools like Phoenix, TruLens, and LangSmith.
- **Synthetic Data Generation**: Test dataset generation using Ragas TestsetGenerator for robust system evaluation.

## Setup and Usage

### Prerequisites
- Python 3.10 or higher.
- Docker and Docker Compose.
- SSH access for distributed processing across networked machines.

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/anand-kamble/automatic-grading-using-llm
   cd automatic-grading-using-llm
   ```
