# Chapter 6: Evaluation Techniques and Tools

This repository contains the scripts and configurations used for evaluating the LLM-driven grading system. It includes tools and methods for Retrieval-Augmented Generation (RAG) evaluation, model tracing, synthetic data generation, and comparative evaluation of different LLM models.

## Repository Structure

- **Evaluation-Implementation.py**: Contains the main setup for embedding models, document loading, and vector indexing to facilitate reproducible evaluation results across different LLM models.
- **Langsmith-Tracing.py**: Implements LangSmith tracing for in-depth monitoring of each RAG evaluation phase, aiding in debugging and improving model response accuracy.
- **Phoenix-Evaluation.py**: Contains code for setting up and executing RAG evaluations using the Phoenix framework. This script includes the setup for document embedding and response generation with the phi3 model.
- **Testset-Generation.py**: Uses the Ragas TestsetGenerator to create synthetic data for comprehensive testing. This script helps simulate a range of student responses for model evaluation.
- **Trulens-Evaluation.py**: Implements TruLens tracing for quality assessment. This script includes feedback functions to measure relevance, groundedness, and context in the model’s responses.

## Evaluation Steps

1. **RAG Evaluation Using Phoenix**  
   The `Phoenix-Evaluation.py` script evaluates RAG capabilities by embedding documents and querying the LLM with specific questions. This setup provides insights into retrieval accuracy and response quality. To run the evaluation:
   ```bash
   python Phoenix-Evaluation.py
   ```

2. **Tracing with LangSmith**  
   The `Langsmith-Tracing.py` script tracks each step of RAG evaluation, enabling detailed inspection of retrieval and response phases. Access the LangSmith dashboard to visualize the results:
   ```bash
   python Langsmith-Tracing.py
   ```

3. **TruLens Tracing**  
   `Trulens-Evaluation.py` uses TruLens for feedback-based quality assessment. This script checks for groundedness, answer relevance, and context relevance in responses.
   ```bash
   python Trulens-Evaluation.py
   ```

4. **Synthetic Data Generation**  
   The `Testset-Generation.py` script generates synthetic datasets using Ragas TestsetGenerator. These datasets simulate varied student response patterns for comprehensive model testing.
   ```bash
   python Testset-Generation.py
   ```

5. **Comparative Evaluation**  
   The `Evaluation-Implementation.py` script facilitates comparative analysis between Llama 3 and Llama 3.1 models. The results are stored as CSV files in the `results` directory, with timing information saved in text files.

## Results

All evaluation results, including CSV files with metric comparisons and timing logs, are saved in the `results` directory with the naming pattern:
```plaintext
results/<DATASET>_query_<QUERY_MODEL>_eval_<EVALUATION_MODEL>.csv
```
