"""
Automated Grading System for Student Assignments
------------------------------------------------

This module provides functions and a main workflow to automate the grading 
of student assignments using a language model pipeline. It utilizes a 
HuggingFace-based language model to compare student answers with correct 
answers, assigning a grade from 0 to 10 based on predefined grading prompts.

Key Components:
    - Loading Functions: Functions to load correct answers and student answers from pickle files.
    - Prompt Formatting: A utility to structure grading prompts that include answer context and expectations.
    - LLM Invocation: Uses the HuggingFacePipeline to generate a grading response based on the student's answer.
    - Main Grading Workflow: The main function handles loading, prompt formatting, grading, and output of results.

This system is designed to run locally with language model pipelines, and it includes detailed tracking with the 
`@traceable` decorator for debugging or performance monitoring.

Author: Anand Kamble
Date: May 28, 2024
Usage: Suitable for automated assignment grading with LLMs, particularly in educational settings where quick and 
       accurate grading feedback is desired.
"""

# %%
import os
from typing import Any, List

from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langsmith import traceable
from utils import constants, pickel_loader

load_dotenv()


def load_correct_answers() -> Any:
    """
    Load the correct answers for assignments from a predefined path.

    This function reads the correct answers stored in a pickle file, 
    returning the data to be used for grading comparison.

    Returns:
    --------
    Any: A data structure containing the correct answers loaded from the pickle file.
    """
    return pickel_loader(constants.CORRECT_ANSWER_PATH)


def load_student_answers(path: str) -> Any:
    """
    Load the student answers for assignments from a given file path.

    This function reads student answers from a specified pickle file, 
    making them available for grading against the correct answers.

    Parameters:
    -----------
    path : str
        The path to the pickle file containing student answers.

    Returns:
    --------
    Any: A data structure containing the student answers loaded from the pickle file.
    """
    return pickel_loader(path)


def format_prompt(
    correct_answer: str, student_answer: str, context: str = "No context provided"
) -> List[dict[str, str]]:
    """
    Format the grading prompt for use by the language model grader.

    This function creates a structured prompt that includes a system role 
    and a user role. The system role provides grading instructions and 
    expected answers, while the user role presents the student’s response 
    to be graded.

    Parameters:
    -----------
    correct_answer : str
        The reference answer for the assignment question.

    student_answer : str
        The answer provided by the student for grading.

    context : str, optional
        Additional contextual information about the assignment, by default "No context provided".

    Returns:
    --------
    List[dict[str, str]]: A formatted list of dictionaries containing the 
    system and user prompts.
    """
    return [
        {
            "role": "system",
            "content": f"You are a assignment grader, you grade explanations from 0 to 10.\nContext:{context} \nThe expected explanation is:{correct_answer}",
        },
        {
            "role": "user",
            "content": f"Please grade the following answer: {student_answer}.",
        },
    ]


@traceable
def invoke_llm(messages: List[dict[str, str]]) -> Any:
    """
    Invoke the language model with the specified grading prompt.

    This function uses the HuggingFacePipeline to invoke a language model 
    with the provided prompts, generating a grading response based on the 
    student's answer.

    Parameters:
    -----------
    messages : List[dict[str, str]]
        A list of dictionaries with system and user prompts.

    Returns:
    --------
    Any: The language model's response to the grading prompt, which typically 
    includes the assigned grade and any additional feedback.
    """
    llm = HuggingFacePipeline.from_model_id(
        model_id="microsoft/Phi-3-mini-128k-instruct",  # The model to be used
        task="text-generation",  # The task to be performed
        pipeline_kwargs={
            "max_length": 1024,  # The maximum length of the generated text
        },
        device_map="auto",  # Automatically select the device to run the model on
    )

    return llm.invoke(messages)


def main():
    """
    Run the main grading process for student assignments.

    This function orchestrates the grading process, loading correct and 
    student answers, formatting the grading prompt, invoking the language 
    model for grading, and displaying the final response.

    Steps:
    ------
    1. Load correct answers and student answers.
    2. Format the grading prompt with the relevant context.
    3. Invoke the language model to grade the student's answer.
    4. Print the grading response to the console.

    Returns:
    --------
    None
    """
    print("Running grader ...")

    # Load correct answers for the assignment
    correct_answers = load_correct_answers()

    # Load student's answers (path is relative to the ./run.sh script)
    student_answers = load_student_answers("./student_code/answers.pkl")

    # Define the context for the grading prompt
    context = ("K-Means and Agglomerative Clustering are two popular unsupervised "
               "machine learning algorithms used for data clustering. While both methods "
               "have their strengths and weaknesses, K-Means is generally more efficient "
               "for large datasets due to its linear time complexity compared to Agglomerative "
               "Clustering's quadratic time complexity. This efficiency allows K-Means to handle "
               "large datasets more quickly and effectively, making it a preferred choice for "
               "production-scale systems. Additionally, K-Means is more scalable and can handle "
               "datasets with over 10,000 data points, whereas Agglomerative Clustering can become "
               "computationally expensive and impractical for such large datasets.")

    # Format the grading prompt using correct and student answers
    messages = format_prompt(
        correct_answers["question1"]["(c) explain"],
        student_answers["question1"]["(c) explain"],
        context=context,
    )

    # Invoke the language model to grade the student's answer
    response = invoke_llm(messages)
    print("Response is: ", response)


if __name__ == "__main__":
    main()
