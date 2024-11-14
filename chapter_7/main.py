"""
Distributed Query Processing Script for LLM-driven Grading System
=================================================================
This script performs distributed query processing across multiple machines, each running an instance of an LLM (Large Language Model).
The script uses a TaskScheduler to parallelize and distribute queries across available machines, recording timings and results for each query.

Key Components:
- Initializes an LLM (Ollama) instance for each machine in the network.
- Distributes queries from the Natural Questions dataset to multiple machines using round-robin scheduling.
- Measures performance metrics (timing) for each query.
- Saves results and timing information to a CSV file for further analysis.

Requirements:
- task_scheduler.py (TaskScheduler class)
- PerfCounterTimer class (for timing)
- Ollama class from llama_index.llms.ollama
"""

#%%
import random
import time
from task_scheduler import TaskScheduler
from llama_index.llms.ollama import Ollama
from timer import PerfCounterTimer
from datasets import load_dataset
import numpy as np
import pandas
#%%
tmr = PerfCounterTimer()
# Available machines
MACHINES: list[str] = [
    "class01", "class02", "class03", "class04", "class05",
    "class06", "class07", "class08", "class09", "class10",
    "class11", "class12", "class13", "class14", "class15",
    "class16", "class17", "class18", "class19"
]
#%%
# LLM
llms: dict[str, Ollama] = dict()
# Initialize LLMs for each machine
# Doing this before running the benchmark since these can be reused for all the queries
# Generating new LLMs for each query would be inefficient
with PerfCounterTimer("llm_init").timeit():
    for machine in MACHINES:
        base_url = f"http://{machine}:11434"
        llms[machine] = Ollama(
            base_url=base_url, model="llama3", request_timeout=240, additional_kwargs={"num_predict": 50}
        )

# Function to query the Ollama API

#%%
def query_api(llm: Ollama, query: str) -> tuple[str, str, float, float, str]:
    start_time = time.perf_counter()
    response = llm.complete(query)
    end_time = time.perf_counter()
    return query, response.text, start_time, end_time, llm.base_url

#%%
# Load the dataset
# I am using 1000 samples from train split
dataset = load_dataset(
    path="google-research-datasets/natural_questions", split="train[:10000]")

queries:list = [x["text"] for x in dataset["question"]]
#%%
ts = TaskScheduler()
exp_start_time: float = time.perf_counter()
# Dynamically add tasks to the scheduler
# for q in queries:
#     llm = llms[random.choice(MACHINES)]
#     ts.add_task(machine, query_api, llm, q)

for i, q in enumerate(queries):
    llm = llms[MACHINES[i % len(MACHINES)]]
    ts.add_task(machine, query_api, llm, q)


ts.execute_tasks()
#%%
results, failed_tasks = ts.get_results()
exp_end_time = time.perf_counter()
#%%
print("Results:", results)
# print("Failed tasks:", failed_tasks)
tmr.report()

# %%
timings = np.array([x[3] - x[2] for x in results])

print(f"Mean time: {np.mean(timings):.4f} seconds")
print(f"Std time: {np.std(timings):.4f} seconds")
print(f"Min time: {np.min(timings):.4f} seconds")
print(f"Max time: {np.max(timings):.4f} seconds")

#%%
df = pandas.DataFrame(results, columns=["query", "response", "start_time", "end_time", "base_url"])
# Add exp_start_time and exp_end_time to the dataframe
df["exp_start_time"] = exp_start_time
df["exp_end_time"] = exp_end_time
# Save the dataframe to a CSV file
df.to_csv(f"random_results_with_10000_{time.time()}.csv")
#%%
print("Total time:", exp_end_time - exp_start_time)
