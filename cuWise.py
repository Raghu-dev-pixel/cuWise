#!/usr/bin/env python3

# An AI based Cuda assistant application

import argparse
import os
from llama_cpp import Llama

def load_model():
    return Llama(
        model_path="deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4
    )

def run_inference(prompt, model):
    output = model(prompt, max_tokens=512)

    result = output["choices"][0]["text"]
    
    #Save the output to a file
    output_path = "/mnt/d/Interview_Prep/Nvidia_interview/Cuda/cuWise/cuwise_output_optimized.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(result)

    return result

#def build_prompt_from_log(log_path):
#    with open(log_path, 'r') as f:
#        log = f.read()
#    prompt = f"### Instruction: Analyze and suggest optimizations for the CUDA kernel profile log:\n{log}\n### Response:"
#    return prompt

def build_prompt_from_log(log_path):
    MAX_LOG_CHARS = 4000  # Optional: limit log length to avoid token overflow

    try:
        with open(log_path, 'r') as f:
            log = f.read()
    except FileNotFoundError:
        return "### Instruction: CUDA log file not found.\n### Response: Please check the file path."

    # Trim long logs for safety
    if len(log) > MAX_LOG_CHARS:
        log = log[:MAX_LOG_CHARS] + "\n... [log truncated]"

    prompt = f"""### Instruction:
    You are a CUDA performance optimization assistant.

    Analyze the following CUDA kernel profiling log and system information. Your goal is to:
    - Identify performance issues (e.g., low occupancy, memory divergence, bank conflicts)
    - Recommend code-level optimizations (e.g., shared memory, grid-stride loops)
    - Suggest improvements to grid/block configurations
    - If helpful, include updated CUDA code snippets
    - Estimate potential performance gains

    Assume the user has intermediate knowledge of CUDA.

    ### Input (Profile Log and GPU Info):
    {log}

    ### Response Format:
    - **Issue Identified:** ...
    - **Optimization Suggestion:** ...
    - **Code Example (if applicable):**
    ```cpp
    // Optimized kernel here
    """
    return prompt


def main():
    parser = argparse.ArgumentParser(description="cuWise: CLI CUDA Optimizer")
    parser.add_argument("--log", type=str, help="Path to the log file (from cuWise runtime)")
    parser.add_argument("--text", type=str, help="Direct prompt input (instead of log file)")
    args = parser.parse_args()

    model = load_model()

    if args.log:
        prompt = build_prompt_from_log(args.log)
    elif args.text:
        prompt = f"### Instruction: {args.text}\n### Response:"
    else:
        print("Please provide either --log or --text input.")
        return
    
    print("\ncuWise LLM Suggestion:\n")
    print(run_inference(prompt, model))


if __name__ == "__main__":
    main()
