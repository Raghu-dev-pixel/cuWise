# cuWise — AI-Powered CUDA Optimization Assistant

`cuWise` is a lightweight CLI tool that uses a local Large Language Model (LLM) to analyze CUDA kernel profiling logs and suggest performance optimizations — **completely offline and private**.

Whether you're profiling unoptimized GPU code or exploring kernel bottlenecks, cuWise generates intelligent insights and code suggestions using models like TinyLLaMA or DeepSeek Coder — no internet, no API keys required.

---

## Why cuWise?

Traditional CUDA profiling tools like Nsight Systems and `nvprof` provide detailed performance metrics, but interpreting those logs requires domain expertise. cuWise simplifies this by using LLMs to:

- Identify performance bottlenecks (e.g. low occupancy, memory divergence)
- Suggest improvements (e.g. shared memory use, better grid/block config)
- Recommend optimized code snippets
- Estimate potential performance gains
- Explain in developer-friendly language

---

---

## 🔍 cuWise vs Other Tools

cuWise is a **focused, offline CUDA optimization assistant**. Here's how it compares with other approaches:

| Feature                                 | **cuWise (This Tool)**                           | **Local LLM (e.g., DeepSeek manually)**        | **ChatGPT / Copilot**                        |
|-----------------------------------------|--------------------------------------------------|------------------------------------------------|----------------------------------------------|
| Offline and private                     | Yes                                              | Yes (if self-hosted)                           | No                                           |
| Accepts `.cu` files + profiling logs    | Yes – directly parsed                            | No – manual input required                     | No – manual input required                   |
| Domain-specific prompting               | Yes – CUDA-optimized prompts built-in            | No – prompts must be crafted manually          | No – general-purpose responses               |
| Command-line automation                 | Yes – designed for CLI use                       | No – interactive only                          | No – chat-based only                         |
| Code + performance suggestions          | Yes – revised kernel code and tuning tips        | Partial – depends on prompt quality            | Sometimes                                    |
| Handles long profiling logs             | Yes – safe truncation built in                   | No – risk of overflow                          | Yes (with token limits)                      |
| Hardware-aware insights                 | Yes – parses GPU info from logs                  | No                                             | No                                           |
| API key required                        | No – works fully offline                         | No (for local models)                          | Yes – OpenAI or GitHub login required        |
| Best suited for                         | CUDA developers seeking repeatable optimization  | ML researchers experimenting with local models | General coding help and suggestions          |


---

## Features

- **Offline Inference** — zero cloud usage
- **Log + Code Input** — combines kernel logs and `.cu` files
- **Actionable Suggestions** — performance + code insights
- **LLM-Powered** — supports GGUF models (e.g., DeepSeek Coder, TinyLLaMA)
- **Readable Output** — results printed + saved as `.txt`

---

## Installation

### Prerequisites

- Python 3.8+
- WSL2 or Linux (Windows works via WSL2)
- `llama-cpp-python` installed
- CUDA-compatible logs (`nvprof`, Nsight, or manual)
- GGUF quantized model (e.g., `*.Q4_K_M.gguf`)

---

### Usage

```bash
- git clone https://github.com/yourusername/cuWise.git
- cd cuWise
- pip install llama-cpp-python
- Download a GGUF Model and place it in the same directory as the cuWise.py file.
- Make it executable.
  chmod +x cuWise.py
- Run the cuWise application along the profiling log.
  ./cuWise.py --log cuwise_input.txt
```

### Sample Output1
```
    -  **Performance Gain Estimation:** ...

    **Issue Identified:** The kernel is utilizing only one thread and one block. This is inefficient as it does not utilize the GPU's full potential.

    **Optimization Suggestion:** To optimize this kernel, we can launch multiple threads in the kernel and divide the workload among them. We can do this by changing the kernel launch configuration to use a grid of blocks and blocks of threads.

    **Code Example:**
    ```cpp
    // Use grid of blocks and blocks of threads
    dim3 gridSize(10, 1, 1);  // 10 blocks
    dim3 blockSize(256, 1, 1);  // each block with 256 threads
    add<<<gridSize, blockSize>>>(N, x, y);
    ```
    This will give us more parallelism and can potentially speed up our computation.

    **Performance Gain Estimation:** The potential performance gain is based on the number of threads and blocks we launch. The more threads and blocks we launch, the more parallel operations we can perform, potentially leading to a speedup. However, the exact performance gain can only be accurately measured by profiling the optimized kernel.

    Please note that while this analysis provides a general direction on how to optimize this kernel, the actual performance gains can vary depending on the specific configuration of the GPU, the size of the arrays, and the specific hardware architecture of the GPU.
```
### Sample Output2
```
    -  **Performance Gain Estimation:** ...

    **Issue Identified:** The kernel is using a large amount of shared memory (49152 bytes). This is more than the maximum amount of shared memory available on the Tesla T4 GPU, which is 48KB. This could lead to bank conflicts and slow memory access.

    **Optimization Suggestion:** To reduce the amount of shared memory used, we could use shared memory to store a subset of the input arrays. This would allow each thread to access its own elements of the input arrays, avoiding bank conflicts.

    **Code Example (if applicable):**
    ```cpp
    // Kernel with shared memory optimization
    __global__
    void add(int n, float* x, float* y)
    {
        __shared__ float x_shared[256];
        __shared__ float y_shared[256];

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            x_shared[threadIdx.x] = x[i];
            y_shared[threadIdx.x] = y[i];
            __syncthreads();
            if (i < n)
                y[i] = x_shared[threadIdx.x] + y_shared[threadIdx.x];
            __syncthreads();
        }
    }
    ```

    **Performance Gain Estimation:** This change should result in improved memory access patterns and reduced memory traffic, potentially leading to improved performance. However, the exact performance gain will depend on the specific characteristics of the input data.

    **Note:** The optimized kernel assumes that the array size (N) is a multiple of the block size (256 threads). If this is not the case, some threads will need to access elements outside the allocated shared memory.

    **Note:** The shared memory optimization is a simple example of how shared memory can be used to reduce memory traffic and improve memory access patterns. In some cases, more sophisticated memory access patterns or data layout changes may be needed to fully leverage shared memory.
```
### Roadmap
 - Log + Kernel File Input

 - Offline LLM Support

 - Summary Output File

 - Nsight .nvvp Parser

 - GUI or VSCode Plugin (planned)
