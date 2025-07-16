# cuWise 🧠⚡ — AI-Powered CUDA Optimization Assistant

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

### Clone the repo

```bash
git clone https://github.com/yourusername/cuWise.git
cd cuWise
