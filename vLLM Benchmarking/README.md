
# vLLM Benchmarking with OpenVINO backend

This explains the process of benchmarking vLLM with OpenVINO.
For additional information, please refer to the following resources:
- OpenVINO [GenAI Pipeline Repository](https://github.com/openvinotoolkit/openvino.genai)
- OpenVINO [Large Language Model Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html)
- OpenVINO [Benchmarking script for vLLM](https://docs.vllm.ai/en/latest/getting_started/openvino-installation.html)
  

## 1. Setup and Installation:
   
#### Step 0: Prepare Environment:

#### First, install Python. For example, on Ubuntu 22.04, you can run:

```
sudo apt-get update  -y
sudo apt-get install python3
```
#### Step 1: Setup environment:

```
python3 -m venv vllm_openvino_env
source ./vllm_openvino_env/bin/activate
pip install --upgrade pip
```
#### Step 2:  Installing vLLM with OpenVINO backend:

```
pip install -r https://github.com/vllm-project/vllm/raw/main/requirements-build.txt \
 --extra-index-url https://download.pytorch.org/whl/cpu
```

```
PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/pre-release" \
VLLM_TARGET_DEVICE=openvino python -m pip install -v .

```
#### Step 3: [optional] Login into huggingface if you need to use non public models:

huggingface-cli login

## 2. LLM benchmarking:

#### Available benchmarking options:

1. benchmark_request_func.py
2. benchmark_latency.py
3. benchmark_prefix_caching.py
4. benchmark_serving.py
5. benchmark_throughput.py

#### Performance benchmarking with vLLM + OpenVINO backend: 

Sample command: throughput performance (with vllm/benchmarks/benchmark_throughput.py):

    VLLM_OPENVINO_KVCACHE_SPACE=100 \
    VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 \
    VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
    python3 vllm/benchmarks/benchmark_throughput.py \
    --model <model-id>/<path-to-ov-model-dir> \
    --dataset <path-to-sample-prompt-file> \
    --enable-chunked-prefill --max-num-batched-tokens 256

Example:
    with huggingface model-id:
    ```
        VLLM_OPENVINO_KVCACHE_SPACE=100 \
        VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 \
        VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
        python3 vllm/benchmarks/benchmark_throughput.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --dataset vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
        --enable-chunked-prefill --max-num-batched-tokens 256
    ```
    
    with openvino.genai llm_bench optimized OpenVINO model (path to local direcotry):
    ```
        VLLM_OPENVINO_KVCACHE_SPACE=40 \
        VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 \
        VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
        python3 vllm/benchmarks/benchmark_throughput.py \
        --model ./openvino.genai/llm_bench/python/meta-llama-3x8b-ov/pytorch/dldt/compressed_weights/OV_FP32-INT8 \
        --dataset ./benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
        --enable-chunked-prefill --max-num-batched-tokens 256  
    ```
#### Input default args:

```
num_prompts: 1000
seed: 0
max_length: 400
temperature: 0.7
top_k: 50
top_p: 1.0
repetition_penalty: 1.0
do_sample: True
num_beams: 1
use_cache: True
use_auth_token: True
prior_resample: False
mode: 'gen'
control: ''
input_basename: 'input'
outfile_basename: 'output'
keep_tokens: True
```
#### Sample output logs:

```
WARNING 08-13 20:31:34 openvino.py:130] OpenVINO IR is available for provided model id ../openvino.genai/llm_bench/python/meta
-llama-3x8b-ov/pytorch/dldt/compressed_weights/OV_FP32-INT8. This IR will be used for inference as-is, all possible options th
at may affect model conversion are ignored.                    
INFO:nncf:Statistics of the bitwidth distribution:                                                                            
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│ Num bits (N)   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
INFO 08-13 20:31:36 openvino_executor.py:74] # CPU blocks: 19275
INFO 08-13 20:31:36 selector.py:122] Cannot use _Backend.FLASH_ATTN backend on OpenVINO.
INFO 08-13 20:31:36 selector.py:70] Using OpenVINO Attention backend.
Processed prompts:  42%|████▌      | 418/1000 [03:54<02:52,  3.38it/s, est. speed input: 428.74 toks/s, output: 179.14 toks/s]
Processed prompts:  42%|████▌      | 420/1000 [03:55<03:29,  2.77it/s, est. speed input: 428.36 toks/s, output: 180.72 toks/s]
Processed prompts: 100%|██████████| 1000/1000 [09:06<00:00,  1.83it/s, est. speed input: 393.69 toks/s, output: 362.86 toks/s]
Throughput: 1.83 requests/s, 755.89 tokens/s
```

**Note**: ```requests/s``` in througput metric indicates, throughput per inference (i.e., per prompt). 
The default value set to ```num_prompts=1000```

#### Key output metrics:

```
Elapsed Time: The total time taken to complete all inference requests. This is measured in seconds.
Number of Requests per Second: The throughput measured in terms of the number of inference requests completed per second.
This is calculated as the total number of requests divided by the elapsed time.

Tokens per Second: The throughput measured in terms of the number of tokens processed per second.
This is calculated as the total number of tokens (input and output combined) divided by the elapsed time.

Total Number of Tokens:The sum of all tokens (input and output) processed across all requests during the benchmark.
Requests per Second (Optional JSON Output):

If specified, the results including elapsed time, number of requests, total number of tokens, requests per second,
and tokens per second are saved to a JSON file.
```
## Additional vLLM env settings:

```
# (CPU backend only) CPU key-value cache space.
# default is 4GB
"VLLM_CPU_KVCACHE_SPACE":
lambda: int(os.getenv("VLLM_CPU_KVCACHE_SPACE", "0")),

# (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31",
# "0,1,2", "0-31,33". CPU cores of different ranks are separated by '|'.
"VLLM_CPU_OMP_THREADS_BIND":
lambda: os.getenv("VLLM_CPU_OMP_THREADS_BIND", "all"),

# OpenVINO key-value cache space
# default is 4GB
"VLLM_OPENVINO_KVCACHE_SPACE":
lambda: int(os.getenv("VLLM_OPENVINO_KVCACHE_SPACE", "0")),

# OpenVINO KV cache precision
# default is bf16 if natively supported by platform, otherwise f16
# To enable KV cache compression, please, explicitly specify u8
"VLLM_OPENVINO_CPU_KV_CACHE_PRECISION":
lambda: os.getenv("VLLM_OPENVINO_CPU_KV_CACHE_PRECISION", None),

# Enables weights compression during model export via HF Optimum
# default is False
"VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS":
lambda: bool(os.getenv("VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS", False)),
```

More info can be found [here](https://docs.vllm.ai/en/latest/serving/env_vars.html)
