![image](https://github.com/user-attachments/assets/f85ee8cc-52ea-4bb1-927b-35ab58cddc09)
# vLLM Benchmarking with OpenVINO backend

This explains the process of benchmarking vLLM with OpenVINO.
For additional information, please refer to the following resources:
- OpenVINO [GenAI Pipeline Repository](https://github.com/openvinotoolkit/openvino.genai)
- OpenVINO [Large Language Model Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html)
- OpenVINO [Benchmarking script for vLLM](https://docs.vllm.ai/en/latest/getting_started/openvino-installation.html)
  
## Table of Contents

- [Table of Content](#table-of-contents)
- [Installation Guide](#-installation-guide)
- [Benchmarking with vLLM and OpenVINO backend](#-benchmarking-vllm-OpenVINO)
- [For Throughput Scenario](#for-throughput-benchmarking)
- [For Model Serving Scenario](#for-model-serving-benchmarking) 
  
##  Installation Guide
   
### Step 1.0: Prepare Environment:

#### First, install Python. For example, on Ubuntu 22.04, you can run:

```
sudo apt-get update  -y
sudo apt-get install python3
```
#### Step 1.1: Setup environment:

```
python3 -m venv vllm_openvino_env
source ./vllm_openvino_env/bin/activate
pip install --upgrade pip
```
#### Step 1.2:  Installing vLLM with OpenVINO backend:

```
pip install -r https://github.com/vllm-project/vllm/raw/main/requirements-build.txt \
 --extra-index-url https://download.pytorch.org/whl/cpu
```

```
PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/pre-release" \
VLLM_TARGET_DEVICE=openvino python -m pip install -v .
```

#### Step 1.3: [Optional] Quick start with Docker:

```
docker build -f Dockerfile.openvino -t vllm-openvino-env .
docker run -it --rm vllm-openvino-env
```

#### Step 1.4: [Optional] Login into huggingface if you need to use non public models:

huggingface-cli login

## Benchmarking with vLLM and OpenVINO backend

### For Throughput Scenario

Sample command args:

    VLLM_OPENVINO_KVCACHE_SPACE=100 \
    VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 \
    VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
    python3 vllm/benchmarks/benchmark_throughput.py \
    --model <model-id>/<path-to-ov-model-dir> \
    --dataset <path-to-sample-prompt-file> \
    --enable-chunked-prefill --max-num-batched-tokens 256

#### with ```--model``` set to huggingface model id
    
    VLLM_OPENVINO_KVCACHE_SPACE=100 \
    VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 \
    VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
    python3 vllm/benchmarks/benchmark_throughput.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dataset vllm/benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
    --enable-chunked-prefill --max-num-batched-tokens 256

#### with ```--model``` set to local directory (openvino.genai optimized OpenVINO model path)
    
    VLLM_OPENVINO_KVCACHE_SPACE=40 \
    VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 \
    VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
    python3 vllm/benchmarks/benchmark_throughput.py \
    --model ./openvino.genai/llm_bench/python/meta-llama-3x8b-ov/pytorch/dldt/compressed_weights/OV_FP32-INT8 \
    --dataset ./benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json \
    --enable-chunked-prefill --max-num-batched-tokens 256  


#### Input default args

```console
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
#### Sample output logs

```console
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

### Additional vLLM env settings

```console
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

## For Model Serving Scenario

### Setup OpenVINO Model Server

Pull public image with CPU only support or including also GPU support.

```bash
docker pull openvino/model_server:latest-gpu
docker pull openvino/model_server:latest
```

### [Optional] Build model server from source and install dependencies

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make release_image RUN_TESTS=0
```

Install dependencies:

```bash
 pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/releases/2024/3/demos/continuous_batching/requirements.txt
```

### Model Preparation

Install python dependencies for conversion scripts (below command using OpenVINO 2024.4.0 Release)

```bash
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
pip3 install "optimum-intel[nncf,openvino]"@git+https://github.com/huggingface/optimum-intel.git@fe77316c5a25c7b0e8ae97c23776688448490be2 openvino_tokenizers==2024.4.0 openvino==2024.4.0
```
### Run optimum-intel to download and quantize HF model (Meta-Llama-3-8B-Instruct)

```bash

cd demos/continuous_batching

convert_tokenizer -o Meta-Llama-3-8B-Instruct \
--utf8_replace_mode replace --with-detokenizer \
--skip-special-tokens --streaming-detokenizer \
--not-add-special-tokens meta-llama/Meta-Llama-3-8B-Instruct

optimum-cli export openvino --disable-convert-tokenizer \
--model meta-llama/Meta-Llama-3-8B-Instruct \
--weight-format fp16 Meta-Llama-3-8B-Instruct
```
Note: Refer to step 1.4 for hf login

### Prepare graph.pbtxt and config.json

```
cp graph.pbtxt Meta-Llama-3-8B-Instruct/
```

```
cat config.json
{
    "model_config_list": [],
    "mediapipe_config_list": [
        {
            "name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "base_path": "Meta-Llama-3-8B-Instruct"
        }
    ]
}
```

### Launch OpenVINO model server

```
docker run -d --rm -p 8000:8000 \
-v $(pwd)/:/workspace:ro openvino/model_server:latest \
--port 9000 --rest_port 8000 \
--config_path /workspace/config.json
```

Wait for the model to load. You can check the status with a simple command:

`bash curl http://localhost:8000/v1/config`

## Launching becnchmark_serving.py

### Clone vLLM repo

`git clone https://github.com/vllm-project/vllm`

### Installing prerequisites and downloading sample dataset

```bash
cd vllm
pip3 install -r requirements-cpu.txt
cd benchmarks
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### Starting client app for benchmarking

```bash
python benchmark_serving.py --host localhost \
--port 8000 --endpoint /v3/chat/completions \
--backend openai-chat \
--model meta-llama/Meta-Llama-3-8B-Instruct \
--dataset ShareGPT_V3_unfiltered_cleaned_split.json \
--num-prompts 100 \
--request-rate inf
```

### Expected sample output benchmark metrics

```console
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  83.82     
Total input tokens:                      40000     
Total generated tokens:                  9364      
Request throughput (req/s):              x.xx      
Output token throughput (tok/s):         xxx.xx    
Total Token throughput (tok/s):          xxx.xx    
---------------Time to First Token----------------
Mean TTFT (ms):                          4xxx.xx  
Median TTFT (ms):                        4xxx.xx  
P99 TTFT (ms):                           8xxx.xx  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2xx.xx    
Median TPOT (ms):                        3xx.xx    
P99 TPOT (ms):                           4xx.xx    
---------------Inter-token Latency----------------
Mean ITL (ms):                           1xx.xx   
Median ITL (ms):                         4xx.xx    
P99 ITL (ms):                            3xx.xx  
==================================================
```
