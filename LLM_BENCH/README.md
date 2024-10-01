
# LLM Benchmarking with OpenVINO

This explains the process of benchmarking LLM with OpenVINO.
For additional information, please refer to the following resources:
- OpenVINO [GenAI Pipeline Repository](https://github.com/openvinotoolkit/openvino.genai)
- OpenVINO [Benchmarking script for LLM](https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python)
- OpenVINO [Large Language Model Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html)
  

#### Step 0: Prepare Environment
```bash
sudo apt update
sudo apt install git-lfs -y
```

#### Step 1: Setup environment
```bash
python3 -m venv python-env
source python-env/bin/activate
pip install update --upgrade
```

#### Step 2:  Setup OpenVINO LLM benchmarking repo
```bash
git clone  https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai/llm_bench/python/
pip install -r requirements.txt  

# Optional, install the latest openvino_nightly if needed
pip uninstall openvino
pip install --upgrade --pre openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```

#### Step 3: Login into huggingface if you need to use non public models.
```bash
huggingface-cli login
```

#### Step 3.1: Download the baseline model.

- The following command will download the original PyTorch model
```bash
huggingface-cli download stabilityai/stablelm-2-1_6b --local-dir models/stablelm-2-1_6b/pytorch/
```

#### Step 4:  Convert the model to OpenVINO format. 

Use `optimum-cli` tool to convert Hugging Face models to the OpenVINO IR format. See detailed info in [Optimum Intel documentation](https://huggingface.co/docs/optimum/main/en/intel/openvino/export).
```bash
optimum-cli export openvino --model <MODEL_NAME> --weight-format <PRECISION> <OUTPUT_DIR>
```

- In the following example we use `stabilityai/stablelm-2-1_6b`
- See more [details on parameters and options here](https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python#2-convert-a-model-to-openvino-ir).
- `--weight-format`: {fp32,fp16,int8,int4}. The weight format of the exported model.
- `--ratio` - If set to 0.8, 80% of the layers will be quantized to int4 while 20% will be quantized to int8.
- `--group_size` - Size of the group of weights that share the same quantization parameters.

**NOTE:** 
- Smaller group_size and ratio values usually improve accuracy at the sacrifice of the model size and inference latency.
- See more details on [OpenVINO LLM weight compression here](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html)
- Models larger than 1 billion parameters are exported to the OpenVINO format with 8-bit weights by default. You can disable it with `--weight-format fp32`.

- The following command will convert the model to FP32. 
```bash
optimum-cli export openvino \
--model stabilityai/stablelm-2-1_6b \
--weight-format fp32 \
 models/stablelm-2-1_6b/fp32
```

- The following command will convert the model to FP16. 
```bash
optimum-cli export openvino \
--model stabilityai/stablelm-2-1_6b \
--weight-format fp16  \
 models/stablelm-2-1_6b/fp16
```

- The following command will convert the model to INT4. 

```bash
optimum-cli export openvino \
--model stabilityai/stablelm-2-1_6b \
--weight-format int4  \
 models/stablelm-2-1_6b/int4
```


#### Step 5: Performance benchmarking:
- See additional [benchmarking parameters here](https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python#3-benchmarking).

```bash
python3 benchmark.py \
-m $OUTPUT_DIR \
-ic $OUTPUT_TOKEN_SIZE \
-lc config.json \
-s $RANDOM_SEED \
-p $PROMPT \
-n $ITERATION_NUMBER
```
- Example benchmarking with PyTorch FP32 model
```bash
python  benchmark.py  \
-m models/stablelm-2-1_6b/pytorch/ \
-ic 512 \
-n 5 \
-p "Give detailed explanation about OpenVINO" \
-f pt
```

- Example benchmarking with OpenVINO INT4 model
```bash
python  benchmark.py  \
-m models/stablelm-2-1_6b/int4 \
-ic 512 \
-n 5 \
-p "Give detailed explanation about OpenVINO"
```


- Example benchmarking with additional OpenVINO settings. See [config.json](https://github.com/junwenwu/LLM_benchmarking/blob/main/config.json).

```bash
python  benchmark.py  \
-m models/stablelm-2-1_6b/int4 \
-ic 512 \
-n 5 \
-lc config.json \
-p "Give detailed explanation about OpenVINO"
```

- Example with prompt file:
```bash
python  benchmark.py  \
-m models/stablelm-2-1_6b/int4 \
-ic 512 \
-n 5 \
-pf prompts/llama-2-7b-chat_l.jsonl 
```

Please refer to the [help file](https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python) on a detailed explanation of the input parameters.
The output provides first token latency, other token latency, and overall latency. Below is sample output:

```bash
[ INFO ] <<< Warm-up iteration is excluded. >>>
[ INFO ] [Total] Iterations: 5
[ INFO ] [Average] Prompt[0] Input token size: 7, 1st token lantency: 49.31 ms/token, 2nd token lantency: 31.16 ms/token, 2nd tokens throughput: 32.10 tokens/s
```
