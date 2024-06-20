
# LLM Benchmarking with OpenVINO

This explains the process of benchmarking LLM with OpenVINO.
For additional information, please refer to the following resources:
- OpenVINO [GenAI Pipeline Repository](https://github.com/openvinotoolkit/openvino.genai)
- OpenVINO [Benchmarking script for LLM](https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python)
- OpenVINO [Large Language Model Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html)
  

Step 0: Prepare Environment
```
sudo apt update
sudo apt install git-lfs -y
```

Step 1: Setup environment
```
python3 -m venv python-env
source python-env/bin/activate
pip install update --upgrade
```

Step 2:  Setup OpenVINO LLM benchmarking repo
```
git clone  https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai/llm_bench/python/
pip install -r requirements.txt  

# Optional, install the latest openvino_nightly if needed
pip uninstall openvino
pip install --upgrade --pre openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
```
Step 4: Login into huggingface if you need to use non public models.
``` 
huggingface-cli login
```

Step 3:  Convert the model to OpenVINO format. 

- Select any huggingface model. In the following example we use `stabilityai/stablelm-2-1_6b`
- The scripts used for benchmarking are located under `llm_bench/python`.
- See more [details on parameters and options here](https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python#2-convert-a-model-to-openvino-ir).
- `--save_orig` will save the pytorch model as well in `<output_dir>/pytorch` subdirectory.
- `--ratio` - Compression ratio between primary and backup precision, e.g. INT4/INT8.
- `--group_size` - Size of the group of weights that share the same quantization parameters

### NOTE: 
- Smaller group_size and ratio values usually improve accuracy at the sacrifice of the model size and inference latency.
- See more details on [OpenVINO LLM weight compression here](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html)
  
```
# assuming you are in llm_bench/python/ directory
python3 convert.py \
--model_id stabilityai/stablelm-2-1_6b \
--output_dir models/stablelm-2-1_6b \
--precision FP16 \
--save_orig
```

- For compressed/quantized model - [available choices](https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python#2-convert-a-model-to-openvino-ir) for quantization are INT8, INT8_ASYM, INT8_SYM, 4BIT_DEFAULT, INT4_ASYM and INT4_SYM.
- The following command will convert the model to FP32, INT8, INT4 and also save the original Pytorch model. 
```
python3 convert.py \
--model_id stabilityai/stablelm-2-1_6b \
--output_dir models/stablelm-2-1_6b \
--compress_weights INT8 4BIT_DEFAULT \
--save_orig
```

### The converted model(s) are located under the `$OUTPUT_DIR/dldt`

Step 4: Performance benchmarking:
- See additional [benchmarking parameters here](https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python#3-benchmarking).

```
python3 benchmark.py \
-m $OUTPUT_DIR/dldt/FP32/ \
-ic $OUTPUT_TOKEN_SIZE \
-lc config.json \
-s $RANDOM_SEED \
-p $PROMPT \
-n $ITERATION_NUMBER
```

- Example benchmarking with OpenVINO FP32 model
```
python  benchmark.py  \
-m models/stablelm-2-1_6b/pytorch/dldt/FP32/ \
-ic 512 \
-n 1 \
-p "Give detailed explanation about OpenVINO"
```

- Example benchmarking with PyTorch FP32 model
```
python  benchmark.py  \
-m models/stablelm-2-1_6b/pytorch/ \
-ic 512 \
-n 1 \
-p "Give detailed explanation about OpenVINO" \
-f pt
```

- Example benchmarking with additional OpenVINO settings. See [config.json](https://github.com/junwenwu/LLM_benchmarking/blob/main/config.json).
```
python  benchmark.py  \
-m models/stablelm-2-1_6b/pytorch/dldt/FP32/ \
-ic 512 \
-n 1 \
-lc config.json \
-p "Give detailed explanation about OpenVINO"
```

- Example with prompt file:
```
python  benchmark.py  \
-m models/stablelm-2-1_6b/pytorch/dldt/FP32/ \
-ic 512 \
-n 1 \
-pf prompts/llama-2-7b-chat_l.jsonl 
```

Please refer to the [help file](https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python) on the detailed explanation of the input parameters.
The output provides first token latency, other token latency and overall latency.
