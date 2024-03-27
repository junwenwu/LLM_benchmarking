
# LLM_benchmarking

## GenAI Pipeline Repository
This explains the process on how to benchmark stability.AI LM2 1.6b model. It uses the GenAI pipeline to evaluate the performance.
GenAI repo is located at:
https://github.com/openvinotoolkit/openvino.genai

Please refer to [how to install OpenVINO](https://docs.openvino.ai/install) on the OpenVINO installation in order to use the OpenVINO GenAI pipeline.
For the usage please refer to the documentation here: https://docs.openvino.ai/2023.3/gen_ai_guide.html.

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
cd llm_bench/python/
pip install -r requirements.txt  

# Optional, install the latest openvino_nightly if needed
pip uninstall openvino
pip install openvino_nightly 
```
Step 4: Login into huggingface if you need to use non public models.
``` 
huggingface-cli login
```

Step 3:  Convert the model to OpenVINO format. Select any huggingface model. In the following example we use `stabilityai/stablelm-2-1_6b`

The scripts used for benchmarking are located under llm_bench/python.


- For FP32/FP16 model. `--save_orig` will save the pytorch model as well.
```
# assuming you are in llm_bench/python/ directory
python3 convert.py \
--model_id stabilityai/stablelm-2-1_6b \
--output_dir models/stablelm-2-1_6b \
-p FP32|FP16
--save_orig
```

- For compressed models: For quantized model (available choices for quantization are INT8, INT8_ASYM, INT_SYM, 4BIT_DEFAULT, INT4_ASYM and INT4_SYM)

- The following with convert the model to FP32, INT8, INT4 and also save the original Pytorch model. 
```
python3 convert.py \
--model_id stabilityai/stablelm-2-1_6b \
--output_dir models/stablelm-2-1_6b \
-c INT8 4BIT_DEFAULT \
--save_orig
```

### The converted model(s) are located under the `$OUTPUT_DIR/dldt`

Step 4: Performance benchmarking:

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
