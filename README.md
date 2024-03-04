# LLM_benchmarking
## GenAI Pipeline Repository
This explains the process on how to benchmark stability.AI LM2 1.6b model. It uses the GenAI pipeline to evaluate the performance.
GenAI repo is located at:
https://github.com/openvinotoolkit/openvino.genai

Please refer to [how to install OpenVINO](https://docs.openvino.ai/install) on the OpenVINO installation in order to use the OpenVINO GenAI pipeline.
For the usage please refer to the documentation here: https://docs.openvino.ai/2023.3/gen_ai_guide.html.

The scrptis used for benchmarking are located under llm_bench/python.
1. Converting the model:

For FP32/FP16 model:
```
        python3 convert.py --model_id stabilityai/stablelm-2-1_6b --output_dir $OUTPUT_DIR -p FP32|FP16
````


2. For quantized model (available choices for quantization are INT8, INT8_ASYM, INT_SYM, 4BIT_DEFAULT, INT4_ASYM and INT4_SYM)
```
        python3 convert.py --model_id stabilityai/stablelm-2-1_6b --output_dir $OUTPUT_DIR -c $COMPRESSION_METHOD
```
The converted model is located under the $OUTPUT_DIR/dldt.

2. Performance benchmarking:
```
{
        python3 benchmark.py -m $OUTPUT_DIR/dldt/FP32/ -ic $OUTPUT_TOKEN_SIZE -lc config.json -s $RANDOM_SEED -p $PROMPT -n $ITERATION_NUMBER
}
```
Please refer to the help file on the detailed explanation of the input parameters.

The output provides first token latency, other token latency and overall latency.
