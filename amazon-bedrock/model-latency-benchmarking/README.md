# Benchmarking Amazon Bedrock `ConverseStream` API

## Introduction

This tool benchmarks Amazon Bedrock models using the **ConverseStream** API, measuring latency and token usage for different scenarios. The code is cutomized from `model-latency-benchmarking/latency-benchmarking-tool.ipynb` at [amazon-bedrock-samples GitHub](https://github.com/aws-samples/amazon-bedrock-samples/tree/main).

## How it works

- Loads scenarios from `scenarios_config.jsonl` and experiment configuration from `experiment_config.jsonl`.
- Runs each scenario multiple times, shuffling the order of scenarios for each run.
- Uses multiple worker threads to send requests to the ConverseStream API endpoint in parallel.
- Records time to first byte, time to last byte, input/output token counts and other response fields for each API call.

ConverseStream provides a consistent API for all supported Amazon Bedrock models but currently AWS CLI does not support streaming operations, including ConverseStream. According to the documentation, Amazon Bedrock does not store any content you provide; data is only used to generate responses.

### Notes

1. **On the inference parameters**

   Inference parameters, inference configuration, inference profile, and performance configurations are all confusing terms.

   - "Inference parameters" refers to the set of configurable options that control how a machine learning model processes input data and generates output during inference.
   - An inference configuration (`InferenceConfiguration`) is described [here](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html).
   - An inference profile (`InferenceProfile`) is conceptually a predefined model setting and its behavior. It includes performance configuration (`PerformanceConfig`) for latency optimimzation.


2. **`modelId` is set to inferenceProfileId for on-demand inference** 

   On-demand throughput isn’t supported: The model you specified cannot be used in on-demand mode (where you pay per use without pre-provisioning resources). If you set a model ID to `modelId`, you will encounter the errors. So `modelId` in a URI request parameter should be `inferenceProfileId`. Refer to: [ConverseStream API Request Parameters](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html#API_runtime_ConverseStream_RequestParameters)


3. We measure only latency performance. 

   Note that optimize model inference for latency is currently **Preview**.
   
   Each model has different response fields. In this benchmark, we are interested in only the time to first byte and time to last byte as well as the number of input and output tokens from ConverseStream API. 

   Default inference profile (i.e., PerformanceConfig) is `standard`. The `optimized` models are currently upported as follows:

   | Provider  | Model                   | Regions supporting inference profile      |
   |-----------|-------------------------|------------------------------------------|
   | Amazon    | Nova Pro                | us-east-1, us-east-2                     |
   | Anthropic | Claude 3.5 Haiku        | us-east-2, us-west-2                     |
   | Meta      | Llama 3.1 405B Instruct | us-east-2                                |
   | Meta      | Llama 3.1 70B Instruct  | us-east-2, us-west-2                     |

   Refer to [Latency-Optimized Inference User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/latency-optimized-inference.html)

   Our current region is `us-east-2`. The Latency Optimized Inference feature is in preview release for Amazon Bedrock and is subject to change.

   Also in this benchmark, it is out of scope for evaluating model responses with model reasoning. Refer to [Inference Reasoning User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-reasoning.html).


## Detail steps

**Prerequisite**

The code is customization of jupyter notebook `model-latency-benchmarking/latency-benchmarking-tool.ipynb` at  
[amazon-bedrock-samples GitHub](https://github.com/aws-samples/amazon-bedrock-samples/tree/main)

Refer to Bedrock setup on your client machine (e.g., this code is tested on Macbook Pro as a local client machine.)


**Step 1**. Check which models and inference profiles are available in your region (e.g., us-east-2)

   ```
   aws bedrock list-foundation-models --region us-east-2 | grep modelId
               "modelId": "amazon.titan-embed-text-v2:0",
               "modelId": "amazon.nova-pro-v1:0",
               "modelId": "amazon.nova-premier-v1:0:8k",
               "modelId": "amazon.nova-premier-v1:0:20k",
               "modelId": "amazon.nova-premier-v1:0:1000k",
               "modelId": "amazon.nova-premier-v1:0:mm",
               "modelId": "amazon.nova-premier-v1:0",
               "modelId": "amazon.nova-lite-v1:0",
               "modelId": "amazon.nova-micro-v1:0",
               "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
               "modelId": "anthropic.claude-3-7-sonnet-20250219-v1:0",
               "modelId": "anthropic.claude-3-haiku-20240307-v1:0:200k",
               "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
               "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
               "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
               "modelId": "anthropic.claude-opus-4-20250514-v1:0",
               "modelId": "anthropic.claude-sonnet-4-20250514-v1:0",
               "modelId": "deepseek.r1-v1:0",
               "modelId": "mistral.pixtral-large-2502-v1:0",
               "modelId": "meta.llama3-1-8b-instruct-v1:0:128k",
               "modelId": "meta.llama3-1-8b-instruct-v1:0",
               "modelId": "meta.llama3-1-70b-instruct-v1:0:128k",
               "modelId": "meta.llama3-1-70b-instruct-v1:0",
               "modelId": "meta.llama3-1-405b-instruct-v1:0",
               "modelId": "meta.llama3-2-11b-instruct-v1:0",
               "modelId": "meta.llama3-2-90b-instruct-v1:0",
               "modelId": "meta.llama3-2-1b-instruct-v1:0",
               "modelId": "meta.llama3-2-3b-instruct-v1:0",
               "modelId": "meta.llama3-3-70b-instruct-v1:0",
               "modelId": "meta.llama4-scout-17b-instruct-v1:0",
               "modelId": "meta.llama4-maverick-17b-instruct-v1:0",
   ```

   ```
   aws bedrock list-inference-profiles --region us-east-2 | grep inferenceProfileId
               "inferenceProfileId": "us.anthropic.claude-3-haiku-20240307-v1:0",
               "inferenceProfileId": "us.meta.llama3-2-1b-instruct-v1:0",
               "inferenceProfileId": "us.meta.llama3-2-11b-instruct-v1:0",
               "inferenceProfileId": "us.meta.llama3-2-3b-instruct-v1:0",
               "inferenceProfileId": "us.meta.llama3-2-90b-instruct-v1:0",
               "inferenceProfileId": "us.meta.llama3-1-8b-instruct-v1:0",
               "inferenceProfileId": "us.meta.llama3-1-70b-instruct-v1:0",          <-- optimization supported
               "inferenceProfileId": "us.amazon.nova-micro-v1:0",
               "inferenceProfileId": "us.amazon.nova-lite-v1:0",
               "inferenceProfileId": "us.amazon.nova-pro-v1:0",                     <-- optimization supported
               "inferenceProfileId": "us.anthropic.claude-3-5-haiku-20241022-v1:0", <-- optimization supported
               "inferenceProfileId": "us.meta.llama3-1-405b-instruct-v1:0",
               "inferenceProfileId": "us.meta.llama3-3-70b-instruct-v1:0",
               "inferenceProfileId": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
               "inferenceProfileId": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
               "inferenceProfileId": "us.deepseek.r1-v1:0",
               "inferenceProfileId": "us.mistral.pixtral-large-2502-v1:0",
               "inferenceProfileId": "us.meta.llama4-scout-17b-instruct-v1:0",
               "inferenceProfileId": "us.meta.llama4-maverick-17b-instruct-v1:0",
               "inferenceProfileId": "us.amazon.nova-premier-v1:0",
               "inferenceProfileId": "us.anthropic.claude-opus-4-20250514-v1:0",
               "inferenceProfileId": "us.anthropic.claude-sonnet-4-20250514-v1:0",
               "inferenceProfileId": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
   ```

**Step 2**. Select and verify the model details

Verify if the model supports latency optimization and other fields.

[Latency-Optimized Inference User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/latency-optimized-inference.html)

   ```
   aws bedrock get-foundation-model --model-identifier amazon.nova-pro-v1:0 --region us-east-2
   aws bedrock get-foundation-model --model-identifier meta.llama3-1-70b-instruct-v1:0 --region us-east-2
   aws bedrock get-foundation-model --model-identifier anthropic.claude-3-5-haiku-20241022-v1:0 --region us-east-2
   ```

**Step 3**. Verify the setup configuration files 

Modify two files accordingly: 

* `scenarios_config.jsonl` for the model-specific inference use cases
* `experiment_config.jsonl` for the experiment-specific test configurations.

**Step 4**. Run the benchmark

   ```
   python benchmark_main.py
   ```

**Step 5**. Run the analysis code

   ```
   python benchmark_analysis.py
   ```
The code inputs the csv files under `results` and outputs a single summary result csv file to `results-analysis`. 

Step 6. Review the results.

* `results` stores per-run results of the API calls.
* `results-analysis` stores a summary results.csv

You can use [`csvlook`](https://csvkit.readthedocs.io/en/latest/scripts/csvlook.html) to view the summary results in a readable table format:

   ```
   csvlook results-analysis/analysis_summary_20250708_112957.csv
   ```

| model                                       | region    | performance_config | sample_size | TTFT_mean | TTFT_p50 | TTFT_p90 | TTFT_std | avg_input_tokens | avg_output_tokens | total_time_mean | total_time_p50 | total_time_p90 | OTPS_mean | OTPS_p50 | OTPS_p90 | OTPS_std |
|---------------------------------------------|-----------|-------------------|-------------|-----------|----------|----------|----------|------------------|-------------------|-----------------|----------------|----------------|-----------|----------|----------|----------|
| us.amazon.nova-pro-v1:0                     | us-east-2 | optimized         |           5 |     0.602 |    0.64  |   0.668  |   0.077  |              10  |                50 |           1.074 |          1.06  |         1.140  |    46.680  |   47.170 |   48.829 |    2.640 |
| us.amazon.nova-pro-v1:0                     | us-east-2 | standard          |           5 |     0.600 |    0.53  |   0.754  |   0.142  |              10  |                50 |           1.248 |          1.15  |         1.494  |    41.088  |   43.478 |   45.895 |    6.608 |
| us.anthropic.claude-3-5-haiku-20241022-v1:0 | us-east-2 | optimized         |           5 |     0.764 |    0.77  |   1.052  |   0.282  |              18  |                50 |           1.472 |          1.40  |         1.788  |    35.241  |   35.714 |   43.045 |    7.675 |
| us.anthropic.claude-3-5-haiku-20241022-v1:0 | us-east-2 | standard          |           5 |     0.918 |    0.95  |   0.970  |   0.072  |              18  |                50 |           1.766 |          1.77  |         1.876  |    28.402  |   28.249 |   30.057 |    1.761 |
| us.meta.llama3-1-70b-instruct-v1:0          | us-east-2 | optimized         |           5 |     0.418 |    0.35  |   0.534  |   0.104  |              26  |                50 |           0.712 |          0.64  |         0.830  |    71.388  |   78.125 |   78.869 |    9.890 |
| us.meta.llama3-1-70b-instruct-v1:0          | us-east-2 | standard          |           5 |     0.852 |    0.82  |   0.994  |   0.146  |              25  |                50 |           2.396 |          2.36  |         2.546  |    20.928  |   21.186 |   21.977 |    1.228 |

**Column descriptions:**
- `TTFT_mean`, `TTFT_p50`, `TTFT_p90`, `TTFT_std`: Time to first token (mean, median, 90th percentile, stddev)
- `total_time_mean`, `total_time_p50`, `total_time_p90`: Total response time (mean, median, 90th percentile)
- `OTPS_mean`, `OTPS_p50`, `OTPS_p90`, `OTPS_std`: Output tokens per second (mean, median, 90th percentile, stddev)
- `avg_input_tokens`, `avg_output_tokens`: Average input/output token counts

This summary helps compare latency and throughput across models and configurations (`standard` vs. `optimized`)


## References
- [amazon-bedrock-samples GitHub](https://github.com/aws-samples/amazon-bedrock-samples/tree/main)
- [ConverseStream API Request Parameters](https://docs.aws.amazon.com/bedrock/latest/APIReference/)
- [InferenceConfiguration API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html)
- [Latency-Optimized Inference User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/latency-optimized-inference.html)


## TODO

- Handle non-exisiting parameter entries
- Make prompts customizable