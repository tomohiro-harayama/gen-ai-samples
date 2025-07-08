#!/usr/bin/env python
"""
A benchmark conducts one experiment for each launch of this script.
For an experiment, a list of scenarios is loaded from the configuration file `scenarios_config.jsonl`.
The experiment performs `num_runs_per_experiment` runs.
The same `scenario_list` is used throughout the experiment, but it is shuffled randomly for each run.
The main loop iterates `num_runs_per_experiment` times, performing one "run" per iteration.
Each run calls the `execute()` function once.
The `execute()` function uses a `ThreadPoolExecutor` with `num_parallel_invocations` worker threads running in parallel.
If the number of scenarios exceeds the number of worker threads, the shuffled scenarios are queued and processed in order as threads become available.
Each scenario assigned to a worker thread is handled by the `process_scenario()` function, which performs `invocations_per_scenario` API invocations.
Thus, each worker thread processes one scenario at a time, invoking the Bedrock ConverseStream API by `invocations_per_scenario` times for a given scenario.
As a result, each run invokes the ConverseStream API a total of
`invocations_per_scenario * len(randomized_scenario_list)` times.
Therefore, the total number of ConverseStream API invocations in an experiment is:
    `num_runs_per_experiment * invocations_per_scenario * len(scenario_list)`.
Each model has a different response fields. In this benchmark, we are interested in only the time to first byte and time to last byte  as well as the number of input and output tokens.
"""
# coding: utf-8

import boto3
import random
import time
import json
import argparse
import pandas as pd
from datetime import datetime
import pytz
import os
import logging
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm
import concurrent.futures


def get_bedrock_client(region):
    config = Config(
        retries=dict(
            max_attempts=1
        )
    )
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=region,
        config=config
    )


def get_body(model_id, file_path, prompt, max_tokens):
    body = [
        {
            'role': 'user',
            'content': [
                {
                    'text': prompt
                },
            ]
        },
    ]
    inference_config = {
        'maxTokens': max_tokens,
        'temperature': 0,
        'topP': 1
    }
    return body, inference_config


def post_invocation(scenario_config):
    logging.info(
        f'Sleeping for {scenario_config["sleep_between_invocations"]} seconds.')
    time.sleep(scenario_config["sleep_between_invocations"])


def invoke_api(bedrock, file_path,
               prompt,
               performance_config,
               max_tokens,
               model_id="",
               stream=True,
               sleep_on_throttling=5):
    api_call_status = 'Success'
    full_error_message = 'Success'
    time_to_first_byte, time_to_last_byte = None, None
    dt = datetime.fromtimestamp(time.time(), tz=pytz.utc)
    job_timestamp_iso = dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    body, inference_config = get_body(model_id, file_path, prompt, max_tokens)
    output_token_size, input_token_size = None, None

    # print("inference_config: ", inference_config)
    # import sys
    # sys.exit(0)

    while True:
        try:
            start_time = time.time()
            response = bedrock.converse_stream(
                messages=body,
                modelId=model_id,  # TODO: This is in fact inferenceProfileId.
                inferenceConfig=inference_config,
                performanceConfig={
                    'latency': performance_config
                }
            )
            first_byte_time = None
            event_stream = response.get('stream')
            for event in event_stream:
                if 'contentBlockDelta' in event:
                    chunk = event['contentBlockDelta']
                    if chunk and not first_byte_time:
                        first_byte_time = time.time()
                elif 'messageStop' in event:
                    pass
                elif 'metadata' in event:
                    metadata = event['metadata']
                    if 'usage' in metadata:
                        output_token_size = metadata['usage'].get(
                            'outputTokens', None)
                        input_token_size = metadata['usage'].get(
                            'inputTokens', None)
            last_byte_time = time.time()

            if first_byte_time is not None:
                time_to_first_byte = round(first_byte_time - start_time, 2)

            if last_byte_time is not None:
                time_to_last_byte = round(last_byte_time - start_time, 2)

        except ClientError as err:
            full_error_message = str(err)
            api_call_status = err.response['Error']['Code']
            logging.error(f"Got Error: {api_call_status}")
            logging.error(f"Full Error Message: {full_error_message}")
            break
        else:
            break
    return time_to_first_byte, time_to_last_byte, job_timestamp_iso, api_call_status, full_error_message, output_token_size, input_token_size


def execute(scenarios_list, scenario_config, num_parallel_invocations=4, early_break=False):
    all_invocations = []

    def process_scenario(scenario):
        bedrock_client = get_bedrock_client(scenario['region'])
        file_path = scenario['file_path']
        prompt = scenario['prompt']
        # TODO: Non-existent key
        performance_config = scenario['performance_config']
        # performance_config=None
        # performance_config=scenario['performance_config'] # TODO: Putting PerformanceConfig value from performance_config.

        # TODO: Non-existent key
        max_tokens = scenario['configured_output_tokens_for_request']
        model_id = scenario['model_id']
        stream = scenario['stream']

        sleep_on_throttling = scenario_config['sleep_between_invocations']
        invocations_per_scenario = scenario_config['invocations_per_scenario']

        scenario_result = []
        for invocation_id in range(invocations_per_scenario):
            try:
                time_to_first_byte, time_to_last_byte, job_timestamp_iso, api_call_status, \
                    full_error_message, model_output_tokens, model_input_tokens = invoke_api(
                        bedrock_client,
                        file_path,
                        prompt,
                        performance_config=performance_config,
                        max_tokens=max_tokens,
                        model_id=model_id,
                        stream=stream,
                        sleep_on_throttling=sleep_on_throttling
                    )

                invocation_result = {
                    'time_to_first_byte': time_to_first_byte,
                    'time_to_last_byte': time_to_last_byte,
                    'job_timestamp_iso': job_timestamp_iso,
                    'configured_output_tokens_for_request': scenario['configured_output_tokens_for_request'],
                    'model_input_tokens': model_input_tokens,
                    'model_output_tokens': model_output_tokens,
                    'model': scenario['model_id'],
                    'region': scenario['region'],
                    'invocation_id': invocation_id,
                    'api_call_status': api_call_status,
                    'full_error_message': full_error_message,
                    'temperature': scenario_config.get('temperature'),
                    'top_p': scenario_config.get('top_p'),
                    'top_k': scenario_config.get('top_k'),
                    'experiment_name': scenario_config.get('experiment_name'),
                    'task_type': scenario['task_type'],
                    'performance_config': scenario['performance_config'],
                }
                scenario_result.append(invocation_result)

                with logging_lock:
                    logging.info(f'Invocation: {invocation_result}')

                post_invocation(scenario_config=scenario_config)

            except Exception as e:
                with logging_lock:
                    logging.error(
                        f"Error while processing scenario: {scenario['model_id']}. Error: {e}")

        return scenario_result

    with ThreadPoolExecutor(max_workers=num_parallel_invocations) as executor:
        future_to_scenario = {executor.submit(process_scenario, scenario): scenario
                              for scenario in scenarios_list}

        logging.info(f"Total scenarios submitted: {len(future_to_scenario)}")
        logging.info(f"Number of parallel workers: {num_parallel_invocations}")

        for future in concurrent.futures.as_completed(future_to_scenario):
            try:
                result = future.result()
                all_invocations.extend(result)
            except Exception as e:
                with logging_lock:
                    logging.error(f"Scenario failed: {e}")

        return all_invocations


if __name__ == "__main__":

    # Create the results directory and log directory if they do not exist
    directory = "results"
    os.makedirs(directory, exist_ok=True)
    os.makedirs(f"{directory}-log", exist_ok=True)

    scenario_config_file = "scenarios_config.jsonl"
    experiment_config_file = "experiment_config.jsonl"

    logging_lock = Lock()
    logging.basicConfig(filename=f"{directory}-log/latency-benchmarking-{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.path.exists(experiment_config_file):
        default_experiment_config = {
            "sleep_between_invocations": 60,
            "invocations_per_scenario": 1,
            "num_parallel_invocations": 4,
            "num_runs_per_experiment": 5,
            "temperature": 1,
            "top_p": 1,
            "top_k": 250,
            "experiment_name": "us-east-2-test-1"
        }
        with open(experiment_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(default_experiment_config) + "\n")

    # Read the experiment configuration
    with open(experiment_config_file, "r", encoding="utf-8") as f:
        experiment_config = json.loads(f.readline().strip())
    num_parallel_invocations = experiment_config.get(
        "num_parallel_invocations", 4)
    num_runs_per_experiment = experiment_config.get(
        "num_runs_per_experiment", 5)

    # Read the scenario configuration
    scenario_list = []
    with open(scenario_config_file, 'r', encoding='utf-8') as f:
        for line in f:
            file = json.loads(line.strip())
            prompt = file.get('text_prompt')
            task_type = file.get('task_type')
            # This is the inferenceProfileId, as this benchmark is for on-demand inference requiring the inference profile.
            model_id = file.get('model_id')
            region = file.get('region')
            performance_config = file.get(
                'performance_config', 'standard')  # This is where
            out_tokens = file.get('expected_output_tokens', 100)
            scenario_list.append({
                "file_path": scenario_config_file,
                "configured_output_tokens_for_request": out_tokens,
                "prompt": prompt,
                "stream": True,
                "model_id": model_id,
                "region": region,
                "task_type": task_type,
                "performance_config": performance_config
            })

    for run_count in tqdm(range(1, num_runs_per_experiment + 1), desc="Experiment Runs"):
        randomized_scenario_list = random.sample(
            scenario_list, k=len(scenario_list))

        with logging_lock:
            logging.info(
                f"{len(randomized_scenario_list)} scenarios x {experiment_config['invocations_per_scenario']} invocations = {len(randomized_scenario_list) * experiment_config['invocations_per_scenario']} total invocations")
            logging.info(
                f"Running {run_count}-th run of {num_runs_per_experiment} runs")

        run_result = execute(
            randomized_scenario_list,
            experiment_config,
            num_parallel_invocations=num_parallel_invocations,
            early_break=False
        )

        df = pd.DataFrame(run_result)
        df['timestamp'] = pd.Timestamp.now()
        df['run_count'] = run_count

        output_file = f"{directory}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)

        with logging_lock:
            logging.info(f"Results written to {output_file}")
            logging.info(
                f"Completed {run_count}-th run of {num_runs_per_experiment} runs")
