#!/usr/bin/env python
import pandas as pd
import glob
import os
import datetime
import numpy as np


def combine_csv_files(directory):
    all_files = glob.glob(os.path.join(directory, "run_*.csv"))
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename)
        df_list.append(df)
    return pd.concat(df_list, axis=0, ignore_index=True)


def calculate_metrics(df, group_columns):
    metrics = df.groupby(group_columns).agg({
        'time_to_first_byte': [
            'count', 'mean', 'median',
            lambda x: x.quantile(0.9),
            lambda x: x.std()
        ],
        'model_input_tokens': ['mean'],
        'model_output_tokens': ['mean'],
        'time_to_last_byte': [
            'mean', 'median',
            lambda x: x.quantile(0.9)
        ]
    }).round(3)

    metrics.columns = [
        'sample_size', 'TTFT_mean', 'TTFT_p50', 'TTFT_p90', 'TTFT_std',
        'avg_input_tokens',
        'avg_output_tokens',
        'total_time_mean', 'total_time_p50', 'total_time_p90'
    ]

    df['OTPS'] = df['model_output_tokens'] / df['time_to_last_byte']
    otps_metrics = df.groupby(group_columns)['OTPS'].agg([
        'mean', 'median',
        lambda x: x.quantile(0.9),
        lambda x: x.std()
    ]).round(3)
    otps_metrics.columns = ['OTPS_mean', 'OTPS_p50', 'OTPS_p90', 'OTPS_std']

    metrics = pd.concat([metrics, otps_metrics], axis=1)
    return metrics


def analyze_latency_metrics(directory):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Loading data from:", directory)
    df = combine_csv_files(directory)
    errored_requests = df[df['api_call_status'] != 'Success']
    errored_count = len(errored_requests)
    throttled_requests = df[df['api_call_status'] == 'ThrottlingException']
    throttled_count = len(throttled_requests)
    df = df[df['api_call_status'] == 'Success']
    df['OTPS'] = df['model_output_tokens'] / df['time_to_last_byte']

    print("\nSummary Statistics")
    print(f"Total API calls: {len(df) + errored_count}")
    print(f"Successful calls: {len(df)}")
    print(
        f"Errored calls: {errored_count} ({(errored_count/(len(df) + errored_count)*100):.1f}%)")
    print(
        f"Throttled calls: {throttled_count} ({(throttled_count/(len(df) + throttled_count)*100):.1f}%)")

    print("\nToken Statistics")
    print(f"Average Input Tokens: {df['model_input_tokens'].mean():.1f}")
    print(f"Max Input Tokens: {df['model_input_tokens'].max():.0f}")
    print(f"Average Output Tokens: {df['model_output_tokens'].mean():.1f}")
    print(f"Max Output Tokens: {df['model_output_tokens'].max():.0f}")

    print("\nModel Information")
    print(f"Number of unique models: {df['model'].nunique()}")
    print("Models:")
    for model in df['model'].unique():
        print(f"  • {model}")

    if 'performance_config' in df.columns:
        print("\nInference Profiles (PerformanceConfig):")
        for profile in df['performance_config'].unique():
            print(f"  • {profile}")

    if 'performance_config' in df.columns:
        print("\nSample Distribution:")
        model_profile_counts = df.groupby(
            ['model', 'performance_config', 'region']).size()
        for (model, profile, region), count in model_profile_counts.items():
            model_display_name = model.split('.')[-1]
            print(
                f"  • {model_display_name} in {region} with ({profile}) inference: {count} samples")

    metrics = calculate_metrics(df, ['model', 'region', 'performance_config'])
    csv_file = os.path.join(f"{directory}-analysis",
                            f'analysis_summary_{timestamp}.csv')
    os.makedirs(f"{directory}-analysis", exist_ok=True)
    metrics.to_csv(csv_file)
    print(f"\nCSV summary saved to: {csv_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "results"
    analyze_latency_metrics(directory)
