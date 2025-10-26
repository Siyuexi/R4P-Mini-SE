import argparse
import os
import datasets
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import verl_utils.data.source as src

def make_map_fn(data_source):

    if 'group' in data_source:
        process_fn = src.batch_prompt.process_fn
    elif 'rm' in data_source:
        process_fn = src.rm_prompt.process_fn
    elif 'vm' in data_source:
        process_fn = src.vm_prompt.process_fn
    elif 'ttsr1' in data_source:
        process_fn = src.tts_round1_prompt.process_fn
    elif 'ttsr2' in data_source:
        process_fn = src.tts_round2_prompt.process_fn
    elif 'pair' in data_source:
        process_fn = src.pair_prompt.process_fn
    else:
        raise ValueError(f"Unknown dataset: {data_source}")

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="agentic")
    parser.add_argument("--file_path", default='data/info_train.parquet')

    args = parser.parse_args()

    dataset = datasets.Dataset.from_parquet(args.file_path)
    dataset = dataset.map(function=make_map_fn(args.data_source), with_indices=True)
    print(dataset)
    dataset.to_parquet(args.file_path.replace("info", "data"))
