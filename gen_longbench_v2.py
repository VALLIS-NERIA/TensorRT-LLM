from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import json
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from functools import partial
import os
import random

# Load the dataset
tokenizer = AutoTokenizer.from_pretrained(
    "/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1-FP4")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def process_item_chunk(items):
    return process_batch(items)


def estimate_token_length(text):
    # Simple estimation function to approximate token count before full tokenization
    # This is a rough estimate to quickly filter very long texts
    # Typical tokenizers produce ~1.3 tokens per word
    return len(text.split()) * 1.3


def load_long_data_collections(max_samples=1024, avg_length=4096, delta=512,
                               name='togethercomputer/Long-Data-Collections', format='trtllm-bench'):
    print(f"Loading dataset {name}...")
    MIN_LENGTH = avg_length - delta
    MAX_LENGTH = avg_length + delta

    dataset = load_dataset(
        "togethercomputer/Long-Data-Collections", split='train', streaming=True)
    table = {'prompt': [], 'isl': []}
    count = 0
    for item in dataset:
        if 'prompt' not in item:
            continue
        prompt = item['prompt']
        if len(prompt) > 16384:
            continue
        input_ids = tokenizer.encode_plus(prompt).input_ids
        prompt_len = len(input_ids)
        if format == 'trtllm-bench':  # trtllm-bench uses input_ids directly
            prompt = input_ids
        if prompt_len > MAX_LENGTH or prompt_len < MIN_LENGTH:
            continue

        table['prompt'].append(prompt)
        table['isl'].append(prompt_len)

        if count > max_samples:
            break
        count += 1
        print(count)
    return table


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', type=str, default='trtllm-bench',
                        choices=['trtllm-bench', 'benchmark_serving'])
    parser.add_argument('--max_samples', type=int, default=1024)
    parser.add_argument('--avg_length', type=int, default=4096)
    parser.add_argument('--delta', type=int, default=512)
    args = parser.parse_args()
    dataset = load_long_data_collections(max_samples=args.max_samples,
                                         avg_length=args.avg_length,
                                         delta=args.delta,
                                         format=args.format)
    df = pd.DataFrame.from_dict(dataset)
    print(df.describe())
    df.to_csv(
        f'long_data_collections_samples_{args.max_samples}_minlen{args.min_length}_maxlen{args.max_length}.csv', index=False)
    i = 0
    if args.format == 'benchmark_serving':
        list = []
        for i in range(len(dataset['prompt'])):
            list.append({
                "prompt": dataset['prompt'][i],
                "prompt_len": dataset['isl'][i],
                "expected_output_len": 1024,
            })
        with open(f"dataset_{args.avg_length}_serving.json", "w") as f:
            json.dump(list, f)
    elif args.format == 'trtllm-bench':
        with open(f"dataset_{args.avg_length}.json", "w") as f:
            for input in df['prompt']:
                f.write('{"task_id":'+str(i)+',"input_ids":['+",".join(
                    [str(x) for x in input])+'],"output_tokens":1024}\n')
                i += 1
    # plt.hist(df['isl'], bins=100)
    # plt.savefig(f'long_data_collections_samples_{MAX_SAMPLES}_minlen{MIN_LENGTH}_maxlen{MAX_LENGTH}.png')
    # pdb.set_trace()
