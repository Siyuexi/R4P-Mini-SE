import pandas as pd
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from verl_utils.reward.extract_answer import *
from verl_utils.reward.reward_fn import get_batch_reward, get_naive_reward

def get_batch_fp_fn(pred_list, gt_list, type):
    fp = 0
    fn = 0
    if not pred_list:
        return 0
    for i in range(4):
        if pred_list[i] and not gt_list[i]:
            fp += 1
        if not pred_list[i] and gt_list[i]:
            fn +=1
    if type == 'fp':
        return fp
    if type == 'fn':
        return fn

def evaluate_batch(output_path):
    df = pd.read_parquet(output_path)
    if 'ground_truth' not in df.columns:
        df['ground_truth'] = df.apply(lambda x: x['reward_model']['ground_truth'], axis=1)
    if 'messages' not in df.columns:
        df['raw_answer'] = df.apply(lambda x: x['responses'][0], axis=1)
    else:
        df['raw_answer'] = df.apply(lambda x: x['messages'][-1]['content'], axis=1)
    print(df.iloc[0]['raw_answer'])
    print(df.iloc[0]['ground_truth'])
    df['answer'] = df.apply(lambda x: extract_batch_combine(x['raw_answer']), axis=1)
    df['acc'] = df.apply(lambda x: get_batch_reward(x['answer'], x['ground_truth'], 0.0) if x['answer'] else 0.0, axis=1)
    df['f1'] = df.apply(lambda x: get_batch_reward(x['answer'], x['ground_truth'], 1.0) if x['answer'] else 0.0, axis=1)
    df['reward'] = df.apply(lambda x: get_batch_reward(x['answer'], x['ground_truth'], 0.5) if x['answer'] else 0.0, axis=1)
    df['fp'] = df.apply(lambda x: get_batch_fp_fn(x['answer'], x['ground_truth'], 'fp'), axis=1)
    df['fn'] = df.apply(lambda x: get_batch_fp_fn(x['answer'], x['ground_truth'], 'fn'), axis=1)
    df['#p'] = df.apply(lambda x: x['ground_truth'].sum(), axis=1)
    df['#n'] = df.apply(lambda x: (~x['ground_truth']).sum(), axis=1)
    fp_num = df['fp'].sum()
    fn_num = df['fn'].sum()
    p_num = df['#p'].sum()
    n_num = df['#n'].sum()
    invalid_num = df['answer'].isna().sum()
    perfect_num = df[df['acc'] == 1.0].shape[0]
    total_num = df.shape[0]
    assert df[df['acc'] == 1.0].shape[0] == df[df['f1'] == 1.0].shape[0], "Calculation Error: F1 or Acc"
    print('-'*80)
    print(f"Name:\t{output_path.split('verl/')[-1].split('/actor')[0]}")
    print('-'*80)
    print(f"Acc:\t{df['acc'].mean()}")
    print(f"F-1:\t{df['f1'].mean()}")
    print(f"Reward:\t{df['reward'].mean()}")
    print(f"FPR:\t{fp_num/n_num}")
    print(f"FNR:\t{fn_num/p_num}")
    print(f"Error:\t{invalid_num}")
    print(f"Exact:\t{perfect_num}")
    print('-'*80)
    print(f"Total:\t{total_num}")
    print(f"#P:\t{p_num}")
    print(f"#N:\t{n_num}")
    print('-'*80)

def evaluate_naive(output_path, extract_fn):
    df = pd.read_parquet(output_path)
    if 'ground_truth' not in df.columns:
        df['ground_truth'] = df.apply(lambda x: x['reward_model']['ground_truth'], axis=1)
    if 'messages' not in df.columns:
        df['raw_answer'] = df.apply(lambda x: x['responses'][0], axis=1)
    else:
        df['raw_answer'] = df.apply(lambda x: x['messages'][-1]['content'], axis=1)
    print('### Raw Response:')
    print(df.iloc[0]['raw_answer'])
    print('### Ground Truth:')
    print(df.iloc[0]['ground_truth'])
    df['answer'] = df.apply(lambda x: extract_fn(x['raw_answer']), axis=1)
    df['acc'] = df.apply(lambda x: get_naive_reward(x['answer'], x['ground_truth']) if x['answer'] else 0.0, axis=1)
    df['fp'] = df.apply(lambda x: 1.0 if x['acc'] == 0.0 and not x['ground_truth'] else 0.0, axis=1)
    df['fn'] = df.apply(lambda x: 1.0 if x['acc'] == 0.0 and x['ground_truth'] else 0.0, axis=1)
    total_num = df.shape[0]
    fp_num = df['fp'].sum()
    fn_num = df['fn'].sum()
    p_num = df['ground_truth'].sum() if not isinstance(df['ground_truth'].sum(), str) else 0
    n_num = total_num - p_num
    invalid_num = df['answer'].isna().sum()
    perfect_num = df[df['acc'] == 1.0].shape[0]
    print('-'*80)
    print(f"Name:\t{output_path.split('verl/')[-1].split('/actor')[0]}")
    print('-'*80)
    print(f"Acc:\t{df['acc'].mean()}")
    print(f"FPR:\t{fp_num/n_num}")
    print(f"FNR:\t{fn_num/p_num}")
    print(f"Error:\t{invalid_num}")
    print(f"Exact:\t{perfect_num}")
    print('-'*80)
    print(f"Total:\t{total_num}")
    print(f"#P:\t{p_num}")
    print(f"#N:\t{n_num}")
    print('-'*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='Path to input parquet dataset')
    parser.add_argument('--type', type=str, choices=['batch', 'naive', 'rm', 'vm'], required=True, help='Type of the data source')
    args = parser.parse_args()
    if args.type != 'batch':
        if args.type == 'naive':
            extract_fn = extract_answer_naive
        elif args.type == 'vm':
            extract_fn = extract_answer_vm
        elif args.type == 'rm':
            extract_fn = extract_answer_rm
        else:
            raise NotImplementedError(f"Type {args.type} not implemented")
        evaluate_naive(args.file_path, extract_fn)
    else:
        evaluate_batch(args.file_path)