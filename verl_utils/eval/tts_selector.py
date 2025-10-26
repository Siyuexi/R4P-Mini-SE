import pandas as pd
import random
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from verl_utils.reward.extract_answer import *

# random.seed(42)

def get_batch_tp_tn_0(pred_list_batch, gt_list, type):
    tp = 0.0
    tn = 0.0
    pred_list = pred_list_batch[0]
    if not pred_list:
        return 0
    for i in range(4):
        if pred_list[i] and gt_list[i]:
            tp += 1.0
        if not pred_list[i] and not gt_list[i]:
            tn += 1.0
    if type == 'tp':
        return tp
    if type == 'tn':
        return tn

def select_1_pred(pred_list_batch, gt_list, p=1, agg='random'):
    pred_list_batch = [pred for pred in pred_list_batch if pred is not None]
    if not pred_list_batch:
        return False
    vote = [0 for i in range(len(pred_list_batch[0]))]
    for pred in pred_list_batch:
        for idx in range(len(pred)):
            if pred[idx]:
                vote[idx] += 1
    max_vote = max(vote)
    max_indices = [i for i, x in enumerate(vote) if x == max_vote]

    if agg == 'random':
        selected_idx = random.choice(max_indices)
    elif agg == 'first':
        selected_idx = max_indices[0]
    elif agg == 'last':
        selected_idx = max_indices[-1]

    if p == 'x2' or p == 'x4':
        return bool(gt_list[selected_idx])
    # return bool(gt_list[selected_idx + (p-1)*4])
    return bool(gt_list[selected_idx * 4 + (p-1)])

def select_1_baseline(gt_list, p=1, agg='random'): # random selecting as a simple baseline
    selected_idx_baseline = random.choice(range(4))
    
    if p == 'x2' or p == 'x4':
        return bool(gt_list[selected_idx_baseline])
    # return bool(gt_list[selected_idx_baseline + (p-1)*4])
    return bool(gt_list[selected_idx_baseline * 4 + (p-1)])

def select_1_best(gt_list, p=1): # ideally upper-bound
    if p == 'x2':
        return True in gt_list[:2]
    elif p == 'x4':
        return True in gt_list[:4]
    elif True in gt_list[(p-1)*4:p*4]:
        return True
    return False

def evaluate_batch(output_path, p=1, agg='random'):
    if 'x' not in p:
        p = int(p)
    df = pd.read_parquet(output_path)
    df['answer'] = df.apply(lambda x: [extract_batch_combine(resp) for resp in x['responses'] if resp is not None], axis=1)
    df['ground_truth'] = df.apply(lambda x: x['reward_model']['ground_truth'], axis=1)
    print(df.iloc[0]['responses'])
    print(df.iloc[0]['answer'])
    print(df.iloc[0]['ground_truth'])
    df['tp'] = df.apply(lambda x: get_batch_tp_tn_0(x['answer'], x['ground_truth'], 'tp'), axis=1)
    df['tn'] = df.apply(lambda x: get_batch_tp_tn_0(x['answer'], x['ground_truth'], 'tn'), axis=1)
    df['#p'] = df.apply(lambda x: x['ground_truth'].sum(), axis=1)
    df['#n'] = df.apply(lambda x: (1-x['ground_truth']).sum(), axis=1)
    df['selected_answer'] = df.apply(lambda x: select_1_pred(x['answer'], x['ground_truth'], p, agg), axis=1)
    df['baseline_answer'] = df.apply(lambda x: select_1_baseline(x['ground_truth'], p), axis=1)
    df['best_answer'] = df.apply(lambda x: select_1_best(x['ground_truth'], p), axis=1)
    num_selected_correct = df[df['selected_answer'] == True].shape[0]
    num_baseline_correct = df[df['baseline_answer'] == True].shape[0]
    num_best_correct = df[df['best_answer'] == True].shape[0]
    invalid_num = df['answer'].apply(lambda x: all(item is None for item in x)).sum()
    total_num = df.shape[0]

    print('-'*80)
    print(f"Name:\t{output_path.split('verl/')[-1].split('/actor')[0]}")
    print('-'*80)
    print(f"TPR[0]:\t{df['tp'].sum()/df['#p'].sum()}")
    print(f"TNR[0]:\t{df['tn'].sum()/df['#n'].sum()}")
    print(f"Select:\t{num_selected_correct}")
    print(f"Random:\t{num_baseline_correct}")
    print(f"Best:\t{num_best_correct}")
    print(f"SR:\t{num_selected_correct/500}")
    print(f"RR:\t{num_baseline_correct/500}")
    print(f"BR:\t{num_best_correct/500}")
    print(f"Error:\t{invalid_num}")
    print('-'*80)
    print(f"Non-empty:\t{total_num}")
    print('-'*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='Path to input parquet dataset')
    parser.add_argument('--p', type=str, required=True, help='Path to input parquet dataset')
    parser.add_argument('--agg', type=str, default='random', help='Aggregation method for selection')
    args = parser.parse_args()
    evaluate_batch(args.file_path, args.p, args.agg)