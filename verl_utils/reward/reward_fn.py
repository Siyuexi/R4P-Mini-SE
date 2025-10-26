import json
from verl_utils.reward.extract_answer import *

def get_naive_reward(p, g): # use Acc
    return str(p).lower() == str(g).lower()

def get_batch_reward(pred_list, gt_list, alpha=0.5): # use Acc & F1
    """
    mixup reward: f1 and acc
    """
    # acc
    element_acc = sum(p == g for p, g in zip(pred_list, gt_list)) / len(pred_list)
    
    # f1
    pred_set = set(i for i, p in enumerate(pred_list) if p)
    gt_set = set(i for i, g in enumerate(gt_list) if g)
    if not pred_set and not gt_set:
        f1 = 1.0 
    elif not pred_set or not gt_set:
        f1 = 0.0
    else:
        intersection = len(pred_set & gt_set)
        f1 = 2.0 * intersection / (len(pred_set) + len(gt_set))
    # mixup
    reward = alpha * f1 + (1 - alpha) * element_acc
    return reward


def get_tool_reward(searched_methods, locs) -> float: # use F1
    num = 0
    remaining_methods = searched_methods.copy()
    try:
        locs = json.loads(locs)
    except:
        return 0.0
    if len(locs) == 0 or len(searched_methods) == 0: # ignore non-method issue
        return 0.0
    for loc in locs:
        file_fqn = loc['file'].replace('.py', '').replace('/', '.')
        func_fqn = loc['func']
        for i in range(len(remaining_methods)):
            arg = remaining_methods[i]
            if arg.startswith(file_fqn) and arg.endswith(func_fqn):
                num += 1
                del remaining_methods[i]
                break
    return 2.0 * num / (len(searched_methods) + len(locs)) # use f1
    # return 1.0 * num / len(locs) # use recall

def reward_naive(data_source: str, solution_str: str, ground_truth: list, extra_info) -> float:
    answer = extract_answer_naive(solution_str)
    if not answer:
        return 0.0
    else:
        return get_naive_reward(answer, ground_truth)

def reward_batch(data_source: str, solution_str: str, ground_truth: list, extra_info) -> float:
    answer = extract_batch_combine(solution_str)
    if not answer:
        # return -1.0
        return 0.0
    else:
        # alpha = extra_info.get('alpha', 0.5)
        alpha = 0.0 # acc only
        # return get_batch_reward(answer, ground_truth, alpha)
        score = get_batch_reward(answer, ground_truth, alpha)
        searched_methods = extract_tool_use(solution_str)
        if searched_methods is not None:
            locs = extra_info.get('loc', "")
            score += get_tool_reward(searched_methods, locs) * 0.2
            return score
        else:
            return 0.0

def reward_batch_pure(data_source: str, solution_str: str, ground_truth: list, extra_info) -> float:
    answer = extract_batch_combine(solution_str)
    if not answer:
        # return -1.0
        return 0.0
    else:
        # alpha = extra_info.get('alpha', 0.5)
        alpha = 0.0 # acc only
        # return get_batch_reward(answer, ground_truth, alpha)
        score = get_batch_reward(answer, ground_truth, alpha)
        return score

def compute_score_force_tool(data_source: str, solution_str: str, ground_truth: list, extra_info) -> float:
    answer = extract_batch_combine(solution_str)
    if not answer:
        # return -1.0
        return 0.0
    else:
        # alpha = extra_info.get('alpha', 0.5)
        alpha = 0.0 # acc only
        # return get_batch_reward(answer, ground_truth, alpha)
        score = get_batch_reward(answer, ground_truth, alpha)
        if extra_info.get('split', 'test') == 'train':
            searched_methods = extract_tool_use(solution_str)
            if not searched_methods: # if there is no tooluse, then return 0.0
                locs = extra_info.get('loc', "")
                score += get_tool_reward(searched_methods, locs) * 0.2
                return score
            else:
                return 0.0
        else:
            return score

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info) -> float:
    if extra_info.get('split', 'test') == 'train' and extra_info.get('need_tools_kwargs', False):
        return reward_batch(data_source, solution_str, ground_truth, extra_info)
    else:
        return reward_batch_pure(data_source, solution_str, ground_truth, extra_info)

def compute_score_naive(data_source: str, solution_str: str, ground_truth: list, extra_info) -> float:
    fmt = extract_tool_format(solution_str)
    if not fmt and extra_info.get('split', 'test') == 'train':
        return -1.0
    answer = extract_answer_naive(solution_str)
    if not answer:
        if extra_info.get('split', 'test') == 'train':
            return -1.0
        else:
            return 0.0
    else:
        return float(get_naive_reward(answer, ground_truth))

def compute_score_verb(data_source: str, solution_str: str, ground_truth: list, extra_info) -> float:
    fmt = extract_tool_format(solution_str)
    if not fmt and extra_info.get('split', 'test') == 'train':
        return 0.0
    answer = extract_batch_combine(solution_str)
    if not answer:
        return 0.0
    else:
        return float(get_batch_reward(answer, ground_truth, 0.0))