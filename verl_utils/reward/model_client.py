import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import requests
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from verl_utils.reward.extract_answer import (extract_patch,
                                              extract_think_format,
                                              extract_tool_format)

random.seed(42)
HARNESS_URL = "http://YOUR_URL:5000/"
SERVER_URL = "http://YOUR_URL:8365/score"
RM_BATCH_SIZE = 4

# --- Pydantic Models ---
class BatchRequest(BaseModel):
    issue: str
    patch_list: List[str]

class BatchItem(BaseModel):
    batch_id: str
    data: BatchRequest

class MultiBatchRequest(BaseModel):
    batches: List[BatchItem]

def compute_score_remote_stage(data_sources, solution_strs, ground_truths, extra_infos):
    scores = compute_score_remote(data_sources, solution_strs, ground_truths, extra_infos)
    new_scores = []
    for score in scores:
        if score == -1.0:
            new_scores.append(0.0)
        elif score == 0.0:
            new_scores.append(0.1)
        else:
            new_scores.append(1.0)
    return new_scores

def compute_score_remote_clip(data_sources, solution_strs, ground_truths, extra_infos):
    scores = compute_score_remote(data_sources, solution_strs, ground_truths, extra_infos)
    scores = [0.0 if score == -1.0 else score for score in scores]
    return scores

def random_reward(data_sources, solution_strs, ground_truths, extra_infos):
    if 'test' in data_sources[0]:
        return compute_score_bench(data_sources, solution_strs, ground_truths, extra_infos)
        # return compute_score_record(data_sources, solution_strs, ground_truths, extra_infos)
    else:
        return compute_score_random(data_sources, solution_strs, ground_truths, extra_infos)

def compute_score_remote(data_sources, solution_strs, ground_truths, extra_infos):
    if 'test' in data_sources[0]:
        return compute_score_bench(data_sources, solution_strs, ground_truths, extra_infos)
        # return compute_score_record(data_sources, solution_strs, ground_truths, extra_infos)
    else:
        return compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos)


def compute_score_random(data_sources, solution_strs, ground_truths, extra_infos):
    return [float(random.choice([0, 1])) if extract_tool_format(sol) and extract_think_format(sol) else 0.0 for sol in solution_strs]

def compute_score_record(data_sources, solution_strs, ground_truths, extra_infos): # if cache only (do not call server for reward calculation.)
    ts = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    patch_strs = [extract_patch(sol) for sol in solution_strs]
    payload = []
    for idx, (patch, extra_info) in enumerate(zip(patch_strs, extra_infos)):
        if patch.strip():
            payload.append({
                "model_name_or_path": 'trae-lite-ossi',
                "instance_id": extra_info["instance_id"],
                "model_patch": patch.strip()
            })

    jsonl_content = "\n".join([json.dumps(p) for p in payload])
    with open(f'cached_submission_{ts}.jsonl', 'w') as f:
        f.write(jsonl_content)
    if os.path.exists('data/rollout'):
        with open(f"data/rollout/cached_submission_{ts}.jsonl", 'w') as f:
            f.write(jsonl_content)
    else:
        print("ERROR: NO MNT FOR SAVING!")

    return [0.0] * len(patch_strs)

def compute_score_bench(data_sources, solution_strs, ground_truths, extra_infos):

    ts = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    
    payload = []
    empty_indices = []
    empty_instances = []
    valid_indices_map = {}

    patch_strs = [extract_patch(sol) for sol in solution_strs]

    for idx, (patch, extra_info) in enumerate(zip(patch_strs, extra_infos)):
        instance_id = extra_info["instance_id"]
        if not patch.strip():
            empty_indices.append(idx)
            empty_instances.append(instance_id)
        else:
            payload.append({
                "model_name_or_path": 'trae-lite-ossi',
                "instance_id": instance_id,
                "model_patch": patch
            })
            if instance_id not in valid_indices_map:
                valid_indices_map[instance_id] = idx

    with open(f'cached_empty_ids_{ts}.txt', 'w') as f:
        f.write('\n'.join(empty_instances))

    if not payload:
        scores = [0.0] * len(patch_strs)
        for idx in empty_indices:
            scores[idx] = -1.0
        return scores

    run_id = None
    try:
        jsonl_content = "\n".join([json.dumps(p) for p in payload])

        with open(f'cached_submission_{ts}.jsonl', 'w') as f:
            f.write(jsonl_content)
        
        data = {"dataset": "SWE-bench/SWE-bench_Verified"}
        files = {'file': ('predictions.jsonl', jsonl_content.encode('utf-8'), 'application/octet-stream')}
        
        print("Uploading patches to evaluation server...")
        submit_url = f"{HARNESS_URL}evaluate"
        response = requests.post(submit_url, data=data, files=files, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        run_id = result.get("run_id")
        if not run_id:
            raise ValueError("Server response did not include a run_id.")
        print(f"Submission successful. Received run_id: {run_id}")

    except Exception as e:
        print(f"Error submitting evaluation request: {e}")
        print("All submissions get 0 reward.")
        return [0.0] * len(patch_strs)

    result_filename = None
    polling_url = f"{HARNESS_URL}progress/{run_id}"
    POLLING_INTERVAL = 15
    MAX_POLLING_ATTEMPTS = 300
    
    print("Polling for results...")
    for attempt in range(MAX_POLLING_ATTEMPTS):
        try:
            response = requests.get(polling_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            status = data.get('status')
            if status == 'completed':
                print("Evaluation completed.")
                result_filename = data.get('result_file')
                break
            elif status == 'error':
                print(f"Evaluation failed on the server. Log: {data.get('output')}")
                print("All submissions get 0 reward.")
                return [0.0] * len(patch_strs)
            elif status == 'running':
                print(f"Attempt {attempt + 1}/{MAX_POLLING_ATTEMPTS}: Status is 'running'. Waiting for {POLLING_INTERVAL}s...")
                time.sleep(POLLING_INTERVAL)
            else:
                print(f"Unknown status received: {status}. Aborting.")
                return [0.0] * len(patch_strs)

        except requests.exceptions.RequestException as e:
            print(f"Error polling for progress: {e}. Retrying in {POLLING_INTERVAL}s...")
            time.sleep(POLLING_INTERVAL)
    
    if not result_filename:
        print("Polling timed out. Could not retrieve results.")
        return [0.0] * len(patch_strs)

    final_scores = [0.0] * len(patch_strs)
    try:
        download_url = f"{HARNESS_URL}download/{result_filename}"
        print(f"Downloading result file from: {download_url}")
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()
        eval_results = response.json()

        with open(f"cached_harness_{ts}.json", 'w') as f:
            f.write(json.dumps(eval_results))
            
        for resolved_id in eval_results['resolved_ids']:
            original_index = valid_indices_map[resolved_id]
            final_scores[original_index] = 1.0

    except requests.exceptions.RequestException as e:
        print(f"Failed to download or parse result file: {e}")
        return [0.0] * len(patch_strs)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing result file: {e}")
        return [0.0] * len(patch_strs)

    for idx in empty_indices:
        final_scores[idx] = -1.0
        
    return final_scores

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    grouped_data = defaultdict(list)
    patch_strs = [extract_patch(sol) for sol in solution_strs]
    tool_format_flags = [extract_tool_format(sol) for sol in solution_strs]
    think_format_flags = [extract_think_format(sol) for sol in solution_strs]
    for idx, (sol, info) in enumerate(zip(patch_strs, extra_infos)):
        instance_id = info["instance_id"]
        issue = info['issue']
        patch = sol
        grouped_data[instance_id].append((idx, issue, patch))
    
    batch_items_pydantic: List[BatchItem] = [] 
    index_mapping = {}
    
    for instance_id, items in grouped_data.items():
        assert len(items) % RM_BATCH_SIZE == 0, f"Instance {instance_id} has {len(items)} rollouts, which can not be divided by batch size {RM_BATCH_SIZE}."
        items.sort(key=lambda x: x[0])
        
        for i in range(0, len(items), RM_BATCH_SIZE):
            batch_id = f"{instance_id}_{i//RM_BATCH_SIZE}"
            batch_items_slice = items[i:i+RM_BATCH_SIZE]
            
            batch_issues = [item[1] for item in batch_items_slice]
            assert len(set(batch_issues)) == 1, "Error occurs when wrapping batches for rewarding."
            batch_solutions = [item[2] for item in batch_items_slice]
            
            request_data = BatchRequest(
                issue=batch_issues[0],
                patch_list=batch_solutions
            )
            batch_item = BatchItem(
                batch_id=batch_id,
                data=request_data
            )
            batch_items_pydantic.append(batch_item)
            
            index_mapping[batch_id] = [item[0] for item in batch_items_slice]
    
    if not batch_items_pydantic:
        return [0.0] * len(patch_strs)

    payload = MultiBatchRequest(batches=batch_items_pydantic)

    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    SERVER_URL,
                    json=payload.model_dump(mode="json"),
                    headers={"Content-Type": "application/json"},
                    proxies={"http": None, "https": None} # do not use proxy
                )
                response.raise_for_status()
                break
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    print(f"Request failed: {str(e)}. Retry attempts: {attempt+1}.")
                else:
                    raise RuntimeError(f"Request failed.") from e
        
        result = response.json()
        batch_results = result.get("scores", {})
        
        if not batch_results:
            raise ValueError("Empty response from reward server.")
    
    except Exception as e:
        print(f"Rewarding failed: {str(e)}")
        return [0.0] * len(patch_strs)
    
    final_scores = [0.0] * len(patch_strs)
    
    for batch in payload.batches:
        batch_id = batch.batch_id
        if batch_id in batch_results:
            scores = batch_results[batch_id]
            orig_indices = index_mapping[batch_id] 
            for score, orig_idx in zip(scores, orig_indices):
                final_scores[orig_idx] = score

    for idx, (patch, tool_flag, think_flag) in enumerate(zip(patch_strs, tool_format_flags, think_format_flags)):
        if not patch or not tool_flag or not think_flag:
            final_scores[idx] = -1.0
    
    return final_scores

if __name__ == '__main__':
    with open('data/rollouts/cached_submission_2025-09-11_15-40-45.jsonl', 'r') as f:
        lines = f.readlines()
    objs = [json.loads(line) for line in lines]
    e = [{"instance_id": obj['instance_id']} for obj in objs]
    t = """{"name": "search_tool", "arguments": {"construct": "class", "entity": "CompoundModel"}}"""
    p = [f"<tool_call>\n{t}\n</tool_call>\n<tool_response>\n[PATCH]\n{obj['model_patch']}\n[/PATCH]\n</tool_response>\n" for obj in objs]
    scores = compute_score_bench("", p, "", e)
    scores_true = [score for score in scores if score == 1.0]
    print(f"Correct patch num: {len(scores_true)}")
    print(f"Total pass@1: {len(scores_true)/500}")
    print(len(scores_true)/len(scores))