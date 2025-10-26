import os
import asyncio
import numpy as np
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from transformers import AutoTokenizer
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
import sys
import re
from ray.exceptions import RayActorError

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from verl_utils.reward.extract_answer import extract_batch_combine

MODEL_PATH = "model/r4p" # modify it
TOKENIZER_PATH = MODEL_PATH
TP_SIZE = 8  # Tensor Parallelism size per engine
DP_SIZE = 1  # Data Parallelism size (number of engine replicas)
NUM_GPUS = 8 # Total GPUs
MAX_CONTEXT_LEN = 32768
MAX_NEW_TOKENS = 8192
TEMPERATURE = 1.0
N_VOTING = 5

SYSTEM_PROMPT = "You are a software expert. You will be given a software issue and some patch candidates in user query. You need to judge which patch(es) can resolve the issue. Carefully review, critic, and compare the given candidates. You need to first think about the reasoning process in the mind until you get the final answer. Finally, put the ID(s) of correct patch candidates within \\boxed{}, e.g., \\boxed{1}, \\boxed{2, 4}, \\boxed{1, 2, 3, 4} (all correct), \\boxed{} (all wrong)."

USER_PROMPT = (
    "<issue>\n{issue}\n</issue>\n"
    "<patch-1>\n{patch1}\n</patch-1>\n"
    "<patch-2>\n{patch2}\n</patch-2>\n"
    "<patch-3>\n{patch3}\n</patch-3>\n"
    "<patch-4>\n{patch4}\n</patch-4>"
)

# --- Pydantic Models ---
class BatchRequest(BaseModel):
    issue: str
    patch_list: List[str]

class BatchItem(BaseModel):
    batch_id: str
    data: BatchRequest

class MultiBatchRequest(BaseModel):
    batches: List[BatchItem]

class ScoreResponse(BaseModel):
    scores: Dict[str, List[float]]

# --- Ray Serve Deployments ---

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 64} # More CPUs might be needed for batching/logic
)
class APIServer:
    def __init__(self, engine_handles: List[DeploymentHandle]):
        self.engine_handles = engine_handles
        self.dp_size = len(engine_handles)

    async def __call__(self, request: MultiBatchRequest) -> ScoreResponse:
        try:
            batches = request.batches
            
            # This handles cases where len(batches) is not a multiple of self.dp_size
            if not batches:
                return ScoreResponse(scores={})

            sub_batch_groups = np.array_split(batches, self.dp_size)
            
            # Create concurrent tasks for each engine
            tasks = []
            for i, sub_batch_group in enumerate(sub_batch_groups):
                if len(sub_batch_group) > 0:
                    # np.array_split returns numpy arrays, convert back to list
                    tasks.append(
                        self.engine_handles[i].process_batch.remote(sub_batch_group.tolist())
                    )

            # Gather results from all engines
            results_list = await asyncio.gather(*tasks)
            
            # Combine results from all engines into a single dictionary
            final_scores = {}
            for result_dict in results_list:
                final_scores.update(result_dict)
            
            return ScoreResponse(scores=final_scores)
        
        except Exception as e:
            # Add more detailed logging here if needed
            print(f"Error in APIServer: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed when processing request: {str(e)}"
            )

# patch functions

def is_only_comment_or_whitespace(patch_str):
    """
    Checks if a patch contains only changes to comments or whitespace.
    Returns True if only noise changes are present, False otherwise.
    """
    # Regex to find all added or removed lines, excluding the diff header lines --- and +++
    changed_lines = re.findall(r'^[+-](?![+-]{2} )(.+)', patch_str, flags=re.MULTILINE)

    if not changed_lines:
        # No lines were added or removed, so it's not a substantive change.
        return True

    for line in changed_lines:
        stripped_line = line.strip()
        # Check if the line is NOT noise. A line is considered noise if it's:
        # 1. Empty or just whitespace.
        # 2. A single-line comment.
        if stripped_line and not stripped_line.startswith('#') and not (stripped_line.startswith('"""') and stripped_line.endswith('"""')) and not (stripped_line.startswith("'''") and stripped_line.endswith("'''")):
            # If we find even one line that is actual code, the patch is valid.
            return False
            
    # If all changed lines were noise, return True to filter this patch out.
    return True

def remove_index_from_patch(patch_str):
    """Remove the 'index xxxx..xxxx' line from patch string using regex"""
    if not isinstance(patch_str, str):
        return patch_str
    
    pattern = r'\nindex [a-f0-9]+\.\.[a-f0-9]+(?: \d+)?\n'
    return re.sub(pattern, '\n', patch_str, flags=re.MULTILINE)

def get_pure_patch(patch_str):
    patches = re.split(r'(?=^diff --git a/)', patch_str, flags=re.MULTILINE)
    filtered_patches = []
    for patch in patches:
        m = re.match(r'diff --git a/(.*?) b/(.*?)\n', patch)
        if not m:
            continue
        origin_filename = m.group(1)
        filename_lower = origin_filename.lower()
        if ".py" not in filename_lower \
            or "test" in filename_lower and 'pytest' not in filename_lower\
            or "reproduce" in filename_lower \
            or origin_filename == '/dev/null' \
            or re.search(r'^new file mode \d{6}$', patch, flags=re.MULTILINE) \
            or re.search(r'^--- /dev/null$', patch, flags=re.MULTILINE):
            continue
        if is_only_comment_or_whitespace(patch):
            continue
            
        filtered_patches.append(remove_index_from_patch(patch))
    patch = '\n'.join(filtered_patches)
    return patch.strip()

@serve.deployment(
    # This `num_replicas` is for a single deployment.
    # We will create DP_SIZE separate deployments.
    num_replicas=1, 
    ray_actor_options={
        "num_gpus": TP_SIZE,
    }
)
class vLLMEngine:
    def __init__(self):
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            tokenizer=TOKENIZER_PATH, # It's good practice to specify tokenizer explicitly
            tensor_parallel_size=TP_SIZE,
            dtype="bfloat16",
            max_model_len=MAX_CONTEXT_LEN,
            gpu_memory_utilization=0.90, # For A100 80G, 0.90 is often safe
            disable_custom_all_reduce=True,
            disable_log_stats=True,
            # This is a crucial performance tuning parameter. It depends on context length and batch size.
            max_num_batched_tokens=MAX_CONTEXT_LEN, # A heuristic
            trust_remote_code=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            n=N_VOTING,
            temperature=TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
        )

    def generate_prompt(self, issue: str, patch_list: List[str]) -> str:
        # Max token of each patch. If exceeds, then remove it and gets 0 reward.
        patch_token_limit = MAX_CONTEXT_LEN // 4
        # Ensure there are always 4 patches, even if input is shorter
        # This makes the USER_PROMPT formatting safe
        assert len(patch_list) == 4, f"Error: patch_list size: {len(patch_list)}, not 4."
        # remove redundant strings for better rm performance
        patch_list = [get_pure_patch(patch_str) for patch_str in patch_list]

        final_patches = []
        for patch_str in patch_list:
            pure_patch = get_pure_patch(patch_str)
            
            patch_token_count = len(self.tokenizer(pure_patch, add_special_tokens=False).input_ids)
            
            if patch_token_count > patch_token_limit:
                print(f"Warning: A patch was too long ({patch_token_count} tokens > limit {patch_token_limit}) and was replaced with an empty string.")
                print(f"The original overlong patch is:\n{pure_patch}")
                final_patches.append("")
            else:
                final_patches.append(pure_patch)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                issue=issue,
                patch1=final_patches[0],
                patch2=final_patches[1],
                patch3=final_patches[2],
                patch4=final_patches[3]
            )}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    async def process_request(self, item: BatchItem) -> List[float]:
        prompt = self.generate_prompt(item.data.issue, item.data.patch_list)
        
        request_id = f"req_{item.batch_id}_{uuid.uuid4().hex}"
        
        result_generator = self.engine.generate(
            prompt, 
            self.sampling_params, 
            request_id=request_id
        )
        
        # The final result is the only one we need from the generator
        final_output = None
        async for request_output in result_generator:
            final_output = request_output
        
        outputs = [output.text for output in final_output.outputs]
        
        all_votes = []
        for text in outputs:
            # Your custom function to parse the model's text output
            result = extract_batch_combine(text)
            if result is not None:
                all_votes.append(result)
        
        if not all_votes:
            print("####### [WRANING] No votes parsed. Return all 0 reward.")
            return [0.0] * 4 # Default to all wrong if no valid votes are parsed
        
        # Perform majority voting
        num_patches = 4 # Assuming always 4 patches
        vote_counts = np.zeros(num_patches)
        for vote in all_votes:
            for i, accepted in enumerate(vote):
                if accepted:
                    vote_counts[i] += 1
        
        acceptance_rates = vote_counts / len(all_votes)
        
        # Score is 1.0 if accepted in more than half the votes, else 0.0
        final_scores = [1.0 if rate > 0.5 else 0.0 for rate in acceptance_rates]
        return final_scores
    
    async def process_batch(self, batch: List[BatchItem]) -> Dict[str, List[float]]:
        # This method runs multiple requests concurrently on a single engine replica
        tasks = [self.process_request(item) for item in batch]
        results = await asyncio.gather(*tasks)
        
        return {
            item.batch_id: score
            for item, score in zip(batch, results)
        }

# This deployment wraps the application in a FastAPI app.
# It's the entrypoint for HTTP requests.
@serve.deployment(num_replicas=1)
@serve.ingress(app := FastAPI())
class FastAPIWrapper:
    def __init__(self, api_server_handle: DeploymentHandle):
        self.api_server_handle = api_server_handle
    
    @app.post("/score", response_model=ScoreResponse)
    async def score(self, request: MultiBatchRequest):
        try:
            return await self.api_server_handle.remote(request)
        except RayActorError as e:
            print(f"A Ray actor error occurred: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"The backend service experienced an error: {e}"
            )
        except Exception as e:
            print(f"An unexpected error occurred in the ingress: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"An internal server error occurred: {str(e)}"
            )

# deployment graph creation
def deploy_app():
    # 1. Create DP_SIZE independent vLLM engine deployments
    # Each deployment is named and assigned its own GPUs
    engine_handles = []
    for i in range(DP_SIZE):
        engine_deployment = vLLMEngine.options(
            name=f"vllm_engine_{i}",
        ).bind()
        engine_handles.append(engine_deployment)
    
    # 2. Create the API server and pass the engine handles to it
    api_server = APIServer.bind(engine_handles=engine_handles)
    
    # 3. Create the FastAPI ingress and pass the API server handle to it
    fastapi_app = FastAPIWrapper.bind(api_server_handle=api_server)
    
    return fastapi_app

# Main block to run the service
# You would run this script with `serve run your_script_name:app`
app_handle = deploy_app()

serve.start(http_options={"host": "::", "port": 8365})

serve.run(app_handle)