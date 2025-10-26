# use trajectory
system_prompt = """You are an expert judge evaluating AI assistant interactions. Your task is to determine if the assistant successfully resolved the user's request.\n\nKey evaluation criteria:\n1. Did the assistant complete the main task requested by the user?\n2. Did the assistant handle all edge cases and requirements specified?\n3. Were there any errors or issues in the final solution?\n4. Did the assistant verify the solution works as intended?\n\nRespond only with \"<judgement>YES</judgement>\" or \"<judgement>NO</judgement>\".
""".strip()

# for swe-gym verifier
spliter = "\n----------------------------------------------------------------------------------------------------\n*** Turn {turn} - {role} ***\n"

# for r2e-gym verifier
# spliter = "\n[STEP]\n{turn}\n[/STEP]\n"

def process_fn(example, idx):

    id = example.pop("instance_id") # unique id
    issue = example.pop("problem_statement") # original issue
    patch = example.pop("patch") # patch with full method context, FQN, and import packages
    resolved = example.pop("resolved") # true or false
    split = example.pop("split") # train, test
    trajs = example.pop("trajs")

    trajs_str = ""

    for turn, msg in enumerate(trajs):
        role = msg['role']
        if isinstance(msg['content'], str):
            content = msg['content']
        else:
            text_content = ""
            tool_content = ""
            if len(msg['content']) != 0:
                text_content = msg['content'][0]['text']
            if 'tool_calls' in msg.keys() and msg['tool_calls']:
                tool_content = str(msg['tool_calls'][0]['function'])
            content = f"Think: {text_content}\n\nTool calls: {tool_content}"
        if role == 'system':
            trajs_str += content # for swe-gym verifier
            # trajs_str = trajs_str + '\n[SYSTEM]\n' + content + '\n[/SYSTEM]\n' # for r2e-gym verifier
        else:
            trajs_str += spliter.format( # for swe-gym verifier
                turn=turn,
                role=msg['role'].upper()
            )
            trajs_str += content

            # if turn % 2 != 0: # for r2e-gym verifier
            #     trajs_str += spliter.format(
            #         turn=turn//2
            #     )
            # trajs_str = trajs_str + f"\n[{msg['role'].upper()}]\n" + content + f"\n[/{msg['role'].upper()}]\n"

    data = {
        "data_source": f'naive_{split}',
        "prompt": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Please evaluate the following interaction between an AI assistant and a user:\n"
                    "\n=== INTERACTION LOG ===\n"
                    "*** System Message that describes the assistant's behavior ***\n"
                    f"{trajs_str}"
                    "\n----------------------------------------------------------------------------------------------------\n" # if use swe-gym
                    "\n=== END INTERACTION ===\n"
                    # "\n=== FINAL PATCH ===\n" # if use r2e-gym
                    # "\n[PATCH]\n" # if use r2e-gym
                    # f"{patch}" # if use r2e-gym
                    # "\n[/PATCH]\n" # if use r2e-gym
                    # "\n=== END FINAL PATCH ===\n" # if use r2e-gym
                    "\nBased on the above interaction, did the assistant successfully resolve the user's initial request? Respond with YES or NO."
                ),
            }
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": resolved},
        "extra_info": {
            "split": split,
            "index": idx,
            "id": id,
        }
    }
    return data