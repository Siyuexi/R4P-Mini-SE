system_prompt = """You are a software expert. You will be given a software issue and two patch candidates in user query. You need to judge which patch can resolve the issue. There must be and only be one correct patch in the given pair. Carefully review, critic, and compare the given candidates. You need to first think about the reasoning process in the mind until you get the final answer. Finally, put the ID(s) of correct patch candidates within \\boxed{}, i.e., \\boxed{1} or \\boxed{2}. You *CANNOT* answer \\boxed{} (empty) or \\boxed{1, 2} (all) because the candidate patch must be one correct and only one correct!
""".strip()

def process_fn(example, idx):

    alpha = 0.5
    id = example.pop("instance_id") # unique id
    repo = example.pop("repo") # repo name
    issue = example.pop("problem_statement") # original issue
    patch_list = example.pop("patch") # patches
    resolved = example.pop("resolved") # true or false
    sha = example.pop("base_commit") # sha of the base commit
    split = example.pop("split") # train, test
    loc = example.pop("oracle_location") # loc

    ans = '1'
    if resolved == 'B':
        ans = '2'

    data = {
        "data_source": f'batch_{split}',
        "prompt": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"<issue>\n{issue.strip()}\n</issue>\n"
                    f"<patch-1>\n{patch_list[0].strip()}\n</patch-1>\n"
                    f"<patch-2>\n{patch_list[1].strip()}\n</patch-2>\n"
                    "Your final answer **MUST BE** either \\boxed{1} or \\boxed{2}. You **CANNOT** answer \\boxed{} or \\boxed{1, 2}!!!"
                ),
            },
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": ans},
        "extra_info": {
            "split": split,
            "index": idx,
            "id": id,
            "repo": repo,
            "sha": sha,
            "alpha": alpha,
            "loc": loc,
        },
    }
    return data