# only issues with correct patches in 16 candidates (268x16 + 1of4 + 1of2/1of4 competition)
system_prompt = """You are a software expert. You will be given a software issue and some patch candidates in user query. You need to judge which patch can resolve the issue. There must be and only be **one** correct patch. Carefully review, critic, and compare the given candidates. You need to first think about the reasoning process in the mind until you get the final answer. Finally, put the ID of the most likely correct patch within \\boxed{}, i.e., \\boxed{1} or \\boxed{2} or \\boxed{3} or \\boxed{4}. If there are multiple *identical* patches that you believe are correct, output with the one that has the smaller ID. You *CANNOT* answer \\boxed{} (empty) or \\boxed{1, 2} (multiple) because you should output only one most likely correct patch!"""

def process_fn(example, idx):

    id = example.pop("instance_id") # unique id
    issue = example.pop("problem_statement") # original issue
    patch_list = example.pop("patch") # patches
    resolved = example.pop("resolved") # true or false
    split = example.pop("split") # train, test

    plist = []
    rlist = []
    for i in range(len(patch_list)):
        if patch_list[i] != '# NO PATCH GENERATED!':
            plist.append(patch_list[i])
            rlist.append(bool(resolved[i]))
    if len(plist) < 16: # auto completion
        delta = 16 - len(plist)
        for i in range(delta):
            plist.append("# NO PATCH GENERATED!")
            rlist.append(False)

    data = {
        "data_source": f'batch_{split}',
        "prompt1": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"<issue>\n{issue.strip()}\n</issue>\n"
                    f"<patch-1>\n{plist[0].strip()}\n</patch-1>\n"
                    f"<patch-2>\n{plist[4].strip()}\n</patch-2>\n"
                    f"<patch-3>\n{plist[8].strip()}\n</patch-3>\n"
                    f"<patch-4>\n{plist[12].strip()}\n</patch-4>\n"
                    "If there are multiple *identical* patches that you believe are correct, output with the one that has the smaller ID. You *CANNOT* answer \\boxed{} (empty) or \\boxed{1, 2} (multiple) because you should output only one most likely correct patch!"
                ),
            },
        ],
        "prompt2": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"<issue>\n{issue.strip()}\n</issue>\n"
                    f"<patch-1>\n{plist[1].strip()}\n</patch-1>\n"
                    f"<patch-2>\n{plist[5].strip()}\n</patch-2>\n"
                    f"<patch-3>\n{plist[9].strip()}\n</patch-3>\n"
                    f"<patch-4>\n{plist[13].strip()}\n</patch-4>\n"
                    "If there are multiple *identical* patches that you believe are correct, output with the one that has the smaller ID. You *CANNOT* answer \\boxed{} (empty) or \\boxed{1, 2} (multiple) because you should output only one most likely correct patch!"
                ),
            },
        ],
        "prompt3": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"<issue>\n{issue.strip()}\n</issue>\n"
                    f"<patch-1>\n{plist[2].strip()}\n</patch-1>\n"
                    f"<patch-2>\n{plist[6].strip()}\n</patch-2>\n"
                    f"<patch-3>\n{plist[10].strip()}\n</patch-3>\n"
                    f"<patch-4>\n{plist[14].strip()}\n</patch-4>\n"
                    "If there are multiple *identical* patches that you believe are correct, output with the one that has the smaller ID. You *CANNOT* answer \\boxed{} (empty) or \\boxed{1, 2} (multiple) because you should output only one most likely correct patch!"
                ),
            },
        ],
        "prompt4": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"<issue>\n{issue.strip()}\n</issue>\n"
                    f"<patch-1>\n{plist[3].strip()}\n</patch-1>\n"
                    f"<patch-2>\n{plist[7].strip()}\n</patch-2>\n"
                    f"<patch-3>\n{plist[11].strip()}\n</patch-3>\n"
                    f"<patch-4>\n{plist[15].strip()}\n</patch-4>\n"
                    "If there are multiple *identical* patches that you believe are correct, output with the one that has the smaller ID. You *CANNOT* answer \\boxed{} (empty) or \\boxed{1, 2} (multiple) because you should output only one most likely correct patch!"
                ),
            },
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": rlist},
        "extra_info": {
            "split": split,
            "index": idx,
            "id": id,
        },
    }
    return data