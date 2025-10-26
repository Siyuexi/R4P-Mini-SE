# only ture (268x16 + 1of4 + 1of2/1of4 competition)
system_promptx2 = """You are a software expert. You will be given a software issue and some patch candidates in user query. You need to judge which patch can resolve the issue. There must be and only be **one** correct patch. Carefully review, critic, and compare the given candidates. You need to first think about the reasoning process in the mind until you get the final answer. Finally, put the ID of the most likely correct patch within \\boxed{}, i.e., \\boxed{1} or \\boxed{2}. If there are multiple *identical* patches that you believe are correct, output with the one that has the smaller ID. You *CANNOT* answer \\boxed{} (empty) or \\boxed{1, 2} (multiple) because you should output only one most likely correct patch!"""

system_promptx4 = """You are a software expert. You will be given a software issue and some patch candidates in user query. You need to judge which patch can resolve the issue. There must be and only be **one** correct patch. Carefully review, critic, and compare the given candidates. You need to first think about the reasoning process in the mind until you get the final answer. Finally, put the ID of the most likely correct patch within \\boxed{}, i.e., \\boxed{1} or \\boxed{2} or \\boxed{3} or \\boxed{4}. If there are multiple *identical* patches that you believe are correct, output with the one that has the smaller ID. You *CANNOT* answer \\boxed{} (empty) or \\boxed{1, 2} (multiple) because you should output only one most likely correct patch!"""

def process_fn(example, idx):

    id = example.pop("instance_id") # unique id
    issue = example.pop("problem_statement") # original issue
    patch_list = example.pop("patch") # patches
    resolved = example.pop("resolved") # true or false
    split = example.pop("split") # train, test
    id1 = example.pop("index1")
    id2 = example.pop("index2")
    id3 = example.pop("index3")
    id4 = example.pop("index4")

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
        "promptx2": [
            {
                "role": "system",
                "content": system_promptx2,
            },
            {
                "role": "user",
                "content": (
                    f"<issue>\n{issue.strip()}\n</issue>\n"
                    f"<patch-1>\n{plist[id1-1].strip()}\n</patch-1>\n"
                    f"<patch-2>\n{plist[id2+3].strip()}\n</patch-2>\n"
                    "If there are multiple *identical* patches that you believe are correct, output with the one that has the smaller ID. You *CANNOT* answer \\boxed{} (empty) or \\boxed{1, 2} (multiple) because you should output only one most likely correct patch!"
                ),
            },
        ],
        "promptx4": [
            {
                "role": "system",
                "content": system_promptx4,
            },
            {
                "role": "user",
                "content": (
                    f"<issue>\n{issue.strip()}\n</issue>\n"
                    f"<patch-1>\n{plist[id1-1].strip()}\n</patch-1>\n"
                    f"<patch-2>\n{plist[id2+3].strip()}\n</patch-2>\n"
                    f"<patch-3>\n{plist[id3+7].strip()}\n</patch-3>\n"
                    f"<patch-4>\n{plist[id4+11].strip()}\n</patch-4>\n"
                    "If there are multiple *identical* patches that you believe are correct, output with the one that has the smaller ID. You *CANNOT* answer \\boxed{} (empty) or \\boxed{1, 2} (multiple) because you should output only one most likely correct patch!"
                ),
            },
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": [rlist[id1-1], rlist[id1+3], rlist[id1+7], rlist[id1+11]]},
        "extra_info": {
            "split": split,
            "index": idx,
            "id": id,
        },
    }
    return data