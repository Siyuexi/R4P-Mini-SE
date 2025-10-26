import pandas as pd
import re
import json

"""
Filter the original bug instances whose golden patch involves new files.
These instances are not suitable for process reward "fault localization".
"""

def is_single_file_patch(patch_str):
    """
    Whether include newly added files or involves more than one file modification.
    """
    patches = re.split(r'(?=^diff --git a/)', patch_str, flags=re.MULTILINE)
    if len(patches) != 2:
        return False
    for patch in patches:
        m = re.match(r'diff --git a/(.*?) b/(.*?)\n', patch)
        if not m:
            continue
        origin_filename = m.group(1)
        filename_lower = origin_filename.lower()
        if ".py" in filename_lower:
            if origin_filename == '/dev/null' \
            or re.search(r'^new file mode \d{6}$', patch, flags=re.MULTILINE) \
            or re.search(r'^--- /dev/null$', patch, flags=re.MULTILINE):
                return False
    return True


def get_classified_instance_ids(path):
    """
    Get the instance ids whose patch does (not) involve new files.
    """
    df = pd.read_parquet(path)
    ids_single = df.apply(lambda x: is_single_file_patch(x['patch']), axis=1)
    ids_multiple = df.apply(lambda x: not is_single_file_patch(x['patch']), axis=1)
    filtered_df_single = df[ids_single]
    filtered_df_multiple = df[ids_multiple]
    ids_list_single = filtered_df_single['instance_id'].tolist()
    ids_list_multiple = filtered_df_multiple['instance_id'].tolist()
    return ids_list_single, ids_list_multiple


if __name__ == '__main__':
    ids_list_single, ids_list_multiple = get_classified_instance_ids('data/datasets/swe-gym.parquet')
    with open('data/instance_ids_single_swe-gym.json', 'w') as f:
        json.dump(ids_list_single, f, indent=2)
    with open('data/instance_ids_multiple_swe-gym.json', 'w') as f:
        json.dump(ids_list_multiple, f, indent=2)
