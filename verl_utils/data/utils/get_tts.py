import pandas as pd
import json
from pathlib import Path
import re

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
        filtered_patches.append(remove_index_from_patch(patch))
        patch = '\n'.join(filtered_patches)
    return patch.strip()

# Base directory and file pattern
path = "data/rollouts/claude_4_space_size_8_413.jsonl"
    
# Read all lines at once and parse JSON
with open(path, 'r') as f:
    data = [json.loads(line) for line in f]

# Create DataFrame directly from the list of dicts
df = pd.DataFrame([{
    "instance_id": item["instance_id"],
    "problem_statement": item["issue"],
    "patch": [get_pure_patch(patch) for patch in item["patches"]],
    "resolved": item["success_id_new"],
    "split": 'test'
} for item in data])

print(df)

# Verify that each list has exactly 16 elements (one from each run)
assert all(df['patch'].apply(len) == 8)
assert all(df['resolved'].apply(len) == 8)

df.to_parquet("data/info_test_trae.parquet")
