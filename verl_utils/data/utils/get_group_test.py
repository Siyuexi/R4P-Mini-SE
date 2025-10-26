from copy import deepcopy
import pandas as pd
import numpy as np
import re
np.random.seed(42)

df1 = pd.read_parquet('data/info_test_01.parquet')
df2 = pd.read_parquet('data/info_test_02.parquet')
df3 = pd.read_parquet('data/info_test_03.parquet')
df4 = pd.read_parquet('data/info_test_04.parquet')

df1 = df1[df1['patch'].fillna('').str.strip() != '']
df2 = df2[df2['patch'].fillna('').str.strip() != '']
df3 = df3[df3['patch'].fillna('').str.strip() != '']
df4 = df4[df4['patch'].fillna('').str.strip() != '']

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

df1['patch'] = df1['patch'].apply(get_pure_patch)
df2['patch'] = df2['patch'].apply(get_pure_patch)
df3['patch'] = df3['patch'].apply(get_pure_patch)
df4['patch'] = df4['patch'].apply(get_pure_patch)

df1 = df1[df1['patch'].fillna('').str.strip() != '']
df2 = df2[df2['patch'].fillna('').str.strip() != '']
df3 = df3[df3['patch'].fillna('').str.strip() != '']
df4 = df4[df4['patch'].fillna('').str.strip() != '']

df_lists = [df1, df2, df3, df4]


df1_instance_list = df1['instance_id'].tolist()
df2_instance_list = df2['instance_id'].tolist()
df3_instance_list = df3['instance_id'].tolist()
df4_instance_list = df4['instance_id'].tolist()

instance_lists = [df1_instance_list, df2_instance_list, df3_instance_list, df4_instance_list]

def get_4_cross_lists(lists):
    return set(lists[0]).intersection(set(lists[1])).intersection(set(lists[2])).intersection(set(lists[3]))

def get_4_merged_df(lists, cross_ids):
    df = deepcopy(lists[0])
    df = df[df['instance_id'].isin(cross_ids)]
    df = df.drop(columns=['patch', 'resolved'])
    df['patch'] = df.apply(lambda row: [lists[0][lists[0]['instance_id'] == row['instance_id']]['patch'].values[0], lists[1][lists[1]['instance_id'] == row['instance_id']]['patch'].values[0], lists[2][lists[2]['instance_id'] == row['instance_id']]['patch'].values[0], lists[3][lists[3]['instance_id'] == row['instance_id']]['patch'].values[0]], axis=1)
    df['resolved'] = df.apply(lambda row: [lists[0][lists[0]['instance_id'] == row['instance_id']]['resolved'].values[0], lists[1][lists[1]['instance_id'] == row['instance_id']]['resolved'].values[0], lists[2][lists[2]['instance_id'] == row['instance_id']]['resolved'].values[0], lists[3][lists[3]['instance_id'] == row['instance_id']]['resolved'].values[0]], axis=1)
    overlap_cnt = df[df['patch'].apply(lambda x: x[0] == x[1] or x[0] == x[2] or x[0] == x[3] or x[1] == x[2] or x[1] == x[3] or x[2] == x[3])].shape[0]
    return df, overlap_cnt

cross_ids = get_4_cross_lists(instance_lists)
merged_df, overlap_cnt = get_4_merged_df(df_lists, cross_ids)

print(overlap_cnt)

patch_instance_count_list = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
}

df = merged_df

patch_instance_count_list['0'] += df[df['resolved'].apply(lambda x: x.count(True) == 0)].shape[0]
patch_instance_count_list['1'] += df[df['resolved'].apply(lambda x: x.count(True) == 1)].shape[0]
patch_instance_count_list['2'] += df[df['resolved'].apply(lambda x: x.count(True) == 2)].shape[0]
patch_instance_count_list['3'] += df[df['resolved'].apply(lambda x: x.count(True) == 3)].shape[0]
patch_instance_count_list['4'] += df[df['resolved'].apply(lambda x: x.count(True) == 4)].shape[0]

print(patch_instance_count_list)

new_df = df.sample(frac=1, random_state=42).reset_index(drop=True) # row shuffle
print(new_df)


def shuffle_patch_resolved(row):
    perm = np.random.permutation(len(row['patch']))
    row['patch'] = [row['patch'][i] for i in perm]
    row['resolved'] = [row['resolved'][i] for i in perm]
    return row

new_df = new_df.apply(shuffle_patch_resolved, axis=1) # column shuffle

print(new_df)


patch_instance_count_list['0'] = new_df[new_df['resolved'].apply(lambda x: x.count(True) == 0)].shape[0]
patch_instance_count_list['1'] = new_df[new_df['resolved'].apply(lambda x: x.count(True) == 1)].shape[0]
patch_instance_count_list['2'] = new_df[new_df['resolved'].apply(lambda x: x.count(True) == 2)].shape[0]
patch_instance_count_list['3'] = new_df[new_df['resolved'].apply(lambda x: x.count(True) == 3)].shape[0]
patch_instance_count_list['4'] = new_df[new_df['resolved'].apply(lambda x: x.count(True) == 4)].shape[0]

print(patch_instance_count_list)
print(new_df[new_df['patch'].apply(lambda x: x[0] == x[1] and x[1] == x[2] and x[2] == x[3])].shape[0])

df_1i = df1[df1["instance_id"].isin(cross_ids)]
df_2i = df2[df2["instance_id"].isin(cross_ids)]
df_3i = df3[df3["instance_id"].isin(cross_ids)]
df_4i = df4[df4["instance_id"].isin(cross_ids)]
df_naive = pd.concat([df_1i, df_2i, df_3i, df_4i], axis=0).reset_index()
print(df_naive)
df_naive.to_parquet('data/info_test_naive.parquet')

new_df.to_parquet('data/info_test_r4p.parquet')