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

df1['patch'] = df1['patch'].apply(remove_index_from_patch)
df2['patch'] = df2['patch'].apply(remove_index_from_patch)
df3['patch'] = df3['patch'].apply(remove_index_from_patch)
df4['patch'] = df4['patch'].apply(remove_index_from_patch)

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

filter_df = merged_df[merged_df['resolved'].apply(lambda x: not (all(x) or not any(x)))]

pair_rows = []

for _, row in filter_df.iterrows():
    resolved = row['resolved']
    patches = row['patch']
    instance_id = row['instance_id']
    
    correct_indices = [i for i, r in enumerate(resolved) if r]
    wrong_indices = [i for i, r in enumerate(resolved) if not r]
    
    base_data = row.drop(labels=['patch', 'resolved']).to_dict()
    
    if len(correct_indices) == 1 and len(wrong_indices) == 3:
        correct_patch = patches[correct_indices[0]]
        for wrong_idx in wrong_indices:
            order = np.random.randint(0, 2)
            pair_data = deepcopy(base_data)
            if order:
                pair_data['patch'] = [correct_patch, patches[wrong_idx]]
                pair_data['resolved'] = 'A'
            else:
                pair_data['patch'] = [patches[wrong_idx], correct_patch]
                pair_data['resolved'] = 'B'
            pair_rows.append(pair_data)
    
    elif len(correct_indices) == 3 and len(wrong_indices) == 1:
        wrong_patch = patches[wrong_indices[0]]
        for correct_idx in correct_indices:
            order = np.random.randint(0, 2)
            pair_data = deepcopy(base_data)
            if order:
                pair_data['patch'] = [patches[correct_idx], wrong_patch]
                pair_data['resolved'] = 'A'
            else:
                pair_data['patch'] = [wrong_patch, patches[correct_idx]]
                pair_data['resolved'] = 'B'
            pair_rows.append(pair_data)
    
    elif len(correct_indices) == 2 and len(wrong_indices) == 2:
        for correct_idx in correct_indices:
            for wrong_idx in wrong_indices:
                order = np.random.randint(0, 2)
                pair_data = deepcopy(base_data)
                if order:
                    pair_data['patch'] = [patches[correct_idx], patches[wrong_idx]]
                    pair_data['resolved'] = 'A'
                else:
                    pair_data['patch'] = [patches[wrong_idx], patches[correct_idx]]
                    pair_data['resolved'] = 'B'
                pair_rows.append(pair_data)

pair_df = pd.DataFrame(pair_rows)

N_original = len(merged_df)
N_current = len(pair_df)

print(N_original)
print(N_current)
print(pair_df.iloc[0])

pair_df.to_parquet('data/info_test_pair.parquet')
