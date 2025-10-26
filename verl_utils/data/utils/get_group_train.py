from copy import deepcopy
import pandas as pd
import numpy as np
import re
np.random.seed(42)

# RL only
df1 = pd.read_parquet('data/info_train_single_01.parquet')
df2 = pd.read_parquet('data/info_train_single_02.parquet')
df3 = pd.read_parquet('data/info_train_single_03.parquet')
df4 = pd.read_parquet('data/info_train_single_04.parquet')
df5 = pd.read_parquet('data/info_train_single_05.parquet')
df6 = pd.read_parquet('data/info_train_single_06.parquet')

df1 = pd.concat([df1, pd.read_parquet('data/info_train_multiple_01.parquet')], ignore_index=True)
df2 = pd.concat([df2, pd.read_parquet('data/info_train_multiple_02.parquet')], ignore_index=True)
df3 = pd.concat([df3, pd.read_parquet('data/info_train_multiple_03.parquet')], ignore_index=True)
df4 = pd.concat([df4, pd.read_parquet('data/info_train_multiple_04.parquet')], ignore_index=True)
df5 = pd.concat([df5, pd.read_parquet('data/info_train_multiple_05.parquet')], ignore_index=True)
df6 = pd.concat([df6, pd.read_parquet('data/info_train_multiple_06.parquet')], ignore_index=True)

df1 = df1[df1['patch'].fillna('').str.strip() != '']
df2 = df2[df2['patch'].fillna('').str.strip() != '']
df3 = df3[df3['patch'].fillna('').str.strip() != '']
df4 = df4[df4['patch'].fillna('').str.strip() != '']
df5 = df5[df5['patch'].fillna('').str.strip() != '']
df6 = df6[df6['patch'].fillna('').str.strip() != '']

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
df5['patch'] = df5['patch'].apply(remove_index_from_patch)
df6['patch'] = df6['patch'].apply(remove_index_from_patch)

df_lists = [df1, df2, df3, df4, df5, df6]


df1_instance_list = df1['instance_id'].tolist()
df2_instance_list = df2['instance_id'].tolist()
df3_instance_list = df3['instance_id'].tolist()
df4_instance_list = df4['instance_id'].tolist()
df5_instance_list = df5['instance_id'].tolist()
df6_instance_list = df6['instance_id'].tolist()

instance_lists = [df1_instance_list, df2_instance_list, df3_instance_list, df4_instance_list, df5_instance_list, df6_instance_list]

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

cross_id_lists = []
merged_df_lists = []
overlap_cnts = 0
for i in range(6):
    for j in range(i+1, 6):
        temp_cross_instance_list = []
        temp_df_lists = []
        for k in range(6):
            if k != i and k != j:
                temp_cross_instance_list.append(deepcopy(instance_lists[k]))
                temp_df_lists.append(deepcopy(df_lists[k]))
        cross_ids = get_4_cross_lists(temp_cross_instance_list)
        cross_id_lists.append(cross_ids)
        merged_df, overlap_cnt = get_4_merged_df(temp_df_lists, cross_ids)
        merged_df_lists.append(merged_df)
        overlap_cnts += overlap_cnt

print(overlap_cnts)

patch_instance_count_list = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
}
for df in merged_df_lists:
    patch_instance_count_list['0'] += df[df['resolved'].apply(lambda x: x.count(True) == 0)].shape[0]
    patch_instance_count_list['1'] += df[df['resolved'].apply(lambda x: x.count(True) == 1)].shape[0]
    patch_instance_count_list['2'] += df[df['resolved'].apply(lambda x: x.count(True) == 2)].shape[0]
    patch_instance_count_list['3'] += df[df['resolved'].apply(lambda x: x.count(True) == 3)].shape[0]
    patch_instance_count_list['4'] += df[df['resolved'].apply(lambda x: x.count(True) == 4)].shape[0]

print(patch_instance_count_list)

new_df = pd.DataFrame()
for df in merged_df_lists:
    # Separate all cases
    df_0_correct = df[df['resolved'].apply(lambda x: x.count(True) == 0)]  # All incorrect
    df_1_correct = df[df['resolved'].apply(lambda x: x.count(True) == 1)]  # 1 correct, 3 incorrect
    df_2_correct = df[df['resolved'].apply(lambda x: x.count(True) == 2)]  # Balanced case
    df_3_correct = df[df['resolved'].apply(lambda x: x.count(True) == 3)]  # 1 incorrect, 3 correct
    df_4_correct = df[df['resolved'].apply(lambda x: x.count(True) == 4)]  # All correct
    
    # Balance 1-correct and 3-correct cases (which are complementary)
    min_1_3 = min(len(df_1_correct), len(df_3_correct))
    if len(df_1_correct) > min_1_3:
        df_1_correct = df_1_correct.sample(n=min_1_3, random_state=42)
    if len(df_3_correct) > min_1_3:
        df_3_correct = df_3_correct.sample(n=min_1_3, random_state=42)
    
    # If do not balance classes
    # min_0_4 = min(len(df_0_correct), len(df_4_correct))
    # df_0_correct = df_0_correct.sample(n=min_0_4, random_state=42)
    # df_4_correct = df_4_correct.sample(n=min_0_4, random_state=42)

    # If balance classes
    reference_size = len(df_1_correct) + len(df_3_correct) + len(df_2_correct)
    if len(df_0_correct) > reference_size // 3:
        df_0_correct = df_0_correct.sample(n=reference_size // 3, random_state=42)
    if len(df_4_correct) > reference_size // 3:
        df_4_correct = df_4_correct.sample(n=reference_size // 3, random_state=42)
    
    new_df = pd.concat([
        new_df,
        df_0_correct,
        df_1_correct,
        df_2_correct,
        df_3_correct,
        df_4_correct
    ], axis=0)
new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True) # row shuffle
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

p = patch_instance_count_list['4'] * 4 + patch_instance_count_list['3'] * 3 + patch_instance_count_list['2'] * 2 + patch_instance_count_list['1'] * 1
n = patch_instance_count_list['0'] * 4 + patch_instance_count_list['1'] * 3 + patch_instance_count_list['2'] * 2 + patch_instance_count_list['3'] * 1

print(p)
print(n)

print(new_df[new_df['patch'].apply(lambda x: x[0] == x[1] and x[1] == x[2] and x[2] == x[3])].shape[0])

new_df.to_parquet('data/info_train_r4p.parquet')