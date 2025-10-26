import pandas as pd
import re

# extract round-1 (x1, from part 1 to part 4) data for round-2 competition (x2 or x4)

def ext(res):
    e = re.findall(r'\\boxed\{(.*?)\}', res[0])
    if len(e) == 0:
        return 1
    e = e[-1]
    if e == '':
        return 1
    return int(e)

df = pd.read_parquet("data/info_test_minise_true_only.parquet")
print(df['patch'])
df = df[df['resolved'].apply(lambda x: True in x)]
df = df.reset_index()
print(df)
print(df['patch'])
df1 = pd.read_parquet("data/new_output_test_minise_competition_1_SB_DAPO_RL_32B_225.parquet")
df2 = pd.read_parquet("data/new_output_test_minise_competition_2_SB_DAPO_RL_32B_225.parquet")
df3 = pd.read_parquet("data/new_output_test_minise_competition_3_SB_DAPO_RL_32B_225.parquet")
df4 = pd.read_parquet("data/new_output_test_minise_competition_4_SB_DAPO_RL_32B_225.parquet")

df1['response1'] = df1['responses']
df1['response2'] = df2['responses']
df1['response3'] = df3['responses']
df1['response4'] = df4['responses']
print(df1.keys())
print(df1['prompt1'])
print(df1['response1'])
print(df1['response1'][267][0])
df['index1'] = df1['response1'].apply(ext)
df['index2'] = df1['response2'].apply(ext)
df['index3'] = df1['response3'].apply(ext)
df['index4'] = df1['response4'].apply(ext)
print(df['index1'])
print(df['index2'])
print(df['index3'])
print(df['index4'])
print(df['patch'])
print(df.keys())
df.to_parquet('data/info_test_minise_competition_x2x4_new.parquet')

