#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import glob
import torch 

from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import NDCGAt, AvgPrecisionAt, RecallAt
from transformers4rec.torch.utils.examples_utils import wipe_memory
from transformers4rec.torch.utils.data_utils import T4RecDataLoader


import nvtabular as nvt
import cudf
import torch.utils.dlpack as dlpack
import tqdm

S3_PATH = f"s3a://zarklab-token-recommender-ohio/buy-sell-updated-2018-11-02-2023-07-31/part-00000-cdec6698-375c-4e9b-9517-b78bada90c52-c000.snappy.parquet"

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/home/sagemaker-user/token_recommender/")

workflow = nvt.Workflow.load('/home/sagemaker-user/token_recommender/preprocessWorkflow/')

model_path= os.environ.get("OUTPUT_DIR", f"{INPUT_DATA_DIR}/saved_model")

model = tr.Model().load(model_path)

dataset = nvt.Dataset(S3_PATH,engine='parquet')

# mydata['timestamp'] = cudf.to_datetime(mydata['timestamp'])

# print("Count with  repeated interactions: {}".format(len(mydata)))

# # Sorts the dataframe by session and timestamp, to remove consecutive repetitions
# # mydata.timestamp = mydata.timestamp.astype(int)
# interactions_df = mydata.sort_values(['recipient', 'timestamp'])
# past_ids = interactions_df['buyAsset'].shift(1).fillna()
# session_past_ids = interactions_df['recipient'].shift(1).fillna()

# # Keeping only no consecutive repeated in session interactions
# interactions_df = interactions_df[~((interactions_df['recipient'] == session_past_ids) & (interactions_df['buyAsset'] == past_ids))]

# print("Count after removed in-session repeated interactions: {}".format(len(interactions_df)))


procesed = workflow.transform(dataset)


dataloader =  T4RecDataLoader.parse('merlin').from_schema(
            procesed.schema,
            procesed,
            1280,
            max_sequence_length=20,
            drop_last=False,
            shuffle=False
        )





all_indices = []
K=10
softmax = torch.nn.Softmax(dim=1)
for i, data in enumerate(dataloader):
    out = model.forward(data[0],testing = False)
    prob = softmax(out)
    top5 = torch.topk(prob, K, dim=1,sorted=True)
    
    # score = top5.values
    indices = top5.indices

    all_indices.append(indices)
    if i % 100==0:
        print(i/len(dataloader))



all_indices_cat = torch.cat(all_indices)

print(all_indices_cat.shape)


column_names = [f'top{x}' for x in range(1,11)]
all_indices_cat_numpy = all_indices_cat.cpu().numpy()

procesed_df = procesed.to_ddf()[["recipient",'buyAsset-list']]


procesed_cudf = procesed_df.compute()

procesed_cudf.shape


procesed_cudf.head()


for i,col in enumerate(column_names):
    procesed_cudf[col] = all_indices_cat_numpy[:, i]

print(procesed_cudf.head())

procesed_cudf.drop(['buyAsset-list'],axis=1).to_parquet(f"s3a://zarklab-token-recommender-ohio/prediction_reuslt/test.parquet")




