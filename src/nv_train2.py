#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import glob
import torch 

import os
import glob
import numpy as np
import pandas as pd
import gc
import calendar
import datetime
import cudf
import cupy
import nvtabular as nvt
from merlin.dag import ColumnSelector
from merlin.schema import Schema, Tags
import nvtabular as nv
from nvtabular.ops import *
import argparse
from datetime import datetime
import boto3
from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import NDCGAt, AvgPrecisionAt, RecallAt
from transformers4rec.torch.utils.examples_utils import wipe_memory
from merlin.schema import Schema
from merlin.io import Dataset
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer
import pickle
import argparse



def remove_duplicates(df):
    print("Count with  repeated interactions: {}".format(len(df)))

    # Sorts the dataframe by session and timestamp, to remove consecutive repetitions
    # mydata.timestamp = mydata.timestamp.astype(int)
    interactions_df = df.sort_values(['recipient', 'timestamp'])
    past_ids = interactions_df['buyAsset'].shift(1).fillna()
    session_past_ids = interactions_df['recipient'].shift(1).fillna()

    # Keeping only no consecutive repeated in session interactions
    interactions_df = interactions_df[~((interactions_df['recipient'] == session_past_ids) & (interactions_df['buyAsset'] == past_ids))]

    print("Count after removed in-session repeated interactions: {}".format(len(interactions_df)))

    return interactions_df

def get_list_files(s3_bucket,s3_prefix ,mykey):
    
    aws_access_key_id = mykey['aws_access_key_id']
    aws_secret_access_key = mykey['aws_secret_access_key']
    region_name = mykey['region_name']

    # Create a session using Boto3
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
    
    # Create an S3 client
    s3_client = session.client('s3')

    # List objects in the S3 bucket based on your prefix
    response = s3_client.list_objects_v2(
        Bucket=s3_bucket,
        Prefix=s3_prefix
    )
    return response


def get_cycled_feature_value_sin(col, max_value):
    value_scaled = (col + 0.000001) / max_value
    value_sin = np.sin(2*np.pi*value_scaled)
    return value_sin

def build_features(SESSIONS_MAX_LENGTH,MINIMUM_SESSION_LENGTH):
    # Encodes categorical features as contiguous integers
    cat_feats = ColumnSelector(['buyAsset','token_category','token_rank','risky_flags']) >> nvt.ops.Categorify()
    # create time features
    session_ts = ColumnSelector(['timestamp'])

    #build continuous features
    txFee_eth_norm =  ColumnSelector(['txFee_eth']) >> nvt.ops.LogOp() >> nvt.ops.Normalize(out_dtype=np.float32) >> nvt.ops.Rename(name='txFee_eth_log_norm')
    buyQty1_norm =  ColumnSelector(['buyQty1']) >> nvt.ops.Clip(min_value=0, max_value=10000000) >> nvt.ops.LogOp()>> nvt.ops.Normalize(out_dtype=np.float32) >> nvt.ops.Rename(name='buyQty1_log_norm')
    buyPrice_norm =  ColumnSelector(['buyPrice']) >>nvt.ops.Clip(min_value=0, max_value=10000000)>> nvt.ops.LogOp() >> nvt.ops.Normalize(out_dtype=np.float32) >> nvt.ops.Rename(name='buyPrice_log_norm')
    
    features = ColumnSelector(['recipient']) + cat_feats + session_ts + txFee_eth_norm + buyQty1_norm + buyPrice_norm

    
    # Define Groupby Operator
    groupby_features = features >> nvt.ops.Groupby(
        groupby_cols=["recipient"], 
        sort_cols=["timestamp"],
        aggs={
            'buyAsset': ["list", "count"],
            'timestamp': ["last"],  
            'txFee_eth_log_norm': ["list"],
            'buyQty1_log_norm': ["list"],
            'buyPrice_log_norm': ["list"],
            'token_category': ["list"],
            'token_rank': ["list"],
            'risky_flags': ["list"]
        },
        name_sep="-")
    
        # Select and truncate the sequential features
    sequence_features_truncated = (
        groupby_features['token_category-list','token_rank-list','risky_flags-list']
        >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH) 
    )

    sequence_features_truncated_item = (
        groupby_features['buyAsset-list']
        >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH) 
        >> TagAsItemID()
    )  
    sequence_features_truncated_cont = (
        groupby_features[ 'txFee_eth_log_norm-list','buyQty1_log_norm-list','buyPrice_log_norm-list'] 
        >> nvt.ops.ListSlice(-SESSIONS_MAX_LENGTH) 
        >> nvt.ops.AddMetadata(tags=[Tags.CONTINUOUS])
    )
    
    
    selected_features = (
        groupby_features['buyAsset-count', 'timestamp-last', 'recipient'] + 
        sequence_features_truncated +
        sequence_features_truncated_item +
        sequence_features_truncated_cont
    )

    filtered_sessions = selected_features >> nvt.ops.Filter(f=lambda df: df["buyAsset-count"] >= MINIMUM_SESSION_LENGTH)

    seq_feats_list = filtered_sessions['buyAsset-list', 'txFee_eth_log_norm-list','buyQty1_log_norm-list','buyPrice_log_norm-list','token_category-list','token_rank-list','risky_flags-list'] >>  nvt.ops.ValueCount()

    workflow = nvt.Workflow(filtered_sessions['recipient', 'timestamp-last'] + seq_feats_list)
    

    return workflow


def load_dataset(s3_bucket ,s3_prefix,mykey):
    df_no_duplicate = []
    
    response = get_list_files(s3_bucket ,s3_prefix,mykey)

    for obj in response.get('Contents', []):
        key = obj['Key']

        # Load only Parquet files based on your partitioning structure
        if key.endswith('.parquet'):

            parquet_file = f's3://{s3_bucket}/{key}'

            mydata = cudf.read_parquet(parquet_file)
            

            df_no_duplicate.append(remove_duplicates(mydata))
                                    

    result_df = cudf.concat(df_no_duplicate, ignore_index=True)
    return result_df

def define_model(schema):
    inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        continuous_projection=64,
        masking="mlm",
        d_output=100,
    )

    # Define XLNetConfig class and set default parameters for HF XLNet config  
    transformer_config = tr.XLNetConfig.build(
        d_model=64, n_head=4, n_layer=2, total_seq_length=20
    )
    # Define the model block including: inputs, masking, projection and transformer block.
    body = tr.SequentialBlock(
        inputs, tr.MLPBlock([64]), tr.TransformerBlock(transformer_config, masking=inputs.masking)
    )

    # Define the evaluation top-N metrics and the cut-offs
    metrics = [NDCGAt(top_ks=[5, 10], labels_onehot=True),  
               RecallAt(top_ks=[5, 10], labels_onehot=True),
               AvgPrecisionAt(top_ks=[5,10], labels_onehot=True)]

    # Define a head related to next item prediction task 
    head = tr.Head(
        body,
        tr.NextItemPredictionTask(weight_tying=True, 
                                  metrics=metrics),
        inputs=inputs,
    )

    # Get the end-to-end Model class 
    model = tr.Model(head)
    
    return model

def main(args):

    # INPUT_TABLE = args.input_table
    # SESSIONS_MAX_LENGTH = args.max_length
    # MINIMUM_SESSION_LENGTH = args.min_length
    # WORKFLOW_PATH = args.workflowPath
    # OUTPUT_TABLE = args.output_table
    # MYKEY = args.mykey_path


    INPUT_TABLE ="s3://zarklab-token-recommender-ohio2/buy-sell-updated-2022-12-01-2023-11-30_repartition2/"
    SESSIONS_MAX_LENGTH = 20
    MINIMUM_SESSION_LENGTH = 2
    WORKFLOW_PATH = "/home/sagemaker-user/token_recommender/preprocessWorkflow"
    OUTPUT_TABLE = "/home/sagemaker-user/token_recommender/buy-sell_agg_train"
    MYKEY = "/home/sagemaker-user/token_recommender/mykey.pkl"
    
    
    per_device_train_batch_size = 256
    per_device_eval_batch_size = 256
    train_paths = "/home/sagemaker-user/token_recommender/buy-sell_agg_train"

    MODEL_PATH="/home/sagemaker-user/token_recommender/saved_model"
    
    

    s3_bucket = INPUT_TABLE.split('/')[2]
    s3_prefix = INPUT_TABLE.split('/')[3]

    with open(MYKEY, 'rb') as file:
        mykey = pickle.load(file)

    result_df = load_dataset(s3_bucket,s3_prefix,mykey)

    dataset = nv.Dataset(result_df)
    workflow = build_features(SESSIONS_MAX_LENGTH,MINIMUM_SESSION_LENGTH)
    workflow.fit(dataset)
    out = workflow.transform(dataset)
    

    workflow.save(WORKFLOW_PATH)
    out.to_parquet(OUTPUT_TABLE)
    
    schema = out.schema
    
    wipe_memory()
    
    

    model = define_model(schema)
    
    
    train_args = T4RecTrainingArguments(data_loader_engine='merlin', 
                                    dataloader_drop_last = True,
                                    gradient_accumulation_steps = 1,
                                    per_device_train_batch_size = per_device_train_batch_size, 
                                    per_device_eval_batch_size = per_device_eval_batch_size,
                                    output_dir = "/home/ec2-user/SageMaker/model/", 
                                    learning_rate=0.0002,
                                    lr_scheduler_type='cosine', 
                                    learning_rate_num_cosine_cycles_by_epoch=1.5,
                                    num_train_epochs=20,
                                    max_sequence_length=20, 
                                    report_to = [],
                                    logging_steps=50,
                                    no_cuda=False)
    
    
    trainer = Trainer(model=model,args=train_args,schema=out.schema,compute_metrics=True)
    
    
    print(train_paths)

    trainer.train_dataset_or_path = train_paths
    trainer.reset_lr_scheduler()
    trainer.train()

    
    model.save(MODEL_PATH)
#     eval_paths = train_paths

#     trainer.eval_dataset_or_path = train_paths
#     train_metrics = trainer.evaluate(metric_key_prefix='eval')
#     print('*'*20)
#     print('\n' + '*'*20 + '\n')
#     for key in sorted(train_metrics.keys()):
#         print(" %s = %s" % (key, str(train_metrics[key]))) 
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference')
    
    parser.add_argument('--input_table', type=str, default="s3://zarklab-token-recommender-ohio2/buy-sell-updated-2022-12-01-2023-11-30_repartition2/", required=False, help='Path for the greeting')
    parser.add_argument('--max_length', type=int, default="20", required=False, help='max sequence length')
    parser.add_argument('--min_length', type=int, default="2", required=False, help='max sequence length')
    parser.add_argument('--workflowPath', type=str, default="/home/sagemaker-user/token_recommender/preprocessWorkflow", required=False, help='Path to save the pre ')
    parser.add_argument('--output_table', type=str, default="s3://zarklab-token-recommender-ohio2/buy-sell-updated-2022-12-01-2023-11-30_agg", required=False, help='max sequence length')

    args = parser.parse_args()
    
    main(args)
