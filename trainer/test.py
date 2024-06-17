from transformers import BertTokenizer, BertModel

import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

model_name = "bert-base-uncased"  # 可以选择其他的预训练模型

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

import torch
# input_text = "Hello, this is a test sentence."
# inputs = tokenizer(input_text, return_tensors="pt")
# outputs = model(**inputs)
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)

import torch

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# 如果可用，打印可用的 GPU 数量和当前 GPU
if cuda_available:
    num_gpus = torch.cuda.device_count()
    current_gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu)
    print(f"Number of GPUs: {num_gpus}")
    print(f"Current GPU ID: {current_gpu}")
    print(f"Current GPU Name: {gpu_name}")
import torch
print(torch.version.cuda)
import torch
print(torch.__version__)
