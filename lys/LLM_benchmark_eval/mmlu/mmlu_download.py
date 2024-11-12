import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all")
print(ds)

# 访问特定数据集的某个分割（如训练集）
train_data = ds['test']

# 查看前几条数据
print(train_data[:5])