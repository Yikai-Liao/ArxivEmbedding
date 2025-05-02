# script/upload2hg.py
# 这个版本使用 huggingface_hub.upload_file 逐个上传文件

import os
import glob
from huggingface_hub import HfApi, create_repo, upload_file

# --- 配置 ---
hub_dataset_id = "lyk/ArxivEmbedding" # 你的用户名/数据集名称
script_dir = os.path.dirname(os.path.abspath(__file__))
local_parquet_dir = os.path.join(script_dir, "../data")
parquet_files_pattern = os.path.join(local_parquet_dir, "*.parquet")

# --- 查找本地 Parquet 文件 ---
local_files = glob.glob(parquet_files_pattern)
local_files = [f for f in local_files if os.path.basename(f) != "current.parquet"]
if not local_files:
    print(f"在 '{local_parquet_dir}' 中没有找到匹配 '{parquet_files_pattern}' 的 Parquet 文件。")
    exit()


print(f"找到本地 Parquet 文件: {local_files}")


# --- 登录提示 (确保已通过 huggingface-cli login 登录) ---
print("确保你已经通过 'huggingface-cli login' 登录。")

# --- 创建或确认 Hub 仓库存在 ---
try:
    repo_url = create_repo(hub_dataset_id, repo_type="dataset", exist_ok=True)
    print(f"Hugging Face Hub 数据集仓库 '{hub_dataset_id}' 已确认存在或已创建: {repo_url}")
except Exception as e:
    print(f"创建或访问 Hub 仓库 '{hub_dataset_id}' 时出错: {e}")
    exit()

# --- 上传每个 Parquet 文件 ---
print("\n开始上传文件...")
for local_file_path in local_files:
    file_name = os.path.basename(local_file_path)
    destination_path_in_repo = file_name # 文件在仓库中的路径 (根目录)

    print(f"  正在上传 '{file_name}' 到仓库 '{hub_dataset_id}'...")
    try:
        upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo=destination_path_in_repo,
            repo_id=hub_dataset_id,
            repo_type="dataset",
            commit_message=f"Upload/Update {file_name}"
        )
        print(f"  '{file_name}' 上传成功。")
    except Exception as e:
        print(f"  上传 '{file_name}' 时出错: {e}")

print("\n所有本地 Parquet 文件处理完毕。")
print("-" * 30)
print("如何从 Hub 加载数据:")
print("加载特定年份 (例如 2020):")
print("  from datasets import load_dataset")
print(f"  ds_2020 = load_dataset('{hub_dataset_id}', data_files='2020.parquet', split='train')")
print("\n加载所有年份 (合并为一个 Dataset):")
print("  from datasets import load_dataset")
print(f"  ds_all = load_dataset('{hub_dataset_id}', data_files='*.parquet', split='train')")
print("-" * 30)
