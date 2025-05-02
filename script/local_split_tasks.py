#!/usr/bin/env python
import os
import sys
import argparse # Use argparse for command-line arguments
from huggingface_hub import HfApi, hf_hub_download
import toml
import json
import math
import polars as pl
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import random

def run_task_splitting(args):
    """Main logic for listing files, checking tasks, and splitting."""

    # --- Configuration from args ---
    repo_id = args.repo_id
    target_matrix_count = args.matrix_count
    target_years_input = args.years
    config_file = args.config_file
    output_task_dir = args.output_dir
    hf_token = os.environ.get('HF_TOKEN') # Still try to get token from env

    if not hf_token:
         print("警告: 未找到 HF_TOKEN 环境变量。如果你需要访问私有仓库，请确保已登录 (huggingface-cli login) 或设置了 HF_TOKEN。")
         # Set token to None if not found, HfApi might work for public repos
         hf_token = None 
         
    print(f"--- 本地任务拆分运行 ---")
    print(f"仓库 ID (Repo ID): {repo_id}")
    print(f"目标矩阵数量 (Matrix Count): {target_matrix_count}")
    print(f"目标年份 (Target Years): '{target_years_input}'")
    print(f"配置文件 (Config File): {config_file}")
    print(f"输出任务目录 (Output Task Dir): {output_task_dir}")
    print(f"--------------------------")

    # 获取需要处理的年份列表（如果有指定）
    target_years = []
    if target_years_input:
        target_years = [year.strip() for year in target_years_input.split(',')]
        print(f"用户指定处理的年份: {target_years}")

    # List files from Hub
    try:
        api = HfApi(token=hf_token) # Pass token if available
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')
    except Exception as e:
         print(f"错误: 无法连接到 Hugging Face Hub 或列出仓库文件: {e}")
         sys.exit(1)
         
    year_files = [f for f in repo_files if f.endswith('.parquet') and f[:-len('.parquet')].isdigit()]

    if target_years:
        year_files = [f for f in year_files if any(year in f for year in target_years)]

    if not year_files:
         print("错误: 在 Hub 上没有找到匹配的年份 Parquet 文件。")
         sys.exit(1)
         
    print(f"找到 Hub 上的年份文件进行处理: {year_files}")

    # 解析配置文件中的模型和维度
    try:
        if not os.path.exists(config_file):
             raise FileNotFoundError(f"配置文件未找到: {config_file}")
        config = toml.load(config_file)
        embedding_configs = config.get("Embedding", {})
        model_dims = {k: v.get("dimension") for k, v in embedding_configs.items() if isinstance(v, dict) and "dimension" in v}
        model_keys = list(model_dims.keys())
    except Exception as e:
        print(f"错误: 加载或解析配置文件 '{config_file}' 失败: {e}")
        sys.exit(1)

    if not model_keys:
        print(f"错误: 无法从 '{config_file}' 中找到带有维度的模型配置。")
        sys.exit(1)

    print(f"找到模型键及维度: {model_dims}")

    # --- Efficient Task Checking using Polars ---
    needed_tasks = []
    total_potential_tasks = 0
    threshold = 1e-7 # Tolerance for zero check

    print(f"\n开始检查文件并使用 Polars 表达式筛选任务 (阈值: {threshold})...")

    # Create local temp directory for downloads if it doesn't exist
    local_temp_dir = "_local_hub_downloads"
    os.makedirs(local_temp_dir, exist_ok=True)

    for year_file in tqdm(year_files, desc="检查年份文件"):
        file_path = None
        try:
            print(f"\n检查文件: {year_file}")
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=year_file,
                repo_type="dataset",
                local_dir=local_temp_dir, # Download to local temp dir
                token=hf_token
            )
            print(f"  已下载到: {file_path}")

            df_year = pl.read_parquet(file_path)
            schema = df_year.schema
            print(f"  文件 {year_file} schema: {list(schema.keys())}")
            total_potential_tasks += df_year.height * len(model_keys) # Rough estimate

            # Iterate through each model we need embeddings for
            for model_key, expected_dim in model_dims.items():
                print(f"  检查模型: {model_key}")

                # Case 1: Column doesn't exist -> All IDs for this model are tasks
                if model_key not in schema:
                    print(f"    列 '{model_key}' 不存在，将所有 {df_year.height} 个 ID 添加到任务列表。")
                    tasks_for_model = [{"id": row_id, "year_file": year_file, "model_key": model_key}
                                       for row_id in df_year.get_column("id").to_list()]
                    needed_tasks.extend(tasks_for_model)
                    continue # Move to the next model

                # Case 2: Column exists, check for null or near-zero vectors
                try:
                     # Define the filter condition using Polars expressions
                     is_null_expr = pl.col(model_key).is_null()

                     # --- FINAL CORRECTED near-zero expression using arr.sum() ---
                     # Check if the sum of the array elements is very close to zero.
                     # This is an approximation but avoids complex element-wise checks.
                     # It assumes non-zero vectors won't coincidentally sum exactly to zero.
                     # Handle null sums explicitly before comparison.
                     is_near_zero_expr = (
                          pl.when(pl.col(model_key).arr.sum().is_null()) 
                         .then(pl.lit(False)) # If sum is null, not near zero
                         .otherwise(pl.col(model_key).arr.sum().abs() < threshold) 
                         .fill_null(False) # Catch any other potential nulls (e.g., if col itself is null)
                     )
                     # --- End FINAL CORRECTED expression ---

                     # --- DEBUGGING START ---
                     # Calculate counts for each condition separately
                     null_count = df_year.filter(is_null_expr).height
                     # We filter with `~is_null_expr` here for accurate debug count.
                     near_zero_count = df_year.filter(is_near_zero_expr & (~is_null_expr)).height
                     print(f"    调试: 模型 '{model_key}' - Null count: {null_count}")
                     print(f"    调试: 模型 '{model_key}' - Near-zero count (sum abs < {threshold}): {near_zero_count}")
                     # --- DEBUGGING END ---

                     # Combine conditions: null OR near-zero
                     filter_condition = is_null_expr | is_near_zero_expr

                     # Apply the filter and select the IDs
                     ids_to_process_df = df_year.filter(filter_condition).select("id")
                     ids_to_process = ids_to_process_df.get_column("id").to_list()

                     count = len(ids_to_process)
                     if count > 0:
                         print(f"    发现 {count} 个需要处理的任务 (null 或 sum abs near-zero)。")
                         tasks_for_model = [{"id": row_id, "year_file": year_file, "model_key": model_key}
                                            for row_id in ids_to_process]
                         needed_tasks.extend(tasks_for_model)
                     else:
                          print(f"    所有行的 '{model_key}' 向量都已存在且 sum abs 非零。")
                          # --- DEBUGGING START: Inspect non-zero rows ---
                          try:
                               # Select rows that were *not* filtered out (should be non-null and not near-zero)
                               not_processed_df = df_year.filter(~filter_condition).select(['id', model_key])
                               sample_size = min(5, not_processed_df.height) # Show up to 5 samples
                               if sample_size > 0:
                                   print(f"    调试: 抽查 {sample_size} 个未被处理的 '{model_key}' 行:")
                                   sample_df = not_processed_df.sample(n=sample_size, seed=42) # Use seed for reproducibility
                                   for row in sample_df.iter_rows(named=True):
                                       row_id = row['id']
                                       value = row[model_key]
                                       if isinstance(value, (list, np.ndarray)):
                                            value_preview = str(np.array(value[:5])) + "..." # Show first 5 elements
                                            value_sum = np.sum(value)
                                            print(f"      - ID: {row_id}, Value Preview: {value_preview}, Sum: {value_sum:.4e}")
                                       else:
                                            print(f"      - ID: {row_id}, Value: {value} (Type: {type(value)})")
                          except Exception as debug_err:
                               print(f"    调试: 抽查未处理行时出错: {debug_err}")
                          # --- DEBUGGING END ---

                except Exception as filter_err:
                     print(f"    处理模型 '{model_key}' 的筛选条件时出错: {filter_err}。将假设所有任务都需要处理。")
                     # Fallback: Add all IDs for this model if filtering fails
                     tasks_for_model = [{"id": row_id, "year_file": year_file, "model_key": model_key}
                                        for row_id in df_year.get_column("id").to_list()]
                     needed_tasks.extend(tasks_for_model)

            file_path = None # Reset path

        except Exception as e:
            print(f"\n!!!!!! 检查 {year_file} 时发生严重错误: {e}。将跳过此文件。!!!!!!")
            # Attempt cleanup even on outer error
            if file_path and os.path.exists(file_path):
                try: 
                    os.remove(file_path)
                    print(f"  (错误后) 已清理下载文件: {file_path}")
                except OSError as clean_err: 
                    print(f"  (错误后) 警告: 无法清理下载文件 {file_path}: {clean_err}")
            # For local debugging, maybe continue to the next file instead of exiting
            # continue 

    # --- End Efficient Task Checking ---

    total_needed_tasks = len(needed_tasks)
    print(f"\n检查完成。总潜在任务数 (估算): {total_potential_tasks}。实际需要处理的任务数: {total_needed_tasks}")

    # --- Task Distribution ---
    actual_matrix_count = 0
    task_matrices_for_strategy = []
    assigned_task_count_total = 0
    target_matrix_limit = args.matrix_count # Total number of matrices to create

    if total_needed_tasks > 0:
        os.makedirs(output_task_dir, exist_ok=True)

        # Shuffle all potential tasks first
        random.shuffle(needed_tasks)

        # Calculate ideal and actual tasks per matrix
        ideal_tasks_per_matrix = math.ceil(total_needed_tasks / target_matrix_limit)
        actual_tasks_per_matrix = min(ideal_tasks_per_matrix, args.max_tasks_per_matrix)

        print(f"目标矩阵数: {target_matrix_limit}, 每个矩阵理想任务数: {ideal_tasks_per_matrix}, 每个矩阵实际任务数 (受 max_tasks_per_matrix={args.max_tasks_per_matrix} 限制): {actual_tasks_per_matrix}")

        # Determine the total tasks that will actually be assigned
        total_tasks_to_assign = min(total_needed_tasks, actual_tasks_per_matrix * target_matrix_limit)

        if total_tasks_to_assign < total_needed_tasks:
             print(f"警告: 由于 max_tasks_per_matrix ({args.max_tasks_per_matrix}) 或 matrix_count ({target_matrix_limit}) 的限制，将丢弃 {total_needed_tasks - total_tasks_to_assign} 个任务。")

        # Select the subset of tasks to distribute
        tasks_to_distribute = needed_tasks[:total_tasks_to_assign]
        assigned_task_count_total = len(tasks_to_distribute) # This is the final count of assigned tasks

        # Write the *actually assigned* tasks to the summary file
        summary_file_path = os.path.join(output_task_dir, "all_tasks_info.json")
        try:
             with open(summary_file_path, "w") as f:
                 json.dump(tasks_to_distribute, f, indent=2)
             print(f"已将分配的 {assigned_task_count_total} 个任务信息写入到 {summary_file_path}")
        except Exception as e:
             print(f"错误: 无法写入任务摘要文件 {summary_file_path}: {e}")
             # Consider if failure here should halt the process

        # Distribute the selected tasks into matrix files
        matrix_index = 0
        start_index = 0
        # Loop until all assigned tasks are distributed OR matrix limit is reached
        while start_index < assigned_task_count_total and matrix_index < target_matrix_limit:
            # Determine the tasks for the current matrix file
            end_index = min(start_index + actual_tasks_per_matrix, assigned_task_count_total)
            matrix_tasks = tasks_to_distribute[start_index:end_index]

            if not matrix_tasks:
                break # Should not happen if assigned_task_count_total > 0, but safe check

            matrix_task_file = os.path.join(output_task_dir, f"matrix_{matrix_index}_tasks.json")
            try:
                with open(matrix_task_file, "w") as f:
                    json.dump(matrix_tasks, f, indent=2)

                # assigned_task_count_total += len(matrix_tasks) # No, total is calculated before loop
                print(f"矩阵 {matrix_index}: 分配了 {len(matrix_tasks)} 个任务，保存到 {matrix_task_file}")
                task_matrices_for_strategy.append({"matrix_id": matrix_index})
                matrix_index += 1

            except Exception as e:
                 print(f"错误: 无法写入任务文件 {matrix_task_file}: {e}")

            start_index = end_index # Move to the next chunk for the next matrix file

        actual_matrix_count = matrix_index # Final count is the number of files created
        print(f"\n总共分配了 {assigned_task_count_total} 个任务到 {actual_matrix_count} 个矩阵文件中 (在 '{output_task_dir}' 目录下)。")

    elif total_needed_tasks == 0:
         print("警告：没有找到任何需要处理的任务。实际矩阵数将为 0。")
         # actual_matrix_count remains 0
         # task_matrices_for_strategy remains []

    # --- Output summary conditionally ---
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # Running in GitHub Actions, use GITHUB_OUTPUT
        github_output_file = os.environ.get("GITHUB_OUTPUT")
        if github_output_file:
             # Ensure task_matrices is valid JSON (it should be)
             task_matrices_json = json.dumps(task_matrices_for_strategy)
             delimiter = "ghadelimiter_" + os.urandom(16).hex() # Use a random delimiter
             with open(github_output_file, 'a') as f: # Append to the output file
                 f.write(f"total_tasks={assigned_task_count_total}\n")
                 f.write(f"matrix_count={actual_matrix_count}\n")
                 # Use delimiter method for task_matrices JSON string
                 f.write(f"task_matrices<<{delimiter}\n")
                 f.write(f"{task_matrices_json}\n")
                 f.write(f"{delimiter}\n")
             print("::debug::Successfully wrote outputs to GITHUB_OUTPUT file using delimiter method.")
        else:
             print("::error::GITHUB_OUTPUT environment variable not found, cannot set outputs for Actions.")
             sys.exit(1) # Exit with error if we are in Actions but cannot set output
    else:
        # Running locally, print to console
        print("\n--- 本地运行任务拆分总结 ---")
        print(f"总任务数 (Total Tasks Found): {total_needed_tasks}")
        print(f"已分配任务数 (Assigned Tasks): {assigned_task_count_total}") # Use the final assigned count
        print(f"实际创建矩阵数 (Matrix Count): {actual_matrix_count}")
        print(f"矩阵策略信息 (Task Matrices for Strategy): {json.dumps(task_matrices_for_strategy)}")
        print("-----------------------------")

    # Clean up download directory if desired
    # try:
    #     import shutil
    #     shutil.rmtree(local_temp_dir)
    #     print(f"已清理下载目录: {local_temp_dir}")
    # except Exception as e:
    #     print(f"警告: 清理下载目录 {local_temp_dir} 失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地运行 Arxiv Embedding 任务筛选和拆分逻辑。")
    parser.add_argument("--repo-id", type=str, default="lyk/ArxivEmbedding", help="Hugging Face Hub 仓库 ID。")
    parser.add_argument("--matrix-count", type=int, default=10, help="目标矩阵数量。")
    parser.add_argument("--years", type=str, default="", help="要处理的年份列表 (逗号分隔)，留空处理所有年份。")
    parser.add_argument("--config-file", type=str, default="config.toml", help="配置文件的路径。")
    parser.add_argument("--output-dir", type=str, default="local_matrix_tasks", help="保存拆分任务 JSON 文件的目录。")
    parser.add_argument("--max-tasks-per-matrix", type=int, default=sys.maxsize, help="每个矩阵 JSON 文件包含的最大任务数 (默认为无限制)。")

    args = parser.parse_args()
    run_task_splitting(args)