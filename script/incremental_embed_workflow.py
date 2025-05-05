#!/usr/bin/env python
import argparse
import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import polars as pl
from loguru import logger
import subprocess
from huggingface_hub import HfApi, hf_hub_download

# Configure Loguru
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO", 
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

def run_command(cmd, desc=None):
    """运行命令并输出结果"""
    if desc:
        logger.info(f"{desc}")
    
    logger.debug(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.debug(f"命令输出: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        raise

def fetch_recent_data(config_file, output_file, days=2):
    """获取最近指定天数的 Arxiv 数据"""
    logger.info(f"开始获取最近 {days} 天的 Arxiv 数据...")
    
    # 计算日期范围
    today = datetime.now().date()
    from_date = today - timedelta(days=days)
    from_date_str = from_date.strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
    
    # 运行爬虫脚本
    cmd = [
        "python", "script/fetch_arxiv_oai_by_date.py",
        "--from-date", from_date_str,
        "--until-date", today_str,
        "--config", config_file,
        "--output-file", output_file
    ]
    
    return run_command(cmd, f"爬取 Arxiv 数据 (从 {from_date_str} 到 {today_str})")

def download_year_file(repo_id, year, local_dir, token=None):
    """下载指定年份的数据文件"""
    filename = f"{year}.parquet"
    logger.info(f"下载 {filename} 从 {repo_id}...")
    
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
            token=token
        )
        logger.info(f"成功下载 {filename} 到 {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"下载 {filename} 失败: {e}")
        return None

def merge_to_year_file(incremental_file, year_file, output_file=None):
    """将增量数据合并到年份文件中"""
    if output_file is None:
        output_file = year_file
    
    logger.info(f"合并增量数据到 {year_file}...")
    
    try:
        # 读取两个数据集
        df_year = pl.read_parquet(year_file)
        df_incremental = pl.read_parquet(incremental_file)
        
        logger.info(f"年份文件包含 {df_year.height} 行记录")
        logger.info(f"增量文件包含 {df_incremental.height} 行记录")
        
        if df_incremental.height == 0:
            logger.warning("增量文件为空，跳过合并")
            return df_year, 0
        
        # 获取年份文件中的嵌入模型列表
        embedding_cols = [col for col in df_year.columns if col not in [
            "id", "title", "authors", "abstract", "date", "categories",
            "created", "updated", "license"
        ]]
        
        # 为增量数据添加嵌入向量列（设为 null）
        for col in embedding_cols:
            if col not in df_incremental.columns:
                # 创建嵌入向量的空列表
                df_incremental = df_incremental.with_columns(
                    pl.lit(None).alias(col)
                )
        
        # 合并数据集，保留最新的记录
        combined_df = pl.concat([df_year, df_incremental])
        # 按 ID 去重，保留最新的记录
        combined_df = combined_df.sort("updated", descending=True).unique(subset=["id"], keep="first")
        
        # 计算新增记录数
        added_count = combined_df.height - df_year.height
        
        logger.info(f"合并后包含 {combined_df.height} 行记录，新增 {added_count} 行")
        
        # 保存合并后的数据
        combined_df.write_parquet(output_file, compression="zstd")
        logger.info(f"已保存合并后的数据到 {output_file}")
        
        return combined_df, added_count
    
    except Exception as e:
        logger.error(f"合并数据失败: {e}")
        raise

def upload_to_hub(file_path, repo_id, token=None):
    """上传文件到 Hugging Face Hub"""
    logger.info(f"上传 {file_path} 到 {repo_id}...")
    
    try:
        api = HfApi(token=token)
        response = api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=Path(file_path).name,
            repo_id=repo_id,
            repo_type="dataset"
        )
        logger.info(f"上传成功: {response}")
        return True
    except Exception as e:
        logger.error(f"上传失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Arxiv 增量数据获取与合并到 Hugging Face 数据集")
    parser.add_argument("--config", type=str, default="config.toml", 
                      help="配置文件路径 (默认: config.toml)")
    parser.add_argument("--days", type=int, default=2, 
                      help="获取最近几天的数据 (默认: 2)")
    parser.add_argument("--repo-id", type=str, default="lyk/ArxivEmbedding", 
                      help="Hugging Face 数据集仓库 ID (默认: lyk/ArxivEmbedding)")
    parser.add_argument("--temp-dir", type=str, default="temp", 
                      help="临时文件目录 (默认: temp)")
    parser.add_argument("--skip-fetch", action="store_true", 
                      help="跳过数据获取步骤")
    parser.add_argument("--skip-upload", action="store_true", 
                      help="跳过数据上传步骤")
    
    args = parser.parse_args()
    
    # 创建临时目录
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(exist_ok=True)
    
    # 文件路径
    incremental_file = temp_dir / "incremental.parquet"
    
    # 获取环境变量中的 HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("未找到 HF_TOKEN 环境变量。如果仓库是私有的，上传可能会失败。")
    
    logger.info(f"=== Arxiv 增量数据获取与合并工作流开始 ===")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"获取天数: {args.days}")
    logger.info(f"仓库 ID: {args.repo_id}")
    logger.info(f"临时目录: {args.temp_dir}")
    
    # 1. 获取最近数据
    if not args.skip_fetch:
        try:
            fetch_recent_data(args.config, incremental_file, args.days)
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return 1
    else:
        logger.info("跳过数据获取步骤")
        if not incremental_file.exists():
            logger.error(f"跳过获取步骤但增量文件 {incremental_file} 不存在")
            return 1
    
    # 2. 获取当前年份
    current_year = datetime.now().year
    year_file = temp_dir / f"{current_year}.parquet"
    
    # 3. 下载当前年份文件
    year_file_path = download_year_file(args.repo_id, current_year, temp_dir, hf_token)
    if not year_file_path:
        # 如果下载失败，创建新的年份文件
        try:
            # 读取增量文件
            df_incremental = pl.read_parquet(incremental_file)
            logger.info(f"创建新的年份文件 {year_file}，包含 {df_incremental.height} 行记录")
            # 直接保存为年份文件
            df_incremental.write_parquet(year_file, compression="zstd")
            added_count = df_incremental.height
        except Exception as e:
            logger.error(f"创建新的年份文件失败: {e}")
            return 1
    else:
        # 4. 合并增量数据到年份文件
        try:
            _, added_count = merge_to_year_file(incremental_file, year_file_path)
        except Exception as e:
            logger.error(f"合并数据失败: {e}")
            return 1
    
    # 5. 上传到 Hub
    if not args.skip_upload and added_count > 0:
        try:
            success = upload_to_hub(year_file, args.repo_id, hf_token)
            if not success:
                logger.error("上传到 Hub 失败")
                return 1
        except Exception as e:
            logger.error(f"上传到 Hub 时出错: {e}")
            return 1
    elif added_count == 0:
        logger.info("没有新增数据，跳过上传")
    else:
        logger.info("跳过数据上传步骤")
    
    logger.info(f"=== Arxiv 增量数据获取与合并工作流完成 ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())