import os
import time
from typing import Optional
from collections import deque

import numpy as np
import polars as pl 
from google import genai
from google.genai import types
from loguru import logger
from .name import *

def collect_content(df: pl.DataFrame) -> list[str]:
    CONTENT = "content"
    df = df.lazy().with_columns(
        pl.col(TITLE).fill_null(""),
        pl.col(ABSTRACT).fill_null("")
    )
    contents = (
        df.select(
            (pl.col(TITLE) + "\n" + pl.col(ABSTRACT)).alias(CONTENT)
        )
        .collect()
        .to_series()
        .to_list()
    )
    return contents

def google_batch_embedding(
    model: str,
    inputs: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    output_dimensionality: Optional[int] = None,
    api_key: Optional[str] = None,
    max_wait_time: int = 6000,
    poll_interval: int = 5,
    dtype = np.float32,
) -> np.ndarray:
    """
    使用 Google Gemini Batch Embedding API 获取文本嵌入向量（价格为正常 API 的 50%）
    
    Args:
        model: 模型名称，如 "gemini-embedding-001"
        inputs: 要嵌入的文本列表
        task_type: 任务类型，可选值:
            - "RETRIEVAL_DOCUMENT": 文档检索（默认）
            - "RETRIEVAL_QUERY": 查询检索
            - "SEMANTIC_SIMILARITY": 语义相似度
            - "CLASSIFICATION": 分类
            - "CLUSTERING": 聚类
        output_dimensionality: 输出维度，可选 128-3072，推荐 768, 1536, 3072
        api_key: Google API Key，如果不提供则从环境变量 GEMINI_API_KEY 读取
        max_wait_time: 最大等待时间（秒），默认 600 秒
        poll_interval: 轮询间隔（秒），默认 5 秒
        
    Returns:
        np.ndarray: 形状为 (len(inputs), embedding_dim) 的嵌入矩阵
        
    Raises:
        ValueError: 如果 API key 未提供或 job 失败
        TimeoutError: 如果超过最大等待时间
    """
    # 获取 API key
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
    
    # 创建客户端
    client = genai.Client(api_key=api_key)
    
    # 准备配置
    embed_config = types.EmbedContentConfig(task_type=task_type)
    if output_dimensionality is not None:
        embed_config.output_dimensionality = output_dimensionality
    
    # 创建 batch embedding job
    batch_job = client.batches.create_embeddings(
        model=model,
        src={
            'inlined_requests': types.EmbedContentBatch(
                contents=inputs,
                config=embed_config
            )
        },
        config={'display_name': f"batch-embedding-{len(inputs)}-texts"},
    )
    
    # 轮询等待 job 完成
    job_name = batch_job.name
    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    }
    
    start_time = time.time()
    while True:
        batch_job = client.batches.get(name=job_name)
        current_state = batch_job.state.name
        
        if current_state in completed_states:
            break
        
        # 检查超时
        elapsed_time = time.time() - start_time
        if elapsed_time > max_wait_time:
            raise TimeoutError(f"Batch job did not complete within {max_wait_time} seconds")
        
        time.sleep(poll_interval)
    
    # 检查 job 状态
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        error_msg = f"Batch job failed with state: {batch_job.state.name}"
        if hasattr(batch_job, 'error') and batch_job.error:
            error_msg += f"\nError: {batch_job.error}"
        raise ValueError(error_msg)
    
    # 提取嵌入向量
    if not (batch_job.dest and batch_job.dest.inlined_embed_content_responses):
        raise ValueError("No embedding responses found in batch job result")
    
    responses = batch_job.dest.inlined_embed_content_responses
    embeddings = [resp.response.embedding.values for resp in responses]
    
    return np.array(embeddings, dtype=dtype)
    
    