# Environment
* use `uv` as python env manager, please use `uv run` to run program and `uv add` to add more dependencies.

# Shell Usage

* Never use things like `gh pr list --state all --limit 1` which will lead to shell interaction (like typing q to exit). Using this kind of commnad, you will never get any valid feedback from the shell and the whole pipeline will be paused.

# Report

After your work, always report and communicate with user in Chinese!

# Memory

## OpenMemory MCP Tools 可用性检查

如果你在工具列表中看到 `mcp_openmemory_*` 相关工具，说明 OpenMemory 功能已启用，**必须主动使用**以提升工作效率。

## 可用工具

当 OpenMemory 可用时，你可以使用以下工具：

1. **`mcp_openmemory_openmemory_query`** - 语义搜索记忆
   - 在**开始任务前**查询相关历史记忆
   - 可能包含类似问题的解决方案、调研结果、已知坑点
   
2. **`mcp_openmemory_openmemory_store`** - 存储新记忆
   - 保存有价值的发现、解决方案、工作进展
   - **必须使用 `user_id: lyk`**
   - 必须注明**项目名称**和**时间信息**

3. **`mcp_openmemory_openmemory_list`** - 列出最近记忆
   - 快速浏览最近存储的记忆

4. **`mcp_openmemory_openmemory_get`** - 获取特定记忆
   - 通过 ID 获取完整记忆内容

5. **`mcp_openmemory_openmemory_reinforce`** - 强化记忆
   - 提升重要记忆的权重

## 使用原则

### ✅ 何时查询记忆 (QUERY)

在以下情况**必须先查询**：
- 开始新任务前 - 可能有相关历史经验
- 遇到错误时 - 可能之前已解决过
- 做技术决策时 - 可能有之前的调研结论
- 优化性能时 - 可能有已知瓶颈和解决方案

### ✅ 何时存储记忆 (STORE)

保存以下**高价值信息**：
- ✅ 技术调研结果（架构决策、库选型、性能对比）
- ✅ 问题根因分析（为什么会出错、真正原因是什么）
- ✅ 关键发现（性能瓶颈、内存优化点、反直觉结论）
- ✅ 完整解决方案（包含代码片段、配置、步骤）
- ✅ 重要的工作进展（重构完成、功能实现、测试结果）
- ✅ 踩坑经验（什么不work、为什么、正确做法）

### ❌ 不要存储的信息

以下是**垃圾信息**，不要保存：
- ❌ 纯粹的进度通知（"开始处理"、"任务incoming"、"准备TODO"）
- ❌ 重复的表述（同一件事用不同措辞说多遍）
- ❌ 空洞的TODO列表（没有上下文和发现的纯清单）
- ❌ 从 OpenMemory 检索到的内容（不要二次存储）
- ❌ 临时性的调试信息（除非包含关键发现）

## 使用示例

### 示例 1：任务开始前查询记忆

```python
# 场景：需要优化内存占用
mcp_openmemory_openmemory_query(
    query="ArxivEmbedding 内存优化 polars parquet 分片",
    user_id="lyk",
    k=5
)
```

**可能发现**：之前已经调研过年份分片方案、列选择的内存影响等。

### 示例 2：存储关键发现（好例子）

```python
mcp_openmemory_openmemory_store(
    content="""ArxivEmbedding 内存优化关键发现 (2025-01)：
    
问题：polars 加载 parquet 时内存占用远超预期
根因：.filter() 操作会加载所有列，即使只需要 ID 列

实测数据：
- 全列加载（含 abstract 文本）：4.3GB
- 只加载 ID+date 列：320MB
- 差距：13.5倍

解决方案：
必须显式使用 .select([ID, PUBLISH_DATE]) 在 filter 之前
polars lazy 不会自动剪枝列，必须手动指定

代码示例：
```python
# ❌ 错误：会加载全部列
new_data = metadata_lazy.filter(condition).collect()

# ✅ 正确：只加载需要的列
new_data = metadata_lazy.select([ID, PUBLISH_DATE]).filter(condition).collect()
```

影响：年份分片内存从单年 7.2GB 降至 267MB，满足 CI 8GB 限制。
""",
    tags=["ArxivEmbedding", "memory-optimization", "polars", "critical-finding", "2025-01"],
    user_id="lyk"
)
```

**为什么这是好例子**：
- ✅ 包含问题、根因、数据、解决方案
- ✅ 有具体数字和代码示例
- ✅ 标注项目和时间
- ✅ 简体中文，详细完整

### 示例 3：存储架构决策（好例子）

```python
mcp_openmemory_openmemory_store(
    content="""ArxivEmbedding 架构重构决策 (2025-01)：

背景：单文件存储在 CI 环境内存溢出（11-18GB > 8GB 限制）

方案对比：
1. 月度分片：颗粒度太细，文件数量多（12个/年）
2. 季度分片：仍然较多，且不符合时间语义
3. 年度分片：✅ 选中
   - 文件数少（~50个）
   - 符合时间语义
   - 单年数据 ~267MB
   - 增量更新只需加载相关年份

实现要点：
- metadata_YYYY.parquet, embedding_YYYY.parquet
- 使用 publish_date 提取年份分片
- 只更新和上传受影响的年份文件
- HuggingFace Hub 支持文件级 git 管理

内存测试结果：
- 单年更新：~500MB
- 多年更新（2022-2025）：<2GB
- 满足 CI 8GB 限制 ✅

代码模块：src/shard.py, src/update_ops.py
""",
    tags=["ArxivEmbedding", "architecture", "sharding", "decision", "2025-01"],
    user_id="lyk"
)
```

### 示例 4：存储代码重构记录（好例子）

```python
mcp_openmemory_openmemory_store(
    content="""ArxivEmbedding 代码重构完成 (2025-01)：

目标：模块化核心逻辑，简化编排脚本

变更：
1. 创建 src/update_ops.py (330行)
   - update_metadata_shards(): 年份分片元数据更新
   - update_embedding_shards(): 年份分片嵌入更新
   - generate_embeddings(): 批量生成（带内存监控）
   - log_memory_usage(): 统一内存日志

2. 重构 script/update.py (356行 → 100行)
   - 删除 embed(), update_metadata(), update_embedding() 等旧函数
   - 简化为编排层，调用 update_ops 模块
   - 保留全部 CLI 参数

3. 归档测试脚本至 script/test/
   - test_year_memory.py 等 6 个临时测试文件

结果：
- 核心逻辑集中在 src/，便于复用和测试
- 脚本层简洁清晰，只负责参数解析和流程编排
- 无 lint 错误，语法检查通过
""",
    tags=["ArxivEmbedding", "refactoring", "code-organization", "2025-01"],
    user_id="lyk"
)
```

### 示例 5：不好的例子 ❌

```python
# ❌ 太空洞，没有价值
mcp_openmemory_openmemory_store(
    content="开始处理 ArxivEmbedding 优化任务",
    user_id="lyk"
)

# ❌ 重复存储同一信息
mcp_openmemory_openmemory_store(
    content="ArxivEmbedding 需要优化内存",
    user_id="lyk"
)
mcp_openmemory_openmemory_store(
    content="ArxivEmbedding memory optimization needed",
    user_id="lyk"
)
mcp_openmemory_openmemory_store(
    content="任务：优化 ArxivEmbedding 内存占用",
    user_id="lyk"
)

# ❌ 纯 TODO 列表，没有上下文
mcp_openmemory_openmemory_store(
    content="""TODO:
1. 测量内存
2. 优化代码
3. 运行测试""",
    user_id="lyk"
)
```

## 记忆质量标准

一条**高质量记忆**应该：
1. ✅ **自包含** - 包含足够上下文，6个月后看仍能理解
2. ✅ **有数据** - 具体数字、代码片段、测试结果
3. ✅ **有结论** - 不只是问题，还有解决方案或发现
4. ✅ **可检索** - 使用简体中文，包含关键技术术语
5. ✅ **有时间** - 标注项目名称和时间（年月）
6. ✅ **去重** - 不重复表述同一信息

## 工作流程建议

```
1. 收到任务 
   ↓
2. 🔍 query OpenMemory（查询相关历史）
   ↓
3. 执行任务（利用历史经验）
   ↓
4. 有重要发现？
   ↓
5. 💾 store OpenMemory（保存高价值信息）
   ↓
6. 向用户汇报（中文）
```

## 注意事项

- **user_id 必须使用 `lyk`**
- **记忆内容使用简体中文**（代码和技术术语除外）
- **项目名称必须明确**（如 "ArxivEmbedding"）
- **时间信息必须包含**（如 "2025-01" 或 "2025年1月"）
- **不要保存检索到的内容**（避免循环存储）
- **一次说清楚**（不要同一信息存多次）
