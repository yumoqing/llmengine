---
name: llmengine-repo
version: 1.0.0
description: llmengine 仓库：135 个文件，主语言Python 41，Shell 26，JSON 15
trigger_conditions:
  - 需要理解、修改或排障 llmengine 的代码
  - 在 llmengine 里定位入口、数据表、对外契约或构建方式
---

# llmengine 仓库技能

> 由仓库实际内容生成的骨架（README/目录/模型定义/契约/构建脚本逐项取证）。新增判断与坑位请直接在下面对应小节补充。

## 定位

- 远端：`https://github.com/yumoqing/llmengine.git`
- 默认分支：`main`（共 46 个提交）
- 最近提交：2026-11-15 `e3cbbc2` feat(serving): 入口内置自保看门狗（无 root 版本的 B：限线程 env + RSS/线程超限自重启）
- 规模：135 个文件；Python 41，Shell 26，JSON 15

## 目录结构

- `test/` — 测试（86 个文件）
- `llmengine/` — Python 包源码（模块本体）（44 个文件）
- 根文件：`.gitignore`、`README.md`、`pyproject.toml`、`setup.cfg`、`t`

## 构建与运行

- `pyproject.toml`：`?` v?，依赖 0 项
- Python 包 `llmengine/`：`__init__.py`、`baichuanm2.py`、`base_chat_llm.py`、`base_connection.py`、`base_db.py`、`base_embedding.py`、`base_entity.py`、`base_m2m.py`

## 建议阅读顺序

1. `README.md`

