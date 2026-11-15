from traceback import format_exc
import os
import sys
import asyncio
import concurrent.futures
import argparse

from llmengine.service_guard import set_default_env, start_guard
set_default_env()   # 必须在下面 import torch（bge_reranker/qwen3_reranker）之前

from llmengine.qwen3_reranker import *
from llmengine.bge_reranker import *
from llmengine.base_reranker import get_llm_class

from appPublic.registerfunction import RegisterFunction
from appPublic.log import debug, exception
from ahserver.serverenv import ServerEnv
from ahserver.webapp import webserver

helptext = """rerank api:
path: /v1/rerank
headers: {
	"Content-Type": "application/json"
}
data:
{
  "model": "rerank-001",
  "query": "什么是量子计算？",
  "documents": [
    "量子计算是一种使用量子比特进行计算的方式。",
    "古典计算机使用的是二进制位。",
    "天气预报依赖于统计模型。",
    "量子计算与物理学密切相关。"
  },
  "top_n": 2
}

response is a json
{
  "data": [
    {
      "index": 0,
      "relevance_score": 0.95
    },
    {
      "index": 3,
      "relevance_score": 0.89
    }
  ],
  "object": "rerank.result",
  "model": "rerank-001",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
"""


def init():
	rf = RegisterFunction()
	rf.register('rerank', rerank)
	rf.register('docs', docs)

# 2026-11-14 生产事故同款修复（与 embedding.py 一致）：
# 每请求 awaitify() → 新 ThreadPoolExecutor → torch 拉新的 libgomp 线程池且不回收
# → 线程/内存无界增长、空闲自旋吃 CPU。改为进程级固定线程池 + 限 torch 线程数。
_EXECUTOR = None


def _get_executor():
	global _EXECUTOR
	if _EXECUTOR is None:
		try:
			n = int(os.environ.get('RERANK_MAX_WORKERS', '1') or 1)
		except ValueError:
			n = 1
		_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
			max_workers=max(1, n), thread_name_prefix='rerank')
	return _EXECUTOR


def _limit_torch_threads():
	try:
		n = int(os.environ.get('RERANK_TORCH_THREADS',
							   os.environ.get('OMP_NUM_THREADS', '4')) or 4)
	except ValueError:
		n = 4
	try:
		import torch
		torch.set_num_threads(max(1, n))
		debug(f'torch.set_num_threads({n})')
	except Exception as e:
		debug(f'torch thread limit skipped: {e}')


async def docs(request, params_kw, *params, **kw):
	return helptext

async def rerank(request, params_kw, *params, **kw):
	se = ServerEnv()
	engine = se.engine
	query = params_kw.query
	if query is None:
		e = Exception(f'query is None')
		raise e
	documents = params_kw.documents
	if documents is None:
		e = Exception(f'documents is None')
		raise e
	if isinstance(documents, str):
		documents = [documents]
	documents = [d if isinstance(d, str) else str(d) for d in documents]
	if not documents:
		e = Exception(f'documents is empty')
		raise e
	top_n = getattr(params_kw, 'top_n', None)
	if top_n is None:
		top_n = 5
	# 2026-10-30：原来把 params_kw.documents（可能是 str）传给引擎，
	# 字符串会被逐字符迭代成 pairs；改为传规范化后的 list。
	loop = asyncio.get_event_loop()
	arr = await loop.run_in_executor(_get_executor(), engine.rerank,
									 query, documents, top_n)
	debug(f'rerank response: {len(arr) if hasattr(arr, "__len__") else "?"} items')
	return arr

def main():
	parser = argparse.ArgumentParser(prog="Rerank")
	parser.add_argument('-w', '--workdir')
	parser.add_argument('-p', '--port')
	parser.add_argument('model_path')
	args = parser.parse_args()
	Klass = get_llm_class(args.model_path)
	if Klass is None:
		e = Exception(f'{args.model_path} has not mapping to a model class')
		exception(f'{e}, {format_exc()}')
		raise e
	se = ServerEnv()
	se.engine = Klass(args.model_path)
	se.engine.use_mps_if_prosible()
	_limit_torch_threads()
	start_guard('rerank', max_rss_mb=3072, max_threads=200)
	workdir = args.workdir or os.getcwd()
	port = args.port 
	debug(f'{args=}')
	webserver(init, workdir, port)

if  __name__ == '__main__':
	main()

