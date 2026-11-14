from traceback import format_exc
import os
import sys
import asyncio
import concurrent.futures
import argparse
from llmengine.qwen3embedding import *
from llmengine.bgeembedding import *
from llmengine.base_embedding import get_llm_class

from appPublic.registerfunction import RegisterFunction
from appPublic.log import debug, exception
from ahserver.serverenv import ServerEnv
from ahserver.globalEnv import stream_response
from ahserver.webapp import webserver

from aiohttp_session import get_session

helptext = """embeddings api:
path: /v1/embeddings
headers: {
	"Content-Type": "application/json"
}
data: {
	"input": "this is a test"
}
 or {
 	"input":[
		"this is first sentence",
		"this is second setence"
	]
}

response is a json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, ...]
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
"""


def init():
	rf = RegisterFunction()
	rf.register('embeddings', embeddings)
	rf.register('docs', docs)

# 2026-11-14 生产事故修复：原来每个请求都走 appPublic.worker.awaitify()，
# 而 awaitify 内部是 `with ThreadPoolExecutor()`——每个请求新建一个线程池。
# torch 在每个新工作线程里首次进 OpenMP 时会各自拉起一套 libgomp 线程，
# 这些线程不随线程池退出回收：实测 1 小时内 230 次请求 → 进程 2804 线程 /
# 12.7GB RSS，且空闲时仍自旋吃 2.5~4.5 核（系统态 CPU 70%），把同机 doit
# 应用和 MariaDB 一起拖慢（/pipeline-llm/ 响应 1.4ms→5s）。
# 修法：进程级固定线程池（默认 1 个 worker，串行推理）+ 启动时限 torch 线程数。
_EXECUTOR = None


def _get_executor():
	global _EXECUTOR
	if _EXECUTOR is None:
		try:
			n = int(os.environ.get('EMB_MAX_WORKERS', '1') or 1)
		except ValueError:
			n = 1
		_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
			max_workers=max(1, n), thread_name_prefix='emb')
	return _EXECUTOR


def _limit_torch_threads():
	try:
		n = int(os.environ.get('EMB_TORCH_THREADS',
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

async def embeddings(request, params_kw, *params, **kw):
	se = ServerEnv()
	engine = se.engine
	input = params_kw.input
	if input is None:
		e = exception(f'input is None')
		raise e
	if isinstance(input, str):
		input = [input]
	# 日志只记条数/字符数：原来 debug(input) 会把整份工具清单（~30KB）灌进日志
	try:
		n_chars = sum(len(str(x)) for x in input)
	except Exception:
		n_chars = -1
	debug(f'embeddings request: n={len(input)}, chars={n_chars}')
	loop = asyncio.get_event_loop()
	arr = await loop.run_in_executor(_get_executor(), engine.embeddings, input)
	debug(f'embeddings response: {len(arr) if hasattr(arr, "__len__") else "?"} items')
	return arr

def main():
	parser = argparse.ArgumentParser(prog="Embedding")
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
	workdir = args.workdir or os.getcwd()
	port = args.port 
	debug(f'{args=}')
	webserver(init, workdir, port)

if  __name__ == '__main__':
	main()

