from traceback import format_exc
import os
import sys
import argparse
from llmengine.qwen3_reranker import *
from llmengine.bge_reranker import *
from llmengine.base_reranker import get_llm_class

from appPublic.registerfunction import RegisterFunction
from appPublic.worker import awaitify
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

async def docs(request, params_kw, *params, **kw):
	return helptext

async def rerank(request, params_kw, *params, **kw):
	debug(f'{params_kw.query=}, {params_kw.documents=}, {params_kw.top_n=}')
	se = ServerEnv()
	engine = se.engine
	f = awaitify(engine.rerank)
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
	arr = await f(query, documents, top_n)
	debug(f'{arr=}, type(arr)')
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
	workdir = args.workdir or os.getcwd()
	port = args.port 
	debug(f'{args=}')
	webserver(init, workdir, port)

if  __name__ == '__main__':
	main()

