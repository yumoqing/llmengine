from traceback import format_exc
import os
import sys
import argparse
from llmengine.qwen3embedding import *
from llmengine.bgeembedding import *
from llmengine.base_embedding import get_llm_class

from appPublic.registerfunction import RegisterFunction
from appPublic.worker import awaitify
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

async def docs(request, params_kw, *params, **kw):
	return helptext

async def embeddings(request, params_kw, *params, **kw):
	debug(f'{params_kw.input=}')
	se = ServerEnv()
	engine = se.engine
	f = awaitify(engine.embeddings)
	input = params_kw.input
	if input is None:
		e = exception(f'input is None')
		raise e
	if isinstance(input, str):
		input = [input]
	arr = await f(input)
	debug(f'{arr=}, type(arr)')
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
	workdir = args.workdir or os.getcwd()
	port = args.port 
	debug(f'{args=}')
	webserver(init, workdir, port)

if  __name__ == '__main__':
	main()

