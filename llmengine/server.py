from traceback import format_exc
import os
import sys
import argparse

from llmengine.base_chat_llm import BaseChatLLM, get_llm_class
from llmengine.gemma3_it import Gemma3LLM
from llmengine.medgemma3_it import MedgemmaLLM
from llmengine.qwen3 import Qwen3LLM
from llmengine.qwen3coder import Qwen3CoderLLM
from llmengine.baiduanm2 import BaichuanM2LLM

from appPublic.registerfunction import RegisterFunction
from appPublic.log import debug, exception
from ahserver.serverenv import ServerEnv
from ahserver.globalEnv import stream_response
from ahserver.webapp import webserver

from aiohttp_session import get_session

def init():
	rf = RegisterFunction()
	rf.register('chat_completions', chat_completions)

async def chat_completions(request, params_kw, *params, **kw):
	se = ServerEnv()
	engine = se.engine
	async def gor():
		async for d in engine.chat_completion_stream(params_kw.messages):
			debug(f'{d=}')
			yield d

	if params_kw.stream:
		return await stream_response(request, gor)
	else:
		return await engine.chat_completion(params_kw.messages)

def main():
	parser = argparse.ArgumentParser(prog="Sage")
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
	webserver(init, workdir, port)

if  __name__ == '__main__':
	main()

