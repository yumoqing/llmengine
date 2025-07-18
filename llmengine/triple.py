from traceback import format_exc
import os
import argparse
import logging
from typing import List
from base_triple import get_llm_class
from mrebeltriple import MRebelTripleExtractor
from appPublic.registerfunction import RegisterFunction
from appPublic.log import debug, exception, error, info
from appPublic.jsonConfig import getConfig
from ahserver.serverenv import ServerEnv
from ahserver.globalEnv import stream_response
from ahserver.webapp import webserver
import aiohttp.web
import time

# 配置日志

helptext = """mREBEL Triplets API:

1. Triplets Endpoint:
path: /v1/triples
headers: {
    "Content-Type": "application/json"
}
data: {
    "text": "知识图谱是一个结构化的语义知识库。"
}
response: {
    "object": "list",
    "data": [
        {
            "head": "知识图谱",
            "head_type": "Concept",
            "type": "is_a",
            "tail": "语义知识库",
            "tail_type": "Concept"
        },
        ...
    ]
}

2. Docs Endpoint:
path: /v1/docs
response: This help text
"""

# 请求计数器
request_count = 0

def init():
    rf = RegisterFunction()
    rf.register('triples', triples)
    rf.register('docs', docs)

async def docs(request, params_kw, *params, **kw):
    return helptext

async def triples(request, params_kw, *params, **kw):
    global request_count
    request_count += 1
    request_id = request_count
    debug(f"Processing request #{request_id}, params_kw: {params_kw}")
    start_time = time.time()
    try:
        # 显式解析请求数据
        if not params_kw:
            try:
                data = await request.json()
                params_kw = data
                debug(f"Request #{request_id} parsed JSON data: {params_kw}")
            except Exception as e:
                error(f"Request #{request_id} failed to parse JSON: {str(e)}")
                raise aiohttp.web.HTTPBadRequest(reason=f"Invalid JSON: {str(e)}")

        se = ServerEnv()
        engine = se.engine
        if engine is None:
            error(f"Request #{request_id} error: Engine not initialized")
            raise ValueError("Engine not initialized")

        text = params_kw.get('text')
        if not text:
            e = ValueError("text cannot be empty")
            error(f"Request #{request_id} error: {str(e)}")
            exception(f'{e}')
            raise e

        triplets = await engine.extract_triplets(text)
        debug(f"Request #{request_id} extracted {len(triplets)} triplets, took {time.time() - start_time:.2f} seconds")
        return {
            "object": "list",
            "data": triplets
        }
    except Exception as e:
        error(f"Request #{request_id} error in triples endpoint: {str(e)}")
        debug(f"Request #{request_id} traceback: {format_exc()}")
        raise
    finally:
        debug(f"Request #{request_id} completed, total time: {time.time() - start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(prog="mREBEL Triplet Service")
    parser.add_argument('-w', '--workdir', default=None)
    parser.add_argument('-p', '--port', type=int, default=9991)
    parser.add_argument('model_path')
    args = parser.parse_args()

    try:
        Klass = get_llm_class(args.model_path)
        if Klass is None:
            e = Exception(f"{args.model_path} has no mapping to a model class")
            exception(f'{e}, {format_exc()}')
            raise e

        se = ServerEnv()
        se.engine = Klass(args.model_path)
        workdir = args.workdir or os.getcwd()
        port = args.port
        webserver(init, workdir, port)
    except Exception as e:
        error(f"Failed to start server: {str(e)}")
        debug(f"Traceback: {format_exc()}")
        raise

if __name__ == "__main__":
    main()