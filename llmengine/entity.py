from traceback import format_exc
import os
import sys
import argparse
from llmengine.ltpentity import *
from llmengine.base_entity import get_ltp_class
from typing import List

from appPublic.registerfunction import RegisterFunction
from appPublic.worker import awaitify
from appPublic.log import debug, exception
from ahserver.serverenv import ServerEnv
from ahserver.globalEnv import stream_response
from ahserver.webapp import webserver

from aiohttp_session import get_session

helptext = """LTP Entities API:

1. Entities Endpoint:
path: /v1/entities
headers: {
    "Content-Type": "application/json"
}
data: {
    "query": "苹果公司在北京开设新店"
}
response: {
    "object": "list",
    "data": [
        "苹果公司",
        "北京",
        "新店",
        "开设",
        ...
    ]
}

2. Docs Endpoint:
path: /v1/docs
response: This help text
"""

def init():
    rf = RegisterFunction()
    rf.register('entities', entities)
    rf.register('docs', docs)
    debug("注册路由: entities, docs")

async def docs(request, params_kw, *params, **kw):
    return helptext

async def entities(request, params_kw, *params, **kw):
    debug(f'{params_kw.query=}')
    se = ServerEnv()
    engine = se.engine
    query = params_kw.query
    if query is None:
        e = exception(f'query is None')
        raise e
    entities = await engine.extract_entities(query)
    debug(f'{entities=}, type(entities)')
    return {
        "object": "list",
        "data": entities
    }

def main():
    parser = argparse.ArgumentParser(prog="LTP Entity Service")
    parser.add_argument('-w', '--workdir')
    parser.add_argument('-p', '--port')
    parser.add_argument('model_path')
    args = parser.parse_args()
    Klass = get_ltp_class(args.model_path)
    if Klass is None:
        e = Exception(f'{args.model_path} has not mapping to a model class')
        exception(f'{e}, {format_exc()}')
        raise e
    se = ServerEnv()
    se.engine = Klass(args.model_path)
    workdir = args.workdir or os.getcwd()
    port = args.port
    debug(f'{args=}')
    webserver(init, workdir, port)

if __name__ == '__main__':
    main()