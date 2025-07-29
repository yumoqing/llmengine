import argparse
import os
from aiohttp import web
from llmengine.db_neo4j import Neo4jConnection
from llmengine.base_db import connection_register, get_connection_class
from appPublic.registerfunction import RegisterFunction
from appPublic.log import debug, error, info
from ahserver.serverenv import ServerEnv
from ahserver.webapp import webserver
import json

helptext = """Neo4j Connection Service API:

1. Initialize Endpoint:
path: /v1/initialize
method: POST
headers: {"Content-Type": "application/json"}
data: {}
response:
- Success: HTTP 200, {
    "status": "success",
    "message": "Neo4j 服务已初始化",
    "collection_name": "neo4j",
    "document_id": "",
    "status_code": 200
}
- Error: HTTP 400, {
    "status": "error",
    "message": "<error message>",
    "collection_name": "neo4j",
    "document_id": "",
    "status_code": 400
}

2. Insert Triples Endpoint:
path: /v1/inserttriples
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "triples": [
        {"head": "entity1", "head_type": "Person", "type": "related_to", "tail": "entity2", "tail_type": "Organization"},
        ...
    ],
    "document_id": "uuid",
    "knowledge_base_id": "kb123",
    "userid": "user123"
}
response:
- Success: HTTP 200, {
    "status": "success",
    "message": "成功插入 <nodes> 个节点和 <rels> 个关系",
    "nodes_created": <int>,
    "rels_created": <int>,
    "collection_name": "neo4j",
    "document_id": "<uuid>",
    "status_code": 200
}
- Error: HTTP 400, {
    "status": "error",
    "message": "<error message>",
    "collection_name": "neo4j",
    "document_id": "<uuid>",
    "status_code": 400
}

3. Delete Document Endpoint:
path: /v1/deletedocument
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "document_id": "uuid"
}
response:
- Success: HTTP 200, {
    "status": "success",
    "message": "成功删除 <nodes> 个节点和 <rels> 个关系",
    "nodes_deleted": <int>,
    "rels_deleted": <int>,
    "collection_name": "neo4j",
    "document_id": "<uuid>",
    "status_code": 200
}
- Error: HTTP 400, {
    "status": "error",
    "message": "<error message>",
    "collection_name": "neo4j",
    "document_id": "<uuid>",
    "status_code": 400
}

4. Delete Knowledge Base Endpoint:
path: /v1/deleteknowledgebase
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "userid": "user123",
    "knowledge_base_id": "kb123"
}
response:
- Success: HTTP 200, {
    "status": "success",
    "message": "成功删除 <nodes> 个节点和 <rels> 个关系",
    "nodes_deleted": <int>,
    "rels_deleted": <int>,
    "collection_name": "neo4j",
    "document_id": "",
    "status_code": 200
}
- Error: HTTP 400, {
    "status": "error",
    "message": "<error message>",
    "collection_name": "neo4j",
    "document_id": "",
    "status_code": 400
}

5. Match Triplets Endpoint:
path: /v1/matchtriplets
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "query": "query text",
    "query_entities": ["entity1", "entity2"],
    "userid": "user123",
    "knowledge_base_id": "kb123"
}
response:
- Success: HTTP 200, {
    "status": "success",
    "message": "找到 <count> 个匹配的三元组",
    "triplets": [
        {"head": "entity1", "type": "related_to", "tail": "entity2", "head_type": "", "tail_type": ""},
        ...
    ],
    "timing": {
        "neo4j_connect": <float>,
        "entity_match": <float>,
        "triplet_query": <float>,
        "embedding": <float>,
        "similarity": <float>,
        "total_time": <float>
    },
    "collection_name": "neo4j",
    "document_id": "",
    "status_code": 200
}
- Error: HTTP 400, {
    "status": "error",
    "message": "<error message>",
    "triplets": [],
    "timing": {},
    "collection_name": "neo4j",
    "document_id": "",
    "status_code": 400
}

6. Connection Endpoint:
path: /v1/connection
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "action": "initialize|insert_triples|delete_document|delete_knowledge_base|match_triplets",
    "params": {...}
}
response:
- Success: HTTP 200, {"status": "success", ...}
- Error: HTTP 400, {"status": "error", "message": "<error message>", "collection_name": "neo4j", "document_id": "", "status_code": 400}

7. Docs Endpoint:
path: /docs
method: GET
response: This help text
"""

def init():
    rf = RegisterFunction()
    rf.register('initialize', initialize)
    rf.register('inserttriples', insert_triples)
    rf.register('deletedocument', delete_document)
    rf.register('deleteknowledgebase', delete_knowledge_base)
    rf.register('matchtriplets', match_triplets)
    rf.register('connection', handle_connection)
    rf.register('docs', docs)

async def docs(request, params_kw, *params, **kw):
    return web.Response(text=helptext, content_type='text/plain')

async def initialize(request, params_kw, *params, **kw):
    debug(f'Received initialize params: {params_kw=}')
    se = ServerEnv()
    engine = se.engine
    debug(f'Engine: {engine}')
    if engine is None or not isinstance(engine, Neo4jConnection):
        error("Neo4jConnection not initialized")
        return web.json_response({
            "status": "error",
            "message": "Neo4j 服务未启动",
            "collection_name": "neo4j",
            "document_id": "",
            "status_code": 400
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)
    try:
        result = await engine.handle_connection("initialize", params_kw)
        debug(f'Initialize result: {result=}')
        status = 200 if result.get("status") == "success" else 400
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=status)
    except Exception as e:
        error(f'初始化失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "message": str(e),
            "collection_name": "neo4j",
            "document_id": "",
            "status_code": 400
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def insert_triples(request, params_kw, *params, **kw):
    debug(f'Received insert_triples params: {params_kw=}')
    se = ServerEnv()
    engine = se.engine
    try:
        required_fields = ['triples', 'document_id', 'knowledge_base_id', 'userid']
        missing_fields = [field for field in required_fields if field not in params_kw or not params_kw[field]]
        if missing_fields:
            raise ValueError(f"缺少必填字段: {', '.join(missing_fields)}")
        result = await engine.handle_connection("insert_triples", params_kw)
        debug(f'Insert triples result: {result=}')
        status = 200 if result.get("status") == "success" else 400
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=status)
    except Exception as e:
        error(f'插入三元组失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "message": str(e),
            "collection_name": "neo4j",
            "document_id": params_kw.get("document_id", ""),
            "status_code": 400
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def delete_document(request, params_kw, *params, **kw):
    debug(f'Received delete_document params: {params_kw=}')
    se = ServerEnv()
    engine = se.engine
    try:
        if not params_kw.get('document_id'):
            raise ValueError("document_id 不能为空")
        result = await engine.handle_connection("delete_document", params_kw)
        debug(f'Delete document result: {result=}')
        status = 200 if result.get("status") == "success" else 400
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=status)
    except Exception as e:
        error(f'删除文档失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "message": str(e),
            "collection_name": "neo4j",
            "document_id": params_kw.get("document_id", ""),
            "status_code": 400
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def delete_knowledge_base(request, params_kw, *params, **kw):
    debug(f'Received delete_knowledge_base params: {params_kw=}')
    se = ServerEnv()
    engine = se.engine
    try:
        required_fields = ['userid', 'knowledge_base_id']
        missing_fields = [field for field in required_fields if field not in params_kw or not params_kw[field]]
        if missing_fields:
            raise ValueError(f"缺少必填字段: {', '.join(missing_fields)}")
        result = await engine.handle_connection("delete_knowledge_base", params_kw)
        debug(f'Delete knowledge base result: {result=}')
        status = 200 if result.get("status") == "success" else 400
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=status)
    except Exception as e:
        error(f'删除知识库失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "message": str(e),
            "collection_name": "neo4j",
            "document_id": "",
            "status_code": 400
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def match_triplets(request, params_kw, *params, **kw):
    debug(f'Received match_triplets params: {params_kw=}')
    se = ServerEnv()
    engine = se.engine
    try:
        required_fields = ['query', 'query_entities', 'userid', 'knowledge_base_id']
        missing_fields = [field for field in required_fields if field not in params_kw or not params_kw[field]]
        if missing_fields:
            raise ValueError(f"缺少必填字段: {', '.join(missing_fields)}")
        result = await engine.handle_connection("match_triplets", params_kw)
        debug(f'Match triplets result: {result=}')
        status = 200 if result.get("status") == "success" else 400
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=status)
    except Exception as e:
        error(f'匹配三元组失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "message": str(e),
            "triplets": [],
            "timing": {},
            "collection_name": "neo4j",
            "document_id": "",
            "status_code": 400
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def handle_connection(request, params_kw, *params, **kw):
    debug(f'Received connection params: {params_kw=}')
    se = ServerEnv()
    engine = se.engine
    try:
        data = await request.json()
        action = data.get('action')
        if not action:
            debug(f'action 未提供')
            return web.json_response({
                "status": "error",
                "message": "action 参数未提供",
                "collection_name": "neo4j",
                "document_id": "",
                "status_code": 400
            }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)
        result = await engine.handle_connection(action, data.get('params', {}))
        debug(f'Connection result: {result=}')
        status = 200 if result.get("status") == "success" else 400
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=status)
    except Exception as e:
        error(f'处理连接操作失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "message": str(e),
            "collection_name": "neo4j",
            "document_id": "",
            "status_code": 400
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

def main():
    parser = argparse.ArgumentParser(prog="Neo4j Connection Service")
    parser.add_argument('-w', '--workdir', help="Working directory")
    parser.add_argument('-p', '--port', default='8885', help="Port to run the server on")
    parser.add_argument('connection_path', help="Connection class path (e.g., Neo4j)")
    args = parser.parse_args()
    debug(f"Arguments: {args}")
    Klass = get_connection_class(args.connection_path)
    se = ServerEnv()
    se.engine = Klass()
    workdir = args.workdir or os.getcwd()
    port = args.port
    debug(f'{args=}')
    webserver(init, workdir, port)

if __name__ == '__main__':
    main()