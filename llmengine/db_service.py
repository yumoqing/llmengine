import argparse
from aiohttp import web
from llmengine.base_db import get_connection_class
from llmengine.milvus_db import MilvusDBConnection
from appPublic.registerfunction import RegisterFunction
from appPublic.log import debug, error, info
from ahserver.serverenv import ServerEnv
from ahserver.webapp import webserver
import os
import json

helptext = """Milvus Database Service API (Port 8886):

1. Create Collection Endpoint:
path: /v1/createcollection
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "db_type": "textdb"  // 可选，若不提供则使用默认集合 ragdb
}
response:
- Success: HTTP 200, {"status": "success", "collection_name": "ragdb" or "ragdb_textdb", "message": "集合 ragdb 或 ragdb_textdb 创建成功"}
- Error: HTTP 400, {"status": "error", "collection_name": "ragdb" or "ragdb_textdb", "message": "<error message>"}

2. Delete Collection Endpoint:
path: /v1/deletecollection
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "db_type": "textdb"  // 可选，若不提供则删除默认集合 ragdb
}
response:
- Success: HTTP 200, {"status": "success", "collection_name": "ragdb" or "ragdb_textdb", "message": "集合 ragdb 或 ragdb_textdb 删除成功"}
- Success (collection does not exist): HTTP 200, {"status": "success", "collection_name": "ragdb" or "ragdb_textdb", "message": "集合 ragdb 或 ragdb_textdb 不存在，无需删除"}
- Error: HTTP 400, {"status": "error", "collection_name": "ragdb" or "ragdb_textdb", "message": "<error message>"}

3. Insert Document Endpoint:
path: /v1/insertdocument
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "userid": "user123",
    "knowledge_base_id": "kb123",
    "document_id": "<uuid>",  // 可选，若不提供则自动生成
    "texts": ["text1", "text2", ...],
    "embeddings": [[float, ...], [float, ...], ...],  // 长度为 1024 的向量
    "filename": "file.txt",
    "file_path": "/path/to/file.txt",
    "upload_time": "<iso_timestamp>",  // 可选，若不提供则使用当前时间
    "file_type": "txt",
    "db_type": "textdb"  // 可选，若不提供则使用默认集合 ragdb
}
response:
- Success: HTTP 200, {"status": "success", "document_id": "<uuid>", "collection_name": "ragdb" or "ragdb_textdb", "message": "成功插入 <count> 个文档到 <collection_name>", "status_code": 200}
- Error: HTTP 400, {"status": "error", "document_id": "", "collection_name": "ragdb" or "ragdb_textdb", "message": "<error message>", "status_code": 400}

4. Delete Document Endpoint:
path: /v1/deletedocument
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "userid": "user123",
    "filename": "file.txt",
    "knowledge_base_id": "kb123",
    "db_type": "textdb"  // 可选，若不提供则使用默认集合 ragdb
}
response:
- Success: HTTP 200, {"status": "success", "document_id": "<uuid1,uuid2>", "collection_name": "ragdb" or "ragdb_textdb", "message": "成功删除 <count> 条 Milvus 记录，userid=<userid>, filename=<filename>, knowledge_base_id=<knowledge_base_id>", "status_code": 200}
- Success (no records): HTTP 200, {"status": "success", "document_id": "", "collection_name": "ragdb" or "ragdb_textdb", "message": "没有找到 userid=<userid>, filename=<filename>, knowledge_base_id=<knowledge_base_id> 的记录，无需删除", "status_code": 200}
- Success (collection missing): HTTP 200, {"status": "success", "document_id": "", "collection_name": "ragdb" or "ragdb_textdb", "message": "集合 <collection_name> 不存在，无需删除", "status_code": 200}
- Error: HTTP 400, {"status": "error", "document_id": "", "collection_name": "ragdb" or "ragdb_textdb", "message": "<error message>", "status_code": 400}

5. Delete Knowledge Base Endpoint:
path: /v1/deleteknowledgebase
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "userid": "user123",
    "knowledge_base_id": "kb123",
    "db_type": "textdb"  // 可选，若不提供则使用默认集合 ragdb
}
response:
- Success: HTTP 200, {"status": "success", "collection_name": "ragdb" or "ragdb_textdb", "deleted_files": ["file1.txt", "file2.pdf"], "message": "成功删除 <count> 条 Milvus 记录，删除文件: <files>, userid=<userid>, knowledge_base_id=<knowledge_base_id>", "status_code": 200}
- Success (no records): HTTP 200, {"status": "success", "collection_name": "ragdb" or "ragdb_textdb", "deleted_files": [], "message": "没有找到 userid=<userid>, knowledge_base_id=<knowledge_base_id> 的记录，无需删除", "status_code": 200}
- Success (collection missing): HTTP 200, {"status": "success", "collection_name": "ragdb" or "ragdb_textdb", "deleted_files": [], "message": "集合 <collection_name> 不存在，无需删除", "status_code": 200}
- Error: HTTP 400, {"status": "error", "collection_name": "ragdb" or "ragdb_textdb", "deleted_files": [], "message": "<error message>", "status_code": 400}

6. Search Query Endpoint:
path: /v1/searchquery
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "query_vector": [float, ...],  // 长度为 1024 的向量
    "userid": "user1",
    "knowledge_base_ids": ["kb123"],
    "limit": 5,
    "offset": 0,
    "db_type": "textdb"  // 可选，若不提供则使用默认集合 ragdb
}
response:
- Success: HTTP 200, {
    "status": "success",
    "results": [
        {
            "text": "<完整文本内容>",
            "distance": 0.95,
            "source": "vector_query",
            "metadata": {
                "userid": "user1",
                "document_id": "<uuid>",
                "filename": "file.txt",
                "file_path": "/path/to/file.txt",
                "upload_time": "<iso_timestamp>",
                "file_type": "txt"
            }
        },
        ...
    ],
    "timing": {
        "collection_load": <float>,
        "vector_search": <float>,
        "deduplication": <float>,
        "total_time": <float>
    },
    "collection_name": "ragdb" or "ragdb_textdb"
}
- Error: HTTP 400, {
    "status": "error",
    "message": "<error message>",
    "collection_name": "ragdb" or "ragdb_textdb"
}

7. List User Files Endpoint:
path: /v1/listuserfiles
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "userid": "user1",
    "db_type": "textdb"  // 可选，若不提供则使用默认集合 ragdb
}
response:
- Success: HTTP 200, {
    "status": "success",
    "files_by_knowledge_base": {
        "kb123": [
            {
                "document_id": "<uuid>",
                "filename": "file1.txt",
                "file_path": "/path/to/file1.txt",
                "upload_time": "<iso_timestamp>",
                "file_type": "txt",
                "knowledge_base_id": "kb123"
            },
            ...
        ],
        "kb456": [...]
    },
    "collection_name": "ragdb" or "ragdb_textdb"
}
- Error: HTTP 400, {
    "status": "error",
    "message": "<error message>",
    "collection_name": "ragdb" or "ragdb_textdb"
}

8. List All Knowledge Bases Endpoint:
path: /v1/listallknowledgebases
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "db_type": "textdb"  // 可选，若不提供则使用默认集合 ragdb
}
response:
- Success: HTTP 200, {
    "status": "success",
    "users_knowledge_bases": {
        "user1": {
            "kb123": [
                {
                    "document_id": "<uuid>",
                    "filename": "file1.txt",
                    "file_path": "/path/to/file1.txt",
                    "upload_time": "<iso_timestamp>",
                    "file_type": "txt",
                    "knowledge_base_id": "kb123"
                },
                ...
            ],
            "kb456": [...]
        },
        "user2": {...}
    },
    "collection_name": "ragdb" or "ragdb_textdb",
    "message": "成功列出 <count> 个用户的知识库和文件",
    "status_code": 200
}
- Error: HTTP 400, {
    "status": "error",
    "users_knowledge_bases": {},
    "collection_name": "ragdb" or "ragdb_textdb",
    "message": "<error message>",
    "status_code": 400
}

9. Connection Endpoint (for compatibility):
path: /v1/connection
method: POST
headers: {"Content-Type": "application/json"}
data: {
    "action": "<initialize|get_params|create_collection|delete_collection|insert_document|delete_document|delete_knowledge_base|search_query|list_user_files|list_all_knowledge_bases>",
    "params": {...}
}
response:
- Success: HTTP 200, {"status": "success", ...}
- Error: HTTP 400, {"status": "error", "message": "<error message>"}

10. Docs Endpoint:
path: /docs
method: GET
response: This help text
"""

def init():
    rf = RegisterFunction()
    rf.register('createcollection', create_collection)
    rf.register('deletecollection', delete_collection)
    rf.register('insertdocument', insert_document)
    rf.register('deletedocument', delete_document)
    rf.register('deleteknowledgebase', delete_knowledge_base)
    rf.register('searchquery', search_query)
    rf.register('listuserfiles', list_user_files)
    rf.register('listallknowledgebases', list_all_knowledge_bases)
    rf.register('connection', handle_connection)
    rf.register('docs', docs)

async def docs(request, params_kw, *params, **kw):
    return web.Response(text=helptext, content_type='text/plain')

async def create_collection(request, params_kw, *params, **kw):
    debug(f'{params_kw=}')
    se = ServerEnv()
    engine = se.engine
    db_type = params_kw.get('db_type', '')
    collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
    try:
        result = await engine.handle_connection("create_collection", {"db_type": db_type})
        debug(f'{result=}')
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, ensure_ascii=False))
    except Exception as e:
        error(f'创建集合失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "collection_name": collection_name,
            "message": str(e)
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def delete_collection(request, params_kw, *params, **kw):
    debug(f'{params_kw=}')
    se = ServerEnv()
    engine = se.engine
    db_type = params_kw.get('db_type', '')
    collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
    try:
        result = await engine.handle_connection("delete_collection", {"db_type": db_type})
        debug(f'{result=}')
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, ensure_ascii=False))
    except Exception as e:
        error(f'删除集合失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "collection_name": collection_name,
            "message": str(e)
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def insert_document(request, params_kw, *params, **kw):
    debug(f'Received params: {params_kw=}')
    se = ServerEnv()
    engine = se.engine
    chunks = params_kw.get('chunks', '')
    db_type = params_kw.get('db_type', '')
    collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
    try:
        required_fields = ['chunks']
        missing_fields = [field for field in required_fields if field not in params_kw or not params_kw[field]]
        if missing_fields:
            raise ValueError(f"缺少必填字段: {', '.join(missing_fields)}")

        result = await engine.handle_connection("insert_document", {
            "chunks": chunks,
            "db_type": db_type
        })
        debug(f'Insert result: {result=}')
        status = 200 if result.get("status") == "success" else 400
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=status)
    except Exception as e:
        error(f'插入文档失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "document_id": "",
            "collection_name": collection_name,
            "message": str(e),
            "status_code": 400
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def delete_document(request, params_kw, *params, **kw):
    debug(f'Received delete_document params: {params_kw=}')
    se = ServerEnv()
    engine = se.engine
    userid = params_kw.get('userid', '')
    file_path = params_kw.get('file_path', '')
    knowledge_base_id = params_kw.get('knowledge_base_id', '')
    document_id = params_kw.get('document_id', '')
    db_type = params_kw.get('db_type', '')
    collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
    try:
        required_fields = ['userid', 'file_path', 'knowledge_base_id', 'document_id']
        missing_fields = [field for field in required_fields if field not in params_kw or not params_kw[field]]
        if missing_fields:
            raise ValueError(f"缺少必填字段: {', '.join(missing_fields)}")

        result = await engine.handle_connection("delete_document", {
            "userid": userid,
            "file_path": file_path,
            "knowledge_base_id": knowledge_base_id,
            "document_id": document_id,
            "db_type": db_type
        })
        debug(f'Delete result: {result=}')
        status = 200 if result.get("status") == "success" else 400
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=status)
    except Exception as e:
        error(f'删除文档失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "collection_name": collection_name,
            "document_id": document_id,
            "message": str(e),
            "status_code": 400
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def delete_knowledge_base(request, params_kw, *params, **kw):
    debug(f'Received delete_knowledge_base params: {params_kw=}')
    se = ServerEnv()
    engine = se.engine
    userid = params_kw.get('userid', '')
    knowledge_base_id = params_kw.get('knowledge_base_id', '')
    db_type = params_kw.get('db_type', '')
    collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
    try:
        required_fields = ['userid', 'knowledge_base_id']
        missing_fields = [field for field in required_fields if field not in params_kw or not params_kw[field]]
        if missing_fields:
            raise ValueError(f"缺少必填字段: {', '.join(missing_fields)}")

        result = await engine.handle_connection("delete_knowledge_base", {
            "userid": userid,
            "knowledge_base_id": knowledge_base_id,
            "db_type": db_type
        })
        debug(f'Delete knowledge base result: {result=}')
        status = 200 if result.get("status") == "success" else 400
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=status)
    except Exception as e:
        error(f'删除知识库失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "collection_name": collection_name,
            "deleted_files": [],
            "message": str(e),
            "status_code": 400
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def search_query(request, params_kw, *params, **kw):
    debug(f'{params_kw=}')
    se = ServerEnv()
    engine = se.engine
    query_vector = params_kw.get('query_vector', [])
    userid = params_kw.get('userid', '')
    knowledge_base_ids = params_kw.get('knowledge_base_ids', [])
    limit = params_kw.get('limit', 5)
    offset = params_kw.get('offset', 0)
    db_type = params_kw.get('db_type', '')
    collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
    try:
        if not query_vector or not userid or not knowledge_base_ids:
            debug(f'query_vector, userid 或 knowledge_base_ids 未提供')
            return web.json_response({
                "status": "error",
                "message": "query_vector, userid 或 knowledge_base_ids 未提供",
                "collection_name": collection_name
            }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)
        result = await engine.handle_connection("search_query", {
            "query_vector": query_vector,
            "userid": userid,
            "knowledge_base_ids": knowledge_base_ids,
            "limit": limit,
            "offset": offset,
            "db_type": db_type
        })
        debug(f'{result=}')
        response = {
            "status": "success",
            "results": result.get("results", []),
            "timing": result.get("timing", {}),
            "collection_name": collection_name
        }
        return web.json_response(response, dumps=lambda obj: json.dumps(obj, ensure_ascii=False))
    except Exception as e:
        error(f'向量搜索失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "message": str(e),
            "collection_name": collection_name
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def list_user_files(request, params_kw, *params, **kw):
    debug(f'{params_kw=}')
    se = ServerEnv()
    engine = se.engine
    userid = params_kw.get('userid', '')
    db_type = params_kw.get('db_type', '')
    collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
    try:
        if not userid:
            debug(f'userid 未提供')
            return web.json_response({
                "status": "error",
                "message": "userid 未提供",
                "collection_name": collection_name
            }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)
        result = await engine.handle_connection("list_user_files", {
            "userid": userid,
            "db_type": db_type
        })
        debug(f'{result=}')
        response = {
            "status": "success",
            "files_by_knowledge_base": result,
            "collection_name": collection_name
        }
        return web.json_response(response, dumps=lambda obj: json.dumps(obj, ensure_ascii=False))
    except Exception as e:
        error(f'列出用户文件失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "message": str(e),
            "collection_name": collection_name
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def list_all_knowledge_bases(request, params_kw, *params, **kw):
    debug(f'{params_kw=}')
    se = ServerEnv()
    engine = se.engine
    db_type = params_kw.get('db_type', '')
    collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
    try:
        result = await engine.handle_connection("list_all_knowledge_bases", {
            "db_type": db_type
        })
        debug(f'{result=}')
        response = {
            "status": result.get("status", "success"),
            "users_knowledge_bases": result.get("users_knowledge_bases", {}),
            "collection_name": collection_name,
            "message": result.get("message", ""),
            "status_code": result.get("status_code", 200)
        }
        return web.json_response(response, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=response["status_code"])
    except Exception as e:
        error(f'列出所有用户知识库失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "users_knowledge_bases": {},
            "collection_name": collection_name,
            "message": str(e),
            "status_code": 400
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

async def handle_connection(request, params_kw, *params, **kw):
    debug(f'{params_kw=}')
    se = ServerEnv()
    engine = se.engine
    try:
        data = await request.json()
        action = data.get('action')
        if not action:
            debug(f'action 未提供')
            return web.json_response({
                "status": "error",
                "message": "action 参数未提供"
            }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)
        result = await engine.handle_connection(action, data.get('params', {}))
        debug(f'{result=}')
        return web.json_response(result, dumps=lambda obj: json.dumps(obj, ensure_ascii=False))
    except Exception as e:
        error(f'处理连接操作失败: {str(e)}')
        return web.json_response({
            "status": "error",
            "message": str(e)
        }, dumps=lambda obj: json.dumps(obj, ensure_ascii=False), status=400)

def main():
    parser = argparse.ArgumentParser(prog="Milvus Database Service")
    parser.add_argument('-w', '--workdir')
    parser.add_argument('-p', '--port', default='8886')
    parser.add_argument('connection_path')
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