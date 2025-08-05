from appPublic.jsonConfig import getConfig
import os
from appPublic.log import debug, error, info
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from threading import Lock
from typing import Dict, List, Any
import uuid
from datetime import datetime
from llmengine.base_db import connection_register, BaseDBConnection
import time

class MilvusDBConnection(BaseDBConnection):
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MilvusDBConnection, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        try:
            config = getConfig()
            self.db_path = config['milvus_db']
        except KeyError as e:
            error(f"配置文件缺少必要字段: {str(e)}")
            raise RuntimeError(f"配置文件缺少必要字段: {str(e)}")
        self._initialize_connection()
        self._initialized = True
        info(f"MilvusDBConnection initialized with db_path: {self.db_path}")

    def _initialize_connection(self):
        """初始化 Milvus 连接，确保单一连接"""
        try:
            db_dir = os.path.dirname(self.db_path)
            if not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                debug(f"创建 Milvus 目录: {db_dir}")
            if not os.access(db_dir, os.W_OK):
                raise RuntimeError(f"Milvus 目录 {db_dir} 不可写")
            if not connections.has_connection("default"):
                connections.connect("default", uri=self.db_path)
                debug(f"已连接到 Milvus Lite，路径: {self.db_path}")
            else:
                debug("已存在 Milvus 连接，跳过重复连接")
        except Exception as e:
            error(f"连接 Milvus 失败: {str(e)}")
            raise RuntimeError(f"连接 Milvus 失败: {str(e)}")

    async def handle_connection(self, action: str, params: Dict = None) -> Dict:
        """处理数据库操作"""
        try:
            debug(f"处理操作: action={action}, params={params}")
            if not params:
                params = {}
            db_type = params.get("db_type", "")
            collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
            if db_type and "_" in db_type:
                return {"status": "error", "message": "db_type 不能包含下划线", "collection_name": collection_name,
                        "document_id": "", "status_code": 400}
            if db_type and len(db_type) > 100:
                return {"status": "error", "message": "db_type 的长度应小于 100", "collection_name": collection_name,
                        "document_id": "", "status_code": 400}

            if action == "initialize":
                return {"status": "success", "message": f"Milvus 连接已初始化，路径: {self.db_path}"}
            elif action == "get_params":
                return {"status": "success", "params": {"uri": self.db_path}}
            elif action == "create_collection":
                return await self._create_collection(db_type)
            elif action == "delete_collection":
                return await self._delete_collection(db_type)
            elif action == "insert_document":
                chunks = params.get("chunks", [])
                return await self._insert_document(chunks, db_type)
            elif action == "delete_document":
                userid = params.get("userid", "")
                file_path = params.get("file_path", "")
                knowledge_base_id = params.get("knowledge_base_id", "")
                document_id = params.get("document_id", "")
                db_type = params.get("db_type", "")
                if not userid or not file_path or not knowledge_base_id or not document_id:
                    return {"status": "error", "message": "userid、file_path document_id和 knowledge_base_id 不能为空1",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                if "_" in userid or "_" in knowledge_base_id:
                    return {"status": "error", "message": "userid 和 knowledge_base_id 不能包含下划线",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                if len(userid) > 100 or len(file_path) > 255 or len(knowledge_base_id) > 100:
                    return {"status": "error", "message": "userid、filename 或 knowledge_base_id 的长度超出限制",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                return await self._delete_document(userid, file_path, knowledge_base_id, document_id, db_type)
            elif action == "delete_knowledge_base":
                userid = params.get("userid", "")
                knowledge_base_id = params.get("knowledge_base_id", "")
                if not userid or not knowledge_base_id:
                    return {"status": "error", "message": "userid 和 knowledge_base_id 不能为空",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                if "_" in userid or "_" in knowledge_base_id:
                    return {"status": "error", "message": "userid 和 knowledge_base_id 不能包含下划线",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                if len(userid) > 100 or len(knowledge_base_id) > 100:
                    return {"status": "error", "message": "userid 或 knowledge_base_id 的长度超出限制",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                return await self._delete_knowledge_base(db_type, userid, knowledge_base_id)
            elif action == "search_query":
                query_vector = params.get("query_vector", [])
                userid = params.get("userid", "")
                knowledge_base_ids = params.get("knowledge_base_ids", [])
                limit = params.get("limit", 5)
                offset = params.get("offset", 0)
                db_type = params.get("db_type", "")
                if not query_vector or not userid or not knowledge_base_ids:
                    return {"status": "error", "message": "query_vector、userid 或 knowledge_base_ids 不能为空",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                if limit < 1 or limit > 16384:
                    return {"status": "error", "message": "limit 必须在 1 到 16384 之间",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                return await self._search_query(query_vector, userid, knowledge_base_ids, limit, offset, db_type)
            elif action == "list_user_files":
                userid = params.get("userid", "")
                if not userid:
                    return {"status": "error", "message": "userid 不能为空", "collection_name": collection_name,
                            "document_id": "", "status_code": 400}
                return await self._list_user_files(userid, db_type)
            elif action == "list_all_knowledge_bases":
                return await self._list_all_knowledge_bases(db_type)
            else:
                return {"status": "error", "message": f"未知的 action: {action}", "collection_name": collection_name,
                        "document_id": "", "status_code": 400}
        except Exception as e:
            error(f"处理操作失败: action={action}, 错误: {str(e)}")
            return {
                "status": "error",
                "message": f"服务器错误: {str(e)}",
                "collection_name": collection_name,
                "document_id": "",
                "status_code": 400
            }

    async def _create_collection(self, db_type: str = "") -> Dict:
        """创建 Milvus 集合"""
        try:
            collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
            if len(collection_name) > 255:
                raise ValueError(f"集合名称 {collection_name} 超过 255 个字符")
            if db_type and "_" in db_type:
                raise ValueError("db_type 不能包含下划线")
            if db_type and len(db_type) > 100:
                raise ValueError("db_type 的长度应小于 100")
            debug(f"集合名称: {collection_name}")

            fields = [
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=36, auto_id=True),
                FieldSchema(name="userid", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="knowledge_base_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=36),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="upload_time", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=64),
            ]
            schema = CollectionSchema(
                fields=fields,
                description="统一数据集合，包含用户ID、知识库ID、document_id 和元数据字段",
                auto_id=True,
                primary_field="pk",
            )

            if utility.has_collection(collection_name):
                try:
                    collection = Collection(collection_name)
                    existing_schema = collection.schema
                    expected_fields = {f.name for f in fields}
                    actual_fields = {f.name for f in existing_schema.fields}
                    vector_field = next((f for f in existing_schema.fields if f.name == "vector"), None)

                    schema_compatible = False
                    if expected_fields == actual_fields and vector_field is not None and vector_field.dtype == DataType.FLOAT_VECTOR:
                        dim = vector_field.params.get('dim', None) if hasattr(vector_field, 'params') and vector_field.params else None
                        schema_compatible = dim == 1024
                        debug(f"检查集合 {collection_name} 的 schema: 字段匹配={expected_fields == actual_fields}, "
                              f"vector_field存在={vector_field is not None}, dtype={vector_field.dtype if vector_field else '无'}, "
                              f"dim={dim if dim is not None else '未定义'}")
                    if not schema_compatible:
                        debug(f"集合 {collection_name} 的 schema 不兼容，原因: "
                              f"字段不匹配: {expected_fields.symmetric_difference(actual_fields) or '无'}, "
                              f"vector_field: {vector_field is not None}, "
                              f"dtype: {vector_field.dtype if vector_field else '无'}, "
                              f"dim: {vector_field.params.get('dim', '未定义') if vector_field and hasattr(vector_field, 'params') and vector_field.params else '未定义'}")
                        utility.drop_collection(collection_name)
                    else:
                        collection.load()
                        debug(f"集合 {collection_name} 已存在并加载成功")
                        return {
                            "status": "success",
                            "collection_name": collection_name,
                            "message": f"集合 {collection_name} 已存在"
                        }
                except Exception as e:
                    error(f"加载集合 {collection_name} 失败: {str(e)}")
                    return {
                        "status": "error",
                        "collection_name": collection_name,
                        "message": str(e)
                    }

            try:
                collection = Collection(collection_name, schema)
                collection.create_index(
                    field_name="vector",
                    index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"}
                )
                for field in ["userid", "knowledge_base_id", "document_id", "filename", "file_path", "upload_time", "file_type"]:
                    collection.create_index(
                        field_name=field,
                        index_params={"index_type": "INVERTED"}
                    )
                collection.load()
                debug(f"成功创建并加载集合: {collection_name}")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "message": f"集合 {collection_name} 创建成功"
                }
            except Exception as e:
                error(f"创建集合 {collection_name} 失败: {str(e)}")
                return {
                    "status": "error",
                    "collection_name": collection_name,
                    "message": str(e)
                }
        except Exception as e:
            error(f"创建集合失败: {str(e)}")
            return {
                "status": "error",
                "collection_name": collection_name,
                "message": str(e)
            }

    async def _delete_collection(self, db_type: str = "") -> Dict:
        """删除 Milvus 集合"""
        try:
            collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
            if len(collection_name) > 255:
                raise ValueError(f"集合名称 {collection_name} 超过 255 个字符")
            if db_type and "_" in db_type:
                raise ValueError("db_type 不能包含下划线")
            if db_type and len(db_type) > 100:
                raise ValueError("db_type 的长度应小于 100")
            debug(f"集合名称: {collection_name}")

            if not utility.has_collection(collection_name):
                debug(f"集合 {collection_name} 不存在")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "message": f"集合 {collection_name} 不存在，无需删除"
                }

            try:
                utility.drop_collection(collection_name)
                debug(f"成功删除集合: {collection_name}")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "message": f"集合 {collection_name} 删除成功"
                }
            except Exception as e:
                error(f"删除集合 {collection_name} 失败: {str(e)}")
                return {
                    "status": "error",
                    "collection_name": collection_name,
                    "message": str(e)
                }
        except Exception as e:
            error(f"删除集合失败: {str(e)}")
            return {
                "status": "error",
                "collection_name": collection_name,
                "message": str(e)
            }

    async def _insert_document(self, chunks: List[Dict], db_type: str = "") -> Dict[str, Any]:
        """插入文档到 Milvus"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        document_id = chunks[0]["document_id"] if chunks else ""
        try:
            # 检查集合是否存在
            create_result = await self._create_collection(db_type)
            if create_result["status"] == "error":
                raise RuntimeError(f"集合创建失败: {create_result['message']}")

            # 检查输入数据
            if not chunks:
                raise ValueError("chunks 不能为空")
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    raise ValueError("每个 chunk 必须是一个字典")
                required_fields = ["text", "vector", "document_id", "filename", "file_path", "upload_time", "file_type",
                                   "userid", "knowledge_base_id"]
                if not all(k in chunk for k in required_fields):
                    raise ValueError(f"chunk 缺少必要字段: {', '.join(set(required_fields) - set(chunk.keys()))}")
                if not isinstance(chunk["vector"], list) or len(chunk["vector"]) != 1024:
                    raise ValueError("vector 必须是长度为 1024 的浮点数列表")

            # 验证 userid 和 knowledge_base_id 一致性
            if len(set(chunk["userid"] for chunk in chunks)) > 1:
                raise ValueError("所有 chunk 的 userid 必须一致")
            if len(set(chunk["knowledge_base_id"] for chunk in chunks)) > 1:
                raise ValueError("所有 chunk 的 knowledge_base_id 必须一致")
            if len(set(chunk["filename"] for chunk in chunks)) > 1:
                raise ValueError("所有 chunk 的 filename 必须一致")

            # 检查是否已存在相同的 userid、knowledge_base_id 和 filename
            collection = Collection(collection_name)
            collection.load()
            expr = f"userid == '{chunks[0]['userid']}' and knowledge_base_id == '{chunks[0]['knowledge_base_id']}' and filename == '{chunks[0]['filename']}'"
            debug(f"检查重复文档: {expr}")
            results = collection.query(expr=expr, output_fields=["document_id"], limit=1)
            if results:
                debug(
                    f"找到重复文档: userid={chunks[0]['userid']}, knowledge_base_id={chunks[0]['knowledge_base_id']}, filename={chunks[0]['filename']}")
                return {
                    "status": "error",
                    "document_id": document_id,
                    "collection_name": collection_name,
                    "message": f"文档已存在: userid={chunks[0]['userid']}, knowledge_base_id={chunks[0]['knowledge_base_id']}, filename={chunks[0]['filename']}",
                    "status_code": 400
                }

            # 提取数据
            userids = [chunk["userid"] for chunk in chunks]
            knowledge_base_ids = [chunk["knowledge_base_id"] for chunk in chunks]
            texts = [chunk["text"] for chunk in chunks]
            embeddings = [chunk["vector"] for chunk in chunks]
            document_ids = [chunk["document_id"] for chunk in chunks]
            filenames = [chunk["filename"] for chunk in chunks]
            file_paths = [chunk["file_path"] for chunk in chunks]
            upload_times = [chunk["upload_time"] for chunk in chunks]
            file_types = [chunk["file_type"] for chunk in chunks]

            # 构造插入数据
            data = {
                "userid": userids,
                "knowledge_base_id": knowledge_base_ids,
                "document_id": document_ids,
                "text": texts,
                "vector": embeddings,
                "filename": filenames,
                "file_path": file_paths,
                "upload_time": upload_times,
                "file_type": file_types,
            }

            schema_fields = [field.name for field in collection.schema.fields if field.name != "pk"]
            debug(f"Schema fields: {schema_fields}")
            debug(f"Data keys: {list(data.keys())}")
            if list(data.keys()) != schema_fields:
                raise ValueError(f"数据字段顺序不匹配，期望: {schema_fields}, 实际: {list(data.keys())}")

            collection.insert([data[field.name] for field in collection.schema.fields if field.name != "pk"])
            collection.flush()
            debug(f"成功插入 {len(texts)} 个文档到集合 {collection_name}")
            return {
                "status": "success",
                "document_id": document_id,
                "collection_name": collection_name,
                "message": f"成功插入 {len(texts)} 个文档到 {collection_name}",
                "status_code": 200
            }
        except MilvusException as e:
            error(f"Milvus 插入失败: {str(e)}, 堆栈: {traceback.format_exc()}")
            return {
                "status": "error",
                "document_id": document_id,
                "collection_name": collection_name,
                "message": f"Milvus 插入失败: {str(e)}",
                "status_code": 400
            }
        except Exception as e:
            error(f"插入文档失败: {str(e)}, 堆栈: {traceback.format_exc()}")
            return {
                "status": "error",
                "document_id": document_id,
                "collection_name": collection_name,
                "message": f"插入文档失败: {str(e)}",
                "status_code": 400
            }

    async def _delete_document(self, userid: str, file_path: str, knowledge_base_id: str, document_id:str, db_type: str = "") -> Dict[
        str, Any]:
        """删除用户指定文件数据，仅处理 Milvus 记录"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        try:
            if not utility.has_collection(collection_name):
                debug(f"集合 {collection_name} 不存在")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "document_id": document_id,
                    "message": f"集合 {collection_name} 不存在，无需删除",
                    "status_code": 200
                }

            try:
                collection = Collection(collection_name)
                collection.load()
                debug(f"加载集合: {collection_name}")
            except Exception as e:
                error(f"加载集合 {collection_name} 失败: {str(e)}")
                return {
                    "status": "error",
                    "collection_name": collection_name,
                    "document_id": document_id,
                    "message": f"加载集合失败: {str(e)}",
                    "status_code": 400
                }

            total_deleted = 0
            try:
                delete_expr = f"document_id == '{document_id}'"
                debug(f"删除表达式: {delete_expr}")
                delete_result = collection.delete(delete_expr)
                deleted_count = delete_result.delete_count
                total_deleted += deleted_count
                info(f"成功删除 document_id={document_id} 的 {deleted_count} 条 Milvus 记录")
            except Exception as e:
                error(f"删除 document_id={document_id} 的 Milvus 记录失败: {str(e)}")

            if total_deleted == 0:
                debug(
                    f"没有删除任何 Milvus 记录，userid={userid}, file_path={file_path}, knowledge_base_id={knowledge_base_id}, document_id={document_id}")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "document_id": document_id,
                    "message": f"没有删除任何记录，userid={userid}, file_path={file_path}, knowledge_base_id={knowledge_base_id}, document_id={document_id}",
                    "status_code": 200
                }

            info(
                f"总计删除 {total_deleted} 条 Milvus 记录，userid={userid}, file_path={file_path}, knowledge_base_id={knowledge_base_id}, document_id={document_id}")
            return {
                "status": "success",
                "collection_name": collection_name,
                "document_id": ",".join(document_ids),
                "message": f"成功删除 {total_deleted} 条 Milvus 记录，userid={userid}, file_path={file_path}, knowledge_base_id={knowledge_base_id}, document_id={document_id}",
                "status_code": 200
            }

        except Exception as e:
            error(f"删除文档失败: {str(e)}")
            return {
                "status": "error",
                "collection_name": collection_name,
                "document_id": document_id,
                "message": f"删除文档失败: {str(e)}",
                "status_code": 400
            }

    async def _delete_knowledge_base(self, db_type: str, userid: str, knowledge_base_id: str) -> Dict[str, Any]:
        """删除用户的整个知识库，仅处理 Milvus 记录"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        try:
            if not utility.has_collection(collection_name):
                debug(f"集合 {collection_name} 不存在")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "deleted_files": [],
                    "message": f"集合 {collection_name} 不存在，无需删除",
                    "status_code": 200
                }

            try:
                collection = Collection(collection_name)
                debug(f"加载集合: {collection_name}")
            except Exception as e:
                error(f"加载集合 {collection_name} 失败: {str(e)}")
                return {
                    "status": "error",
                    "collection_name": collection_name,
                    "deleted_files": [],
                    "message": f"加载集合失败: {str(e)}",
                    "status_code": 400
                }

            deleted_files = []
            try:
                expr = f"userid == '{userid}' and knowledge_base_id == '{knowledge_base_id}'"
                debug(f"查询表达式: {expr}")
                results = collection.query(
                    expr=expr,
                    output_fields=["file_path"],
                    limit=1000
                )
                if results:
                    deleted_files = list(set(result["file_path"] for result in results if "file_path" in result))
                    debug(f"找到 {len(deleted_files)} 个唯一文件: {deleted_files}")
                else:
                    debug(f"没有找到 userid={userid}, knowledge_base_id={knowledge_base_id} 的记录")
            except Exception as e:
                error(f"查询 file_path 失败: {str(e)}")
                return {
                    "status": "error",
                    "collection_name": collection_name,
                    "deleted_files": [],
                    "message": f"查询 file_path 失败: {str(e)}",
                    "status_code": 400
                }

            total_deleted = 0
            try:
                delete_expr = f"userid == '{userid}' and knowledge_base_id == '{knowledge_base_id}'"
                debug(f"删除表达式: {delete_expr}")
                delete_result = collection.delete(delete_expr)
                total_deleted = delete_result.delete_count
                info(f"成功删除 {total_deleted} 条 Milvus 记录")
            except Exception as e:
                error(f"删除 Milvus 记录失败: {str(e)}")
                return {
                    "status": "error",
                    "collection_name": collection_name,
                    "deleted_files": deleted_files,
                    "message": f"删除 Milvus 记录失败: {str(e)}",
                    "status_code": 400
                }

            if total_deleted == 0:
                debug(f"没有删除任何记录，userid={userid}, knowledge_base_id={knowledge_base_id}")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "deleted_files": [],
                    "message": f"没有找到 userid={userid}, knowledge_base_id={knowledge_base_id} 的记录，无需删除",
                    "status_code": 200
                }

            info(
                f"总计删除 {total_deleted} 条 Milvus 记录，删除文件: {deleted_files}, userid={userid}, knowledge_base_id={knowledge_base_id}")
            return {
                "status": "success",
                "collection_name": collection_name,
                "deleted_files": deleted_files,
                "message": f"成功删除 {total_deleted} 条 Milvus 记录，删除文件: {deleted_files}, userid={userid}, knowledge_base_id={knowledge_base_id}",
                "status_code": 200
            }

        except Exception as e:
            error(f"删除知识库失败: {str(e)}")
            return {
                "status": "error",
                "collection_name": collection_name,
                "deleted_files": [],
                "message": f"删除知识库失败: {str(e)}",
                "status_code": 400
            }

    async def _search_query(self, query_vector: List[float], userid: str,
                           knowledge_base_ids: List[str], limit: int = 5, offset: int = 0, db_type: str = "") -> Dict[str, Any]:
        """基于向量搜索 Milvus 集合"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        timing_stats = {}
        start_time = time.time()
        try:
            if not query_vector or not isinstance(query_vector, list) or len(query_vector) != 1024:
                raise ValueError("query_vector 必须是长度为 1024 的浮点数列表")
            if not userid:
                raise ValueError("userid 不能为空")
            if not knowledge_base_ids:
                raise ValueError("knowledge_base_ids 不能为空")
            if "_" in userid:
                raise ValueError("userid 不能包含下划线")
            if len(userid) > 100:
                raise ValueError("userid 的长度超出限制")
            if limit <= 0 or limit > 16384:
                raise ValueError("limit 必须在 1 到 16384 之间")
            if offset < 0:
                raise ValueError("offset 不能为负数")
            if limit + offset > 16384:
                raise ValueError("limit + offset 不能超过 16384")
            for kb_id in knowledge_base_ids:
                if not isinstance(kb_id, str):
                    raise ValueError(f"knowledge_base_id 必须是字符串: {kb_id}")
                if len(kb_id) > 100:
                    raise ValueError(f"knowledge_base_id 长度超出 100 个字符: {kb_id}")
                if "_" in kb_id:
                    raise ValueError(f"knowledge_base_id 不能包含下划线: {kb_id}")

            if not utility.has_collection(collection_name):
                debug(f"集合 {collection_name} 不存在")
                return {"results": [], "timing": timing_stats}

            try:
                collection = Collection(collection_name)
                collection.load()
                debug(f"加载集合: {collection_name}")
                timing_stats["collection_load"] = time.time() - start_time
                debug(f"集合加载耗时: {timing_stats['collection_load']:.3f} 秒")
            except Exception as e:
                error(f"加载集合 {collection_name} 失败: {str(e)}")
                return {"results": [], "timing": timing_stats}

            search_start = time.time()
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            kb_id_expr = " or ".join([f"knowledge_base_id == '{kb_id}'" for kb_id in knowledge_base_ids])
            expr = f"userid == '{userid}' and ({kb_id_expr})"
            debug(f"搜索表达式: {expr}")

            try:
                results = collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param=search_params,
                    limit=100,
                    expr=expr,
                    output_fields=["text", "userid", "document_id", "filename", "file_path", "upload_time",
                                   "file_type"],
                    offset=offset
                )
            except Exception as e:
                error(f"搜索失败: {str(e)}")
                return {"results": [], "timing": timing_stats}
            timing_stats["vector_search"] = time.time() - search_start
            debug(f"向量搜索耗时: {timing_stats['vector_search']:.3f} 秒")

            search_results = []
            for hits in results:
                for hit in hits:
                    metadata = {
                        "userid": hit.entity.get("userid"),
                        "document_id": hit.entity.get("document_id"),
                        "filename": hit.entity.get("filename"),
                        "file_path": hit.entity.get("file_path"),
                        "upload_time": hit.entity.get("upload_time"),
                        "file_type": hit.entity.get("file_type")
                    }
                    result = {
                        "text": hit.entity.get("text"),
                        "distance": hit.distance,
                        "source": "vector_query",
                        "metadata": metadata
                    }
                    search_results.append(result)
                    debug(
                        f"命中: text={result['text'][:100]}..., distance={hit.distance}, filename={metadata['filename']}")

            dedup_start = time.time()
            unique_results = []
            seen_texts = set()
            for result in sorted(search_results, key=lambda x: x['distance'], reverse=True):
                if result['text'] not in seen_texts:
                    unique_results.append(result)
                    seen_texts.add(result['text'])
            timing_stats["deduplication"] = time.time() - dedup_start
            debug(f"去重耗时: {timing_stats['deduplication']:.3f} 秒")
            info(f"去重后结果数量: {len(unique_results)} (原始数量: {len(search_results)})")

            timing_stats["total_time"] = time.time() - start_time
            info(f"向量搜索完成，返回 {len(unique_results)} 条结果，总耗时: {timing_stats['total_time']:.3f} 秒")
            return {"results": unique_results[:limit], "timing": timing_stats}

        except Exception as e:
            error(f"向量搜索失败: {str(e)}")
            return {"results": [], "timing": timing_stats}

    async def _list_user_files(self, userid: str, db_type: str = "") -> Dict[str, List[Dict]]:
        """列出用户的所有知识库及其文件，按 knowledge_base_id 分组"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        try:
            info(f"列出用户文件: userid={userid}, db_type={db_type}")

            if not userid:
                raise ValueError("userid 不能为空")
            if "_" in userid or (db_type and "_" in db_type):
                raise ValueError("userid 和 db_type 不能包含下划线")
            if (db_type and len(db_type) > 100) or len(userid) > 100:
                raise ValueError("userid 或 db_type 的长度超出限制")

            if not utility.has_collection(collection_name):
                debug(f"集合 {collection_name} 不存在")
                return {}

            try:
                collection = Collection(collection_name)
                collection.load()
                debug(f"加载集合: {collection_name}")
            except Exception as e:
                error(f"加载集合 {collection_name} 失败: {str(e)}")
                return {}

            expr = f"userid == '{userid}'"
            debug(f"查询表达式: {expr}")

            try:
                results = collection.query(
                    expr=expr,
                    output_fields=["document_id", "filename", "file_path", "upload_time", "file_type", "knowledge_base_id"],
                    limit=1000
                )
            except Exception as e:
                error(f"查询用户文件失败: {str(e)}")
                return {}

            files_by_kb = {}
            seen_document_ids = set()
            for result in results:
                document_id = result.get("document_id")
                kb_id = result.get("knowledge_base_id")
                if document_id not in seen_document_ids:
                    seen_document_ids.add(document_id)
                    file_info = {
                        "document_id": document_id,
                        "filename": result.get("filename"),
                        "file_path": result.get("file_path"),
                        "upload_time": result.get("upload_time"),
                        "file_type": result.get("file_type"),
                        "knowledge_base_id": kb_id
                    }
                    if kb_id not in files_by_kb:
                        files_by_kb[kb_id] = []
                    files_by_kb[kb_id].append(file_info)
                    debug(f"找到文件: document_id={document_id}, filename={result.get('filename')}, knowledge_base_id={kb_id}")

            info(f"找到 {len(seen_document_ids)} 个文件，userid={userid}, 知识库数量={len(files_by_kb)}")
            return files_by_kb

        except Exception as e:
            error(f"列出用户文件失败: {str(e)}")
            return {}

    async def _list_all_knowledge_bases(self, db_type: str = "") -> Dict[str, Any]:
        """列出数据库中所有用户的知识库及其文件，按用户分组"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        try:
            info(f"列出所有用户的知识库: db_type={db_type}")

            if db_type and "_" in db_type:
                raise ValueError("db_type 不能包含下划线")
            if db_type and len(db_type) > 100:
                raise ValueError("db_type 的长度应小于 100")

            if not utility.has_collection(collection_name):
                debug(f"集合 {collection_name} 不存在")
                return {
                    "status": "success",
                    "users_knowledge_bases": {},
                    "collection_name": collection_name,
                    "message": f"集合 {collection_name} 不存在",
                    "status_code": 200
                }

            try:
                collection = Collection(collection_name)
                collection.load()
                debug(f"加载集合: {collection_name}")
            except Exception as e:
                error(f"加载集合 {collection_name} 失败: {str(e)}")
                return {
                    "status": "error",
                    "users_knowledge_bases": {},
                    "collection_name": collection_name,
                    "message": f"加载集合失败: {str(e)}",
                    "status_code": 400
                }

            expr = "userid != ''"
            debug(f"查询表达式: {expr}")
            try:
                results = collection.query(
                    expr=expr,
                    output_fields=["userid", "knowledge_base_id", "document_id", "filename", "file_path", "upload_time",
                                   "file_type"],
                    limit=10000
                )
            except Exception as e:
                error(f"查询所有用户文件失败: {str(e)}")
                return {
                    "status": "error",
                    "users_knowledge_bases": {},
                    "collection_name": collection_name,
                    "message": f"查询失败: {str(e)}",
                    "status_code": 400
                }

            users_knowledge_bases = {}
            seen_document_ids = set()
            for result in results:
                userid = result.get("userid")
                kb_id = result.get("knowledge_base_id")
                document_id = result.get("document_id")
                if document_id not in seen_document_ids:
                    seen_document_ids.add(document_id)
                    file_info = {
                        "document_id": document_id,
                        "filename": result.get("filename"),
                        "file_path": result.get("file_path"),
                        "upload_time": result.get("upload_time"),
                        "file_type": result.get("file_type"),
                        "knowledge_base_id": kb_id
                    }
                    if userid not in users_knowledge_bases:
                        users_knowledge_bases[userid] = {}
                    if kb_id not in users_knowledge_bases[userid]:
                        users_knowledge_bases[userid][kb_id] = []
                    users_knowledge_bases[userid][kb_id].append(file_info)
                    debug(
                        f"找到文件: userid={userid}, knowledge_base_id={kb_id}, document_id={document_id}, filename={result.get('filename')}")

            info(f"找到 {len(seen_document_ids)} 个文件，涉及 {len(users_knowledge_bases)} 个用户")
            return {
                "status": "success",
                "users_knowledge_bases": users_knowledge_bases,
                "collection_name": collection_name,
                "message": f"成功列出 {len(users_knowledge_bases)} 个用户的知识库和文件",
                "status_code": 200
            }

        except Exception as e:
            error(f"列出所有用户知识库失败: {str(e)}")
            return {
                "status": "error",
                "users_knowledge_bases": {},
                "collection_name": collection_name,
                "message": f"列出所有用户知识库失败: {str(e)}",
                "status_code": 400
            }

connection_register('Milvus', MilvusDBConnection)
info("MilvusDBConnection registered")