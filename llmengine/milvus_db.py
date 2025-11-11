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
import traceback
import hashlib

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
            debug(f"dbpath: {self.db_path}")
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
            debug(f"db_dir: {db_dir}")
            if not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                debug(f"创建 Milvus 目录: {db_dir}")
            if not os.access(db_dir, os.W_OK):
                raise RuntimeError(f"Milvus 目录 {db_dir} 不可写")
                debug(f"不可写")
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
            fields = params.get("fields", None)  # 获取 fields 参数
            collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
            if db_type and "_" in db_type:
                return {"status": "error", "message": "db_type 不能包含下划线", "collection_name": collection_name,
                        "document_id": "", "status_code": 400}
            if db_type and len(db_type) > 100:
                return {"status": "error", "message": "db_type 的长度应小于 100", "collection_name": collection_name,
                        "document_id": "", "status_code": 400}

            if action == "initialize":
                return {"status": "success", "message": f"Milvus 连接已初始化，路径: {self.db_path}"}
            elif action == "create_collection":
                return await self._create_collection(db_type, fields)
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
                if not userid or not knowledge_base_id or not document_id:
                    return {"status": "error", "message": "userid、document_id 和 knowledge_base_id 不能为空",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                if len(userid) > 100 or len(file_path) > 255 or len(knowledge_base_id) > 100:
                    return {"status": "error", "message": "userid、filename 或 knowledge_base_id 的长度超出限制",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                return await self._delete_document(userid, knowledge_base_id, document_id, file_path, db_type)
            elif action == "delete_knowledge_base":
                userid = params.get("userid", "")
                knowledge_base_id = params.get("knowledge_base_id", "")
                if not userid or not knowledge_base_id:
                    return {"status": "error", "message": "userid 和 knowledge_base_id 不能为空",
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
                if not query_vector or not userid:
                    return {"status": "error", "message": "query_vector或userid不能为空",
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

    async def _create_collection(self, db_type: str = "", fields: List[Any] = None) -> Dict:
        """创建 Milvus 集合，支持动态 fields（JSON 或 FieldSchema）"""
        try:
            # 确定集合名称
            collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
            if len(collection_name) > 255:
                raise ValueError(f"集合名称 {collection_name} 超过 255 个字符")
            if db_type and "_" in db_type:
                raise ValueError("db_type 不能包含下划线")
            if db_type and len(db_type) > 100:
                raise ValueError("db_type 的长度应小于 100")
            debug(f"初始集合名称: {collection_name}")

            # 检查集合是否已存在
            if utility.has_collection(collection_name):
                collection = Collection(collection_name)
                collection.load()
                debug(f"集合 {collection_name} 已存在，直接加载")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "message": f"集合 {collection_name} 已存在，直接使用现有 schema"
                }

            # 如果集合不存在，处理 fields
            if db_type == "":
                # db_type 为空，使用默认 fields
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
                debug("db_type 为空，使用默认 fields")
            else:
                # db_type 不为空，集合不存在时必须提供 fields
                if fields is None:
                    debug(f"集合 {collection_name} 不存在，且未提供 fields")
                    raise ValueError(f"集合 {collection_name} 不存在，必须提供 fields 参数")

            # 定义 dtype 映射
            dtype_map = {
                "VARCHAR": DataType.VARCHAR,
                "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
                "BINARY_VECTOR": DataType.BINARY_VECTOR,
                "INT64": DataType.INT64,
                "INT32": DataType.INT32,
                "FLOAT": DataType.FLOAT,
                "DOUBLE": DataType.DOUBLE,
                "BOOL": DataType.BOOL,
                "JSON": DataType.JSON
            }

            # 如果 fields 是 JSON 格式，转换为 FieldSchema 对象
            if all(isinstance(f, dict) for f in fields):
                try:
                    converted_fields = []
                    for f in fields:
                        if f["dtype"] not in dtype_map:
                            debug(f"无效的 dtype: {f['dtype']}")
                            raise ValueError(f"无效的 dtype: {f['dtype']}")
                        dtype = dtype_map[f["dtype"]]
                        params = {
                            "name": f["name"],
                            "dtype": dtype,
                            "is_primary": f.get("is_primary", False),
                            "auto_id": f.get("auto_id", False)
                        }
                        if dtype == DataType.VARCHAR:
                            if "max_length" not in f or not isinstance(f["max_length"], int) or f["max_length"] <= 0:
                                debug(f"VARCHAR 字段 {f['name']} 必须指定有效的 max_length")
                                raise ValueError(f"VARCHAR 字段 {f['name']} 必须指定有效的 max_length")
                            params["max_length"] = f["max_length"]
                        if dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                            if "dim" not in f or not isinstance(f["dim"], int) or f["dim"] <= 0:
                                debug(f"vector 字段 {f['name']} 必须指定有效的 dim")
                                raise ValueError(f"vector 字段 {f['name']} 必须指定有效的 dim")
                            params["dim"] = f["dim"]
                        converted_fields.append(FieldSchema(**params))
                    fields = converted_fields
                    debug("成功将 JSON fields 转换为 FieldSchema 对象")
                except KeyError as e:
                    debug(f"fields 中的 JSON 格式缺少必要字段: {str(e)}")
                    raise ValueError(f"fields 中的 JSON 格式缺少必要字段: {str(e)}")

            # 验证 fields
            if not isinstance(fields, list) or not all(isinstance(f, FieldSchema) for f in fields):
                raise ValueError("fields 必须是 FieldSchema 对象的列表")

            # 检查 primary key
            primary_keys = [f for f in fields if f.is_primary]
            if len(primary_keys) != 1:
                raise ValueError("fields 必须正好包含一个 primary key 字段")
            primary_key = primary_keys[0]
            if primary_key.dtype not in [DataType.INT64, DataType.VARCHAR]:
                raise ValueError("primary key 字段类型必须是 INT64 或 VARCHAR")
            if primary_key.dtype == DataType.VARCHAR and not hasattr(primary_key, 'max_length'):
                raise ValueError("VARCHAR primary key 必须指定 max_length")

            # 检查 vector 字段
            vector_fields = [f for f in fields if f.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]]
            if not vector_fields:
                raise ValueError("fields 必须包含至少一个 vector 字段")
            for vf in vector_fields:
                if 'dim' not in vf.params or not isinstance(vf.params['dim'], int) or vf.params['dim'] <= 0:
                    raise ValueError(f"vector 字段 {vf.name} 必须指定有效的 dim 参数")

            # 检查字段名称唯一性
            field_names = [f.name for f in fields]
            if len(field_names) != len(set(field_names)):
                raise ValueError("fields 中的 name 必须唯一")

            # 检查其他字段约束
            for f in fields:
                if f.dtype == DataType.VARCHAR and not hasattr(f, 'max_length'):
                    raise ValueError(f"VARCHAR 字段 {f.name} 必须指定 max_length")
                if f.name == "" or len(f.name) > 255:
                    raise ValueError(f"字段名称 {f.name} 无效（不能为空或超过 255 字符）")

            debug(f"fields 验证通过，字段数量: {len(fields)}")

            # 创建集合
            schema = CollectionSchema(
                fields=fields,
                description="统一数据集合，包含用户ID、知识库ID、document_id 和元数据字段",
                auto_id=any(f.auto_id for f in primary_keys),
                primary_field=primary_key.name,
            )

            collection = Collection(collection_name, schema)
            # 创建 vector 索引
            for f in vector_fields:
                collection.create_index(
                    field_name=f.name,
                    index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"}
                )
            # 创建 scalar 索引（可选）
            for f in fields:
                if f.dtype not in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR, DataType.JSON]:
                    collection.create_index(
                        field_name=f.name,
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
        """插入文档到 Milvus，支持动态 schema"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        document_id = chunks[0].get("pk", chunks[0].get("document_id", "")) if chunks else ""
        try:
            debug(f"_insert_document called with db_type={db_type}, chunks_count={len(chunks) if chunks else 0}")

            # 检查集合是否存在 - 不传递 fields，依赖 _create_collection 的逻辑
            create_result = await self._create_collection(db_type)
            if create_result["status"] == "error":
                raise RuntimeError(f"集合创建失败: {create_result['message']}")
            collection_name = create_result["collection_name"]
            debug(f"Collection ready: {collection_name}")

            # 检查输入数据
            if not chunks:
                raise ValueError("chunks 不能为空")
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    raise ValueError("每个 chunk 必须是一个字典")

            # # 字段映射
            # for chunk in chunks:
            #     if 'orgid' in chunk and 'userid' not in chunk:
            #         chunk['userid'] = chunk['orgid']
            #     if 'fiid' in chunk and 'knowledge_base_id' not in chunk:
            #         chunk['knowledge_base_id'] = chunk['fiid']
            #     if 'embedding' in chunk and 'vector' not in chunk:
            #         chunk['vector'] = chunk['embedding']
            #     if 'pk' not in chunk:
            #         chunk['pk'] = chunk.get('document_id', str(uuid.uuid4()))
            # debug(f"Sample chunk after mapping: {chunks[0] if chunks else {} }")

            # 加载集合并获取 schema
            collection = Collection(collection_name)
            collection.load()
            schema_fields = [field.name for field in collection.schema.fields if not field.auto_id]
            vector_fields = [field.name for field in collection.schema.fields if
                             field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]]
            debug(f"Schema fields: {schema_fields}, Vector fields: {vector_fields}")

            # 验证 chunks 是否包含所有必要字段
            for chunk in chunks:
                missing_fields = [f for f in schema_fields if f not in chunk]
                if missing_fields:
                    raise ValueError(f"chunk 缺少必要字段: {', '.join(missing_fields)}")
                for vector_field in vector_fields:
                    vector_dim = next(
                        field.params["dim"] for field in collection.schema.fields if field.name == vector_field)
                    if not isinstance(chunk[vector_field], list) or len(chunk[vector_field]) != vector_dim:
                        raise ValueError(f"{vector_field} 必须是长度为 {vector_dim} 的浮点数列表")

            # 构造插入数据
            data = {field: [] for field in schema_fields}
            for chunk in chunks:
                for field in schema_fields:
                    data[field].append(chunk[field])

            # 插入数据
            collection.insert([data[field] for field in schema_fields])
            collection.flush()
            debug(f"成功插入 {len(chunks)} 个文档到集合 {collection_name}")
            return {
                "status": "success",
                "document_id": document_id,
                "collection_name": collection_name,
                "message": f"成功插入 {len(chunks)} 个文档到 {collection_name}",
                "status_code": 200
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

    async def _delete_document(self, userid: str, knowledge_base_id: str, document_id:str, file_path: "", db_type: str = "") -> Dict[
        str, Any]:
        """删除用户指定文件数据，仅处理 Milvus 记录"""
        collection_name = "ragdb" if not db_type or db_type == "ragdb" else f"ragdb_{db_type}"
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
                "document_id":document_id,
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
        collection_name = "ragdb" if not db_type or db_type == "ragdb" else f"ragdb_{db_type}"
        try:
            if not userid:
                raise ValueError("userid 不能为空")
            if not knowledge_base_id:
                raise ValueError("knowledge_base_id 不能为空")
            if len(userid) > 100 or len(knowledge_base_id) > 100 or (db_type and len(db_type) > 100):
                raise ValueError("userid, knowledge_base_id 或 db_type 的长度超出限制")

            debug(
                f"删除知识库参数: userid={userid}, knowledge_base_id={knowledge_base_id}, db_type={db_type}, collection_name={collection_name}")

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
                collection.load()
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

            # 检查集合 schema
            schema_fields = [field.name for field in collection.schema.fields]
            debug(f"集合 {collection_name} schema: {schema_fields}")

            deleted_files = []
            if collection_name == "ragdb" and "file_path" in schema_fields:
                try:
                    expr = f"userid == '{userid}' and knowledge_base_id == '{knowledge_base_id}'"
                    debug(f"查询表达式: {expr}")
                    results = collection.query(
                        expr=expr,
                        output_fields=["file_path"],
                        limit=1000
                    )
                    if results:
                        deleted_files = list(set(result["file_path"] for result in results if result["file_path"]))
                        debug(f"找到 {len(deleted_files)} 个唯一文件: {deleted_files}")
                    else:
                        debug(f"没有找到 userid={userid}, knowledge_base_id={knowledge_base_id} 的记录")
                except Exception as e:
                    error(f"查询 file_path 失败: {str(e)}")
                    # 继续执行删除操作，不因查询失败而中断
                    deleted_files = []

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
                            knowledge_base_ids: List[str] | None, limit: int = 5, offset: int = 0, db_type: str = "") -> Dict[str, Any]:
        """基于向量搜索 Milvus 集合（knowledge_base_ids 可为空）"""
        timing_stats = {}
        start_time = time.time()
        try:
            # ---------- 参数校验 ----------
            if not query_vector or not isinstance(query_vector, list) or len(query_vector) != 1024:
                raise ValueError("query_vector 必须是长度为 1024 的浮点数列表")
            if not userid:
                raise ValueError("userid 不能为空")
            if len(userid) > 100:
                raise ValueError("userid 的长度超出限制")
            if limit <= 0 or limit > 16384:
                raise ValueError("limit 必须在 1 到 16384 之间")
            if offset < 0:
                raise ValueError("offset 不能为负数")
            if limit + offset > 16384:
                raise ValueError("limit + offset 不能超过 16384")
            if knowledge_base_ids is not None:
                for kb_id in knowledge_base_ids:
                    if not isinstance(kb_id, str) or len(kb_id) > 100:
                        raise ValueError(f"knowledge_base_id 必须是字符串且长度 ≤ 100: {kb_id}")

            # ---------- 集合名称 ----------
            collection_name = "ragdb" if not db_type or db_type == "ragdb" else f"ragdb_{db_type}"
            debug(f"目标集合: {collection_name}")

            if not utility.has_collection(collection_name):
                debug(f"集合 {collection_name} 不存在")
                return {"results": [], "timing": timing_stats}

            # ---------- 加载集合 ----------
            try:
                collection = Collection(collection_name)
                collection.load()
                timing_stats["collection_load"] = time.time() - start_time
            except Exception as e:
                error(f"加载集合 {collection_name} 失败: {str(e)}")
                return {"results": [], "timing": timing_stats}

            # ---------- 动态 schema ----------
            schema_fields = [f.name for f in collection.schema.fields]
            is_ragdb = collection_name == "ragdb"
            output_fields = (
                ["text", "userid", "document_id", "filename", "file_path", "upload_time", "file_type",
                 "knowledge_base_id"]
                if is_ragdb else
                ["document_id", "userid", "knowledge_base_id", "text"]
            )
            output_fields = [f for f in output_fields if f in schema_fields]

            # ---------- 动态构建表达式 ----------
            if is_ragdb:
                expr_parts = [f"userid == '{userid}'"]
                if knowledge_base_ids:
                    kb_expr = " or ".join([f"knowledge_base_id == '{kb}'" for kb in knowledge_base_ids])
                    expr_parts.append(f"({kb_expr})")
                expr = " and ".join(expr_parts)
            else:
                expr_parts = [f"userid == '{userid}'"]
                if knowledge_base_ids:
                    kb_expr = " or ".join([f"knowledge_base_id == '{kb}'" for kb in knowledge_base_ids])
                    expr_parts.append(f"({kb_expr})")
                expr = " and ".join(expr_parts)

            debug(f"搜索表达式: {expr}")

            # ---------- 保留你指定的 search 调用 ----------
            search_start = time.time()
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

            try:
                results = collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param=search_params,
                    limit=100,  # 保留
                    expr=expr,  # 动态
                    output_fields=output_fields,  # 动态
                    offset=offset  # 保留
                )
            except Exception as e:
                error(f"搜索失败: {str(e)}")
                return {"results": [], "timing": timing_stats}

            timing_stats["vector_search"] = time.time() - search_start

            # ---------- 统一结果结构 + 去重 ----------
            raw_results = []
            for hits in results:
                for hit in hits:
                    if is_ragdb:
                        metadata = {
                            "userid": hit.entity.get("userid"),
                            "document_id": hit.entity.get("document_id"),
                            "filename": hit.entity.get("filename"),
                            "file_path": hit.entity.get("file_path"),
                            "upload_time": hit.entity.get("upload_time"),
                            "file_type": hit.entity.get("file_type"),
                            "knowledge_base_id": hit.entity.get("knowledge_base_id")
                        }
                        raw_results.append({
                            "text": hit.entity.get("text"),
                            "distance": 1.0 - hit.distance,
                            "metadata": metadata
                        })
                    else:
                        metadata = {
                            "document_id": hit.entity.get("document_id"),
                            "userid": hit.entity.get("userid"),
                            "knowledge_base_id": hit.entity.get("knowledge_base_id"),
                        }
                        raw_results.append({
                            "text": hit.entity.get("text"),
                            "distance": 1.0 - hit.distance,
                            "metadata": metadata
                        })

            # ---------- 去重+ limit ----------
            dedup_start = time.time()
            unique_results = []
            seen_texts = set()
            for result in sorted(raw_results, key=lambda x: x['distance'], reverse=True):
                if result['text'] not in seen_texts:
                    unique_results.append(result)
                    seen_texts.add(result['text'])
            timing_stats["deduplication"] = time.time() - dedup_start
            debug(f"去重耗时: {timing_stats['deduplication']:.3f} 秒")
            info(f"去重后结果数量: {len(unique_results)} (原始数量: {len(raw_results)})")

            timing_stats["total_time"] = time.time() - start_time
            info(f"向量搜索完成，返回 {len(unique_results)} 条结果，总耗时: {timing_stats['total_time']:.3f} 秒")
            return {"results": unique_results[:limit], "timing": timing_stats}

        except Exception as e:
            error(f"向量搜索失败: {str(e)}")
            return {"results": [], "timing": timing_stats}

    # async def _search_query(self, query_vector: List[float], userid: str,
    #                        knowledge_base_ids: List[str], limit: int = 5, offset: int = 0, db_type: str = "") -> Dict[str, Any]:
    #     """基于向量搜索 Milvus 集合"""
    #     collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
    #     timing_stats = {}
    #     start_time = time.time()
    #     try:
    #         if not query_vector or not isinstance(query_vector, list) or len(query_vector) != 1024:
    #             raise ValueError("query_vector 必须是长度为 1024 的浮点数列表")
    #         if not userid:
    #             raise ValueError("userid 不能为空")
    #         if not knowledge_base_ids:
    #             raise ValueError("knowledge_base_ids 不能为空")
    #         if len(userid) > 100:
    #             raise ValueError("userid 的长度超出限制")
    #         if limit <= 0 or limit > 16384:
    #             raise ValueError("limit 必须在 1 到 16384 之间")
    #         if offset < 0:
    #             raise ValueError("offset 不能为负数")
    #         if limit + offset > 16384:
    #             raise ValueError("limit + offset 不能超过 16384")
    #         for kb_id in knowledge_base_ids:
    #             if not isinstance(kb_id, str):
    #                 raise ValueError(f"knowledge_base_id 必须是字符串: {kb_id}")
    #             if len(kb_id) > 100:
    #                 raise ValueError(f"knowledge_base_id 长度超出 100 个字符: {kb_id}")
    #
    #         if not utility.has_collection(collection_name):
    #             debug(f"集合 {collection_name} 不存在")
    #             return {"results": [], "timing": timing_stats}
    #
    #         try:
    #             collection = Collection(collection_name)
    #             collection.load()
    #             debug(f"加载集合: {collection_name}")
    #             timing_stats["collection_load"] = time.time() - start_time
    #             debug(f"集合加载耗时: {timing_stats['collection_load']:.3f} 秒")
    #         except Exception as e:
    #             error(f"加载集合 {collection_name} 失败: {str(e)}")
    #             return {"results": [], "timing": timing_stats}
    #
    #         search_start = time.time()
    #         search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    #         kb_id_expr = " or ".join([f"knowledge_base_id == '{kb_id}'" for kb_id in knowledge_base_ids])
    #         expr = f"userid == '{userid}' and ({kb_id_expr})"
    #         debug(f"搜索表达式: {expr}")
    #
    #         try:
    #             results = collection.search(
    #                 data=[query_vector],
    #                 anns_field="vector",
    #                 param=search_params,
    #                 limit=100,
    #                 expr=expr,
    #                 output_fields=["text", "userid", "document_id", "filename", "file_path", "upload_time",
    #                                "file_type"],
    #                 offset=offset
    #             )
    #         except Exception as e:
    #             error(f"搜索失败: {str(e)}")
    #             return {"results": [], "timing": timing_stats}
    #         timing_stats["vector_search"] = time.time() - search_start
    #         debug(f"向量搜索耗时: {timing_stats['vector_search']:.3f} 秒")
    #
    #         search_results = []
    #         for hits in results:
    #             for hit in hits:
    #                 metadata = {
    #                     "userid": hit.entity.get("userid"),
    #                     "document_id": hit.entity.get("document_id"),
    #                     "filename": hit.entity.get("filename"),
    #                     "file_path": hit.entity.get("file_path"),
    #                     "upload_time": hit.entity.get("upload_time"),
    #                     "file_type": hit.entity.get("file_type")
    #                 }
    #                 result = {
    #                     "text": hit.entity.get("text"),
    #                     "distance": hit.distance,
    #                     "source": "vector_query",
    #                     "metadata": metadata
    #                 }
    #                 search_results.append(result)
    #                 debug(
    #                     f"命中: text={result['text'][:100]}..., distance={hit.distance}, filename={metadata['filename']}")
    #
    #         dedup_start = time.time()
    #         unique_results = []
    #         seen_texts = set()
    #         for result in sorted(search_results, key=lambda x: x['distance'], reverse=True):
    #             if result['text'] not in seen_texts:
    #                 unique_results.append(result)
    #                 seen_texts.add(result['text'])
    #         timing_stats["deduplication"] = time.time() - dedup_start
    #         debug(f"去重耗时: {timing_stats['deduplication']:.3f} 秒")
    #         info(f"去重后结果数量: {len(unique_results)} (原始数量: {len(search_results)})")
    #
    #         timing_stats["total_time"] = time.time() - start_time
    #         info(f"向量搜索完成，返回 {len(unique_results)} 条结果，总耗时: {timing_stats['total_time']:.3f} 秒")
    #         return {"results": unique_results[:limit], "timing": timing_stats}
    #
    #     except Exception as e:
    #         error(f"向量搜索失败: {str(e)}")
    #         return {"results": [], "timing": timing_stats}

    async def _list_user_files(self, userid: str, db_type: str = "") -> Dict[str, Any]:
        """列出用户的所有知识库及其文件，按 knowledge_base_id 分组（仅 ragdb）"""
        try:
            debug(f"列出用户文件: userid={userid}, db_type={db_type}")

            if not userid:
                raise ValueError("userid 不能为空")
            if (db_type and len(db_type) > 100) or len(userid) > 100:
                raise ValueError("userid 或 db_type 的长度超出限制")

            collection_name = "ragdb" if not db_type or db_type == "ragdb" else f"ragdb_{db_type}"
            debug(f"目标集合: {collection_name}")

            if not utility.has_collection(collection_name):
                debug(f"集合 {collection_name} 不存在")
                return {
                    "status": "success",
                    "files_by_knowledge_base": {} if collection_name == "ragdb" else [],
                    "collection_name": collection_name
                }

            try:
                collection = Collection(collection_name)
                collection.load()
            except Exception as e:
                error(f"加载集合 {collection_name} 失败: {str(e)}")
                return {"status": "error", "message": str(e)}

            schema_fields = [f.name for f in collection.schema.fields]
            debug(f"集合 {collection_name} schema: {schema_fields}")

            is_ragdb = collection_name == "ragdb"
            output_fields = (
                ["userid", "document_id", "filename", "file_path", "upload_time", "file_type", "knowledge_base_id"]
                if is_ragdb else
                ["document_id", "userid", "knowledge_base_id", "text"]
            )
            output_fields = [f for f in output_fields if f in schema_fields]

            # 动态构建表达式
            if "userid" in schema_fields:
                expr = f"userid == '{userid}'"
                debug(f"使用顶级字段过滤: {expr}")
            else:
                expr = f'metadata["user_id"] == "{userid}"'
                debug(f"使用 JSON 嵌套字段过滤: {expr}")

            results = collection.query(expr=expr, output_fields=output_fields, limit=1000)

            # 正确初始化（不嵌套）
            if is_ragdb:
                files_by_knowledge_base = {}  # {kb_id: [file,...]}
            else:
                files_by_knowledge_base = []  # [file1, file2, ...]

            seen_document_ids = set()

            for result in results:
                document_id = result.get("document_id") or result.get("id")
                if not document_id or document_id in seen_document_ids:
                    continue
                seen_document_ids.add(document_id)

                kb_id = result.get("knowledge_base_id")
                if is_ragdb and not kb_id:
                    continue

                if is_ragdb:
                    file_info = {
                        "userid": result.get("userid"),
                        "document_id": document_id,
                        "filename": result.get("filename", ""),
                        "file_path": result.get("file_path", ""),
                        "upload_time": result.get("upload_time", ""),
                        "file_type": result.get("file_type", ""),
                        "knowledge_base_id": kb_id
                    }
                    files_by_knowledge_base.setdefault(kb_id, []).append(file_info)
                else:
                    file_info = {
                        "document_id": document_id,
                        "userid": result.get("userid", ""),
                        "knowledge_base_id": kb_id,
                        "text": result.get("text", ""),
                    }
                    files_by_knowledge_base.append(file_info)

                debug(f"找到文件: document_id={document_id}, kb_id={kb_id}, file_info={file_info}")

            info(f"找到 {len(seen_document_ids)} 个文件")
            return {
                "status": "success",
                "files_by_knowledge_base": files_by_knowledge_base,
                "collection_name": collection_name
            }

        except Exception as e:
            error(f"列出用户文件失败: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _list_all_knowledge_bases(self, db_type: str = "") -> Dict[str, Any]:
        try:
            debug(f"列出所有知识库: db_type={db_type}")

            collection_names = utility.list_collections()
            if db_type:
                target = "ragdb" if db_type == "ragdb" else f"ragdb_{db_type}"
                collection_names = [n for n in collection_names if n == target]
            else:
                collection_names = [n for n in collection_names if n.startswith("ragdb")]

            if not collection_names:
                return {
                    "status": "success",
                    "collections": {},
                    "collection_names": "none",
                    "message": "未找到任何以 ragdb 开头的集合",
                    "status_code": 200
                }

            collections = {}
            total_documents = 0
            skipped_collections = []

            for collection_name in collection_names:
                if collection_name.startswith("ragdb_"):
                    db_part = collection_name[len("ragdb_"):]
                    if "_" in db_part or len(db_part) > 100:
                        skipped_collections.append(collection_name)
                        continue

                try:
                    collection = Collection(collection_name)
                    collection.load()
                except Exception as e:
                    error(f"加载集合 {collection_name} 失败: {str(e)}")
                    skipped_collections.append(collection_name)
                    continue

                schema_fields = [f.name for f in collection.schema.fields]
                debug(f"集合{collection_name}的schme是：{schema_fields}")
                is_ragdb = collection_name == "ragdb"
                output_fields = (
                    ["document_id", "knowledge_base_id", "filename", "text", "userid"]
                    if is_ragdb else
                    ["document_id", "userid", "knowledge_base_id", "text"]
                )
                output_fields = [f for f in output_fields if f in schema_fields]

                results = collection.query(expr="", output_fields=output_fields, limit=10000)
                debug(f"集合{collection_name}的查询到{len(results)}条记录")
                for r in results:
                    debug(f"集合{collection_name}找到文档：{r}")
                # 正确初始化
                if is_ragdb:
                    collections[collection_name] = {"knowledge_bases": {}}
                else:
                    collections[collection_name] = {"knowledge_bases": []}

                seen = set()
                for result in results:
                    doc_id = result.get("document_id") or result.get("id")
                    if not doc_id or doc_id in seen:
                        continue
                    seen.add(doc_id)

                    kb_id = result.get("knowledge_base_id")
                    if is_ragdb and not kb_id:
                        continue

                    if is_ragdb:
                        file_info = {
                            "document_id": doc_id,
                            "text": result.get("text", "")[:50],
                            "userid": result.get("userid", ""),
                            "filename": result.get("filename", ""),
                            "knowledge_base_id": kb_id
                        }
                        collections[collection_name]["knowledge_bases"].setdefault(kb_id, []).append(file_info)
                    else:
                        file_info = {
                            "document_id": doc_id,
                            "userid": result.get("userid", ""),
                            "knowledge_base_id": kb_id,
                            "text": result.get("text", ""),
                        }
                        collections[collection_name]["knowledge_bases"].append(file_info)

                    total_documents += 1

            debug(f"找到 {total_documents} 个文档，跨 {len(collections)} 个集合")
            return {
                "status": "success",
                "collections": collections,
                "collection_names": ",".join(collection_names),
                "message": f"成功列出 {len(collections)} 个集合，{total_documents} 个文档",
                "status_code": 200
            }

        except Exception as e:
            error(f"列出所有知识库失败: {str(e)}")
            return {
                "status": "error",
                "collections": {},
                "collection_names": "none",
                "message": f"列出所有知识库失败: {str(e)}",
                "status_code": 400
            }

connection_register('Milvus', MilvusDBConnection)
info("MilvusDBConnection registered")
