from appPublic.jsonConfig import getConfig
import os
from appPublic.log import debug, error, info
import yaml
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from threading import Lock
from llmengine.base_connection import connection_register
from typing import Dict, List, Any
import aiohttp
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from datetime import datetime
from filetxt.loader import fileloader
from llmengine.kgc import KnowledgeGraph
import numpy as np
from py2neo import Graph
from scipy.spatial.distance import cosine
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import traceback
import asyncio
import re

# 嵌入缓存
EMBED_CACHE = {}

class MilvusConnection:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MilvusConnection, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        try:
            config = getConfig()
            self.db_path = config['milvus_db']
            self.neo4j_uri = config['neo4j']['uri']
            self.neo4j_user = config['neo4j']['user']
            self.neo4j_password = config['neo4j']['password']
        except KeyError as e:
            error(f"配置文件缺少必要字段: {str(e)}")
            raise RuntimeError(f"配置文件缺少必要字段: {str(e)}")
        self._initialize_connection()
        self._initialized = True
        info(f"MilvusConnection initialized with db_path: {self.db_path}")

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
            # 通用 db_type 验证
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
                file_path = params.get("file_path", "")
                userid = params.get("userid", "")
                knowledge_base_id = params.get("knowledge_base_id", "")
                if not file_path or not userid or not knowledge_base_id:
                    return {"status": "error", "message": "file_path、userid 和 knowledge_base_id 不能为空",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                if "_" in userid or "_" in knowledge_base_id:
                    return {"status": "error", "message": "userid 和 knowledge_base_id 不能包含下划线",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                if len(knowledge_base_id) > 100:
                    return {"status": "error", "message": "knowledge_base_id 的长度应小于 100",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                return await self._insert_document(file_path, userid, knowledge_base_id, db_type)
            elif action == "delete_document":
                userid = params.get("userid", "")
                filename = params.get("filename", "")
                knowledge_base_id = params.get("knowledge_base_id", "")
                if not userid or not filename or not knowledge_base_id:
                    return {"status": "error", "message": "userid、filename 和 knowledge_base_id 不能为空",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                if "_" in userid or "_" in knowledge_base_id:
                    return {"status": "error", "message": "userid 和 knowledge_base_id 不能包含下划线",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                if len(userid) > 100 or len(filename) > 255 or len(knowledge_base_id) > 100:
                    return {"status": "error", "message": "userid、filename 或 knowledge_base_id 的长度超出限制",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                return await self._delete_document(db_type, userid, filename, knowledge_base_id)
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
            elif action == "fused_search":
                query = params.get("query", "")
                userid = params.get("userid", "")
                knowledge_base_ids = params.get("knowledge_base_ids", [])
                limit = params.get("limit", 5)
                if not query or not userid or not knowledge_base_ids:
                    return {
                        "status": "error",
                        "message": "query、userid 或 knowledge_base_ids 不能为空",
                        "collection_name": "ragdb" if not params.get("db_type","") else f"ragdb_{params.get('db_type')}",
                        "document_id": "",
                        "status_code": 400
                    }
                if limit < 1 or limit > 16384:
                    return {
                        "status": "error",
                        "message": "limit 必须在 1 到 16384 之间",
                        "collection_name": "ragdb" if not params.get("db_type","") else f"ragdb_{params.get('db_type')}",
                        "document_id": "",
                        "status_code": 400
                    }
                return await self._fused_search(
                    query,
                    userid,
                    params.get("db_type", ""),
                    knowledge_base_ids,
                    limit,
                    params.get("offset", 0),
                    params.get("use_rerank", True)
                )
            elif action == "search_query":
                query = params.get("query", "")
                userid = params.get("userid", "")
                limit = params.get("limit", "")
                knowledge_base_ids = params.get("knowledge_base_ids", [])
                if not query or not userid or not knowledge_base_ids:
                    return {"status": "error", "message": "query、userid 或 knowledge_base_ids 不能为空",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                return await self._search_query(
                    query,
                    userid,
                    db_type,
                    knowledge_base_ids,
                    limit,
                    params.get("offset", 0),
                    params.get("use_rerank", True)
                )
            elif action == "list_user_files":
                userid = params.get("userid", "")
                if not userid:
                    return {"status": "error", "message": "userid 不能为空", "collection_name": collection_name,
                            "document_id": "", "status_code": 400}
                return await self._list_user_files(userid)
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
                "collection_name": params.get("db_type", "ragdb") if params else "ragdb",
                "document_id": "",
                "status_code": 400
            }

    async def _create_collection(self, db_type: str = "") -> Dict:
        """创建 Milvus 集合"""
        try:
            # 根据 db_type 决定集合名称
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
                "collection_name":collection_name,
                "message": str(e)
            }

    async def _delete_collection(self, db_type: str = "") -> Dict:
        """删除 Milvus 集合"""
        try:
            # 根据 db_type 决定集合名称
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

    async def _insert_document(self, file_path: str, userid: str, knowledge_base_id: str, db_type: str = "") -> Dict[
        str, Any]:
        """将文档插入 Milvus 并抽取三元组到 Neo4j"""
        document_id = str(uuid.uuid4())
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        debug(
            f'Inserting document: file_path={file_path}, userid={userid}, db_type={db_type}, knowledge_base_id={knowledge_base_id}, document_id={document_id}')

        timings = {}
        start_total = time.time()
        start_neo4j = None

        try:
            # 检查是否已存在相同的 file_path、userid 和 knowledge_base_id
            collection = Collection(collection_name)
            expr = f'file_path == "{file_path}" && userid == "{userid}" && knowledge_base_id == "{knowledge_base_id}"'
            debug(f"检查重复文档: {expr}")
            start_check = time.time()
            results = collection.query(expr=expr, output_fields=["document_id"])
            timings["check_duplicate"] = time.time() - start_check
            debug(f"检查重复文档耗时: {timings['check_duplicate']:.2f} 秒")
            if results:
                raise ValueError(
                    f"文档已存在: file_path={file_path}, userid={userid}, knowledge_base_id={knowledge_base_id}")

            if not os.path.exists(file_path):
                raise ValueError(f"文件 {file_path} 不存在")

            supported_formats = {'pdf', 'docx', 'xlsx', 'pptx', 'csv', 'txt'}
            ext = file_path.rsplit('.', 1)[1].lower() if '.' in file_path else ''
            if ext not in supported_formats:
                raise ValueError(f"不支持的文件格式: {ext}, 支持的格式: {', '.join(supported_formats)}")

            info(f"生成 document_id: {document_id} for file: {file_path}")

            # 文件加载
            debug(f"加载文件: {file_path}")
            start_load = time.time()
            text = fileloader(file_path)
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.;,\n]', '', text)
            timings["load_file"] = time.time() - start_load
            debug(f"加载文件耗时: {timings['load_file']:.2f} 秒, 文本长度: {len(text)}")
            if not text or not text.strip():
                raise ValueError(f"文件 {file_path} 加载为空")

            # 文本分片
            document = Document(page_content=text)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len,
            )
            debug("开始分片文件内容")
            start_split = time.time()
            chunks = text_splitter.split_documents([document])
            timings["split_text"] = time.time() - start_split
            debug(
                f"文本分片耗时: {timings['split_text']:.2f} 秒, 分片数量: {len(chunks)}, 分片内容: {[chunk.page_content[:50] for chunk in chunks[:5]]}")
            if not chunks:
                raise ValueError(f"文件 {file_path} 未生成任何文档块")

            filename = os.path.basename(file_path).rsplit('.', 1)[0]
            upload_time = datetime.now().isoformat()
            documents = []
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'userid': userid,
                    'knowledge_base_id': knowledge_base_id,
                    'document_id': document_id,
                    'filename': filename + '.' + ext,
                    'file_path': file_path,
                    'upload_time': upload_time,
                    'file_type': ext,
                })
                documents.append(chunk)
                debug(f"文档块 {i} 元数据: {chunk.metadata}")

            # 确保集合存在
            debug(f"确保集合 {collection_name} 存在")
            start_create = time.time()
            create_result = await self._create_collection(db_type)
            timings["create_collection"] = time.time() - start_create
            debug(f"集合创建耗时: {timings['create_collection']:.2f} 秒")
            if create_result["status"] == "error":
                raise RuntimeError(f"集合创建失败: {create_result['message']}")

            # 生成嵌入
            debug("调用嵌入服务生成向量")
            texts = [doc.page_content for doc in documents]
            start_embed = time.time()
            embeddings = await self._get_embeddings(texts)
            timings["generate_embeddings"] = time.time() - start_embed
            debug(f"生成嵌入耗时: {timings['generate_embeddings']:.2f} 秒")

            # 插入 Milvus
            start_milvus = time.time()
            await self._insert_to_milvus(collection_name, documents, embeddings)
            timings["insert_milvus"] = time.time() - start_milvus
            debug(f"Milvus 插入耗时: {timings['insert_milvus']:.2f} 秒")
            info(f"成功插入 {len(documents)} 个文档块到 {collection_name}")

            # 三元组抽取
            debug("调用三元组抽取服务")
            start_triples = time.time()
            try:
                chunk_texts = [doc.page_content for doc in documents]
                debug(f"处理 {len(chunk_texts)} 个分片进行三元组抽取")

                tasks = [self._extract_triples(chunk) for chunk in chunk_texts]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                triples = []
                for i, result in enumerate(results):
                    if isinstance(result, list):
                        triples.extend(result)
                        debug(f"分片 {i + 1} 抽取到 {len(result)} 个三元组: {result[:5]}")
                    else:
                        error(f"分片 {i + 1} 处理失败: {str(result)}")

                # 去重
                unique_triples = []
                seen = set()
                for t in triples:
                    identifier = (t['head'].lower(), t['tail'].lower(), t['type'].lower())
                    if identifier not in seen:
                        seen.add(identifier)
                        unique_triples.append(t)
                    else:
                        for existing in unique_triples:
                            if (existing['head'].lower() == t['head'].lower() and
                                    existing['tail'].lower() == t['tail'].lower() and
                                    len(t['type']) > len(existing['type'])):
                                unique_triples.remove(existing)
                                unique_triples.append(t)
                                debug(f"替换三元组为更具体类型: {t}")
                                break

                timings["extract_triples"] = time.time() - start_triples
                debug(
                    f"三元组抽取耗时: {timings['extract_triples']:.2f} 秒, 抽取到 {len(unique_triples)} 个三元组: {unique_triples[:5]}")

                # Neo4j 插入
                debug(f"抽取到 {len(unique_triples)} 个三元组，插入 Neo4j")
                start_neo4j = time.time()
                if unique_triples:
                    kg = KnowledgeGraph(triples=unique_triples, document_id=document_id,
                                        knowledge_base_id=knowledge_base_id, userid=userid)
                    kg.create_graphnodes()
                    kg.create_graphrels()
                    kg.export_data()
                    info(f"文件 {file_path} 三元组成功插入 Neo4j")
                else:
                    debug(f"文件 {file_path} 未抽取到三元组")
                timings["insert_neo4j"] = time.time() - start_neo4j if start_neo4j is not None else 0
                debug(f"Neo4j 插入耗时: {timings['insert_neo4j']:.2f} 秒")

            except Exception as e:
                timings["extract_triples"] = time.time() - start_triples if "extract_triples" not in timings else \
                timings["extract_triples"]
                timings["insert_neo4j"] = time.time() - start_neo4j if start_neo4j is not None else 0
                debug(f"处理三元组或 Neo4j 插入失败: {str(e)}, 堆栈: {traceback.format_exc()}")
                timings["total"] = time.time() - start_total
                return {
                    "status": "success",
                    "document_id": document_id,
                    "collection_name": collection_name,
                    "timings": timings,
                    "unique_triples": unique_triples,
                    "message": f"文件 {file_path} 成功嵌入，但三元组处理或 Neo4j 插入失败: {str(e)}",
                    "status_code": 200
                }

            timings["total"] = time.time() - start_total
            debug(f"总耗时: {timings['total']:.2f} 秒")
            return {
                "status": "success",
                "document_id": document_id,
                "collection_name": collection_name,
                "timings": timings,
                "unique_triples": unique_triples,
                "message": f"文件 {file_path} 成功嵌入并处理三元组",
                "status_code": 200
            }

        except Exception as e:
            error(f"插入文档失败: {str(e)}")
            timings["total"] = time.time() - start_total
            debug(f"总耗时: {timings['total']:.2f} 秒")
            return {
                "status": "error",
                "document_id": document_id,
                "collection_name": collection_name,
                "timings": timings,
                "message": f"插入文档失败: {str(e)}",
                "status_code": 400
            }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, RuntimeError)),
        before_sleep=lambda retry_state: debug(f"重试三元组抽取服务，第 {retry_state.attempt_number} 次")
    )

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """调用嵌入服务获取文本的向量，带缓存"""
        try:
            # 检查缓存
            uncached_texts = [text for text in texts if text not in EMBED_CACHE]
            if uncached_texts:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:9998/v1/embeddings",
                        headers={"Content-Type": "application/json"},
                        json={"input": uncached_texts}
                    ) as response:
                        if response.status != 200:
                            error(f"嵌入服务调用失败，状态码: {response.status}")
                            raise RuntimeError(f"嵌入服务调用失败: {response.status}")
                        result = await response.json()
                        if result.get("object") != "list" or not result.get("data"):
                            error(f"嵌入服务响应格式错误: {result}")
                            raise RuntimeError("嵌入服务响应格式错误")
                        embeddings = [item["embedding"] for item in result["data"]]
                        for text, embedding in zip(uncached_texts, embeddings):
                            EMBED_CACHE[text] = np.array(embedding) / np.linalg.norm(embedding)
                        debug(f"成功获取 {len(embeddings)} 个新嵌入向量，缓存大小: {len(EMBED_CACHE)}")
            # 返回缓存中的嵌入
            return [EMBED_CACHE[text] for text in texts]
        except Exception as e:
            error(f"嵌入服务调用失败: {str(e)}")
            raise RuntimeError(f"嵌入服务调用失败: {str(e)}")

    async def _extract_triples(self, text: str) -> List[Dict]:
        """调用三元组抽取服务，无超时限制"""
        request_id = str(uuid.uuid4())  # 为每个请求生成唯一 ID
        start_time = time.time()
        debug(f"Request #{request_id} started for triples extraction")
        try:
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=30),
                timeout=aiohttp.ClientTimeout(total=None)  # 无限等待
            ) as session:
                async with session.post(
                    "http://localhost:9991/v1/triples",
                    headers={"Content-Type": "application/json; charset=utf-8"},
                    json={"text": text}
                ) as response:
                    elapsed_time = time.time() - start_time
                    debug(f"Request #{request_id} received response, status: {response.status}, took {elapsed_time:.2f} seconds")
                    if response.status != 200:
                        error_text = await response.text()
                        error(f"Request #{request_id} failed, status: {response.status}, response: {error_text}")
                        raise RuntimeError(f"三元组抽取服务调用失败: {response.status}, {error_text}")
                    result = await response.json()
                    if result.get("object") != "list" or not result.get("data"):
                        error(f"Request #{request_id} invalid response format: {result}")
                        raise RuntimeError("三元组抽取服务响应格式错误")
                    triples = result["data"]
                    debug(f"Request #{request_id} extracted {len(triples)} triples, total time: {elapsed_time:.2f} seconds")
                    return triples
        except Exception as e:
            elapsed_time = time.time() - start_time
            error(f"Request #{request_id} failed to extract triples: {str(e)}, took {elapsed_time:.2f} seconds")
            debug(f"Request #{request_id} traceback: {traceback.format_exc()}")
            raise RuntimeError(f"三元组抽取服务调用失败: {str(e)}")

    async def _insert_to_milvus(self, collection_name: str, documents: List[Document],
                                embeddings: List[List[float]]) -> None:
        """将文档和嵌入向量插入 Milvus 集合"""
        try:
            if not connections.has_connection("default"):
                self._initialize_connection()
            collection = Collection(collection_name)
            collection.load()
            data = {
                "userid": [doc.metadata["userid"] for doc in documents],
                "knowledge_base_id": [doc.metadata["knowledge_base_id"] for doc in documents],
                "document_id": [doc.metadata["document_id"] for doc in documents],
                "text": [doc.page_content for doc in documents],
                "vector": embeddings,
                "filename": [doc.metadata["filename"] for doc in documents],
                "file_path": [doc.metadata["file_path"] for doc in documents],
                "upload_time": [doc.metadata["upload_time"] for doc in documents],
                "file_type": [doc.metadata["file_type"] for doc in documents],
            }
            collection.insert([data[field.name] for field in collection.schema.fields if field.name != "pk"])
            collection.flush()
            debug(f"成功插入 {len(documents)} 个文档到集合 {collection_name}")
        except Exception as e:
            error(f"插入 Milvus 失败: {str(e)}")
            raise RuntimeError(f"插入 Milvus 失败: {str(e)}")

    async def _delete_document(self, db_type: str, userid: str, filename: str, knowledge_base_id: str) -> Dict[
        str, Any]:
        """删除用户指定文件数据，包括 Milvus 和 Neo4j 中的记录"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        try:
            if not utility.has_collection(collection_name):
                debug(f"集合 {collection_name} 不存在")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "document_id": "",
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
                    "document_id": "",
                    "message": f"加载集合失败: {str(e)}",
                    "status_code": 400
                }

            expr = f"userid == '{userid}' and filename == '{filename}' and knowledge_base_id == '{knowledge_base_id}'"
            debug(f"查询表达式: {expr}")
            try:
                results = collection.query(
                    expr=expr,
                    output_fields=["document_id"],
                    limit=1000
                )
                if not results:
                    debug(
                        f"没有找到 userid={userid}, filename={filename}, knowledge_base_id={knowledge_base_id} 的记录")
                    return {
                        "status": "success",
                        "collection_name": collection_name,
                        "document_id": "",
                        "message": f"没有找到 userid={userid}, filename={filename}, knowledge_base_id={knowledge_base_id} 的记录，无需删除",
                        "status_code": 200
                    }
                document_ids = list(set(result["document_id"] for result in results if "document_id" in result))
                debug(f"找到 {len(document_ids)} 个 document_id: {document_ids}")
            except Exception as e:
                error(f"查询 document_id 失败: {str(e)}")
                return {
                    "status": "error",
                    "collection_name": collection_name,
                    "document_id": "",
                    "message": f"查询失败: {str(e)}",
                    "status_code": 400
                }

            total_deleted = 0
            neo4j_deleted_nodes = 0
            neo4j_deleted_rels = 0
            for doc_id in document_ids:
                try:
                    # 删除 Milvus 记录
                    delete_expr = f"document_id == '{doc_id}'"
                    debug(f"删除表达式: {delete_expr}")
                    delete_result = collection.delete(delete_expr)
                    deleted_count = delete_result.delete_count
                    total_deleted += deleted_count
                    info(f"成功删除 document_id={doc_id} 的 {deleted_count} 条 Milvus 记录")

                    # 删除 Neo4j 三元组
                    try:
                        graph = Graph(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
                        query = """
                        MATCH (n {document_id: $document_id})
                        OPTIONAL MATCH (n)-[r {document_id: $document_id}]->()
                        WITH collect(r) AS rels, collect(n) AS nodes
                        FOREACH (r IN rels | DELETE r)
                        FOREACH (n IN nodes | DELETE n)
                        RETURN size(nodes) AS node_count, size(rels) AS rel_count, [r IN rels | type(r)] AS rel_types
                        """
                        result = graph.run(query, document_id=doc_id).data()
                        nodes_deleted = result[0]['node_count'] if result else 0
                        rels_deleted = result[0]['rel_count'] if result else 0
                        rel_types = result[0]['rel_types'] if result else []
                        info(
                            f"成功删除 document_id={doc_id} 的 {nodes_deleted} 个 Neo4j 节点和 {rels_deleted} 个关系，关系类型: {rel_types}")
                        neo4j_deleted_nodes += nodes_deleted
                        neo4j_deleted_rels += rels_deleted
                    except Exception as e:
                        error(f"删除 document_id={doc_id} 的 Neo4j 三元组失败: {str(e)}")
                        continue
                except Exception as e:
                    error(f"删除 document_id={doc_id} 的 Milvus 记录失败: {str(e)}")
                    continue

            if total_deleted == 0:
                debug(
                    f"没有删除任何 Milvus 记录，userid={userid}, filename={filename}, knowledge_base_id={knowledge_base_id}")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "document_id": "",
                    "message": f"没有删除任何记录，userid={userid}, filename={filename}, knowledge_base_id={knowledge_base_id}",
                    "status_code": 200
                }

            info(
                f"总计删除 {total_deleted} 条 Milvus 记录，{neo4j_deleted_nodes} 个 Neo4j 节点，{neo4j_deleted_rels} 个 Neo4j 关系，userid={userid}, filename={filename}, knowledge_base_id={knowledge_base_id}")
            return {
                "status": "success",
                "collection_name": collection_name,
                "document_id": ",".join(document_ids),
                "message": f"成功删除 {total_deleted} 条 Milvus 记录，{neo4j_deleted_nodes} 个 Neo4j 节点，{neo4j_deleted_rels} 个 Neo4j 关系，userid={userid}, filename={filename}, knowledge_base_id={knowledge_base_id}",
                "status_code": 200
            }

        except Exception as e:
            error(f"删除文档失败: {str(e)}")
            return {
                "status": "error",
                "collection_name": collection_name,
                "document_id": "",
                "message": f"删除文档失败: {str(e)}",
                "status_code": 400
            }

    async def _delete_knowledge_base(self, db_type: str, userid: str, knowledge_base_id: str) -> Dict[str, Any]:
        """删除用户的整个知识库，包括 Milvus 和 Neo4j 中的记录"""
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

            # 查询被删除的文件列表
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

            # 删除 Milvus 记录
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

            # 删除 Neo4j 数据
            neo4j_deleted_nodes = 0
            neo4j_deleted_rels = 0
            try:
                debug(f"尝试连接 Neo4j: uri={self.neo4j_uri}, user={self.neo4j_user}")
                graph = Graph(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
                debug("Neo4j 连接成功")
                query = """
                   MATCH (n {userid: $userid, knowledge_base_id: $knowledge_base_id})
                   OPTIONAL MATCH (n)-[r {userid: $userid, knowledge_base_id: $knowledge_base_id}]->()
                   WITH collect(r) AS rels, collect(n) AS nodes
                   FOREACH (r IN rels | DELETE r)
                   FOREACH (n IN nodes | DELETE n)
                   RETURN size(nodes) AS node_count, size(rels) AS rel_count, [r IN rels | type(r)] AS rel_types
                   """
                result = graph.run(query, userid=userid, knowledge_base_id=knowledge_base_id).data()
                nodes_deleted = result[0]['node_count'] if result else 0
                rels_deleted = result[0]['rel_count'] if result else 0
                rel_types = result[0]['rel_types'] if result else []
                neo4j_deleted_nodes += nodes_deleted
                neo4j_deleted_rels += rels_deleted
                info(f"成功删除 {nodes_deleted} 个 Neo4j 节点和 {rels_deleted} 个关系，关系类型: {rel_types}")
            except Exception as e:
                error(f"删除 Neo4j 数据失败: {str(e)}")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "deleted_files": deleted_files,
                    "message": f"成功删除 {total_deleted} 条 Milvus 记录，{neo4j_deleted_nodes} 个 Neo4j 节点，{neo4j_deleted_rels} 个 Neo4j 关系，但 Neo4j 删除失败: {str(e)}",
                    "status_code": 200
                }

            if total_deleted == 0 and neo4j_deleted_nodes == 0 and neo4j_deleted_rels == 0:
                debug(f"没有删除任何记录，userid={userid}, knowledge_base_id={knowledge_base_id}")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "deleted_files": [],
                    "message": f"没有找到 userid={userid}, knowledge_base_id={knowledge_base_id} 的记录，无需删除",
                    "status_code": 200
                }

            info(
                f"总计删除 {total_deleted} 条 Milvus 记录，{neo4j_deleted_nodes} 个 Neo4j 节点，{neo4j_deleted_rels} 个 Neo4j 关系，删除文件: {deleted_files}, userid={userid}, knowledge_base_id={knowledge_base_id}")
            return {
                "status": "success",
                "collection_name": collection_name,
                "deleted_files": deleted_files,
                "message": f"成功删除 {total_deleted} 条 Milvus 记录，{neo4j_deleted_nodes} 个 Neo4j 节点，{neo4j_deleted_rels} 个 Neo4j 关系，删除文件: {deleted_files}, userid={userid}, knowledge_base_id={knowledge_base_id}",
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

    async def _extract_entities(self, query: str) -> List[str]:
        """调用实体识别服务"""
        try:
            if not query:
                raise ValueError("查询文本不能为空")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:9990/v1/entities",
                    headers={"Content-Type": "application/json"},
                    json={"query": query}
                ) as response:
                    if response.status != 200:
                        error(f"实体识别服务调用失败，状态码: {response.status}")
                        raise RuntimeError(f"实体识别服务调用失败: {response.status}")
                    result = await response.json()
                    if result.get("object") != "list" or not result.get("data"):
                        error(f"实体识别服务响应格式错误: {result}")
                        raise RuntimeError("实体识别服务响应格式错误")
                    entities = result["data"]
                    unique_entities = list(dict.fromkeys(entities))  # 去重
                    debug(f"成功提取 {len(unique_entities)} 个唯一实体: {unique_entities}")
                    return unique_entities
        except Exception as e:
            error(f"实体识别服务调用失败: {str(e)}")
            return []

    async def _match_triplets(self, query: str, query_entities: List[str], userid: str, knowledge_base_id: str) -> List[Dict]:
        """匹配查询实体与 Neo4j 中的三元组"""
        start_time = time.time()  # 记录开始时间
        matched_triplets = []
        ENTITY_SIMILARITY_THRESHOLD = 0.8

        try:
            graph = Graph(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            debug(f"已连接到 Neo4j: {self.neo4j_uri}")
            neo4j_connect_time = time.time() - start_time
            debug(f"Neo4j 连接耗时: {neo4j_connect_time:.3f} 秒")

            matched_names = set()
            entity_match_start = time.time()
            for entity in query_entities:
                normalized_entity = entity.lower().strip()
                query = """
                MATCH (n {userid: $userid, knowledge_base_id: $knowledge_base_id})
                WHERE toLower(n.name) CONTAINS $entity
                   OR apoc.text.levenshteinSimilarity(toLower(n.name), $entity) > 0.7
                RETURN n.name, apoc.text.levenshteinSimilarity(toLower(n.name), $entity) AS sim
                ORDER BY sim DESC
                LIMIT 100
                """
                try:
                    results = graph.run(query, userid=userid, knowledge_base_id=knowledge_base_id, entity=normalized_entity).data()
                    for record in results:
                        matched_names.add(record['n.name'])
                        debug(f"实体 {entity} 匹配节点: {record['n.name']} (Levenshtein 相似度: {record['sim']:.2f})")
                except Exception as e:
                    debug(f"模糊匹配实体 {entity} 失败: {str(e)}")
                    continue
            entity_match_time = time.time() - entity_match_start
            debug(f"实体匹配耗时: {entity_match_time:.3f} 秒")

            triplets = []
            if matched_names:
                triplet_query_start = time.time()
                query = """
                MATCH (h {userid: $userid, knowledge_base_id: $knowledge_base_id})-[r {userid: $userid, knowledge_base_id: $knowledge_base_id}]->(t {userid: $userid, knowledge_base_id: $knowledge_base_id})
                WHERE h.name IN $matched_names OR t.name IN $matched_names
                RETURN h.name AS head, r.name AS type, t.name AS tail
                LIMIT 100
                """
                try:
                    results = graph.run(query, userid=userid, knowledge_base_id=knowledge_base_id, matched_names=list(matched_names)).data()
                    seen = set()
                    for record in results:
                        head, type_, tail = record['head'], record['type'], record['tail']
                        triplet_key = (head.lower(), type_.lower(), tail.lower())
                        if triplet_key not in seen:
                            seen.add(triplet_key)
                            triplets.append({
                                'head': head,
                                'type': type_,
                                'tail': tail,
                                'head_type': '',
                                'tail_type': ''
                            })
                    debug(f"从 Neo4j 加载三元组: knowledge_base_id={knowledge_base_id}, 数量={len(triplets)}")
                except Exception as e:
                    error(f"检索三元组失败: knowledge_base_id={knowledge_base_id}, 错误: {str(e)}")
                    return []
                triplet_query_time = time.time() - triplet_query_start
                debug(f"Neo4j 三元组查询耗时: {triplet_query_time:.3f} 秒")

            if not triplets:
                debug(f"知识库 knowledge_base_id={knowledge_base_id} 无匹配三元组")
                return []

            embedding_start = time.time()
            texts_to_embed = query_entities + [t['head'] for t in triplets] + [t['tail'] for t in triplets]
            embeddings = await self._get_embeddings(texts_to_embed)
            entity_vectors = {entity: embeddings[i] for i, entity in enumerate(query_entities)}
            head_vectors = {t['head']: embeddings[len(query_entities) + i] for i, t in enumerate(triplets)}
            tail_vectors = {t['tail']: embeddings[len(query_entities) + len(triplets) + i] for i, t in enumerate(triplets)}
            debug(f"成功获取 {len(embeddings)} 个嵌入向量（{len(query_entities)} entities + {len(triplets)} heads + {len(triplets)} tails）")
            embedding_time = time.time() - embedding_start
            debug(f"嵌入向量生成耗时: {embedding_time:.3f} 秒")

            similarity_start = time.time()
            for entity in query_entities:
                entity_vec = entity_vectors[entity]
                for d_triplet in triplets:
                    d_head_vec = head_vectors[d_triplet['head']]
                    d_tail_vec = tail_vectors[d_triplet['tail']]
                    head_similarity = 1 - cosine(entity_vec, d_head_vec)
                    tail_similarity = 1 - cosine(entity_vec, d_tail_vec)

                    if head_similarity >= ENTITY_SIMILARITY_THRESHOLD or tail_similarity >= ENTITY_SIMILARITY_THRESHOLD:
                        matched_triplets.append(d_triplet)
                        debug(f"匹配三元组: {d_triplet['head']} - {d_triplet['type']} - {d_triplet['tail']} "
                              f"(entity={entity}, head_sim={head_similarity:.2f}, tail_sim={tail_similarity:.2f})")
            similarity_time = time.time() - similarity_start
            debug(f"相似度计算耗时: {similarity_time:.3f} 秒")

            unique_matched = []
            seen = set()
            for t in matched_triplets:
                identifier = (t['head'].lower(), t['type'].lower(), t['tail'].lower())
                if identifier not in seen:
                    seen.add(identifier)
                    unique_matched.append(t)

            total_time = time.time() - start_time
            debug(f"_match_triplets 总耗时: {total_time:.3f} 秒")
            info(f"找到 {len(unique_matched)} 个匹配的三元组")
            return unique_matched

        except Exception as e:
            error(f"匹配三元组失败: {str(e)}")
            return []

    async def _rerank_results(self, query: str, results: List[Dict], top_n: int) -> List[Dict]:
        """调用重排序服务"""
        try:
            if not results:
                debug("无结果需要重排序")
                return results

            if not isinstance(top_n, int) or top_n < 1:
                debug(f"无效的 top_n 参数: {top_n}，使用 len(results)={len(results)}")
                top_n = len(results)
            else:
                top_n = min(top_n, len(results))
            debug(f"重排序 top_n={top_n}, 原始结果数={len(results)}")

            documents = [result["text"] for result in results]
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        "http://localhost:9997/v1/rerank",
                        headers={"Content-Type": "application/json"},
                        json={
                            "model": "rerank-001",
                            "query": query,
                            "documents": documents,
                            "top_n": top_n
                        }
                ) as response:
                    if response.status != 200:
                        error(f"重排序服务调用失败，状态码: {response.status}")
                        raise RuntimeError(f"重排序服务调用失败: {response.status}")
                    result = await response.json()
                    if result.get("object") != "rerank.result" or not result.get("data"):
                        error(f"重排序服务响应格式错误: {result}")
                        raise RuntimeError("重排序服务响应格式错误")
                    rerank_data = result["data"]
                    reranked_results = []
                    for item in rerank_data:
                        index = item["index"]
                        if index < len(results):
                            results[index]["rerank_score"] = item["relevance_score"]
                            reranked_results.append(results[index])
                    debug(f"成功重排序 {len(reranked_results)} 条结果")
                    return reranked_results[:top_n]
        except Exception as e:
            error(f"重排序服务调用失败: {str(e)}")
            return results

    async def _fused_search(self, query: str, userid: str, db_type: str, knowledge_base_ids: List[str], limit: int = 5,
                            offset: int = 0, use_rerank: bool = True) -> Dict[str, Any]:
        """融合搜索，将查询与所有三元组拼接后向量化搜索"""
        start_time = time.time()  # 记录开始时间
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        timing_stats = {}  # 记录各步骤耗时
        try:
            info(
                f"开始融合搜索: query={query}, userid={userid}, db_type={db_type}, knowledge_base_ids={knowledge_base_ids}, limit={limit}, offset={offset}, use_rerank={use_rerank}")

            if not query or not userid or not knowledge_base_ids:
                raise ValueError("query、userid 和 knowledge_base_ids 不能为空")
            if "_" in userid or (db_type and "_" in db_type):
                raise ValueError("userid 和 db_type 不能包含下划线")
            if (db_type and len(db_type) > 100) or len(userid) > 100:
                raise ValueError("db_type 或 userid 的长度超出限制")
            if limit < 1 or limit > 16384 or offset < 0:
                raise ValueError("limit 必须在 1 到 16384 之间，offset 必须大于或等于 0")

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

            entity_extract_start = time.time()
            query_entities = await self._extract_entities(query)
            timing_stats["entity_extraction"] = time.time() - entity_extract_start
            debug(f"提取实体: {query_entities}, 耗时: {timing_stats['entity_extraction']:.3f} 秒")

            all_triplets = []
            triplet_match_start = time.time()
            for kb_id in knowledge_base_ids:
                debug(f"处理知识库: {kb_id}")
                matched_triplets = await self._match_triplets(query, query_entities, userid, kb_id)
                debug(f"知识库 {kb_id} 匹配三元组: {len(matched_triplets)} 条")
                all_triplets.extend(matched_triplets)
            timing_stats["triplet_matching"] = time.time() - triplet_match_start
            debug(f"三元组匹配总耗时: {timing_stats['triplet_matching']:.3f} 秒")

            if not all_triplets:
                debug("未找到任何匹配的三元组")
                return {"results": [], "timing": timing_stats}

            triplet_text_start = time.time()
            triplet_texts = []
            for triplet in all_triplets:
                head = triplet.get('head', '')
                type_ = triplet.get('type', '')
                tail = triplet.get('tail', '')
                if head and type_ and tail:
                    triplet_texts.append(f"{head} {type_} {tail}")
                else:
                    debug(f"无效三元组: {triplet}")
            combined_text = query
            if triplet_texts:
                combined_text += " [三元组] " + "; ".join(triplet_texts)
            debug(
                f"拼接文本: {combined_text[:200]}... (总长度: {len(combined_text)}, 三元组数量: {len(triplet_texts)})")
            timing_stats["triplet_text_combine"] = time.time() - triplet_text_start
            debug(f"拼接三元组文本耗时: {timing_stats['triplet_text_combine']:.3f} 秒")

            embedding_start = time.time()
            embeddings = await self._get_embeddings([combined_text])
            query_vector = embeddings[0]
            debug(f"拼接文本向量维度: {len(query_vector)}")
            timing_stats["embedding_generation"] = time.time() - embedding_start
            debug(f"嵌入向量生成耗时: {timing_stats['embedding_generation']:.3f} 秒")

            search_start = time.time()
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            kb_expr = " or ".join([f"knowledge_base_id == '{kb_id}'" for kb_id in knowledge_base_ids])
            expr = f"userid == '{userid}' and ({kb_expr})"
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
                error(f"向量搜索失败: {str(e)}")
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
                        "source": "fused_query_with_triplets",
                        "metadata": metadata
                    }
                    search_results.append(result)
                    debug(
                        f"搜索命中: text={result['text'][:100]}..., distance={hit.distance}, source={result['source']}")

            unique_results = []
            seen_texts = set()
            dedup_start = time.time()
            for result in sorted(search_results, key=lambda x: x['distance'], reverse=True):
                if result['text'] not in seen_texts:
                    unique_results.append(result)
                    seen_texts.add(result['text'])
            timing_stats["deduplication"] = time.time() - dedup_start
            debug(f"去重耗时: {timing_stats['deduplication']:.3f} 秒")
            info(f"去重后结果数量: {len(unique_results)} (原始数量: {len(search_results)})")

            if use_rerank and unique_results:
                rerank_start = time.time()
                debug("开始重排序")
                unique_results = await self._rerank_results(combined_text, unique_results, limit)
                unique_results = sorted(unique_results, key=lambda x: x.get('rerank_score', 0), reverse=True)
                timing_stats["reranking"] = time.time() - rerank_start
                debug(f"重排序耗时: {timing_stats['reranking']:.3f} 秒")
                debug(f"重排序分数分布: {[round(r.get('rerank_score', 0), 3) for r in unique_results]}")
            else:
                unique_results = [{k: v for k, v in r.items() if k != 'rerank_score'} for r in unique_results]

            timing_stats["total_time"] = time.time() - start_time
            info(f"融合搜索完成，返回 {len(unique_results)} 条结果，总耗时: {timing_stats['total_time']:.3f} 秒")
            return {"results": unique_results[:limit], "timing": timing_stats}

        except Exception as e:
            error(f"融合搜索失败: {str(e)}")
            return {"results": [], "timing": timing_stats}

    async def _search_query(self, query: str, userid: str, db_type: str = "", knowledge_base_ids: List[str] = [], limit: int = 5,
                           offset: int = 0, use_rerank: bool = True) -> Dict[str, Any]:
        """纯向量搜索，基于查询文本在指定知识库中搜索相关文本块"""
        start_time = time.time()  # 记录开始时间
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        timing_stats = {}  # 记录各步骤耗时
        try:
            info(
                f"开始纯向量搜索: query={query}, userid={userid}, db_type={db_type}, knowledge_base_ids={knowledge_base_ids}, limit={limit}, offset={offset}, use_rerank={use_rerank}")

            if not query:
                raise ValueError("查询文本不能为空")
            if not userid:
                raise ValueError("userid 不能为空")
            if "_" in userid or (db_type and "_" in db_type):
                raise ValueError("userid 和 db_type 不能包含下划线")
            if (db_type and len(db_type) > 100) or len(userid) > 100:
                raise ValueError("userid 或 db_type 的长度超出限制")
            if limit <= 0 or limit > 16384:
                raise ValueError("limit 必须在 1 到 16384 之间")
            if offset < 0:
                raise ValueError("offset 不能为负数")
            if limit + offset > 16384:
                raise ValueError("limit + offset 不能超过 16384")
            if not knowledge_base_ids:
                raise ValueError("knowledge_base_ids 不能为空")
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

            embedding_start = time.time()
            embeddings = await self._get_embeddings([query])
            query_vector = embeddings[0]
            debug(f"查询向量维度: {len(query_vector)}")
            timing_stats["embedding_generation"] = time.time() - embedding_start
            debug(f"嵌入向量生成耗时: {timing_stats['embedding_generation']:.3f} 秒")

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

            if use_rerank and unique_results:
                rerank_start = time.time()
                debug("开始重排序")
                unique_results = await self._rerank_results(query, unique_results, limit)
                unique_results = sorted(unique_results, key=lambda x: x.get('rerank_score', 0), reverse=True)
                timing_stats["reranking"] = time.time() - rerank_start
                debug(f"重排序耗时: {timing_stats['reranking']:.3f} 秒")
                debug(f"重排序分数分布: {[round(r.get('rerank_score', 0), 3) for r in unique_results]}")
            else:
                unique_results = [{k: v for k, v in r.items() if k != 'rerank_score'} for r in unique_results]

            timing_stats["total_time"] = time.time() - start_time
            info(f"纯向量搜索完成，返回 {len(unique_results)} 条结果，总耗时: {timing_stats['total_time']:.3f} 秒")
            return {"results": unique_results[:limit], "timing": timing_stats}

        except Exception as e:
            error(f"纯向量搜索失败: {str(e)}")
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

            # 查询所有用户的文件，按 userid 和 knowledge_base_id 分组
            expr = "userid != ''"  # 查询所有非空用户
            debug(f"查询表达式: {expr}")
            try:
                results = collection.query(
                    expr=expr,
                    output_fields=["userid", "knowledge_base_id", "document_id", "filename", "file_path", "upload_time",
                                   "file_type"],
                    limit=10000  # 假设最大 10000 条记录，需根据实际数据量调整
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

connection_register('Milvus', MilvusConnection)
info("MilvusConnection registered")