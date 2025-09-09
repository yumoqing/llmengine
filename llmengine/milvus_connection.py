import os
from appPublic.log import debug, error, info
from llmengine.base_connection import connection_register
from typing import Dict, List, Any
import numpy as np
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from datetime import datetime
from filetxt.loader import fileloader
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import traceback
import asyncio
import re
# 嵌入缓存
EMBED_CACHE = {}

class MilvusConnection:
    def __init__(self):
        pass

    @retry(stop = stop_after_attempt(3))
    async def _make_neo4japi_request(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        debug(f"开始API请求：action={action}, params={params}")
        try:
            async with ClientSession(timeout=ClientTimeout(total=300)) as session:
                url = f"http://localhost:8885/v1/{action}"
                debug(f"发起POST请求：{url}")
                async with session.post(
                    url,
                    headers={'Content-Type': 'application/json'},
                    json=params
                ) as response:
                    debug(f"收到相应: status={response.status}, headers={response.headers}")
                    respose_text = await response.text()
                    debug(f"响应内容: {respose_text}")
                    result = await response.json()
                    debug(f"API响应内容: {result}")
                    if response.status == 400:
                        debug(f"客户端错误，状态码: {response.status},返回响应: {result}")
                        return result
                    if response.status != 200:
                        error(f"API 调用失败，动作: {action}, 状态码: {response.status}, 响应: {response_text}")
                        raise RuntimeError(f"API 调用失败: {response.status}")
                    debug(f"API 调用成功: {action}, 响应: {result}")
                    return result
        except Exception as e:
            error(f"API 调用失败: {action}, 错误: {str(e)}, 堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"API 调用失败: {str(e)}")

    @retry(stop=stop_after_attempt(3))
    async def _make_api_request(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        debug(f"开始 API 请求: action={action}, params={params}")
        try:
            async with ClientSession(timeout=ClientTimeout(total=300)) as session:
                url = f"http://localhost:8886/v1/{action}"
                debug(f"发起 POST 请求: {url}")
                async with session.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        json=params
                ) as response:
                    debug(f"收到响应: status={response.status}, headers={response.headers}")
                    response_text = await response.text()
                    debug(f"响应内容: {response_text}")
                    result = await response.json()
                    debug(f"API 响应内容: {result}")
                    if response.status == 400:  # 客户端错误，直接返回
                        debug(f"客户端错误，状态码: {response.status}, 返回响应: {result}")
                        return result
                    if response.status != 200:
                        error(f"API 调用失败，动作: {action}, 状态码: {response.status}, 响应: {response_text}")
                        raise RuntimeError(f"API 调用失败: {response.status}")
                    debug(f"API 调用成功: {action}, 响应: {result}")
                    return result
        except Exception as e:
            error(f"API 调用失败: {action}, 错误: {str(e)}, 堆栈: {traceback.format_exc()}")
            raise RuntimeError(f"API 调用失败: {str(e)}")

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
                return {"status": "success", "message": "Milvus 服务已初始化"}
            elif action == "get_params":
                return {"status": "success", "params": {}}
            elif action == "create_collection":
                return await self._create_collection(db_type)
            elif action == "delete_collection":
                return await self._delete_collection(db_type)
            elif action == "insert_document":
                file_path = params.get("file_path", "")
                userid = params.get("userid", "")
                knowledge_base_id = params.get("knowledge_base_id", "")
                document_id = params.get("document_id", "")
                if not file_path or not userid or not knowledge_base_id or not document_id:
                    return {"status": "error", "message": "file_path、userid document_id和 knowledge_base_id 不能为空",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                if len(knowledge_base_id) > 100:
                    return {"status": "error", "message": "knowledge_base_id 的长度应小于 100",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                return await self._insert_document(file_path, userid, knowledge_base_id, document_id, db_type)
            elif action == "delete_document":
                userid = params.get("userid", "")
                file_path = params.get("file_path", "")
                knowledge_base_id = params.get("knowledge_base_id", "")
                document_id = params.get("document_id", "")
                if not userid or not file_path or not knowledge_base_id or not document_id:
                    return {"status": "error", "message": "userid、file_path document_id和 knowledge_base_id 不能为空",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                if len(userid) > 100 or len(file_path) > 255 or len(knowledge_base_id) > 100:
                    return {"status": "error", "message": "userid、file_path 或 knowledge_base_id 的长度超出限制",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                return await self._delete_document(userid, file_path, knowledge_base_id, document_id, db_type)
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
                query = params.get("query", "")
                userid = params.get("userid", "")
                knowledge_base_ids = params.get("knowledge_base_ids", [])
                limit = params.get("limit", 5)
                offset = params.get("offset", 0)
                db_type = params.get("db_type", "")
                use_rerank = params.get("use_rerank", True)
                if not query or not userid or not knowledge_base_ids:
                    return {"status": "error", "message": "query、userid 或 knowledge_base_ids 不能为空",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                return await self._search_query(query, userid, knowledge_base_ids, limit, offset, use_rerank, db_type)
            elif action == "fused_search":
                query = params.get("query", "")
                userid = params.get("userid", "")
                knowledge_base_ids = params.get("knowledge_base_ids", [])
                limit = params.get("limit", 5)
                offset = params.get("offset", 0)
                db_type = params.get("db_type", "")
                use_rerank = params.get("use_rerank", True)
                if not query or not userid or not knowledge_base_ids:
                    return {"status": "error", "message": "query、userid 或 knowledge_base_ids 不能为空",
                            "collection_name": collection_name, "document_id": "", "status_code": 400}
                return await self._fused_search(query, userid, knowledge_base_ids, limit, offset, use_rerank, db_type)
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

    async def _create_collection(self, db_type: str = "") -> Dict[str, Any]:
        """创建 Milvus 集合"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        try:
            if len(collection_name) > 255:
                raise ValueError(f"集合名称 {collection_name} 超过 255 个字符")
            if len(db_type) > 100:
                raise ValueError("db_type 的长度应小于 100")
            debug(f"调用创建集合端点: {collection_name}, 参数: {{'db_type': '{db_type}'}}")
            result = await self._make_api_request("createcollection", {"db_type": db_type})
            return result
        except Exception as e:
            error(f"创建集合失败: {str(e)}, 堆栈: {traceback.format_exc()}")
            return {
                "status": "error",
                "collection_name": collection_name,
                "message": f"创建集合失败: {str(e)}",
                "status_code": 400
            }

    async def _delete_collection(self, db_type: str = "") -> Dict:
        """删除 Milvus 集合通过服务化端点"""
        try:
            collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
            if len(collection_name) > 255:
                raise ValueError(f"集合名称 {collection_name} 超过 255 个字符")
            if db_type and "_" in db_type:
                raise ValueError("db_type 不能包含下划线")
            if db_type and len(db_type) > 100:
                raise ValueError("db_type 的长度应小于 100")
            debug(f"调用删除集合端点: {collection_name}")

            result = await self._make_api_request("deletecollection", {"db_type": db_type})
            return result
        except Exception as e:
            error(f"删除集合失败: {str(e)}")
            return {
                "status": "error",
                "collection_name": collection_name,
                "message": str(e),
                "status_code": 400
            }

    async def _insert_document(self, file_path: str, userid: str, knowledge_base_id: str, document_id:str, db_type: str = "") -> Dict[
        str, Any]:
        """将文档插入 Milvus 并抽取三元组到 Neo4j"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        debug(
            f'Inserting document: file_path={file_path}, userid={userid}, db_type={db_type}, knowledge_base_id={knowledge_base_id}, document_id={document_id}')

        timings = {}
        start_total = time.time()

        try:
            # 验证参数
            if not userid or not knowledge_base_id:
                raise ValueError("userid 和 knowledge_base_id 不能为空")
            if len(userid) > 100 or len(knowledge_base_id) > 100:
                raise ValueError("userid 或 knowledge_base_id 的长度超出限制")
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

            # 生成嵌入向量
            debug("调用嵌入服务生成向量")
            start_embedding = time.time()
            texts = [chunk.page_content for chunk in chunks]
            embeddings = await self._get_embeddings(texts)
            if not embeddings or not all(len(vec) == 1024 for vec in embeddings):
                raise ValueError("所有嵌入向量必须是长度为 1024 的浮点数列表")
            timings["generate_embeddings"] = time.time() - start_embedding
            debug(f"生成嵌入向量耗时: {timings['generate_embeddings']:.2f} 秒, 嵌入数量: {len(embeddings)}")

            # 构造 chunks 参数（展平结构）
            chunks_data = []
            for i, chunk in enumerate(chunks):
                chunks_data.append({
                    "userid": userid,
                    "knowledge_base_id": knowledge_base_id,
                    "text": chunk.page_content,
                    "vector": embeddings[i].tolist(),
                    "document_id": document_id,
                    "filename": filename + '.' + ext,
                    "file_path": file_path,
                    "upload_time": upload_time,
                    "file_type": ext,
                })

            # 调用 Milvus 插入端点
            debug(f"调用插入文件端点: {file_path}")
            start_milvus = time.time()
            result = await self._make_api_request("insertdocument", {
                "chunks": chunks_data,
                "db_type": db_type,
            })
            timings["insert_milvus"] = time.time() - start_milvus
            debug(f"Milvus 插入耗时: {timings['insert_milvus']:.2f} 秒")

            if result.get("status") != "success":
                timings["total"] = time.time() - start_total
                return {
                    "status": "error",
                    "document_id": document_id,
                    "collection_name": collection_name,
                    "timings": timings,
                    "message": result.get("message", "未知错误"),
                    "status_code": 400
                }

            # 三元组抽取
            debug("调用三元组抽取服务")
            start_triples = time.time()
            try:
                chunk_texts = [doc.page_content for doc in chunks]
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
                debug(f"抽取到 {len(unique_triples)} 个三元组，调用Neo4j服务插入")
                start_neo4j = time.time()
                if unique_triples:
                    neo4j_result = await self._make_neo4japi_request("inserttriples", {
                        "triples":unique_triples,
                        "document_id": document_id,
                        "knowledge_base_id": knowledge_base_id,
                        "userid": userid
                    })
                    debug(f"Neo4j服务响应: {neo4j_result}")
                    if neo4j_result.get("status") != "success":
                        timings["insert_neo4j"] = time.time() - start_neo4j
                        timings["total"] = time.time() - start_total
                        return{
                            "status": "error",
                            "document_id": document_id,
                            "collection_name": collection_name,
                            "timings": timings,
                            "message": f"Neo4j 三元组插入失败: {neo4j_result.get('message', '未知错误')}",
                            "status_code": 400
                        }
                    info(f"文件 {file_path} 三元组成功插入 Neo4j: {neo4j_result.get('message')}")
                else:
                    debug(f"文件 {file_path} 未抽取到三元组")
                timings["insert_neo4j"] = time.time() - start_neo4j
                debug(f"Neo4j 插入耗时: {timings['insert_neo4j']:.2f} 秒")

            except Exception as e:
                timings["extract_triples"] = time.time() - start_triples if "extract_triples" not in timings else \
                timings["extract_triples"]
                timings["insert_neo4j"] = time.time() - start_neo4j
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
            error(f"插入文档失败: {str(e)}, 堆栈: {traceback.format_exc()}")
            timings["total"] = time.time() - start_total
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

    async def _delete_document(self, userid: str, file_path: str, knowledge_base_id: str, document_id:str, db_type: str = "") -> Dict[str, Any]:
        """删除用户指定文件数据，包括 Milvus 和 Neo4j 中的记录"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        try:
            # 调用 Milvus 删除文件端点
            debug(f"调用删除文件端点: userid={userid}, file_path={file_path}, knowledge_base_id={knowledge_base_id}, document_id={document_id}")
            milvus_result = await self._make_api_request("deletedocument", {
                "userid": userid,
                "file_path": file_path,
                "knowledge_base_id": knowledge_base_id,
                "document_id": document_id,
                "db_type": db_type
            })

            if milvus_result.get("status") != "success":
                error(f"Milvus 删除文件失败: {milvus_result.get('message', '未知错误')}")
                return milvus_result

            # 调用 Neo4j 删除端点
            neo4j_deleted_nodes = 0
            neo4j_deleted_rels = 0
            try:
                debug(f"调用 Neo4j 删除文档端点: document_id={document_id}")
                neo4j_result = await self._make_neo4japi_request("deletedocument", {
                    "document_id": document_id
                })
                if neo4j_result.get("status") != "success":
                    error(
                        f"Neo4j 删除文档失败: document_id={document_id}, 错误: {neo4j_result.get('message', '未知错误')}")
                nodes_deleted = neo4j_result.get("nodes_deleted", 0)
                rels_deleted = neo4j_result.get("rels_deleted", 0)
                neo4j_deleted_nodes += nodes_deleted
                neo4j_deleted_rels += rels_deleted
                info(f"成功删除 document_id={document_id} 的 {nodes_deleted} 个 Neo4j 节点和 {rels_deleted} 个关系")
            except Exception as e:
                error(f"删除 document_id={document_id} 的 Neo4j 数据失败: {str(e)}")

            return {
                "status": "success",
                "collection_name": collection_name,
                "document_id": document_id,
                "message": f"成功删除 Milvus 记录，{neo4j_deleted_nodes} 个 Neo4j 节点，{neo4j_deleted_rels} 个 Neo4j 关系",
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
        """删除用户的整个知识库，包括 Milvus 和 Neo4j 中的记录"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        try:
            # 调用 Milvus 删除知识库端点
            debug(f"调用删除知识库端点: userid={userid}, knowledge_base_id={knowledge_base_id}")
            milvus_result = await self._make_api_request("deleteknowledgebase", {
                "userid": userid,
                "knowledge_base_id": knowledge_base_id,
                "db_type": db_type
            })

            if milvus_result.get("status") != "success":
                error(f"Milvus 删除知识库失败: {milvus_result.get('message', '未知错误')}")
                return milvus_result

            deleted_files = milvus_result.get("deleted_files", [])

            # 新增：调用 Neo4j 删除知识库端点
            neo4j_deleted_nodes = 0
            neo4j_deleted_rels = 0
            try:
                debug(f"调用 Neo4j 删除知识库端点: userid={userid}, knowledge_base_id={knowledge_base_id}")
                neo4j_result = await self._make_neo4japi_request("deleteknowledgebase", {
                    "userid": userid,
                    "knowledge_base_id": knowledge_base_id
                })
                if neo4j_result.get("status") == "success":
                    neo4j_deleted_nodes = neo4j_result.get("nodes_deleted", 0)
                    neo4j_deleted_rels = neo4j_result.get("rels_deleted", 0)
                    info(f"成功删除 {neo4j_deleted_nodes} 个 Neo4j 节点和 {neo4j_deleted_rels} 个关系")
                else:
                    error(f"Neo4j 删除知识库失败: {neo4j_result.get('message', '未知错误')}")
                    return {
                        "status": "success",
                        "collection_name": collection_name,
                        "deleted_files": deleted_files,
                        "message": f"成功删除 Milvus 知识库，{neo4j_deleted_nodes} 个 Neo4j 节点，{neo4j_deleted_rels} 个 Neo4j 关系，但 Neo4j 删除失败: {neo4j_result.get('message')}",
                        "status_code": 200
                    }
            except Exception as e:
                error(f"Neo4j 删除知识库失败: {str(e)}")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "deleted_files": deleted_files,
                    "message": f"成功删除 Milvus 知识库，{neo4j_deleted_nodes} 个 Neo4j 节点，{neo4j_deleted_rels} 个 Neo4j 关系，但 Neo4j 删除失败: {str(e)}",
                    "status_code": 200
                }

            if not deleted_files and neo4j_deleted_nodes == 0 and neo4j_deleted_rels == 0:
                debug(f"没有删除任何记录，userid={userid}, knowledge_base_id={knowledge_base_id}")
                return {
                    "status": "success",
                    "collection_name": collection_name,
                    "deleted_files": [],
                    "message": f"没有找到 userid={userid}, knowledge_base_id={knowledge_base_id} 的记录，无需删除",
                    "status_code": 200
                }

            info(
                f"总计删除 Milvus 记录，{neo4j_deleted_nodes} 个 Neo4j 节点，{neo4j_deleted_rels} 个 Neo4j 关系，删除文件: {deleted_files}")
            return {
                "status": "success",
                "collection_name": collection_name,
                "deleted_files": deleted_files,
                "message": f"成功删除 Milvus 记录，{neo4j_deleted_nodes} 个 Neo4j 节点，{neo4j_deleted_rels} 个 Neo4j 关系，删除文件: {deleted_files}",
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

    async def _search_query(self, query: str, userid: str, knowledge_base_ids: List[str], limit: int = 5,
                            offset: int = 0, use_rerank: bool = True, db_type: str = "") -> Dict[str, Any]:
        """纯向量搜索，调用服务化端点"""
        start_time = time.time()
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        timing_stats = {}
        try:
            info(
                f"开始纯向量搜索: query={query}, userid={userid}, db_type={db_type}, knowledge_base_ids={knowledge_base_ids}, limit={limit}, offset={offset}, use_rerank={use_rerank}")

            if not query:
                raise ValueError("查询文本不能为空")
            if not userid:
                raise ValueError("userid 不能为空")
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

            # 将查询文本转换为向量
            vector_start = time.time()
            query_vector = await self._get_embeddings([query])
            if not query_vector or not all(len(vec) == 1024 for vec in query_vector):
                raise ValueError("查询向量必须是长度为 1024 的浮点数列表")
            query_vector = query_vector[0]  # 取第一个向量
            timing_stats["vector_generation"] = time.time() - vector_start
            debug(f"生成查询向量耗时: {timing_stats['vector_generation']:.3f} 秒")

            # 调用纯向量搜索端点
            search_start = time.time()
            result = await self._make_api_request("searchquery", {
                "query_vector": query_vector.tolist(),
                "userid": userid,
                "knowledge_base_ids": knowledge_base_ids,
                "limit": limit,
                "offset": offset,
                "db_type": db_type
            })
            timing_stats["vector_search"] = time.time() - search_start
            debug(f"向量搜索耗时: {timing_stats['vector_search']:.3f} 秒")

            if result.get("status") != "success":
                error(f"纯向量搜索失败: {result.get('message', '未知错误')}")
                return {"results": [], "timing": timing_stats}

            unique_results = result.get("results", [])
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
            error(f"纯向量搜索失败: {str(e)}, 堆栈: {traceback.format_exc()}")
            return {"results": [], "timing": timing_stats}

    async def _fused_search(self, query: str, userid: str, knowledge_base_ids: List[str], limit: int = 5,
                            offset: int = 0, use_rerank: bool = True, db_type: str = "") -> Dict[str, Any]:
        """融合搜索，调用服务化端点"""
        start_time = time.time()
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        timing_stats = {}
        try:
            info(
                f"开始融合搜索: query={query}, userid={userid}, db_type={db_type}, knowledge_base_ids={knowledge_base_ids}, limit={limit}, offset={offset}, use_rerank={use_rerank}")

            if not query or not userid or not knowledge_base_ids:
                raise ValueError("query、userid 和 knowledge_base_ids 不能为空")
            if (db_type and len(db_type) > 100) or len(userid) > 100:
                raise ValueError("db_type 或 userid 的长度超出限制")
            if limit < 1 or limit > 16384 or offset < 0:
                raise ValueError("limit 必须在 1 到 16384 之间，offset 必须大于或等于 0")

            # 提取实体
            entity_extract_start = time.time()
            query_entities = await self._extract_entities(query)
            timing_stats["entity_extraction"] = time.time() - entity_extract_start
            debug(f"提取实体: {query_entities}, 耗时: {timing_stats['entity_extraction']:.3f} 秒")

            # 调用 Neo4j 服务进行三元组匹配
            all_triplets = []
            triplet_match_start = time.time()
            for kb_id in knowledge_base_ids:
                debug(f"调用 Neo4j 三元组匹配: knowledge_base_id={kb_id}")
                try:
                    neo4j_result = await self._make_neo4japi_request("matchtriplets", {
                        "query": query,
                        "query_entities": query_entities,
                        "userid": userid,
                        "knowledge_base_id": kb_id
                    })
                    if neo4j_result.get("status") == "success":
                        triplets = neo4j_result.get("triplets", [])
                        all_triplets.extend(triplets)
                        debug(f"知识库 {kb_id} 匹配到 {len(triplets)} 个三元组: {triplets[:5]}")
                    else:
                        error(
                            f"Neo4j 三元组匹配失败: knowledge_base_id={kb_id}, 错误: {neo4j_result.get('message', '未知错误')}")
                except Exception as e:
                    error(f"Neo4j 三元组匹配失败: knowledge_base_id={kb_id}, 错误: {str(e)}")
                    continue
            timing_stats["triplet_matching"] = time.time() - triplet_match_start
            debug(f"三元组匹配总耗时: {timing_stats['triplet_matching']:.3f} 秒")

            # 拼接三元组文本
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

            # 将拼接文本转换为向量
            vector_start = time.time()
            query_vector = await self._get_embeddings([combined_text])
            if not query_vector or not all(len(vec) == 1024 for vec in query_vector):
                raise ValueError("查询向量必须是长度为 1024 的浮点数列表")
            query_vector = query_vector[0]  # 取第一个向量
            timing_stats["vector_generation"] = time.time() - vector_start
            debug(f"生成查询向量耗时: {timing_stats['vector_generation']:.3f} 秒")

            # 调用融合搜索端点
            search_start = time.time()
            result = await self._make_api_request("searchquery", {
                "query_vector": query_vector.tolist(),
                "userid": userid,
                "knowledge_base_ids": knowledge_base_ids,
                "limit": limit,
                "offset": offset,
                "db_type": db_type
            })
            timing_stats["vector_search"] = time.time() - search_start
            debug(f"向量搜索耗时: {timing_stats['vector_search']:.3f} 秒")

            if result.get("status") != "success":
                error(f"融合搜索失败: {result.get('message', '未知错误')}")
                return {"results": [], "timing": timing_stats}

            unique_results = result.get("results", [])
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
            error(f"融合搜索失败: {str(e)}, 堆栈: {traceback.format_exc()}")
            return {"results": [], "timing": timing_stats}

    async def _list_user_files(self, userid: str, db_type: str = "") -> Dict[str, List[Dict]]:
        """列出用户的所有知识库及其文件，按 knowledge_base_id 分组"""
        collection_name = "ragdb" if not db_type else f"ragdb_{db_type}"
        try:
            info(f"列出用户文件: userid={userid}, db_type={db_type}")

            if not userid:
                raise ValueError("userid 不能为空")
            if (db_type and len(db_type) > 100) or len(userid) > 100:
                raise ValueError("userid 或 db_type 的长度超出限制")

            # 调用列出用户文件端点
            result = await self._make_api_request("listuserfiles", {
                "userid": userid,
                "db_type": db_type
            })

            if result.get("status") != "success":
                error(f"列出用户文件失败: {result.get('message', '未知错误')}")
                return {}

            return result.get("files_by_knowledge_base", {})

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

            # 调用列出所有知识库端点
            result = await self._make_api_request("listallknowledgebases", {
                "db_type": db_type
            })

            return result

        except Exception as e:
            error(f"列出所有用户知识库失败: {str(e)}")
            return {
                "status": "error",
                "users_knowledge_bases": {},
                "collection_name": collection_name,
                "message": f"列出所有用户知识库失败: {str(e)}",
                "status_code": 400
            }

connection_register('Rag', MilvusConnection)
info("MilvusConnection registered")