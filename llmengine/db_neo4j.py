from typing import Dict, List, Any, Set, Tuple
from appPublic.jsonConfig import getConfig
from appPublic.log import debug, error, info
from py2neo import Graph, Node, Relationship
import re
from llmengine.base_db import BaseDBConnection, connection_register
from threading import Lock
from scipy.spatial.distance import cosine
import numpy as np
import aiohttp
import asyncio
import time

# 嵌入缓存
EMBED_CACHE = {}

class Neo4jConnection(BaseDBConnection):
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Neo4jConnection, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        try:
            config = getConfig()
            self.neo4j_uri = config['neo4j']['uri']
            self.neo4j_user = config['neo4j']['user']
            self.neo4j_password = config['neo4j']['password']
            self.g = Graph(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            info("Neo4jConnection initialized")
            self._initialized = True
        except KeyError as e:
            error(f"配置文件缺少必要字段: {str(e)}")
            self._initialized = False
            raise RuntimeError(f"配置文件缺少必要字段: {str(e)}")
        except Exception as e:
            error(f"Neo4jConnection 初始化失败: {str(e)}")
            self._initialized = False
            raise RuntimeError(f"Neo4jConnection 初始化失败: {str(e)}")

    def _normalize_label(self, entity_type: str) -> str:
        """规范化实体类型为 Neo4j 标签"""
        if not entity_type or not entity_type.strip():
            return 'Entity'
        entity_type = re.sub(r'[^\w\s]', '', entity_type.strip())
        words = entity_type.split()
        label = '_'.join(word.capitalize() for word in words if word)
        return label or 'Entity'

    def _clean_relation(self, relation: str) -> Tuple[str, str]:
        """清洗关系，返回 (rel_type, rel_name)，确保 rel_type 合法"""
        relation = relation.strip()
        if not relation:
            return 'RELATED_TO', '相关'
        cleaned_relation = re.sub(r'[^\w\s]', '', relation).strip()
        if not cleaned_relation:
            return 'RELATED_TO', '相关'
        if 'instance of' in relation.lower():
            return 'INSTANCE_OF', '实例'
        elif 'subclass of' in relation.lower():
            return 'SUBCLASS_OF', '子类'
        elif 'part of' in relation.lower():
            return 'PART_OF', '部分'
        rel_type = re.sub(r'\s+', '_', cleaned_relation).upper()
        if rel_type and rel_type[0].isdigit():
            rel_type = f'REL_{rel_type}'
        if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', rel_type):
            debug(f"非法关系类型 '{rel_type}'，替换为 'RELATED_TO'")
            return 'RELATED_TO', relation
        return rel_type, relation

    def _read_nodes(self, triples: List[Dict]) -> Tuple[Dict[str, Set], Dict[str, List], List[Dict]]:
        """从三元组列表中读取节点和关系"""
        nodes_by_label = {}
        relations_by_type = {}
        valid_triples = []
        try:
            for triple in triples:
                if not all(key in triple for key in ['head', 'head_type', 'type', 'tail', 'tail_type']):
                    debug(f"无效三元组: {triple}")
                    continue
                head, relation, tail, head_type, tail_type = (
                    triple['head'], triple['type'], triple['tail'], triple['head_type'], triple['tail_type']
                )
                head_label = self._normalize_label(head_type)
                tail_label = self._normalize_label(tail_type)
                debug(f"实体类型: {head_type} -> {head_label}, {tail_type} -> {tail_label}")
                if head_label not in nodes_by_label:
                    nodes_by_label[head_label] = set()
                if tail_label not in nodes_by_label:
                    nodes_by_label[tail_label] = set()
                nodes_by_label[head_label].add(head)
                nodes_by_label[tail_label].add(tail)
                rel_type, rel_name = self._clean_relation(relation)
                if rel_type not in relations_by_type:
                    relations_by_type[rel_type] = []
                relations_by_type[rel_type].append({
                    'head': head,
                    'tail': tail,
                    'head_label': head_label,
                    'tail_label': tail_label,
                    'rel_name': rel_name
                })
                valid_triples.append({
                    'head': head,
                    'relation': relation,
                    'tail': tail,
                    'head_type': head_type,
                    'tail_type': tail_type
                })
            info(f"读取节点: {sum(len(nodes) for nodes in nodes_by_label.values())} 个")
            info(f"读取关系: {sum(len(rels) for rels in relations_by_type.values())} 条")
            return nodes_by_label, relations_by_type, valid_triples
        except Exception as e:
            error(f"读取三元组失败: {str(e)}")
            raise RuntimeError(f"读取三元组失败: {str(e)}")

    def _create_node(self, label: str, nodes: Set[str], document_id: str, knowledge_base_id: str, userid: str) -> int:
        """创建节点，包含 document_id、knowledge_base_id 和 userid 属性"""
        count = 0
        for node_name in nodes:
            query = (
                f"MATCH (n:{label} {{name: $name, document_id: $doc_id, "
                f"knowledge_base_id: $kb_id, userid: $userid}}) RETURN n"
            )
            try:
                if self.g.run(query, name=node_name, doc_id=document_id,
                            kb_id=knowledge_base_id, userid=userid).data():
                    continue
                node = Node(
                    label,
                    name=node_name,
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    userid=userid
                )
                self.g.create(node)
                count += 1
                debug(f"创建节点: {label} - {node_name} (document_id: {document_id}, "
                      f"knowledge_base_id: {knowledge_base_id}, userid: {userid})")
            except Exception as e:
                error(f"创建节点失败: {label} - {node_name}, 错误: {str(e)}")
        info(f"创建 {label} 节点: {count}/{len(nodes)} 个")
        return count

    def _create_relationship(self, rel_type: str, relations: List[Dict], document_id: str, knowledge_base_id: str, userid: str) -> int:
        """创建关系，包含 document_id、knowledge_base_id 和 userid 属性"""
        count = 0
        total = len(relations)
        seen_edges = set()
        for rel in relations:
            head, tail, head_label, tail_label, rel_name = (
                rel['head'], rel['tail'], rel['head_label'], rel['tail_label'], rel['rel_name']
            )
            edge_key = f"{head_label}:{head}###{tail_label}:{tail}###{rel_type}"
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            query = (
                f"MATCH (p:{head_label} {{name: $head, document_id: $doc_id, "
                f"knowledge_base_id: $kb_id, userid: $userid}}), "
                f"(q:{tail_label} {{name: $tail, document_id: $doc_id, "
                f"knowledge_base_id: $kb_id, userid: $userid}}) "
                f"CREATE (p)-[r:{rel_type} {{name: $rel_name, document_id: $doc_id, "
                f"knowledge_base_id: $kb_id, userid: $userid}}]->(q)"
            )
            try:
                self.g.run(query, head=head, tail=tail, rel_name=rel_name,
                          doc_id=document_id, kb_id=knowledge_base_id,
                          userid=userid)
                count += 1
                debug(f"创建关系: {head} -[{rel_type}]-> {tail} (document_id: {document_id}, "
                      f"knowledge_base_id: {knowledge_base_id}, userid: {userid})")
            except Exception as e:
                error(f"创建关系失败: {query}, 错误: {str(e)}")
        info(f"创建 {rel_type} 关系: {count}/{total} 条")
        return count

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """调用嵌入服务获取文本的向量，带缓存"""
        try:
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
            return [EMBED_CACHE[text] for text in texts]
        except Exception as e:
            error(f"嵌入服务调用失败: {str(e)}")
            raise RuntimeError(f"嵌入服务调用失败: {str(e)}")

    async def handle_connection(self, action: str, params: Dict = None) -> Dict:
        """处理 Neo4j 数据库操作"""
        if not params:
            params = {}
        try:
            debug(f"处理 Neo4j 操作: action={action}, params={params}")
            collection_name = "neo4j"  # 为了与 MilvusConnection 的返回格式一致
            document_id = params.get("document_id", "")
            if not self._initialized:
                error("Neo4j 服务未启动")
                return {
                    "status": "error",
                    "message": "Neo4j 服务未启动",
                    "collection_name": collection_name,
                    "document_id": document_id,
                    "status_code": 400
                }
            if action == "initialize":
                return {
                    "status": "success",
                    "message": "Neo4j 服务已初始化",
                    "collection_name": collection_name,
                    "document_id": document_id,
                    "status_code": 200
                }
            elif action == "insert_triples":
                triples = params.get("triples", [])
                document_id = params.get("document_id", "")
                knowledge_base_id = params.get("knowledge_base_id", "")
                userid = params.get("userid", "")
                if not triples or not document_id or not knowledge_base_id or not userid:
                    return {
                        "status": "error",
                        "message": "triples、document_id、knowledge_base_id 和 userid 不能为空",
                        "collection_name": collection_name,
                        "document_id": document_id,
                        "status_code": 400
                    }
                return self._insert_triples(triples, document_id, knowledge_base_id, userid)
            elif action == "delete_document":
                document_id = params.get("document_id", "")
                if not document_id:
                    return {
                        "status": "error",
                        "message": "document_id 不能为空",
                        "collection_name": collection_name,
                        "document_id": document_id,
                        "status_code": 400
                    }
                return self._delete_document(document_id)
            elif action == "delete_knowledge_base":
                userid = params.get("userid", "")
                knowledge_base_id = params.get("knowledge_base_id", "")
                if not userid or not knowledge_base_id:
                    return {
                        "status": "error",
                        "message": "userid 和 knowledge_base_id 不能为空",
                        "collection_name": collection_name,
                        "document_id": document_id,
                        "status_code": 400
                    }
                return self._delete_knowledge_base(userid, knowledge_base_id)
            elif action == "match_triplets":
                query = params.get("query", "")
                query_entities = params.get("query_entities", [])
                userid = params.get("userid", "")
                knowledge_base_id = params.get("knowledge_base_id", "")
                if not query or not query_entities or not userid or not knowledge_base_id:
                    return {
                        "status": "error",
                        "message": "query、query_entities、userid 和 knowledge_base_id 不能为空",
                        "collection_name": collection_name,
                        "document_id": document_id,
                        "status_code": 400
                    }
                return await self._match_triplets(query, query_entities, userid, knowledge_base_id)
            else:
                return {
                    "status": "error",
                    "message": f"未知的 action: {action}",
                    "collection_name": collection_name,
                    "document_id": document_id,
                    "status_code": 400
                }
        except Exception as e:
            error(f"处理 Neo4j 操作失败: action={action}, 错误: {str(e)}")
            return {
                "status": "error",
                "message": f"服务器错误: {str(e)}",
                "collection_name": collection_name,
                "document_id": document_id,
                "status_code": 400
            }

    def _insert_triples(self, triples: List[Dict], document_id: str, knowledge_base_id: str, userid: str) -> Dict:
        """插入三元组到 Neo4j"""
        try:
            if not triples:
                return {
                    "status": "success",
                    "message": "没有三元组需要插入",
                    "nodes_created": 0,
                    "rels_created": 0,
                    "collection_name": "neo4j",
                    "document_id": document_id,
                    "status_code": 200
                }
            nodes_by_label, relations_by_type, _ = self._read_nodes(triples)
            total_nodes = 0
            for label, nodes in nodes_by_label.items():
                total_nodes += self._create_node(label, nodes, document_id, knowledge_base_id, userid)
            total_rels = 0
            for rel_type, relations in relations_by_type.items():
                total_rels += self._create_relationship(rel_type, relations, document_id, knowledge_base_id, userid)
            info(f"成功插入 {total_nodes} 个节点和 {total_rels} 个关系")
            return {
                "status": "success",
                "message": f"成功插入 {total_nodes} 个节点和 {total_rels} 个关系",
                "nodes_created": total_nodes,
                "rels_created": total_rels,
                "collection_name": "neo4j",
                "document_id": document_id,
                "status_code": 200
            }
        except Exception as e:
            error(f"插入三元组失败: {str(e)}")
            return {
                "status": "error",
                "message": f"插入三元组失败: {str(e)}",
                "collection_name": "neo4j",
                "document_id": document_id,
                "status_code": 400
            }

    def _delete_document(self, document_id: str) -> Dict:
        """删除 Neo4j 中指定文档的数据"""
        try:
            query = """
            MATCH (n {document_id: $document_id})
            OPTIONAL MATCH (n)-[r {document_id: $document_id}]->()
            WITH collect(r) AS rels, collect(n) AS nodes
            FOREACH (r IN rels | DELETE r)
            FOREACH (n IN nodes | DELETE n)
            RETURN size(nodes) AS node_count, size(rels) AS rel_count, [r IN rels | type(r)] AS rel_types
            """
            result = self.g.run(query, document_id=document_id).data()
            nodes_deleted = result[0]['node_count'] if result else 0
            rels_deleted = result[0]['rel_count'] if result else 0
            rel_types = result[0]['rel_types'] if result else []
            info(f"成功删除 document_id={document_id} 的 {nodes_deleted} 个节点和 {rels_deleted} 个关系，关系类型: {rel_types}")
            return {
                "status": "success",
                "message": f"成功删除 {nodes_deleted} 个节点和 {rels_deleted} 个关系",
                "nodes_deleted": nodes_deleted,
                "rels_deleted": rels_deleted,
                "collection_name": "neo4j",
                "document_id": document_id,
                "status_code": 200
            }
        except Exception as e:
            error(f"删除 document_id={document_id} 的 Neo4j 数据失败: {str(e)}")
            return {
                "status": "error",
                "message": f"删除 Neo4j 数据失败: {str(e)}",
                "collection_name": "neo4j",
                "document_id": document_id,
                "status_code": 400
            }

    def _delete_knowledge_base(self, userid: str, knowledge_base_id: str) -> Dict:
        """删除 Neo4j 中指定用户和知识库的数据"""
        try:
            query = """
            MATCH (n {userid: $userid, knowledge_base_id: $knowledge_base_id})
            OPTIONAL MATCH (n)-[r {userid: $userid, knowledge_base_id: $knowledge_base_id}]->()
            WITH collect(r) AS rels, collect(n) AS nodes
            FOREACH (r IN rels | DELETE r)
            FOREACH (n IN nodes | DELETE n)
            RETURN size(nodes) AS node_count, size(rels) AS rel_count, [r IN rels | type(r)] AS rel_types
            """
            result = self.g.run(query, userid=userid, knowledge_base_id=knowledge_base_id).data()
            nodes_deleted = result[0]['node_count'] if result else 0
            rels_deleted = result[0]['rel_count'] if result else 0
            rel_types = result[0]['rel_types'] if result else []
            info(f"成功删除 userid={userid}, knowledge_base_id={knowledge_base_id} 的 {nodes_deleted} 个节点和 {rels_deleted} 个关系")
            return {
                "status": "success",
                "message": f"成功删除 {nodes_deleted} 个节点和 {rels_deleted} 个关系",
                "nodes_deleted": nodes_deleted,
                "rels_deleted": rels_deleted,
                "collection_name": "neo4j",
                "document_id": "",
                "status_code": 200
            }
        except Exception as e:
            error(f"删除 Neo4j 数据失败: {str(e)}")
            return {
                "status": "error",
                "message": f"删除 Neo4j 数据失败: {str(e)}",
                "collection_name": "neo4j",
                "document_id": "",
                "status_code": 400
            }

    async def _match_triplets(self, query: str, query_entities: List[str], userid: str, knowledge_base_id: str) -> Dict:
        """匹配查询实体与 Neo4j 中的三元组"""
        start_time = time.time()
        timing_stats = {}
        matched_triplets = []
        ENTITY_SIMILARITY_THRESHOLD = 0.8
        try:
            debug(f"已连接到 Neo4j: {self.neo4j_uri}")
            timing_stats["neo4j_connect"] = time.time() - start_time

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
                    results = self.g.run(query, userid=userid, knowledge_base_id=knowledge_base_id, entity=normalized_entity).data()
                    for record in results:
                        matched_names.add(record['n.name'])
                        debug(f"实体 {entity} 匹配节点: {record['n.name']} (Levenshtein 相似度: {record['sim']:.2f})")
                except Exception as e:
                    debug(f"模糊匹配实体 {entity} 失败: {str(e)}")
                    continue
            timing_stats["entity_match"] = time.time() - entity_match_start
            debug(f"实体匹配耗时: {timing_stats['entity_match']:.3f} 秒")

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
                    results = self.g.run(query, userid=userid, knowledge_base_id=knowledge_base_id, matched_names=list(matched_names)).data()
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
                    return {
                        "status": "error",
                        "message": f"检索三元组失败: {str(e)}",
                        "triplets": [],
                        "timing": timing_stats,
                        "collection_name": "neo4j",
                        "document_id": "",
                        "status_code": 400
                    }
                timing_stats["triplet_query"] = time.time() - triplet_query_start
                debug(f"Neo4j 三元组查询耗时: {timing_stats['triplet_query']:.3f} 秒")

            if not triplets:
                debug(f"知识库 knowledge_base_id={knowledge_base_id} 无匹配三元组")
                return {
                    "status": "success",
                    "message": f"知识库 knowledge_base_id={knowledge_base_id} 无匹配三元组",
                    "triplets": [],
                    "timing": timing_stats,
                    "collection_name": "neo4j",
                    "document_id": "",
                    "status_code": 200
                }

            embedding_start = time.time()
            texts_to_embed = query_entities + [t['head'] for t in triplets] + [t['tail'] for t in triplets]
            embeddings = await self._get_embeddings(texts_to_embed)
            entity_vectors = {entity: embeddings[i] for i, entity in enumerate(query_entities)}
            head_vectors = {t['head']: embeddings[len(query_entities) + i] for i, t in enumerate(triplets)}
            tail_vectors = {t['tail']: embeddings[len(query_entities) + len(triplets) + i] for i, t in enumerate(triplets)}
            debug(f"成功获取 {len(embeddings)} 个嵌入向量（{len(query_entities)} entities + {len(triplets)} heads + {len(triplets)} tails）")
            timing_stats["embedding"] = time.time() - embedding_start
            debug(f"嵌入向量生成耗时: {timing_stats['embedding']:.3f} 秒")

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
            timing_stats["similarity"] = time.time() - similarity_start
            debug(f"相似度计算耗时: {timing_stats['similarity']:.3f} 秒")

            unique_matched = []
            seen = set()
            for t in matched_triplets:
                identifier = (t['head'].lower(), t['type'].lower(), t['tail'].lower())
                if identifier not in seen:
                    seen.add(identifier)
                    unique_matched.append(t)

            timing_stats["total_time"] = time.time() - start_time
            info(f"找到 {len(unique_matched)} 个匹配的三元组，总耗时: {timing_stats['total_time']:.3f} 秒")
            return {
                "status": "success",
                "message": f"找到 {len(unique_matched)} 个匹配的三元组",
                "triplets": unique_matched,
                "timing": timing_stats,
                "collection_name": "neo4j",
                "document_id": "",
                "status_code": 200
            }

        except Exception as e:
            error(f"匹配三元组失败: {str(e)}")
            timing_stats["total_time"] = time.time() - start_time
            return {
                "status": "error",
                "message": f"匹配三元组失败: {str(e)}",
                "triplets": [],
                "timing": timing_stats,
                "collection_name": "neo4j",
                "document_id": "",
                "status_code": 400
            }

connection_register('Neo4j', Neo4jConnection)
info("Neo4jconnection registered")