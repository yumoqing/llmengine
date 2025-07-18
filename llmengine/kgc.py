import os
import re
from py2neo import Graph, Node, Relationship
from typing import Set, List, Dict, Tuple
from appPublic.jsonConfig import getConfig
from appPublic.log import debug, error, info


class KnowledgeGraph:
    def __init__(self, triples: List[Dict], document_id: str, knowledge_base_id: str, userid: str):
        self.triples = triples
        self.document_id = document_id
        self.knowledge_base_id = knowledge_base_id
        self.userid = userid
        config = getConfig()
        self.neo4j_uri = config['neo4j']['uri']
        self.neo4j_user = config['neo4j']['user']
        self.neo4j_password = config['neo4j']['password']
        self.g = Graph(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        info(f"开始构建知识图谱，document_id: {self.document_id}, knowledge_base_id: {self.knowledge_base_id}, userid: {self.userid}, 三元组数量: {len(triples)}")

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

    def read_nodes(self) -> Tuple[Dict[str, Set], Dict[str, List], List[Dict]]:
        """从三元组列表中读取节点和关系"""
        nodes_by_label = {}
        relations_by_type = {}
        triples = []

        try:
            for triple in self.triples:
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

                triples.append({
                    'head': head,
                    'relation': relation,
                    'tail': tail,
                    'head_type': head_type,
                    'tail_type': tail_type
                })

            info(f"读取节点: {sum(len(nodes) for nodes in nodes_by_label.values())} 个")
            info(f"读取关系: {sum(len(rels) for rels in relations_by_type.values())} 条")
            return nodes_by_label, relations_by_type, triples

        except Exception as e:
            error(f"读取三元组失败: {str(e)}")
            raise RuntimeError(f"读取三元组失败: {str(e)}")

    def create_node(self, label: str, nodes: Set[str]):
        """创建节点，包含 document_id、knowledge_base_id 和 userid 属性"""
        count = 0
        for node_name in nodes:
            query = (
                f"MATCH (n:{label} {{name: $name, document_id: $doc_id, "
                f"knowledge_base_id: $kb_id, userid: $userid}}) RETURN n"
            )
            try:
                if self.g.run(query, name=node_name, doc_id=self.document_id,
                            kb_id=self.knowledge_base_id, userid=self.userid).data():
                    continue
                node = Node(
                    label,
                    name=node_name,
                    document_id=self.document_id,
                    knowledge_base_id=self.knowledge_base_id,
                    userid=self.userid
                )
                self.g.create(node)
                count += 1
                debug(f"创建节点: {label} - {node_name} (document_id: {self.document_id}, "
                      f"knowledge_base_id: {self.knowledge_base_id}, userid: {self.userid})")
            except Exception as e:
                error(f"创建节点失败: {label} - {node_name}, 错误: {str(e)}")
        info(f"创建 {label} 节点: {count}/{len(nodes)} 个")
        return count

    def create_relationship(self, rel_type: str, relations: List[Dict]):
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
                          doc_id=self.document_id, kb_id=self.knowledge_base_id,
                          userid=self.userid)
                count += 1
                debug(f"创建关系: {head} -[{rel_type}]-> {tail} (document_id: {self.document_id}, "
                      f"knowledge_base_id: {self.knowledge_base_id}, userid: {self.userid})")
            except Exception as e:
                error(f"创建关系失败: {query}, 错误: {str(e)}")
        info(f"创建 {rel_type} 关系: {count}/{total} 条")
        return count

    def create_graphnodes(self):
        """创建所有节点"""
        nodes_by_label, _, _ = self.read_nodes()
        total = 0
        for label, nodes in nodes_by_label.items():
            total += self.create_node(label, nodes)
        info(f"总计创建节点: {total} 个")
        return total

    def create_graphrels(self):
        """创建所有关系"""
        _, relations_by_type, _ = self.read_nodes()
        total = 0
        for rel_type, relations in relations_by_type.items():
            total += self.create_relationship(rel_type, relations)
        info(f"总计创建关系: {total} 条")
        return total

    def export_data(self):
        """导出节点到文件，包含 document_id、knowledge_base_id 和 userid"""
        nodes_by_label, _, _ = self.read_nodes()
        os.makedirs('dict', exist_ok=True)
        for label, nodes in nodes_by_label.items():
            with open(f'dict/{label.lower()}.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(f"{name}\t{self.document_id}\t{self.knowledge_base_id}\t{self.userid}"
                                 for name in sorted(nodes)))
            info(f"导出 {label} 节点到 dict/{label.lower()}.txt: {len(nodes)} 个")
        return