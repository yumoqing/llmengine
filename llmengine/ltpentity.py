from ltp import LTP
from typing import List
from appPublic.log import debug, info, error
from appPublic.worker import awaitify
from llmengine.base_entity import BaseLtp, ltp_register
import asyncio

class LtpEntity(BaseLtp):
    def __init__(self, model_id):
        # Load LTP model for CWS, POS, and NER
        self.ltp = LTP(model_id)
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]

    async def extract_entities(self, query: str) -> List[str]:
        """
        从查询文本中抽取实体，包括：
        - LTP NER 识别的实体（所有类型）。
        - LTP POS 标注为名词（'n'）的词。
        - LTP POS 标注为动词（'v'）的词。
        - 连续名词合并（如 '苹果 公司' -> '苹果公司'），移除子词。
        """
        try:
            if not query:
                raise ValueError("查询文本不能为空")

            # 定义同步 pipeline 函数，正确传递 tasks 参数
            def sync_pipeline(query, tasks):
                return self.ltp.pipeline([query], tasks=tasks)

            # 使用 run_in_executor 运行同步 pipeline
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: sync_pipeline(query, ["cws", "pos", "ner"])
            )

            # 解析结果
            words = result.cws[0]
            pos_list = result.pos[0]
            ner = result.ner[0]

            entities = []
            subword_set = set()

            debug(f"NER 结果: {ner}")
            for entity_type, entity, start, end in ner:
                entities.append(entity)

            combined = ""
            combined_words = []
            for i in range(len(words)):
                if pos_list[i] == 'n':
                    combined += words[i]
                    combined_words.append(words[i])
                    if i + 1 < len(words) and pos_list[i + 1] == 'n':
                        continue
                    if combined:
                        entities.append(combined)
                        subword_set.update(combined_words)
                        debug(f"合并连续名词: {combined}, 子词: {combined_words}")
                        combined = ""
                        combined_words = []
                else:
                    combined = ""
                    combined_words = []
            debug(f"连续名词子词集合: {subword_set}")

            for word, pos in zip(words, pos_list):
                if pos == 'n' and word not in subword_set:
                    entities.append(word)

            for word, pos in zip(words, pos_list):
                if pos == 'v':
                    entities.append(word)

            unique_entities = list(dict.fromkeys(entities))
            info(f"从查询中提取到 {len(unique_entities)} 个唯一实体: {unique_entities}")
            return unique_entities

        except Exception as e:
            error(f"实体抽取失败: {str(e)}")
            raise  # 抛出异常以便调试，而不是返回空列表

ltp_register('LTP', LtpEntity)