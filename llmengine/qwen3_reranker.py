import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmengine.base_reranker import BaseReranker, llm_register


class Qwen3Reranker(BaseReranker):
    def __init__(self, model_id, max_length=1024):  # 设置 max_length 为 1024
        if 'Qwen3-Reranker' not in model_id:
            raise Exception(f'{model_id} is not a Qwen3-Reranker')

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')

        # Use FP16 on GPU, FP32 on CPU
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id
            ).eval()

        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        self.max_length = max_length

        # 初始化前缀和后缀标记
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )
        return output

    def process_inputs(self, pairs, batch_size=8):
        all_inputs = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = self.tokenizer(
                batch_pairs,
                padding=False,
                truncation='longest_first',
                return_attention_mask=False,
                max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
            )
            for j, ele in enumerate(inputs['input_ids']):
                inputs['input_ids'][j] = self.prefix_tokens + ele + self.suffix_tokens
            inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                return_tensors="pt",
                max_length=self.max_length
            )
            for key in inputs:
                inputs[key] = inputs[key].to(self.model.device)
            all_inputs.append(inputs)
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return all_inputs

    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def build_pairs(self, query, docs, sys_prompt="", task=""):
        pairs = [self.format_instruction(task, query, doc) for doc in docs]
        return pairs

    def rerank(self, query, docs, top_n, sys_prompt="", task="", batch_size=8):
        if query is None:
            raise Exception("query is None")
        if docs is None or not docs:
            raise Exception("documents is None or empty")
        if not isinstance(docs, list):
            docs = [docs]
        if top_n is None or top_n <= 0:
            top_n = len(docs)

        pairs = self.build_pairs(query, docs, sys_prompt=sys_prompt, task=task)
        print(f"Number of documents: {len(docs)}")
        for i, p in enumerate(pairs):
            print(f"Pair {i} token length: {len(self.tokenizer.encode(p))}")
        scores = []
        with torch.no_grad():
            input_batches = self.process_inputs(pairs, batch_size=batch_size)
            for inputs in input_batches:
                batch_scores = self.compute_logits(inputs)
                scores.extend(batch_scores)
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        data = [
            {'index': i, 'relevance_score': s}
            for i, s in enumerate(scores)
        ]
        data = sorted(data, key=lambda x: x["relevance_score"], reverse=True)
        if len(data) > top_n:
            data = data[:top_n]

        return {
            "data": data,
            "object": "rerank.result",
            "model": self.model_name,
            "usage": {
                "prompt_tokens": sum(len(self.tokenizer.encode(p)) for p in pairs),
                "total_tokens": sum(len(self.tokenizer.encode(p)) for p in pairs)
            }
        }


llm_register('Qwen3-Reranker', Qwen3Reranker)