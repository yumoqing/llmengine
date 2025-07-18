import torch

model_pathMap = {
}


def llm_register(model_key, Klass):
    model_pathMap[model_key] = Klass


def get_llm_class(model_path):
    for k, klass in model_pathMap.items():
        if len(model_path.split(k)) > 1:
            return klass
    print(f'{model_pathMap=}')
    return None


class BaseReranker:
    def __init__(self, model_id, **kw):
        self.model_id = model_id

    def use_mps_if_prosible(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.model = self.model.to(device, dtype=torch.float16)  
        else:
            raise Exception("GPU not available, but required for FP16 inference")

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length
        )
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    def build_sys_prompt(self, sys_prompt):
        return f"<|im_start|>system\n{sys_prompt}\n<|im_end|>"

    def build_user_prompt(self, query, doc, instruct=''):
        return f'<|im_start|>user\n<Instruct>: {instruct}\n<Query>:{query}\n<Document>:\n{doc}<|im_end|>'

    def build_assistant_prompt(self):
        return "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def build_pairs(self, query, docs, sys_prompt="", task=""):
        sys_str = self.build_sys_prompt(sys_prompt)
        ass_str = self.build_assistant_prompt()
        pairs = [sys_str + '\n' + self.build_user_prompt(task, query, doc) + '\n' + ass_str for doc in docs]
        return pairs

    def rerank(self, query, docs, top_n, sys_prompt="", task=""):
        pairs = self.build_pairs(query, docs, sys_prompt=sys_prompt, task=task)
        with torch.no_grad():
            inputs = self.process_inputs(pairs)
            scores = self.compute_logits(inputs)
        data = []
        for i, s in enumerate(scores):
            d = {
                'index': i,
                'relevance_score': s
            }
            data.append(d)
        data = sorted(data,
                      key=lambda x: x["relevance_score"],
                      reverse=True)
        if len(data) > top_n:
            data = data[:top_n]
        ret = {
            "data": data,
            "object": "rerank.result",
            "model": self.model_name,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        }
        return ret